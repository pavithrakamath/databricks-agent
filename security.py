"""
Security and validation utilities for SQL query safety.

Provides functions for:
- SQL identifier sanitization
- Search term sanitization
- SQL query validation
- Query complexity checking
- Rate limiting
"""

import re
import time
from collections import defaultdict
from threading import Lock
from typing import Optional

from config import MAX_QUERIES_PER_MINUTE, RATE_LIMIT_WINDOW

# Module-level state for rate limiting
_query_rate_limiter: dict[str, list[float]] = defaultdict(list)
_rate_limit_lock = Lock()


def sanitize_identifier(identifier: str) -> str:
    """Sanitize SQL identifier to prevent SQL injection."""
    if not identifier:
        raise ValueError("Identifier cannot be empty")
    if not re.match(r'^[a-zA-Z0-9_-]+$', identifier):
        raise ValueError(
            f"Invalid identifier: {identifier}. "
            "Only alphanumeric characters, underscores, and hyphens are allowed."
        )
    return identifier


def sanitize_search_term(term: str) -> str:
    """Sanitize search term to prevent SQL injection."""
    term = re.sub(r"[';]|--", "", term)
    term = re.sub(r"[^a-zA-Z0-9\s-]", "", term)
    return term.strip()


def check_query_complexity(query: str) -> tuple[bool, str, int]:
    """Check query complexity to prevent long-running queries."""
    query_upper = query.upper()
    complexity_score = 0
    
    # Count JOINs
    join_count = len(re.findall(r'\bJOIN\b', query_upper))
    complexity_score += join_count * 15
    
    # Count subqueries
    subquery_count = query_upper.count('(SELECT')
    complexity_score += subquery_count * 20
    
    # Count aggregate functions
    aggregate_functions = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP BY', 'HAVING']
    for func in aggregate_functions:
        if re.search(rf'\b{func}\b', query_upper):
            complexity_score += 10
    
    # Count window functions
    window_functions = ['ROW_NUMBER', 'RANK', 'DENSE_RANK', 'LAG', 'LEAD', 'OVER']
    for func in window_functions:
        if re.search(rf'\b{func}\b', query_upper):
            complexity_score += 15
    
    # Count UNION/UNION ALL
    union_count = len(re.findall(r'\bUNION\s+(?:ALL\s+)?SELECT\b', query_upper))
    complexity_score += union_count * 25
    
    # Check for SELECT *
    if re.search(r'\bSELECT\s+\*', query_upper):
        complexity_score += 10
    
    # Check for ORDER BY without LIMIT
    if re.search(r'\bORDER\s+BY\b', query_upper) and not re.search(r'\bLIMIT\b', query_upper):
        complexity_score += 15
    
    # Block queries that are too complex
    if complexity_score > 50:
        return False, (
            f"Query is too complex (complexity score: {complexity_score}/100). "
            "Queries with multiple JOINs, subqueries, or heavy aggregates are not allowed. "
            "Please simplify your query or use LIMIT to reduce result size."
        ), complexity_score
    
    return True, "", complexity_score


def check_rate_limit(session_id: str = "default") -> tuple[bool, str]:
    """Check if query rate limit is exceeded."""
    with _rate_limit_lock:
        current_time = time.time()
        
        # Clean old entries
        _query_rate_limiter[session_id] = [
            timestamp for timestamp in _query_rate_limiter[session_id]
            if current_time - timestamp < RATE_LIMIT_WINDOW
        ]
        
        # Check if limit exceeded
        if len(_query_rate_limiter[session_id]) >= MAX_QUERIES_PER_MINUTE:
            return False, (
                f"Rate limit exceeded. Maximum {MAX_QUERIES_PER_MINUTE} queries per minute allowed. "
                "Please wait before executing another query."
            )
        
        # Record this query
        _query_rate_limiter[session_id].append(current_time)
        return True, ""


def validate_sql_query(query: str) -> tuple[bool, str]:
    """Validate SQL query for safety - only SELECT queries allowed."""
    if not query or not isinstance(query, str):
        return False, "Query must be a non-empty string"
    
    query_upper = query.upper().strip()
    
    # Only allow SELECT statements
    if not query_upper.startswith("SELECT"):
        return False, (
            "Only SELECT queries are allowed. "
            "DDL (CREATE, DROP, ALTER) and DML (INSERT, UPDATE, DELETE) operations are not permitted."
        )
    
    # Block DDL operations
    ddl_keywords = [
        "CREATE", "DROP", "ALTER", "TRUNCATE", "RENAME", 
        "COMMENT", "GRANT", "REVOKE", "ANALYZE", "EXPLAIN"
    ]
    
    # Block DML operations
    dml_keywords = ["INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT"]
    
    # Block other dangerous operations
    other_dangerous = ["EXEC", "EXECUTE", "CALL", "SHOW", "DESCRIBE", "DESC"]
    
    # Check for blocked keywords
    for keyword in ddl_keywords + dml_keywords + other_dangerous:
        if re.search(rf'\b{keyword}\b', query_upper):
            return False, (
                f"Operation '{keyword}' is not allowed. "
                "Only SELECT queries are permitted."
            )
    
    # Check for semicolons (potential SQL injection)
    if ";" in query:
        if query.count(";") > 1 or (query.count(";") == 1 and not query.rstrip().endswith(";")):
            return False, (
                "Multiple statements or unexpected semicolons detected. "
                "Only single SELECT statements are allowed."
            )
    
    # Limit query length
    if len(query) > 10000:
        return False, "Query is too long. Maximum length is 10,000 characters."
    
    return True, ""

