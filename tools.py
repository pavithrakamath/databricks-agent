"""
Agent tools for database operations.

Provides tools for:
- SQL query execution
- Schema and table listing
- Column search
- Natural language table search
"""

import json
import time
from typing import Optional

from security import (
    check_query_complexity,
    check_rate_limit,
    sanitize_identifier,
    sanitize_search_term,
    validate_sql_query,
)
from term_expansion import expand_search_terms, load_term_expansions
from utils import get_spark_session


def execute_sql_query(query: str, max_rows: int = 1000, session_id: str = "default") -> str:
    """
    Execute a SQL SELECT query with validation and safety checks.
    
    IMPORTANT: Only SELECT queries are allowed. DDL and DML operations are blocked.
    
    Safety features:
    - Only SELECT queries (read-only)
    - Rate limiting: Max 10 queries per minute per session
    - Query timeout: Max 30 seconds execution time
    - Complexity checks: Blocks complex queries
    - Result limits: Default 1000 rows, max 10,000
    
    Args:
        query: SQL SELECT query to execute
        max_rows: Maximum number of rows to return (default: 1000, max: 10000)
        session_id: Session identifier for rate limiting
    
    Returns:
        JSON string with query results or error information
    """
    start_time = time.time()
    max_rows = min(max_rows, 10000)
    
    # Check rate limit
    rate_allowed, rate_error = check_rate_limit(session_id)
    if not rate_allowed:
        return json.dumps({"error": rate_error, "query": query[:200]})
    
    # Validate query
    is_valid, error_msg = validate_sql_query(query)
    if not is_valid:
        return json.dumps({"error": error_msg, "query": query[:200]})
    
    # Check complexity
    complexity_allowed, complexity_error, complexity_score = check_query_complexity(query)
    if not complexity_allowed:
        return json.dumps({
            "error": complexity_error, 
            "query": query[:200], 
            "complexity_score": complexity_score
        })
    
    spark = get_spark_session()
    
    try:
        execution_start = time.time()
        df = spark.sql(query)
        rows = df.limit(max_rows).collect()
        execution_time = (time.time() - execution_start) * 1000
        
        # Check timeout
        if execution_time > 30000:
            return json.dumps({
                "error": (
                    f"Query execution exceeded timeout limit (30 seconds). "
                    f"Execution time: {execution_time/1000:.2f}s"
                ),
                "query": query[:200],
                "execution_time_ms": execution_time
            })
        
        # Convert to JSON
        if len(rows) == 0:
            return json.dumps({
                "rows": [],
                "row_count": 0,
                "columns": df.columns,
                "truncated": False,
                "execution_time_ms": execution_time,
                "complexity_score": complexity_score
            })
        
        result = {
            "rows": [dict(row.asDict()) for row in rows],
            "row_count": len(rows),
            "columns": df.columns,
            "truncated": len(rows) >= max_rows,
            "execution_time_ms": round(execution_time, 2),
            "complexity_score": complexity_score
        }
        return json.dumps(result)
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        return json.dumps({
            "error": str(e),
            "query": query[:200],
            "type": type(e).__name__,
            "execution_time_ms": execution_time
        })


def list_schemas(catalog: str = "samples") -> str:
    """List all schemas (databases) in the specified catalog."""
    catalog = sanitize_identifier(catalog)
    query = f"""
    SELECT DISTINCT table_schema as schema_name
    FROM {catalog}.information_schema.tables
    ORDER BY schema_name
    """
    
    spark = get_spark_session()
    try:
        df = spark.sql(query)
        schemas = [row.schema_name for row in df.collect()]
        return json.dumps({"schemas": schemas, "count": len(schemas), "catalog": catalog})
    except Exception as e:
        return json.dumps({"error": str(e), "type": type(e).__name__})


def list_tables(catalog: str = "samples", schema: str = "bakehouse") -> str:
    """List all tables in the specified catalog and schema."""
    catalog = sanitize_identifier(catalog)
    schema = sanitize_identifier(schema)
    
    query = f"""
    SELECT DISTINCT table_name
    FROM {catalog}.information_schema.tables
    WHERE table_schema = '{schema}'
    ORDER BY table_name
    """
    
    spark = get_spark_session()
    try:
        df = spark.sql(query)
        tables = [row.table_name for row in df.collect()]
        return json.dumps({
            "tables": tables, 
            "count": len(tables), 
            "catalog": catalog, 
            "schema": schema
        })
    except Exception as e:
        return json.dumps({"error": str(e), "type": type(e).__name__})


def find_tables_with_column(
    column_name_partial: str,
    catalog: str = "samples",
    tb_schema: str = "bakehouse"
) -> str:
    """Find tables with columns matching the partial column name."""
    catalog = sanitize_identifier(catalog)
    tb_schema = sanitize_identifier(tb_schema)
    column_name_partial = sanitize_search_term(column_name_partial)
    
    if not column_name_partial:
        return json.dumps([])
    
    query = f"""
    SELECT table_name, column_name
    FROM {catalog}.information_schema.columns
    WHERE table_schema = '{tb_schema}'
      AND column_name LIKE '%{column_name_partial}%'
    """
    
    spark = get_spark_session()
    df = spark.sql(query)
    result = [
        {"table_name": row.table_name, "column_name": row.column_name}
        for row in df.collect()
    ]
    return json.dumps(result)


def search_tables_by_description(
    description: str,
    catalog: str = "samples",
    tb_schema: Optional[str] = None,
    expansion_config_path: Optional[str] = None
) -> str:
    """
    Search for tables/columns by natural language description.
    
    Intelligently expands description into multiple related search terms.
    """
    catalog = sanitize_identifier(catalog)
    if tb_schema:
        tb_schema = sanitize_identifier(tb_schema)
    
    # Load term expansions
    try:
        term_expansions = load_term_expansions(expansion_config_path)
    except (ValueError, IOError):
        term_expansions = {}
    
    # Expand search terms
    search_terms = expand_search_terms(description, term_expansions)
    if not search_terms:
        return json.dumps([])
    
    # Build SQL query
    max_terms_in_query = 30
    terms_to_use = search_terms[:max_terms_in_query]
    escaped_terms = [term.replace("'", "''") for term in terms_to_use]
    like_conditions = " OR ".join([
        f"LOWER(column_name) LIKE '%{term}%'" for term in escaped_terms
    ])
    
    schema_filter = f"AND table_schema = '{tb_schema}'" if tb_schema else ""
    
    query = f"""
    SELECT DISTINCT table_schema as schema_name, table_name, column_name
    FROM {catalog}.information_schema.columns
    WHERE ({like_conditions})
    {schema_filter}
    ORDER BY schema_name, table_name, column_name
    """
    
    spark = get_spark_session()
    df = spark.sql(query)
    rows = df.collect()
    
    result = [
        {
            "schema_name": row.schema_name,
            "table_name": row.table_name,
            "column_name": row.column_name
        }
        for row in rows
    ]
    return json.dumps(result)


def greet_me(name: str) -> str:
    """Greet the user by name."""
    name = sanitize_search_term(name)
    return f"Hello {name}!"


def get_all_tools(include_uc_tools: Optional[bool] = None):
    """
    Get all agent tools.
    
    Args:
        include_uc_tools: If True, include Databricks UC tools.
                         If None, auto-detects based on environment.
                         If False, excludes UC tools (for standalone mode).
    
    Returns:
        List of agent tools
    """
    from environment import is_databricks_mode
    
    tools = []
    
    # Include UC tools if in Databricks mode (or explicitly requested)
    if include_uc_tools is None:
        include_uc_tools = is_databricks_mode()
    
    if include_uc_tools:
        try:
            from databricks_langchain import UCFunctionToolkit
            UC_TOOL_NAMES = ["system.ai.python_exec"]
            uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
            tools.extend(uc_toolkit.tools)
        except ImportError:
            # UC tools not available, skip them
            pass
    
    # Add standard tools
    tools.extend([
        greet_me,
        list_schemas,
        list_tables,
        find_tables_with_column,
        search_tables_by_description,
        execute_sql_query
    ])
    return tools

