"""
Configuration module for the LangGraph Agent.

Contains all configuration constants including:
- LLM endpoint/provider configuration
- System prompts
- Middleware configuration for human-in-the-loop
- Rate limiting settings
"""

import os
from typing import Any, Optional, Union

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

# ============================================================================
# LLM Configuration
# ============================================================================

# LLM Provider: "databricks", "openai", "azure_openai", "anthropic", "ollama", or None for auto-detect
LLM_PROVIDER: Optional[str] = os.getenv("LLM_PROVIDER", None)

# LLM Endpoint/Model Name
# For Databricks: endpoint name (e.g., "my-test-endpoint")
# For OpenAI/Anthropic: model name (e.g., "gpt-4", "claude-3-5-sonnet-20241022")
# For Azure OpenAI: deployment name (e.g., "gpt-4-deployment")
LLM_ENDPOINT_NAME = os.getenv("LLM_ENDPOINT_NAME", "my-test-endpoint")
# TODO: Replace with your model serving endpoint
# LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"  # For Databricks
# LLM_ENDPOINT_NAME = "gpt-4"  # For OpenAI
# LLM_ENDPOINT_NAME = "gpt-4-deployment"  # For Azure OpenAI
# LLM_ENDPOINT_NAME = "claude-3-5-sonnet-20241022"  # For Anthropic

# Azure OpenAI Configuration (only used when LLM_PROVIDER="azure_openai")
# These can also be set via environment variables:
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_DEPLOYMENT_NAME (or use LLM_ENDPOINT_NAME)
# - AZURE_OPENAI_API_KEY
# - AZURE_OPENAI_API_VERSION (defaults to "2023-05-15")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", None)
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", None)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", None)
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

# Execution Mode: "databricks", "standalone", or "auto" (default: auto-detect)
EXECUTION_MODE = os.getenv("AGENT_EXECUTION_MODE", "auto")


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are a helpful data scientist assistant that can run Python and SQL code and figure out questions asked about different schemas and columns. 

When users ask about database structure (schemas, tables, columns), you MUST use the available tools:
- Use `list_schemas` to list all schemas in a catalog
- Use `list_tables` to list tables in a schema
- Use `search_tables_by_description` to find tables by description
- Use `find_tables_with_column` to find tables with specific columns
- Use `execute_sql_query` to run custom SELECT queries

You can infer columns and use tools to check if there are any matching columns. e.g., if the user asks about the "customer" column, you can infer that the user is asking about the "customer_id", "cust_id", "cust_no", etc. column.

Always use tools to answer questions about database structure - never say you don't have access."""


# ============================================================================
# Middleware Configuration for Human-in-the-Loop
# ============================================================================

# Format: {tool_name: {"allowed_decisions": ["approve", "edit", "reject"], "required": True/False}}
MIDDLEWARE_CONFIG: dict[str, Union[bool, dict[str, Any]]] = {
    # Require approval for execute_sql_query - SQL must be approved before execution
    "execute_sql_query": {
        "allowed_decisions": ["approve", "edit", "reject"],
        "required": True
    },
    # No approval needed for read-only tools
    "list_schemas": False,
    "list_tables": False,
    "find_tables_with_column": False,
    "search_tables_by_description": False,
    "greet_me": False,
}


# ============================================================================
# Rate Limiting Configuration
# ============================================================================

MAX_QUERIES_PER_MINUTE = 10
RATE_LIMIT_WINDOW = 60  # seconds

