import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict, Union

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
)
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

############################################
# Define your LLM endpoint and system prompt
############################################
# TODO: Replace with your model serving endpoint
# LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
LLM_ENDPOINT_NAME = "my-test-endpoint"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# TODO: Update with your system prompt
system_prompt = """You are a helpful data scientist assistant that can run Python and SQL code and figure out questions asked about different schemas and columns. You can infer columns and use tools to check if there are any matching columns. e.g., if the user asks about the "customer" column, you can infer that the user is asking about the "customer_id", "cust_id", "cust_no", etc. column."""

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################
tools = []

def _sanitize_identifier(identifier: str) -> str:
    """
    Sanitize SQL identifier (catalog, schema, table names) to prevent SQL injection.
    Only allows alphanumeric characters, underscores, and hyphens.
    
    Args:
        identifier: SQL identifier to sanitize
    
    Returns:
        Sanitized identifier
    
    Raises:
        ValueError: If identifier contains invalid characters
    """
    if not identifier:
        raise ValueError("Identifier cannot be empty")
    
    # Only allow alphanumeric, underscore, and hyphen
    if not re.match(r'^[a-zA-Z0-9_-]+$', identifier):
        raise ValueError(f"Invalid identifier: {identifier}. Only alphanumeric characters, underscores, and hyphens are allowed.")
    
    return identifier


def _sanitize_search_term(term: str) -> str:
    """
    Sanitize search term to prevent SQL injection.
    Removes or escapes special SQL characters.
    
    Args:
        term: Search term to sanitize
    
    Returns:
        Sanitized search term
    """
    # Remove SQL injection characters: quotes, semicolons, comments
    # Escape dash to avoid FutureWarning about set difference
    term = re.sub(r"[';]|--", "", term)
    # Remove any remaining non-alphanumeric except spaces and hyphens
    term = re.sub(r"[^a-zA-Z0-9\s-]", "", term)
    return term.strip()


def greet_me(name: str) -> str:
    """Greets the user by name"""
    # Sanitize input to prevent injection
    name = _sanitize_search_term(name)
    return f"Hello {name}!"

def find_tables_with_column(
    column_name_partial: str,
    catalog: str = "samples",
    tb_schema: str = "bakehouse"
) -> str:
    """
    Returns a JSON string of tables in the specified catalog and tb_schema
    that have a column name containing the given partial string.
    """
    # Sanitize inputs to prevent SQL injection
    catalog = _sanitize_identifier(catalog)
    tb_schema = _sanitize_identifier(tb_schema)
    column_name_partial = _sanitize_search_term(column_name_partial)
    
    if not column_name_partial:
        return json.dumps([])
    
    query = f"""
    SELECT table_name, column_name
    FROM {catalog}.information_schema.columns
    WHERE table_schema = '{tb_schema}'
      AND column_name LIKE '%{column_name_partial}%'
    """
    # Use DatabricksSession to avoid warnings in Databricks notebooks
    try:
        from databricks.connect import DatabricksSession
        spark = DatabricksSession.builder.getOrCreate()
    except (ImportError, AttributeError):
        # Fallback to SparkSession if DatabricksSession is not available
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
    df = spark.sql(query)
    result = [
        {"table_name": row.table_name, "column_name": row.column_name}
        for row in df.collect()
    ]
    return json.dumps(result)


# Module-level cache for term expansions
_term_expansions_cache: Optional[dict[str, list[str]]] = None
_default_config_path = Path(__file__).parent / "term_expansions.json"


def _load_term_expansions(config_path: Optional[str] = None) -> dict[str, list[str]]:
    """
    Load term expansions from a JSON config file or Databricks table.
    Results are cached at module level for performance.
    
    Args:
        config_path: Optional path to JSON config file. If None, uses default location.
    
    Returns:
        Dictionary mapping terms to their expansions
    """
    global _term_expansions_cache
    
    # Use cache if available and config_path matches default
    if _term_expansions_cache is not None and config_path is None:
        return _term_expansions_cache
    
    if config_path is None:
        config_path = str(_default_config_path)
    else:
        # Validate and sanitize file path to prevent path traversal attacks
        config_path = os.path.abspath(config_path)
        # Ensure path is within the project directory or explicitly allowed
        project_root = Path(__file__).parent.resolve()
        if not config_path.startswith(str(project_root)):
            raise ValueError(f"Config path must be within project directory: {config_path}")
    
    # Try loading from file first
    if os.path.exists(config_path) and os.path.isfile(config_path):
        try:
            with open(config_path, "r") as f:
                expansions = json.load(f)
                # Validate structure
                if not isinstance(expansions, dict):
                    raise ValueError("Term expansions file must contain a JSON object")
                # Cache if using default path
                if config_path == str(_default_config_path):
                    _term_expansions_cache = expansions
                return expansions
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in term expansions file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading term expansions file: {e}")
    
    # Optionally load from Databricks table (uncomment and configure as needed)
    # from pyspark.sql import SparkSession
    # spark = SparkSession.builder.getOrCreate()
    # try:
    #     df = spark.sql("SELECT term, expansions FROM your_catalog.your_schema.term_expansions")
    #     expansions = {row.term: json.loads(row.expansions) for row in df.collect()}
    #     _term_expansions_cache = expansions
    #     return expansions
    # except Exception:
    #     pass
    
    # Return empty dict if no config found
    return {}


def _expand_search_terms(
    description: str, 
    term_expansions: dict[str, list[str]], 
    max_terms: int = 50
) -> list[str]:
    """
    Expand a natural language description into a prioritized list of search terms.
    
    Args:
        description: Natural language description
        term_expansions: Dictionary of term expansions
        max_terms: Maximum number of terms to return (default: 50)
    
    Returns:
        Prioritized list of search terms (most relevant first)
    """
    # Validate input
    if not description or not isinstance(description, str):
        return []
    
    # Limit description length to prevent abuse
    if len(description) > 500:
        description = description[:500]
    
    # Convert to lowercase and split into words
    words = description.lower().replace("_", " ").replace("-", " ").split()
    
    # Priority 1: Original words from description
    priority_terms = set(words)
    
    # Priority 2: Direct expansions of original words
    direct_expansions = set()
    for word in words:
        if word in term_expansions:
            direct_expansions.update(term_expansions[word])
    
    # Priority 3: Transitive expansions (terms found in other expansions)
    transitive_expansions = set()
    for word in words:
        for key, expansions in term_expansions.items():
            if word in expansions or any(word in exp for exp in expansions):
                transitive_expansions.update(expansions)
    
    # Combine and prioritize: original terms first, then direct expansions, then transitive
    all_terms = list(priority_terms) + list(direct_expansions - priority_terms) + list(transitive_expansions - direct_expansions - priority_terms)
    
    # Remove very short terms (less than 2 characters) except common ones
    filtered_terms = [term for term in all_terms if len(term) >= 2 or term in ["id", "pk"]]
    
    # Sanitize all terms to prevent SQL injection
    sanitized_terms = [_sanitize_search_term(term) for term in filtered_terms]
    sanitized_terms = [term for term in sanitized_terms if term]  # Remove empty strings
    
    # Limit to max_terms to prevent SQL query bloat
    return sanitized_terms[:max_terms]


def search_tables_by_description(
    description: str,
    catalog: str = "samples",
    tb_schema: Optional[str] = None,
    expansion_config_path: Optional[str] = None
) -> str:
    """
    USE THIS TOOL when users ask about finding tables or columns by description or natural language.
    Searches across ALL schemas in the specified catalog (or a specific tb_schema if provided) to find 
    tables and columns that match a natural language description. The function intelligently expands
    the description into multiple related search terms automatically.
    
    If tb_schema is not specified, it searches across all schemas automatically.
    It also expands terms (e.g., "fares" will search for "fare", "price", "cost", "charge", etc.)
    
    Usage examples:
    - "tables with geolocation columns" -> description="geolocation"
    - "columns related to fares" -> description="fares"
    - "tables with customer information" -> description="customer"
    - "find tables with price or cost columns" -> description="price cost"
    - "which tables have date or time columns" -> description="date time"
    
    Args:
        description: Natural language description of what to search for. Can be single word or phrase such as "geolocation", "fares", "customer information", "date time", or "timestamp"
        catalog: The catalog to search in (default: "samples")
        tb_schema: Optional tb_schema name to limit search to. If None, searches all schemas in the catalog.
        expansion_config_path: Optional path to JSON config file with term expansions.
                              If None, uses default term_expansions.json
    
    Returns:
        JSON string containing list of dictionaries with schema_name, table_name, and column_name.
        Returns empty list [] if no matches found.
    """
    from pyspark.sql import SparkSession
    
    # Validate and sanitize inputs
    catalog = _sanitize_identifier(catalog)
    if tb_schema:
        tb_schema = _sanitize_identifier(tb_schema)
    
    # Load term expansions (cached)
    try:
        term_expansions = _load_term_expansions(expansion_config_path)
    except (ValueError, IOError) as e:
        # If config loading fails, continue with empty expansions
        term_expansions = {}
    
    # Expand search terms
    search_terms = _expand_search_terms(description, term_expansions)
    
    if not search_terms:
        return json.dumps([])
    
    # Build SQL query with multiple LIKE conditions
    # Limit search terms to prevent SQL query bloat (max 50 terms from _expand_search_terms)
    # Use case-insensitive matching for better coverage
    max_terms_in_query = 30  # Limit to prevent SQL query bloat
    terms_to_use = search_terms[:max_terms_in_query] if len(search_terms) > max_terms_in_query else search_terms
    
    # Terms are already sanitized by _expand_search_terms, but escape single quotes for SQL
    escaped_terms = [term.replace("'", "''") for term in terms_to_use]
    like_conditions = " OR ".join([f"LOWER(column_name) LIKE '%{term}%'" for term in escaped_terms])
    
    # Add tb_schema filter if specified
    schema_filter = f"AND table_schema = '{tb_schema}'" if tb_schema else ""
    
    # Use DatabricksSession to avoid warnings in Databricks notebooks
    # The warning suggests using DatabricksSession.builder.getOrCreate() with no parameters
    try:
        from databricks.connect import DatabricksSession
        spark = DatabricksSession.builder.getOrCreate()
    except (ImportError, AttributeError):
        # Fallback to SparkSession if DatabricksSession is not available
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
    
    query = f"""
    SELECT DISTINCT table_schema as schema_name, table_name, column_name
    FROM {catalog}.information_schema.columns
    WHERE ({like_conditions})
    {schema_filter}
    ORDER BY schema_name, table_name, column_name
    """
    
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

UC_TOOL_NAMES = ["system.ai.python_exec"]
uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
tools.extend(uc_toolkit.tools)
tools.extend([greet_me, find_tables_with_column, search_tables_by_description])
VECTOR_SEARCH_TOOLS = []
tools.extend(VECTOR_SEARCH_TOOLS)
#####################
## Define agent logic
#####################


class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]


def create_tool_calling_agent(
    model: ChatDatabricks,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
):
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: AgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


class LangGraphResponsesAgent(ResponsesAgent):
    def __init__(self, agent):
        self.agent = agent

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])

        for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates", "messages"]):
            if event[0] == "updates":
                for node_data in event[1].values():
                    if len(node_data.get("messages", [])) > 0:
                        yield from output_to_responses_items_stream(node_data["messages"])
            # filter the streamed messages to just the generated text messages
            elif event[0] == "messages":
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id),
                        )
                except Exception as e:
                    print(e)


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
mlflow.langchain.autolog()
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphResponsesAgent(agent)
mlflow.models.set_model(AGENT)
