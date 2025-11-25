# Databricks LangGraph Agent

A Databricks LangGraph agent built with LangChain components and MLflow integration, featuring tool calling capabilities for data science tasks.

## Setup

This project uses Poetry for dependency management.

### Prerequisites

- Python 3.10 or 3.11
- Poetry (install via: `curl -sSL https://install.python-poetry.org | python3 -`)

### Installation

```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell

# Or run commands directly
poetry run python agent.py
```

## Dependencies

- `databricks-agents` - Databricks Agents framework
- `databricks-langchain` - Databricks LangChain integration
- `langchain` - LangChain framework (v1.0.0+)
- `langchain-core` - LangChain core (v1.0.0+)
- `langgraph` - LangGraph for agent workflows (uses StateGraph for agent orchestration)
- `mlflow-skinny[databricks]` - MLflow with Databricks extras (uses ResponsesAgent framework)

## Usage

See `agent.py` for the main agent implementation.

## Architecture Decisions

Architecture Decision Records (ADRs) are documented in [`docs/adr/`](docs/adr/):
- [ADR-0001: Use LangGraph for Agent Orchestration](docs/adr/0001-use-langgraph-for-agent-orchestration.md)

## Term Expansions Configuration

The `search_tables_by_description` tool uses a configurable term expansion system to map natural language descriptions to database column search terms.

### Managing Term Expansions

Term expansions are stored in `term_expansions.json`. To add or modify expansions:

1. **Edit the JSON file directly:**
   ```json
   {
     "your_term": ["term1", "term2", "term3", "synonym1", "synonym2"]
   }
   ```

2. **The file is automatically loaded and cached** - no code changes needed.

3. **For large-scale management**, you can optionally load from a Databricks table:
   - Uncomment the table loading code in `_load_term_expansions()`
   - Create a table with columns: `term` (string), `expansions` (JSON string)
   - Update the SQL query to point to your table

### Example

To add support for "financial" terms:
```json
{
  "financial": ["financial", "finance", "fiscal", "money", "currency", "dollar", "euro", "payment", "transaction"],
  "payment": ["payment", "pay", "transaction", "financial", "invoice", "billing"]
}
```

The system will automatically:
- Cache expansions for performance
- Expand search terms when users query
- Search across all schemas in the catalog

