# LangGraph Agent with Human-in-the-Loop

A flexible LangGraph agent built with LangChain components, featuring:
- **Multi-provider LLM support** (Databricks, OpenAI, Azure OpenAI, Anthropic, Ollama)
- **Human-in-the-loop middleware** for tool approval and answer verification
- **MLflow integration** for deployment (optional)
- **Modular architecture** for easy maintenance and extension
- **Environment-agnostic** - runs in Databricks or standalone mode

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

**Core:**
- `langchain` - LangChain framework (v1.0.0+)
- `langchain-core` - LangChain core (v1.0.0+)
- `langgraph` - LangGraph for agent workflows
- `python-dotenv` - Environment variable management

**Optional (provider-specific):**
- `databricks-langchain` - For Databricks provider
- `langchain-openai` - For OpenAI/Azure OpenAI providers
- `langchain-anthropic` - For Anthropic provider
- `langchain-community` - For Ollama provider
- `mlflow-skinny[databricks]` - For MLflow integration (optional)

## Quick Start

### Configuration

The agent supports configuration via environment variables or a `.env` file:

```bash
# .env file
LLM_PROVIDER=azure_openai
LLM_ENDPOINT_NAME=gpt-4-deployment
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AGENT_EXECUTION_MODE=standalone
```

### Basic Usage

```python
from agent import agent

config = {"configurable": {"thread_id": "1"}}
result = agent.invoke({
    "messages": [{"role": "user", "content": "List all schemas"}]
}, config)
```

### Human-in-the-Loop

The agent includes built-in human-in-the-loop middleware:
- **SQL queries** require approval before execution (configurable)
- **Final answers** can be verified before sending to users

See [`USAGE.md`](USAGE.md) for detailed usage instructions and [`AZURE_OPENAI_SETUP.md`](AZURE_OPENAI_SETUP.md) for Azure OpenAI configuration.

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

