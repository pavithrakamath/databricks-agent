# ADR-0001: Use LangGraph for Agent Orchestration

## Status
Accepted

## Context
We need to build an agent system that can:
1. Accept user queries in natural language
2. Call multiple tools in sequence or parallel
3. Maintain conversation state across multiple interactions
4. Handle conditional logic (continue tool calling vs. end conversation)
5. Integrate with MLflow's ResponsesAgent framework for deployment

The agent requires a multi-step workflow pattern:
- Agent receives user message
- Agent decides to call tools (if needed)
- Tools execute and return results
- Agent processes results and decides next action
- Loop continues until agent decides conversation is complete

## Decision
We will use **LangGraph** for agent orchestration instead of:
- Manual while loops with state management
- LangChain's AgentExecutor (less flexible for complex workflows)
- Custom orchestration framework

## Consequences

### Positive
- **Clean workflow definition**: Declarative graph-based approach makes the agent flow easy to understand
- **Built-in state management**: `AgentState` automatically persists across nodes
- **Conditional routing**: Easy to implement `should_continue()` logic with conditional edges
- **Tool integration**: `ToolNode` handles parallel tool execution automatically
- **Streaming support**: Built-in support for streaming responses
- **Extensibility**: Easy to add new nodes (e.g., validation, logging, error handling)
- **Debugging**: Better visibility into agent execution flow
- **MLflow compatibility**: Works seamlessly with MLflow's ResponsesAgent framework

### Negative
- **Additional dependency**: Requires `langgraph` package
- **Learning curve**: Team needs to understand LangGraph concepts (nodes, edges, state)
- **Slight overhead**: Graph compilation adds minimal overhead compared to direct function calls

### Neutral
- Part of LangChain ecosystem, so integrates well with existing LangChain components
- Active development and good community support

## Implementation Details

The agent workflow is defined as:

```
[Entry] → [Agent Node] → [Conditional Edge]
                              ↓
                    ┌─────────┴─────────┐
                    │                   │
              [Continue]            [End]
                    │                   │
              [Tools Node]         [Exit]
                    │
                    ↓
              [Agent Node] (loop back)
```

Key components:
- **AgentState**: TypedDict managing messages and custom inputs/outputs
- **StateGraph**: LangGraph's graph builder for workflow definition
- **ToolNode**: Prebuilt node for executing tool calls
- **Conditional edges**: Routes based on `should_continue()` function

## Alternatives Considered

### 1. Manual While Loop
```python
messages = [user_message]
while True:
    response = llm.invoke(messages)
    if response.tool_calls:
        # Execute tools manually
        # Manage state manually
    else:
        break
```
**Rejected because**: Too much boilerplate, error-prone, hard to extend

### 2. LangChain AgentExecutor
**Rejected because**: Less flexible, harder to customize workflow, limited control over state

### 3. Custom Orchestration Framework
**Rejected because**: Reinventing the wheel, maintenance burden, less battle-tested

## References
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [MLflow ResponsesAgent](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ResponsesAgent)

