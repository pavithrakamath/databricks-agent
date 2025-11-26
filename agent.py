"""
LangGraph Agent with Human-in-the-Loop Middleware

Main agent module that orchestrates the LangGraph workflow with:
- Tool-calling agent creation
- Human-in-the-loop middleware integration
- MLflow integration for deployment (optional)
- Multi-provider LLM support (Databricks, OpenAI, Anthropic, etc.)
"""

from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict, Union

from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_ENDPOINT,
    EXECUTION_MODE,
    LLM_ENDPOINT_NAME,
    LLM_PROVIDER,
    MIDDLEWARE_CONFIG,
    SYSTEM_PROMPT,
)
from llm_provider import create_llm
from middleware import get_tools_requiring_approval, is_exit_message, clear_session
from tools import get_all_tools

# Optional MLflow imports (only if available)
try:
    import mlflow
    from mlflow.pyfunc import ResponsesAgent
    from mlflow.types.responses import (
        ResponsesAgentRequest,
        ResponsesAgentResponse,
        ResponsesAgentStreamEvent,
        output_to_responses_items_stream,
        to_chat_completions_input,
    )
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# ============================================================================
# Agent State & Workflow
# ============================================================================

class AgentState(TypedDict):
    """Agent state schema."""
    messages: Annotated[Sequence[AnyMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]


def create_tool_calling_agent(
    model: Any,  # Changed from ChatDatabricks to Any for multi-provider support
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
    checkpointer: Optional[Any] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    middleware_config: Optional[dict[str, Union[bool, dict[str, Any]]]] = None,
    verify_final_answer: bool = True,
):
    """
    Create a tool-calling agent with human-in-the-loop middleware support.
    
    Args:
        model: The language model to use (any LangChain-compatible chat model)
        tools: Tools for the agent to use
        system_prompt: Optional system prompt
        checkpointer: Optional checkpointer for state persistence (required for interrupts)
        interrupt_before: Optional list of node names to interrupt before execution
        interrupt_after: Optional list of node names to interrupt after execution
        middleware_config: Optional middleware configuration dict for tool-specific approval
        verify_final_answer: If True, interrupt after agent node to verify final answer
    
    Returns:
        Compiled LangGraph agent with optional interrupt support
    """
    # Use provided middleware_config or fall back to global config
    if middleware_config is None:
        middleware_config = MIDDLEWARE_CONFIG
    
    # Extract tools list for middleware configuration
    tools_list = []
    if isinstance(tools, ToolNode):
        if hasattr(tools, 'tools'):
            tools_list = tools.tools
        elif hasattr(tools, '_tools'):
            tools_list = tools._tools
    elif isinstance(tools, (list, tuple)):
        tools_list = list(tools)
    elif hasattr(tools, '__iter__'):
        tools_list = list(tools)
    
    # Get tools requiring approval
    tools_requiring_approval = (
        get_tools_requiring_approval(tools_list, middleware_config) 
        if tools_list else set()
    )
    
    # Configure interrupts
    # Note: interrupt_before tools is disabled to allow automatic tool execution
    # Tool-specific approval can be handled within the tool execution logic if needed
    # For now, we allow all tools to execute automatically without interruption
    # if tools_requiring_approval and checkpointer is not None:
    #     if interrupt_before is None:
    #         interrupt_before = ["tools"]
    
    # Initialize interrupt_before if not provided (but don't add tools to it)
    if interrupt_before is None:
        interrupt_before = []
    
    model = model.bind_tools(tools)

    def should_continue(state: AgentState):
        """Determine if agent should continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        return "end"

    def filter_incomplete_tool_calls(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
        """
        Filter out AIMessages with tool_calls that don't have corresponding ToolMessages.
        This prevents Azure OpenAI errors when stale state contains incomplete tool calls.
        """
        filtered = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            
            # If this is an AIMessage with tool_calls, check if all tool calls are responded to
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_call_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
                
                # Look ahead to find ToolMessages responding to these tool calls
                j = i + 1
                found_tool_call_ids = set()
                while j < len(messages):
                    next_msg = messages[j]
                    if isinstance(next_msg, ToolMessage):
                        # ToolMessage has tool_call_id attribute
                        tool_call_id = getattr(next_msg, "tool_call_id", None)
                        if tool_call_id:
                            found_tool_call_ids.add(tool_call_id)
                    elif isinstance(next_msg, AIMessage):
                        # Stop at next AIMessage (new turn)
                        break
                    j += 1
                
                # If not all tool calls are responded to, create AIMessage without tool_calls
                if tool_call_ids and not tool_call_ids.issubset(found_tool_call_ids):
                    # Create a new AIMessage without tool_calls to avoid Azure OpenAI error
                    filtered.append(AIMessage(content=msg.content or ""))
                else:
                    # All tool calls have responses, include the original message
                    filtered.append(msg)
            else:
                # Not an AIMessage with tool_calls, include as-is
                filtered.append(msg)
            
            i += 1
        
        return filtered

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [SystemMessage(content=system_prompt)] + filter_incomplete_tool_calls(state["messages"])
        )
    else:
        preprocessor = RunnableLambda(lambda state: filter_incomplete_tool_calls(state["messages"]))
    model_runnable = preprocessor | model

    def call_model(state: AgentState, config: RunnableConfig):
        """Call the language model."""
        # Exit handling is done in invoke_with_exit_handling wrapper
        # Just call the model normally here
        response = model_runnable.invoke(state, config)
        return {"messages": [response]}

    # Build workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")

    # Compile with checkpointer and interrupts
    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before
    
    # For verify_final_answer: Don't interrupt after agent when tool calls are present
    # This allows tools to execute automatically. Only interrupt for final answers.
    # Since LangGraph's interrupt_after doesn't support conditionals, we remove "agent"
    # from interrupt_after to allow automatic tool execution.
    if verify_final_answer and checkpointer is not None:
        if interrupt_after is None:
            interrupt_after = []
        elif isinstance(interrupt_after, list) and "agent" in interrupt_after:
            # Remove "agent" from interrupt_after to allow tools to execute automatically
            interrupt_after = [node for node in interrupt_after if node != "agent"]
        elif isinstance(interrupt_after, dict) and "agent" in interrupt_after:
            del interrupt_after["agent"]
    
    if interrupt_after:
        compile_kwargs["interrupt_after"] = interrupt_after

    compiled_agent = workflow.compile(**compile_kwargs)
    
    # Store metadata and checkpointer reference for session management
    compiled_agent.middleware_config = middleware_config
    compiled_agent.tools_requiring_approval = tools_requiring_approval
    compiled_agent.verify_final_answer = verify_final_answer
    if checkpointer is not None:
        compiled_agent.checkpointer = checkpointer  # Store reference for easy access
    
    # Wrap the invoke method to handle exit messages and clear sessions
    original_invoke = compiled_agent.invoke
    
    def invoke_with_exit_handling(input_data, config=None, **kwargs):
        """Wrapper that clears session if user says exit/goodbye."""
        # Check if the last user message is an exit message BEFORE invoking
        messages = input_data.get("messages", []) if isinstance(input_data, dict) else []
        is_exit = False
        
        if messages and config:
            last_user_msg = None
            for msg in reversed(messages):
                if isinstance(msg, dict):
                    msg_role = msg.get("role", msg.get("type", ""))
                    msg_content = msg.get("content", "")
                    if msg_role in ("user", "human") and msg_content:
                        last_user_msg = msg_content
                        break
                elif hasattr(msg, "content"):
                    msg_type = getattr(msg, "type", None) or getattr(msg, "role", None)
                    if msg_type in ("human", "user"):
                        last_user_msg = getattr(msg, "content", None) or str(msg)
                        break
            
            # If exit message detected, mark it
            if last_user_msg and is_exit_message(last_user_msg):
                is_exit = True
        
        # Normal flow: invoke the agent
        result = original_invoke(input_data, config, **kwargs)
        
        # If this was an exit message, clear the session AFTER invoke
        # This ensures we delete all state including the exit message and goodbye
        if is_exit and config:
            try:
                # Clear session - try multiple times to ensure it works
                clear_result = clear_session(compiled_agent, config)
                
                # Verify clearing worked by checking if state still exists
                try:
                    state = compiled_agent.get_state(config)
                    if state.values.get("messages"):
                        # Still has messages, try direct deletion
                        checkpointer = getattr(compiled_agent, "checkpointer", None)
                        if checkpointer:
                            thread_id = config.get("configurable", {}).get("thread_id")
                            storage = getattr(checkpointer, "_storage", None)
                            if storage and isinstance(storage, dict) and thread_id in storage:
                                del storage[thread_id]
                except Exception:
                    pass
            except Exception as e:
                # If clearing fails, try direct access as last resort
                try:
                    checkpointer = getattr(compiled_agent, "checkpointer", None)
                    if checkpointer:
                        thread_id = config.get("configurable", {}).get("thread_id")
                        storage = getattr(checkpointer, "_storage", None)
                        if storage and isinstance(storage, dict) and thread_id in storage:
                            del storage[thread_id]
                except Exception:
                    pass
        
        return result
    
    compiled_agent.invoke = invoke_with_exit_handling
    
    return compiled_agent


# ============================================================================
# MLflow Integration (Optional)
# ============================================================================

if MLFLOW_AVAILABLE:
    class LangGraphResponsesAgent(ResponsesAgent):
        """MLflow ResponsesAgent wrapper for LangGraph agent."""
        
        def __init__(self, agent):
            self.agent = agent

        def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
            """Predict without streaming."""
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
            """Stream predictions."""
            cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])

            for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates", "messages"]):
                if event[0] == "updates":
                    for node_data in event[1].values():
                        if len(node_data.get("messages", [])) > 0:
                            yield from output_to_responses_items_stream(node_data["messages"])
                elif event[0] == "messages":
                    try:
                        chunk = event[1][0]
                        if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                            yield ResponsesAgentStreamEvent(
                                **self.create_text_delta(delta=content, item_id=chunk.id),
                            )
                    except Exception as e:
                        print(e)
else:
    # Dummy class when MLflow is not available
    LangGraphResponsesAgent = None


# ============================================================================
# Agent Initialization
# ============================================================================

# Initialize MLflow if available
if MLFLOW_AVAILABLE:
    mlflow.langchain.autolog()

# Initialize LLM (auto-detects provider based on environment)
# For Azure OpenAI, pass additional configuration
llm_kwargs = {}
if LLM_PROVIDER and LLM_PROVIDER.lower() in ("azure_openai", "azure-openai", "azure"):
    if AZURE_OPENAI_ENDPOINT:
        llm_kwargs["azure_endpoint"] = AZURE_OPENAI_ENDPOINT
    if AZURE_OPENAI_DEPLOYMENT_NAME:
        llm_kwargs["azure_deployment"] = AZURE_OPENAI_DEPLOYMENT_NAME
    elif LLM_ENDPOINT_NAME:
        llm_kwargs["azure_deployment"] = LLM_ENDPOINT_NAME
    if AZURE_OPENAI_API_KEY:
        llm_kwargs["api_key"] = AZURE_OPENAI_API_KEY
    if AZURE_OPENAI_API_VERSION:
        llm_kwargs["api_version"] = AZURE_OPENAI_API_VERSION

llm = create_llm(
    provider=LLM_PROVIDER,
    endpoint=LLM_ENDPOINT_NAME,
    model_name=LLM_ENDPOINT_NAME,  # Used for non-Databricks providers
    **llm_kwargs
)

# Get all tools (auto-detects whether to include UC tools)
tools = get_all_tools()

# Create checkpointer for state persistence (required for interrupts)
checkpointer = MemorySaver()

# Create agent with automatic tool execution
# Note: verify_final_answer is set to False to allow automatic tool execution
# Tool-specific approval can still be handled via MIDDLEWARE_CONFIG if needed
agent = create_tool_calling_agent(
    llm, 
    tools, 
    SYSTEM_PROMPT,
    checkpointer=checkpointer,
    verify_final_answer=False  # Disable to allow automatic tool execution
)

# Create MLflow agent wrapper (if MLflow is available)
if MLFLOW_AVAILABLE and LangGraphResponsesAgent is not None:
    AGENT = LangGraphResponsesAgent(agent)
    mlflow.models.set_model(AGENT)
else:
    # For standalone mode, expose the agent directly
    AGENT = agent
