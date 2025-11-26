"""
Middleware and interrupt helpers for human-in-the-loop functionality.

Provides functions for:
- Checking interrupt state
- Managing tool call approvals
- Managing final answer verification
"""

from typing import Any, Optional

from langchain.messages import AIMessage

from config import MIDDLEWARE_CONFIG


def get_tools_requiring_approval(
    tools,
    middleware_config: dict[str, Any] = None
) -> set[str]:
    """
    Get set of tool names that require human approval.
    
    Handles both LangChain BaseTool objects (with .name attribute) and
    raw Python functions (with __name__ attribute).
    """
    if middleware_config is None:
        middleware_config = MIDDLEWARE_CONFIG
    
    tools_requiring_approval = set()
    
    # Extract tool names - handle both BaseTool objects and raw functions
    tool_names = set()
    for tool in tools:
        # Check if it's a LangChain BaseTool (has .name attribute)
        if hasattr(tool, 'name'):
            tool_names.add(tool.name)
        # Check if it's a raw Python function (has __name__ attribute)
        elif hasattr(tool, '__name__'):
            tool_names.add(tool.__name__)
        # If neither, skip (shouldn't happen, but be defensive)
        else:
            continue
    
    for tool_name, config in middleware_config.items():
        if tool_name in tool_names:
            if config is False:
                continue
            if isinstance(config, dict) and config.get("required", False):
                tools_requiring_approval.add(tool_name)
    
    return tools_requiring_approval


def check_interrupt_state(agent: Any, config: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Check if the agent is currently in an interrupted state."""
    try:
        state = agent.get_state(config)
        if state.next:
            messages = state.values.get("messages", [])
            pending_tool_calls = []
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage) and last_message.tool_calls:
                    pending_tool_calls = [
                        {
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                            "id": tc.get("id", "")
                        }
                        for tc in last_message.tool_calls
                    ]
            
            return {
                "interrupted": True,
                "next_nodes": state.next,
                "pending_tool_calls": pending_tool_calls,
                "state": state.values
            }
        return None
    except Exception as e:
        return {"error": str(e)}


def get_pending_tool_calls(agent: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Get list of pending tool calls waiting for approval."""
    interrupt_info = check_interrupt_state(agent, config)
    if interrupt_info and interrupt_info.get("interrupted"):
        return interrupt_info.get("pending_tool_calls", [])
    return []


def approve_tool_calls(agent: Any, config: dict[str, Any], decision: str = "approve") -> dict[str, Any]:
    """Approve pending tool calls and continue execution."""
    interrupt_info = check_interrupt_state(agent, config)
    if not interrupt_info or not interrupt_info.get("interrupted"):
        return {"error": "No interrupt pending", "status": "no_interrupt"}
    
    if decision == "approve":
        try:
            result = agent.invoke(None, config)
            return {"status": "approved", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    elif decision == "reject":
        try:
            current_state = interrupt_info["state"]
            messages = current_state.get("messages", [])
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage):
                    new_message = AIMessage(content=last_message.content)
                    updated_messages = messages[:-1] + [new_message]
                    agent.update_state(config, {"values": {"messages": updated_messages}})
            return {"status": "rejected"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    else:
        return {"error": f"Unknown decision: {decision}. Use 'approve', 'edit', or 'reject'"}


def edit_tool_call(
    agent: Any, 
    config: dict[str, Any], 
    tool_call_id: str, 
    new_args: dict[str, Any]
) -> dict[str, Any]:
    """Edit a specific tool call's arguments before approval."""
    interrupt_info = check_interrupt_state(agent, config)
    if not interrupt_info or not interrupt_info.get("interrupted"):
        return {"error": "No interrupt pending", "status": "no_interrupt"}
    
    try:
        current_state = interrupt_info["state"]
        messages = current_state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                updated_tool_calls = []
                for tc in last_message.tool_calls:
                    if tc.get("id") == tool_call_id:
                        updated_tc = dict(tc)
                        updated_tc["args"] = new_args
                        updated_tool_calls.append(updated_tc)
                    else:
                        updated_tool_calls.append(tc)
                
                new_message = AIMessage(
                    content=last_message.content,
                    tool_calls=updated_tool_calls
                )
                updated_messages = messages[:-1] + [new_message]
                agent.update_state(config, {"values": {"messages": updated_messages}})
                return {"status": "edited", "tool_call_id": tool_call_id}
        
        return {"error": "Tool call not found", "status": "error"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_final_answer(agent: Any, config: dict[str, Any]) -> Optional[str]:
    """Get the final answer from the agent state (when interrupted after agent node)."""
    try:
        state = agent.get_state(config)
        messages = state.values.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                # Final answer has no tool calls
                if not last_message.tool_calls:
                    return last_message.content
        return None
    except Exception:
        return None


def approve_final_answer(
    agent: Any, 
    config: dict[str, Any], 
    decision: str = "approve", 
    edited_answer: Optional[str] = None
) -> dict[str, Any]:
    """Approve or edit the final answer before sending to user."""
    interrupt_info = check_interrupt_state(agent, config)
    if not interrupt_info or not interrupt_info.get("interrupted"):
        return {"error": "No interrupt pending", "status": "no_interrupt"}
    
    # Check if this is a final answer interrupt (no tool calls in last message)
    messages = interrupt_info.get("state", {}).get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            # This is a tool call interrupt, not final answer
            return {"error": "Not a final answer interrupt", "status": "wrong_interrupt"}
    
    try:
        current_state = interrupt_info["state"]
        messages = current_state.get("messages", [])
        
        if decision == "approve":
            result = agent.invoke(None, config)
            return {"status": "approved", "result": result}
        
        elif decision == "edit":
            if edited_answer is None:
                return {
                    "error": "edited_answer required when decision is 'edit'", 
                    "status": "error"
                }
            
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage):
                    new_message = AIMessage(content=edited_answer)
                    updated_messages = messages[:-1] + [new_message]
                    agent.update_state(config, {"values": {"messages": updated_messages}})
                    result = agent.invoke(None, config)
                    return {"status": "edited", "result": result}
            
            return {"error": "Could not find final answer to edit", "status": "error"}
        
        elif decision == "reject":
            if messages:
                updated_messages = messages[:-1]
                agent.update_state(config, {"values": {"messages": updated_messages}})
            return {"status": "rejected"}
        
        else:
            return {
                "error": f"Unknown decision: {decision}. Use 'approve', 'edit', or 'reject'",
                "status": "error"
            }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}

