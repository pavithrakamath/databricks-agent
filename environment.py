"""
Environment detection and configuration for Databricks vs standalone execution.

Provides utilities to detect the execution environment and configure
the agent accordingly.
"""

import os
from enum import Enum
from typing import Optional

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it


class ExecutionMode(Enum):
    """Execution mode for the agent."""
    DATABRICKS = "databricks"
    STANDALONE = "standalone"
    AUTO = "auto"


def detect_databricks_environment() -> bool:
    """
    Detect if running in Databricks environment.
    
    Returns:
        True if running in Databricks, False otherwise
    """
    # Check for Databricks-specific environment variables
    databricks_indicators = [
        "DATABRICKS_RUNTIME_VERSION",
        "DATABRICKS_ROOT_VIRTUALENV_ENV",
        "SPARK_DATABRICKS_WORKSPACE_URL",
    ]
    
    if any(os.getenv(indicator) for indicator in databricks_indicators):
        return True
    
    # Check if databricks.connect is available and can create a session
    try:
        from databricks.connect import DatabricksSession
        # Try to get or create session (will fail if not in Databricks)
        try:
            DatabricksSession.builder.getOrCreate()
            return True
        except Exception:
            return False
    except ImportError:
        return False


def get_execution_mode(mode: Optional[str] = None) -> ExecutionMode:
    """
    Get the execution mode, either from parameter or auto-detect.
    
    Args:
        mode: Optional mode string ("databricks", "standalone", "auto")
              If None, uses AUTO_DETECT_MODE from config
    
    Returns:
        ExecutionMode enum value
    """
    if mode is None:
        mode = os.getenv("AGENT_EXECUTION_MODE", "auto")
    
    mode_lower = mode.lower()
    if mode_lower == "databricks":
        return ExecutionMode.DATABRICKS
    elif mode_lower == "standalone":
        return ExecutionMode.STANDALONE
    else:  # auto
        if detect_databricks_environment():
            return ExecutionMode.DATABRICKS
        return ExecutionMode.STANDALONE


def is_databricks_mode(mode: Optional[str] = None) -> bool:
    """
    Check if agent should run in Databricks mode.
    
    Args:
        mode: Optional mode string, defaults to auto-detect
    
    Returns:
        True if Databricks mode, False if standalone
    """
    execution_mode = get_execution_mode(mode)
    return execution_mode == ExecutionMode.DATABRICKS


def get_available_providers() -> list[str]:
    """
    Get list of available LLM providers based on installed packages.
    
    Returns:
        List of available provider names
    """
    providers = []
    
    # If LLM_PROVIDER is explicitly set, return it (if available)
    explicit_provider = os.getenv("LLM_PROVIDER")
    if explicit_provider:
        # Still check if the provider is actually available
        provider_lower = explicit_provider.lower()
        if provider_lower == "databricks":
            try:
                from databricks_langchain import ChatDatabricks  # type: ignore
                return ["databricks"]
            except ImportError:
                pass
        elif provider_lower in ("openai", "azure_openai", "azure-openai", "azure"):
            try:
                from langchain_openai import ChatOpenAI, AzureChatOpenAI  # type: ignore
                if provider_lower.startswith("azure"):
                    return ["azure_openai"]
                return ["openai"]
            except ImportError:
                pass
        elif provider_lower == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic  # type: ignore
                return ["anthropic"]
            except ImportError:
                pass
        elif provider_lower == "ollama":
            try:
                from langchain_community.chat_models import ChatOllama  # type: ignore
                return ["ollama"]
            except ImportError:
                pass
        # If explicit provider is set but not available, return empty list
        return []
    
    # Check for Databricks
    try:
        from databricks_langchain import ChatDatabricks  # type: ignore
        providers.append("databricks")
    except ImportError:
        pass
    
    # Check for OpenAI
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        providers.append("openai")
    except ImportError:
        pass
    
    # Check for Azure OpenAI
    try:
        from langchain_openai import AzureChatOpenAI  # type: ignore
        providers.append("azure_openai")
    except ImportError:
        pass
    
    # Check for Anthropic
    try:
        from langchain_anthropic import ChatAnthropic  # type: ignore
        providers.append("anthropic")
    except ImportError:
        pass
    
    # Check for other providers
    try:
        from langchain_community.chat_models import ChatOllama  # type: ignore
        providers.append("ollama")
    except ImportError:
        pass
    
    return providers

