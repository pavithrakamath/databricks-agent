"""
LLM Provider abstraction for multi-provider support.

Supports:
- Databricks (ChatDatabricks)
- OpenAI (ChatOpenAI)
- Azure OpenAI (ChatOpenAI with Azure configuration)
- Anthropic (ChatAnthropic)
- Other LangChain-compatible providers
"""

from typing import Any, Optional

from environment import is_databricks_mode


def create_llm(
    provider: Optional[str] = None,
    endpoint: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Create an LLM instance based on provider and environment.
    
    Args:
        provider: Provider name ("databricks", "openai", "azure_openai", "anthropic", etc.)
                  If None, auto-detects based on environment
        endpoint: Endpoint name (for Databricks) or model name
        model_name: Model name (for non-Databricks providers)
        **kwargs: Additional provider-specific arguments
                  For Azure OpenAI: azure_endpoint, azure_deployment, api_version, api_key
    
    Returns:
        LangChain-compatible chat model instance
    
    Raises:
        ValueError: If provider is not available or invalid
    """
    # Auto-detect provider if not specified
    if provider is None:
        if is_databricks_mode():
            provider = "databricks"
        else:
            # Try to find available provider
            from environment import get_available_providers
            available = get_available_providers()
            if available:
                provider = available[0]  # Use first available
            else:
                raise ValueError(
                    "No LLM provider available. "
                    "Please install one of: databricks-langchain, langchain-openai, langchain-anthropic"
                )
    
    provider_lower = provider.lower()
    
    # Databricks provider
    if provider_lower == "databricks":
        try:
            from databricks_langchain import ChatDatabricks
            endpoint = endpoint or kwargs.pop("endpoint", None)
            if not endpoint:
                raise ValueError("endpoint is required for Databricks provider")
            return ChatDatabricks(endpoint=endpoint, **kwargs)
        except ImportError:
            raise ValueError(
                "Databricks provider not available. "
                "Install with: pip install databricks-langchain"
            )
    
    # OpenAI provider
    elif provider_lower == "openai":
        try:
            from langchain_openai import ChatOpenAI
            model_name = model_name or endpoint or kwargs.pop("model_name", "gpt-4")
            return ChatOpenAI(model=model_name, **kwargs)
        except ImportError:
            raise ValueError(
                "OpenAI provider not available. "
                "Install with: pip install langchain-openai"
            )
    
    # Azure OpenAI provider
    elif provider_lower in ("azure_openai", "azure-openai", "azure"):
        print("Creating Azure OpenAI provider")
        try:
            from langchain_openai import AzureChatOpenAI
            import os
            
            # Azure OpenAI requires specific configuration
            azure_endpoint = kwargs.pop("azure_endpoint", None) or kwargs.pop("azure_openai_endpoint", None) or os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_deployment = kwargs.pop("azure_deployment", None) or kwargs.pop("deployment_name", None) or kwargs.pop("model_name", None) or model_name or endpoint or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            api_version = kwargs.pop("api_version", None) or os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
            api_key = kwargs.pop("api_key", None) or os.getenv("AZURE_OPENAI_API_KEY")
            
            if not azure_endpoint:
                raise ValueError(
                    "azure_endpoint is required for Azure OpenAI provider. "
                    "Set AZURE_OPENAI_ENDPOINT environment variable or pass azure_endpoint parameter."
                )
            if not azure_deployment:
                raise ValueError(
                    "azure_deployment is required for Azure OpenAI provider. "
                    "Set AZURE_OPENAI_DEPLOYMENT_NAME environment variable or pass azure_deployment parameter."
                )
            if not api_key:
                raise ValueError(
                    "api_key is required for Azure OpenAI provider. "
                    "Set AZURE_OPENAI_API_KEY environment variable or pass api_key parameter."
                )
            
            return AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                api_version=api_version,
                api_key=api_key,
                **kwargs
            )
        except ImportError:
            raise ValueError(
                "Azure OpenAI provider not available. "
                "Install with: pip install langchain-openai"
            )
    
    # Anthropic provider
    elif provider_lower == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            model_name = model_name or endpoint or kwargs.pop("model_name", "claude-3-5-sonnet-20241022")
            return ChatAnthropic(model=model_name, **kwargs)
        except ImportError:
            raise ValueError(
                "Anthropic provider not available. "
                "Install with: pip install langchain-anthropic"
            )
    
    # Ollama provider (for local models)
    elif provider_lower == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
            model_name = model_name or endpoint or kwargs.pop("model_name", "llama2")
            return ChatOllama(model=model_name, **kwargs)
        except ImportError:
            raise ValueError(
                "Ollama provider not available. "
                "Install with: pip install langchain-community"
            )
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported providers: databricks, openai, azure_openai, anthropic, ollama"
        )

