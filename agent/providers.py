import openai

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from utils.config import ConfigLoader
from utils.env import EnvLoader
from typing import Optional

config = ConfigLoader()
env = EnvLoader()


def get_llm_model(model_choices: Optional[str] = None) -> OpenAIModel:
    llm_choice = model_choices or config.get(
        'LLM_CHOICE', 'gpt-4-turbo-preview')
    base_url = config.get("LLM_BASE_URL", "https://api.openai.com/v1'")
    api_key = env.get("LLM_API_KEY", "ollama")

    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIModel(llm_choice, provider=provider)


def get_embedding_model() -> str:
    """
    Get embedding model name from environment.
    
    Returns:
        Embedding model name
    """
    return config.get('EMBEDDING_MODEL', 'text-embedding-3-small')


def get_embedding_client() -> openai.AsyncOpenAI:
    """
    Get embedding client configuration based on environment variables.
    
    Returns:
        Configured OpenAI-compatible client for embeddings
    """
    base_url = config.get('EMBEDDING_BASE_URL', 'https://api.openai.com/v1')
    api_key = env.get('EMBEDDING_API_KEY', 'ollama')

    return openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )


def get_embedding_provider() -> str:
    """Get the embedding provider name."""
    return config.get('EMBEDDING_PROVIDER', 'openai')


def get_ingestion_model() -> OpenAIModel:
    """
    Get ingestion-specific LLM model (can be faster/cheaper than main model).
    
    Returns:
        Configured model for ingestion tasks
    """
    ingestion_choice = config.get("INGESTION_LLM_CHOICE")

    # If no specific ingestion model, use the main model
    if not ingestion_choice:
        return get_llm_model()

    return get_llm_model(model_choice=ingestion_choice)


def get_llm_provider() -> str:
    """Get the LLM provider name"""
    return config.get("LLM_PROVIDER", "openai")


def get_model_info() -> dict:
    """
    Get information about current model configuration.
    
    Returns:
        Dictionary with model configuration info
    """
    return {
        "llm_provider": get_llm_provider(),
        "llm_model": config.get('LLM_CHOICE'),
        "llm_base_url": config.get('LLM_BASE_URL'),
        "embedding_provider": get_embedding_provider(),
        "embedding_model": get_embedding_model(),
        "embedding_base_url": config.get('EMBEDDING_BASE_URL'),
        "ingestion_model": config.get('INGESTION_LLM_CHOICE', 'same as main'),
    }
