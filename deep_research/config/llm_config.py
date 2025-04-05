"""
LLM configuration and factory functions for the Deep Research Agent.
"""
from langchain.llms import Ollama
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Default LLM constants
DEFAULT_MODEL = "phi4:latest"
DEFAULT_API_BASE = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_CONTEXT_SIZE = 40960  # Maximum context window

# Configuration presets for different tasks
LLM_CONFIG_PRESETS = {
    "default": {
        "temperature": 0.3,
    },
    "plan_generation": {
        "temperature": 0.4,
    },
    "research_analysis": {
        "temperature": 0.4,
    },
    "knowledge_synthesis": {
        "temperature": 0.4,
    },
    "report_generation": {
        "temperature": 0.4,
    }
}

def get_ollama_llm(
    model_name: str = DEFAULT_MODEL,
    preset: Optional[str] = None,
    **kwargs
) -> Ollama:
    """
    Factory function to create an Ollama LLM with appropriate configuration.
    
    Args:
        model_name: Name of the Ollama model to use
        preset: Optional configuration preset (default, plan_generation, etc.)
        **kwargs: Additional parameters to override defaults
        
    Returns:
        Configured Ollama LLM instance
    """
    # Start with base configuration
    config = {
        "base_url": DEFAULT_API_BASE,
        "model": model_name,
        "temperature": DEFAULT_TEMPERATURE,
        "num_ctx": DEFAULT_CONTEXT_SIZE,
    }
    
    # Apply preset if specified
    if preset and preset in LLM_CONFIG_PRESETS:
        config.update(LLM_CONFIG_PRESETS[preset])
    
    # Override with any passed kwargs
    config.update(kwargs)
    
    try:
        # Initialize and test the LLM
        llm = Ollama(**config)
        llm.invoke("test")  # Quick test to verify connection
        return llm
    except Exception as e:
        error_msg = f"Failed to initialize Ollama model {model_name}: {e}"
        logger.error(error_msg)
        raise RuntimeError(
            f"{error_msg}. Please ensure Ollama is running "
            f"and the specified model is installed. You can install models using "
            f"`ollama pull {model_name}`"
        )

def test_ollama_connection(model_name: str = DEFAULT_MODEL) -> bool:
    """
    Test if Ollama is running and the model is available.
    
    Args:
        model_name: Model name to test
        
    Returns:
        Boolean indicating if connection was successful
    """
    try:
        llm = Ollama(
            base_url=DEFAULT_API_BASE,
            model=model_name
        )
        llm.invoke("test")
        return True
    except Exception as e:
        logger.warning(f"Ollama connection test failed: {e}")
        return False