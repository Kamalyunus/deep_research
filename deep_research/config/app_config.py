<<<<<<< Updated upstream
"""
Application configuration and constants for the Deep Research Agent.
"""
import os
from typing import Dict, Any, List

# Output directory for research reports
RESEARCH_OUTPUT_DIR = "research_outputs"

# Ensure the directory exists
os.makedirs(RESEARCH_OUTPUT_DIR, exist_ok=True)

# Search configuration
DEFAULT_MAX_PAPERS = 2
DEFAULT_MAX_WEB_RESULTS = 3
DEFAULT_SEARCH_SOURCES = ["ArXiv", "Web Search"]

# Research configuration
MAX_ITERATIONS_PER_TASK = 5
DEFAULT_ITERATIONS_PER_TASK = 2
DEFAULT_DETAIL_LEVEL = "Standard"
DETAIL_LEVEL_OPTIONS = ["Basic", "Standard", "Comprehensive"]

# Detail level adjustments
DETAIL_LEVEL_ADJUSTMENTS = {
    "Basic": {
        "papers_adjustment": -1,
        "web_results_adjustment": -1,
    },
    "Standard": {
        "papers_adjustment": 0,
        "web_results_adjustment": 0,
    },
    "Comprehensive": {
        "papers_adjustment": 1,
        "web_results_adjustment": 2,
    }
}

# Model options for the UI
MODEL_OPTIONS = [
    "qwq:32b-q8_0",
    "phi4:14b-q8_0"
]

# Task iteration configuration
ITERATION_CONFIG_HELP = {
    "iterations": "More iterations will explore the topic in greater depth, but will take longer to complete.",
    "detail_level": "Basic: Less content but faster. Standard: Balanced approach. Comprehensive: More detailed content but slower."
}

# Session state keys
SESSION_KEYS = {
    "messages": "messages",
    "research_plan": "research_plan",
    "plan_approved": "plan_approved",
    "research_iterations": "research_iterations",
    "agent": "agent",
    "all_papers": "all_papers",
    "all_web_results": "all_web_results",
    "plan_generated": "plan_generated",
    "task_iterations": "task_iterations",
    "iterations_configured": "iterations_configured",
    "research_complete": "research_complete",
    "debug_mode": "debug_mode"
}

def get_sanitized_filename(topic: str, timestamp: str) -> str:
    """
    Create a sanitized filename for research outputs based on topic and timestamp.
    
    Args:
        topic: Research topic
        timestamp: Timestamp string
        
    Returns:
        Sanitized filename
    """
    sanitized_topic = "".join([c if c.isalnum() else "_" for c in topic])
    return f"{RESEARCH_OUTPUT_DIR}/{sanitized_topic}_{timestamp}.md"

def reset_session_state(state_dict: Dict[str, Any]) -> None:
    """
    Reset all tracked session state variables.
    
    Args:
        state_dict: Streamlit session state dictionary
    """
    for key in SESSION_KEYS.values():
        if key in state_dict:
            state_dict[key] = None if key != "messages" else []
            
    # Special cases
    state_dict[SESSION_KEYS["messages"]] = []
    state_dict[SESSION_KEYS["plan_approved"]] = False
    state_dict[SESSION_KEYS["research_iterations"]] = 0
    state_dict[SESSION_KEYS["all_papers"]] = []
    state_dict[SESSION_KEYS["all_web_results"]] = []
    state_dict[SESSION_KEYS["plan_generated"]] = False
    state_dict[SESSION_KEYS["task_iterations"]] = {}
    state_dict[SESSION_KEYS["iterations_configured"]] = False
    state_dict[SESSION_KEYS["research_complete"]] = False
    state_dict[SESSION_KEYS["debug_mode"]] = False
    
    # If agent exists, reset its state too
    if state_dict.get(SESSION_KEYS["agent"]):
        agent = state_dict[SESSION_KEYS["agent"]]
        agent.accumulated_knowledge = ""
        agent.task_knowledge = {}
        agent.current_papers = []
        agent.current_web_results = []
=======
"""
Application configuration and constants for the Deep Research Agent.
"""
import os
from typing import Dict, Any, List

# Output directory for research reports
RESEARCH_OUTPUT_DIR = "research_outputs"

# Ensure the directory exists
os.makedirs(RESEARCH_OUTPUT_DIR, exist_ok=True)

# Search configuration
DEFAULT_MAX_PAPERS = 3
DEFAULT_SEARCH_SOURCES = ["ArXiv"]

# Research configuration
MAX_ITERATIONS_PER_TASK = 5
DEFAULT_ITERATIONS_PER_TASK = 2
DEFAULT_DETAIL_LEVEL = "Standard"
DETAIL_LEVEL_OPTIONS = ["Basic", "Standard", "Comprehensive"]

# Detail level adjustments
DETAIL_LEVEL_ADJUSTMENTS = {
    "Basic": {
        "papers_adjustment": -1,
    },
    "Standard": {
        "papers_adjustment": 0,
    },
    "Comprehensive": {
        "papers_adjustment": 2,
    }
}

# Model options for the UI
MODEL_OPTIONS = [
    "phi4:latest",
    "gemma3:12b"
]

# Task iteration configuration
ITERATION_CONFIG_HELP = {
    "iterations": "More iterations will explore the topic in greater depth, but will take longer to complete.",
    "detail_level": "Basic: Less content but faster. Standard: Balanced approach. Comprehensive: More detailed content but slower."
}

# Session state keys
SESSION_KEYS = {
    "messages": "messages",
    "research_plan": "research_plan",
    "plan_approved": "plan_approved",
    "research_iterations": "research_iterations",
    "agent": "agent",
    "all_papers": "all_papers",
    "plan_generated": "plan_generated",
    "task_iterations": "task_iterations",
    "iterations_configured": "iterations_configured",
    "research_complete": "research_complete",
    "debug_mode": "debug_mode"
}

def get_sanitized_filename(topic: str, timestamp: str) -> str:
    """
    Create a sanitized filename for research outputs based on topic and timestamp.
    
    Args:
        topic: Research topic
        timestamp: Timestamp string
        
    Returns:
        Sanitized filename
    """
    sanitized_topic = "".join([c if c.isalnum() else "_" for c in topic])
    return f"{RESEARCH_OUTPUT_DIR}/{sanitized_topic}_{timestamp}.md"

def reset_session_state(state_dict: Dict[str, Any]) -> None:
    """
    Reset all tracked session state variables.
    
    Args:
        state_dict: Streamlit session state dictionary
    """
    for key in SESSION_KEYS.values():
        if key in state_dict:
            state_dict[key] = None if key != "messages" else []
            
    # Special cases
    state_dict[SESSION_KEYS["messages"]] = []
    state_dict[SESSION_KEYS["plan_approved"]] = False
    state_dict[SESSION_KEYS["research_iterations"]] = 0
    state_dict[SESSION_KEYS["all_papers"]] = []
    state_dict[SESSION_KEYS["plan_generated"]] = False
    state_dict[SESSION_KEYS["task_iterations"]] = {}
    state_dict[SESSION_KEYS["iterations_configured"]] = False
    state_dict[SESSION_KEYS["research_complete"]] = False
    state_dict[SESSION_KEYS["debug_mode"]] = False
    
    # If agent exists, reset its state too
    if state_dict.get(SESSION_KEYS["agent"]):
        agent = state_dict[SESSION_KEYS["agent"]]
        agent.accumulated_knowledge = ""
        agent.task_knowledge = {}
        agent.current_papers = []
>>>>>>> Stashed changes
