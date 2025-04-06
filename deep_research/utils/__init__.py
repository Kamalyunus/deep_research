"""
Utility modules for the Deep Research Agent.
"""
from deep_research.utils.logger import (
    setup_logger,
    get_logger,
    configure_root_logger,
    set_log_level
)
from deep_research.utils.text_processing import (
    remove_thinking_tags,
    escape_curly_braces,
    escape_math_expressions,
    escape_template_variables,
    safe_process_text,
    chunk_text,
    clean_web_text
)