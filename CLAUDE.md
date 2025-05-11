# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Install: `pip install -e .`
- Run app: `streamlit run deep_research/app.py`

## Code Style Guidelines
- **Imports**: Standard library first, then third-party, then project imports
- **Type Annotations**: Use typing module (List, Dict, Optional) and Pydantic models
- **Error Handling**: Use specialized exception classes from error_handling.py
- **Retry Logic**: Use the @with_retry decorator for operations that might fail
- **Naming**: Classes: PascalCase, functions/variables: snake_case
- **Documentation**: Google-style docstrings with Args/Returns sections
- **Validation**: Use Pydantic validators for data validation (@validator decorator)
- **Parsing**: Use extract_json_from_text and parse_pydantic_from_llm for LLM responses
- **Logging**: Use the logger with appropriate levels (debug, info, warning, error)

## Project Architecture
- Agent-based architecture using LangChain and LangGraph
- Modular components with clear separation of concerns
- Pydantic models for structured data validation
- Robust error handling with custom exception classes