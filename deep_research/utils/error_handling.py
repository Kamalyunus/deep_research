"""
Error handling utilities for the Deep Research Agent.
"""
import logging
import traceback
import json, re
from typing import Dict, Any, Optional, Type, TypeVar, Callable, List, Union
from pydantic import BaseModel, ValidationError
import functools
import time

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class ResearchError(Exception):
    """Base exception class for research agent errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

class SourceError(ResearchError):
    """Exception raised for errors in source retrieval."""
    pass

class ParsingError(ResearchError):
    """Exception raised for errors in parsing LLM responses."""
    pass

class LLMError(ResearchError):
    """Exception raised for errors in LLM invocation."""
    pass

class PlanError(ResearchError):
    """Exception raised for errors in research plan generation."""
    pass

class ExecutionError(ResearchError):
    """Exception raised for errors in research execution."""
    pass

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from text that may contain markdown and other content.
    Improved with more robust extraction patterns and handling.
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Extracted JSON as a dictionary
        
    Raises:
        ParsingError: If JSON cannot be extracted
    """
    # Log the input for debugging
    logger.debug(f"Attempting to extract JSON from text: {text[:300]}...")
    
    # Pattern 1: Look for content between ```json and ```
    json_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    json_blocks = re.findall(json_block_pattern, text)
    
    if json_blocks:
        for block in json_blocks:
            try:
                # Clean the block of any invalid control characters
                clean_block = re.sub(r'[\x00-\x1F\x7F]', '', block.strip())
                # Fix common JSON formatting issues
                clean_block = clean_block.replace('\\"', '"').replace('\\n', '\n')
                
                result = json.loads(clean_block)
                logger.info("Successfully extracted JSON from code block")
                return result
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON from code block: {e}")
                continue
    
    # Pattern 2: Try to find JSON by looking for { and } braces (full object)
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            # Clean the string of any invalid characters
            json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
            # Fix common issues with escaped quotes
            json_str = json_str.replace('\\"', '"').replace('\\n', '\n')
            
            result = json.loads(json_str)
            logger.info("Successfully extracted JSON object using brace matching")
            return result
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse JSON from brace matching: {e}")
    
    # Pattern 3: Try to extract a JSON object with relaxed parsing
    try:
        # Find all potential JSON objects
        potential_objects = re.findall(r'({[^{}]*(?:{[^{}]*})*[^{}]*})', text)
        for obj in potential_objects:
            try:
                # Clean the string
                clean_obj = re.sub(r'[\x00-\x1F\x7F]', '', obj)
                # Fix common issues
                clean_obj = clean_obj.replace('\\"', '"').replace('\\n', '\n')
                
                result = json.loads(clean_obj)
                logger.info("Successfully extracted JSON with relaxed parsing")
                return result
            except:
                continue
    except Exception as e:
        logger.debug(f"Relaxed JSON parsing failed: {e}")
    
    # Pattern 4: Try to find JSON array by looking for [ and ] brackets
    try:
        start_idx = text.find('[')
        end_idx = text.rfind(']') + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            # Clean the string
            json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
            # Fix common issues
            json_str = json_str.replace('\\"', '"').replace('\\n', '\n')
            
            result = json.loads(json_str)
            logger.info("Successfully extracted JSON array")
            return result
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse JSON array: {e}")
    
    # Pattern 5: Try to construct a JSON object from key-value patterns
    try:
        # Look for key: value patterns
        patterns = re.findall(r'"?([a-zA-Z_][a-zA-Z0-9_]*)"?\s*:\s*("[^"]*"|\'[^\']*\'|\[[^\]]*\]|{[^}]*}|-?\d+\.?\d*|\w+)', text)
        if patterns:
            json_obj = {}
            for key, value in patterns:
                key = key.strip('"\'')
                # Try to parse value
                try:
                    # If it looks like a JSON object or array
                    if (value.startswith('{') and value.endswith('}')) or (value.startswith('[') and value.endswith(']')):
                        json_obj[key] = json.loads(value)
                    # If it's a quoted string
                    elif (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                        json_obj[key] = value[1:-1]
                    # If it's a number
                    elif re.match(r'-?\d+\.?\d*', value):
                        if '.' in value:
                            json_obj[key] = float(value)
                        else:
                            json_obj[key] = int(value)
                    # Otherwise treat as string
                    else:
                        json_obj[key] = value
                except:
                    json_obj[key] = value
            
            if json_obj:
                logger.info("Successfully constructed JSON object from key-value patterns")
                return json_obj
    except Exception as e:
        logger.debug(f"Key-value pattern extraction failed: {e}")
    
    # If all patterns fail, construct a minimal object
    try:
        # Look for what might be the topic and objective
        topic_match = re.search(r'topic["\s:=]+"?([^"]+)"?', text, re.IGNORECASE)
        objective_match = re.search(r'objective["\s:=]+"?([^"]+)"?', text, re.IGNORECASE)
        
        if topic_match or objective_match:
            logger.warning("Constructing minimal JSON object from extracted fields")
            minimal_json = {}
            
            if topic_match:
                minimal_json["topic"] = topic_match.group(1).strip()
            else:
                minimal_json["topic"] = "Research Topic"
                
            if objective_match:
                minimal_json["objective"] = objective_match.group(1).strip()
            else:
                minimal_json["objective"] = "Research Objective"
                
            # Look for any task-like structures
            task_matches = re.findall(r'task[_\s]?(\d+)|description["\s:=]+"?([^"]+)"?', text, re.IGNORECASE)
            
            if task_matches:
                minimal_json["tasks"] = [{"task_id": 1, "description": "Research task", "queries": ["Research query"]}]
            
            return minimal_json
    except Exception as e:
        logger.debug(f"Minimal JSON construction failed: {e}")
    
    # If all patterns fail, raise an error
    logger.error("Could not extract valid JSON from text")
    raise ParsingError("Could not extract valid JSON from text", {"text_preview": text[:500]})

def parse_pydantic_from_llm(
    text: str, 
    model_class: Type[T], 
    fallback_generator: Optional[Callable[[], T]] = None
) -> T:
    """
    Parse a Pydantic model from LLM-generated text with improved validation handling.
    
    Args:
        text: Text from LLM potentially containing structured data
        model_class: Pydantic model class to parse into
        fallback_generator: Optional function to generate a fallback instance
        
    Returns:
        Instance of the specified Pydantic model
    """
    # Log the entire text for debugging
    logger.debug(f"Attempting to parse text into {model_class.__name__}: {text[:500]}...")
    
    # Strategy 1: Try direct parsing
    try:
        instance = model_class.parse_raw(text)
        logger.info(f"Successfully parsed {model_class.__name__} with direct parsing")
        return instance
    except Exception as direct_error:
        logger.debug(f"Direct parsing failed: {direct_error}")
    
    # Strategy 2: Try to extract JSON and parse
    try:
        json_data = extract_json_from_text(text)
        logger.debug(f"Extracted JSON data: {json.dumps(json_data)[:500]}...")
        
        # For ResearchPlan, handle task objects specifically
        if model_class.__name__ == "ResearchPlan" and "tasks" in json_data:
            # Make sure tasks is a list
            if not isinstance(json_data["tasks"], list):
                logger.warning("Tasks is not a list, converting to list")
                if isinstance(json_data["tasks"], dict):
                    json_data["tasks"] = [json_data["tasks"]]
                else:
                    json_data["tasks"] = []
            
            # Process each task
            processed_tasks = []
            for task in json_data["tasks"]:
                # Ensure task has required fields
                processed_task = {}
                processed_task["task_id"] = task.get("task_id", len(processed_tasks) + 1)
                processed_task["description"] = task.get("description", f"Research task {processed_task['task_id']}")
                
                # Handle queries
                queries = task.get("queries", [])
                if not isinstance(queries, list):
                    logger.warning(f"Queries for task {processed_task['task_id']} is not a list, converting")
                    if isinstance(queries, str):
                        queries = [queries]
                    else:
                        queries = [f"Research on {processed_task['description']}"]
                
                # Ensure all queries are strings
                processed_task["queries"] = [str(q) for q in queries]
                processed_tasks.append(processed_task)
            
            # Replace tasks in json_data
            json_data["tasks"] = processed_tasks
        
        try:
            # Try to parse with full validation
            instance = model_class.parse_obj(json_data)
            logger.info(f"Successfully parsed {model_class.__name__} from extracted JSON")
            return instance
        except ValidationError as validation_error:
            logger.warning(f"Validation error parsing {model_class.__name__}: {validation_error}")
            
            # Special handling for ResearchPlan
            if model_class.__name__ == "ResearchPlan":
                try:
                    # Extract the base fields
                    topic = json_data.get("topic", "Research Topic")
                    objective = json_data.get("objective", f"Research on {topic}")
                    
                    # Handle tasks
                    from deep_research.agent.models import ResearchTask  # Import here to avoid circular imports
                    tasks = []
                    
                    task_data = json_data.get("tasks", [])
                    for i, task in enumerate(task_data):
                        try:
                            # Create task with minimal validation
                            task_id = task.get("task_id", i+1)
                            description = task.get("description", f"Research task {task_id}")
                            
                            # Ensure queries is a list of strings
                            queries = task.get("queries", [])
                            if not isinstance(queries, list):
                                queries = [str(queries)]
                            else:
                                queries = [str(q) for q in queries]
                            
                            # If queries is empty, add a default query
                            if not queries:
                                queries = [f"Research on {description}"]
                                
                            # Create the task object manually
                            task_obj = ResearchTask(
                                task_id=task_id,
                                description=description,
                                queries=queries
                            )
                            tasks.append(task_obj)
                        except Exception as task_error:
                            logger.warning(f"Error creating task {i}: {task_error}")
                            # Add a minimal valid task
                            tasks.append(ResearchTask(
                                task_id=i+1,
                                description=f"Research task {i+1}",
                                queries=[f"Research on topic {i+1}"]
                            ))
                    
                    # If no tasks were created, add a default task
                    if not tasks:
                        tasks.append(ResearchTask(
                            task_id=1,
                            description=f"Research on {topic}",
                            queries=[f"{topic} research", f"{topic} methods", f"{topic} applications"]
                        ))
                    
                    # Create the plan manually
                    from deep_research.agent.models import ResearchPlan  # Import here to avoid circular imports
                    plan = ResearchPlan(
                        topic=topic,
                        objective=objective,
                        tasks=tasks
                    )
                    
                    logger.info(f"Created ResearchPlan with manual object creation: {len(tasks)} tasks")
                    return plan
                    
                except Exception as manual_error:
                    logger.error(f"Manual object creation failed: {manual_error}")
    except Exception as json_error:
        logger.debug(f"JSON extraction parsing failed: {json_error}")
    
    # Strategy 3: Look for field patterns in the text
    try:
        # Get field names from the model
        field_dict = {}
        for field_name in model_class.__fields__:
            # Look for field_name: value patterns
            pattern = rf'["\']?{field_name}["\']?\s*[:=]\s*["\']?(.*?)["\']?(?:[,}}]|\n)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                field_dict[field_name] = matches[0].strip()
        
        # Try to create an instance with the extracted fields
        if field_dict:
            try:
                instance = model_class.parse_obj(field_dict)
                logger.info(f"Successfully parsed {model_class.__name__} from field patterns")
                return instance
            except ValidationError:
                logger.warning(f"Validation failed for field pattern extraction")
    except Exception as field_error:
        logger.debug(f"Field pattern extraction failed: {field_error}")
    
    # Strategy 4: Use the fallback generator if provided
    if fallback_generator:
        try:
            logger.warning(f"All parsing strategies failed, using fallback generator for {model_class.__name__}")
            return fallback_generator()
        except Exception as fallback_error:
            logger.error(f"Fallback generator failed: {fallback_error}")
    
    # If all strategies fail, raise a detailed error
    raise ParsingError(
        f"Failed to parse {model_class.__name__} from LLM output",
        {"text_preview": text[:500], "model": model_class.__name__}
    )

def with_retry(
    max_attempts: int = 3, 
    backoff_factor: float = 2.0,
    exceptions_to_retry: List[Type[Exception]] = None
):
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        backoff_factor: Factor for exponential backoff
        exceptions_to_retry: List of exception types to retry on (defaults to Exception)
        
    Returns:
        Decorated function
    """
    if exceptions_to_retry is None:
        exceptions_to_retry = [Exception]
        
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except tuple(exceptions_to_retry) as e:
                    last_exception = e
                    
                    # Don't sleep after the last attempt
                    if attempt < max_attempts - 1:
                        sleep_time = backoff_factor ** attempt
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed with error: {e}. "
                            f"Retrying in {sleep_time:.2f} seconds."
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed. Last error: {e}")
            
            # If we get here, all attempts failed
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator

def format_error_for_display(
    error: Exception, 
    debug_mode: bool = False
) -> str:
    """
    Format an error for display in the UI.
    
    Args:
        error: The exception
        debug_mode: Whether to include detailed debug info
        
    Returns:
        Formatted error message
    """
    if isinstance(error, ResearchError):
        base_msg = f"Error: {error.message}"
        
        if debug_mode and error.details:
            details = json.dumps(error.details, indent=2)
            return f"{base_msg}\n\nDetails:\n```json\n{details}\n```"
        return base_msg
    
    # Format other exceptions
    error_msg = f"Error: {str(error)}"
    
    if debug_mode:
        stack_trace = traceback.format_exc()
        return f"{error_msg}\n\nStacktrace:\n```\n{stack_trace}\n```"
    
    return error_msg

def log_and_format_error(
    error: Exception, 
    context: str, 
    debug_mode: bool = False
) -> str:
    """
    Log an error with context and format it for display.
    
    Args:
        error: The exception
        context: Description of what was happening when the error occurred
        debug_mode: Whether to include detailed debug info
        
    Returns:
        Formatted error message
    """
    # Log the error
    logger.error(f"Error in {context}: {error}", exc_info=True)
    
    # Format for display
    return format_error_for_display(error, debug_mode)