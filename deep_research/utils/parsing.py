"""
Response parsing utilities for the Deep Research Agent.
"""
import json
import re
from typing import Dict, List, Any, Optional, Type, TypeVar, Callable, Union
import logging
from pydantic import BaseModel, ValidationError

from .error_handling import ParsingError, extract_json_from_text

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

def parse_pydantic_from_text(
    text: str, 
    model_class: Type[T], 
    fallback_generator: Optional[Callable[[], T]] = None
) -> T:
    """
    Parse a Pydantic model from text with multiple fallback strategies.
    
    Args:
        text: Text potentially containing structured data
        model_class: Pydantic model class to parse into
        fallback_generator: Optional function to generate a fallback instance
        
    Returns:
        Instance of the specified Pydantic model
        
    Raises:
        ParsingError: If parsing fails and no fallback is provided
    """
    # Strategy 1: Try direct parsing as JSON
    try:
        return model_class.parse_raw(text)
    except (ValidationError, json.JSONDecodeError) as direct_error:
        logger.debug(f"Direct parsing failed: {direct_error}")
    
    # Strategy 2: Try to extract JSON and parse
    try:
        json_data = extract_json_from_text(text)
        return model_class.parse_obj(json_data)
    except (ParsingError, ValidationError) as json_error:
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
            return model_class.parse_obj(field_dict)
    except Exception as field_error:
        logger.debug(f"Field pattern extraction failed: {field_error}")
    
    # Strategy 4: Use the fallback generator if provided
    if fallback_generator:
        try:
            return fallback_generator()
        except Exception as fallback_error:
            logger.warning(f"Fallback generator failed: {fallback_error}")
    
    # If all strategies fail, raise a detailed error
    raise ParsingError(
        f"Failed to parse {model_class.__name__} from text",
        {"text_preview": text[:500], "model": model_class.__name__}
    )

def extract_code_blocks(text: str, language: Optional[str] = None) -> List[str]:
    """
    Extract code blocks from markdown text.
    
    Args:
        text: Markdown text containing code blocks
        language: Optional language to filter by
        
    Returns:
        List of extracted code blocks
    """
    if language:
        # Match code blocks with the specified language
        pattern = rf'```{language}\s*([\s\S]*?)\s*```'
    else:
        # Match any code blocks
        pattern = r'```(?:\w+)?\s*([\s\S]*?)\s*```'
    
    code_blocks = re.findall(pattern, text)
    return code_blocks

def extract_list_items(text: str) -> List[str]:
    """
    Extract list items from markdown text.
    
    Args:
        text: Markdown text containing list items
        
    Returns:
        List of extracted items
    """
    # Match both unordered and ordered list items
    pattern = r'^\s*(?:[-*+]|\d+\.)\s+(.*?)$'
    
    items = []
    for line in text.split('\n'):
        match = re.match(pattern, line)
        if match:
            items.append(match.group(1).strip())
    
    return items

def extract_section_content(text: str, section_name: str) -> Optional[str]:
    """
    Extract content from a specific section in markdown text.
    
    Args:
        text: Markdown text with sections
        section_name: Name of the section to extract
        
    Returns:
        Content of the section or None if not found
    """
    # Create pattern for the section heading (supports different heading levels)
    patterns = [
        rf'#+\s+{re.escape(section_name)}\s*\n+([\s\S]*?)(?=\n+#+\s+|$)',  # Standard heading
        rf'{re.escape(section_name)}:?\s*\n+([\s\S]*?)(?=\n\w+:?\s*\n+|$)'  # Title with colon
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None

def extract_key_value_pairs(text: str) -> Dict[str, str]:
    """
    Extract key-value pairs from text.
    
    Args:
        text: Text containing key-value pairs
        
    Returns:
        Dictionary of extracted key-value pairs
    """
    # Pattern for "key: value" or "key = value" formats
    pattern = r'(\w+[\s\w]*?)[:=]\s*(.*?)(?=\n\w+[\s\w]*?[:=]|\Z)'
    
    pairs = {}
    for match in re.finditer(pattern, text, re.DOTALL):
        key = match.group(1).strip()
        value = match.group(2).strip()
        pairs[key] = value
    
    return pairs

def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """
    Validate that required fields are present in data.
    
    Args:
        data: Dictionary of data
        required_fields: List of required field names
        
    Returns:
        List of missing field names
    """
    missing_fields = []
    
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == "":
            missing_fields.append(field)
    
    return missing_fields

def clean_and_parse_json(text: str) -> Dict[str, Any]:
    """
    Clean text and attempt to parse JSON.
    
    Args:
        text: Text to clean and parse
        
    Returns:
        Parsed JSON data
        
    Raises:
        ParsingError: If cleaning and parsing fails
    """
    # Remove any non-JSON prefix/suffix content
    try:
        # Find the first { and last }
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_text = text[start_idx:end_idx]
            
            # Handle potential escape issues
            json_text = json_text.replace('\\"', '"')
            json_text = re.sub(r'\\([^"])', r'\1', json_text)
            
            # Parse the cleaned JSON
            return json.loads(json_text)
    except json.JSONDecodeError:
        # Try alternative approaches if the first method fails
        pass
    
    # Try removing markdown code block syntax
    cleaned_text = re.sub(r'```(?:json)?\s*', '', text)
    cleaned_text = cleaned_text.replace('```', '')
    
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass
    
    # If all attempts fail, raise an error
    raise ParsingError("Failed to clean and parse JSON", {"text_preview": text[:500]})