"""
Text processing utilities for the Deep Research Agent.
"""
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional

def remove_thinking_tags(text: Optional[str]) -> str:
    """
    Remove any content between <think> and </think> tags.
    
    Args:
        text: The input text
        
    Returns:
        Text with thinking sections removed
    """
    if text is None:
        return ""
        
    if not isinstance(text, str):
        text = str(text)
        
    # Remove thinking sections using regex
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Also remove any standalone tags that might remain
    cleaned_text = cleaned_text.replace('<think>', '').replace('</think>', '')
    
    return cleaned_text

def escape_curly_braces(text: Optional[str]) -> str:
    """
    Escape any standalone curly braces in the text to prevent string formatting errors.
    
    Args:
        text: The input text that might contain standalone curly braces
        
    Returns:
        Text with standalone curly braces escaped
    """
    if text is None:
        return ""
    
    # Double any standalone curly braces to escape them
    # First replace {{ with a temporary marker to avoid double-escaping
    text = text.replace('{{', '!!DOUBLE_OPEN!!')
    text = text.replace('}}', '!!DOUBLE_CLOSE!!')
    
    # Now replace single braces
    text = text.replace('{', '{{')
    text = text.replace('}', '}}')
    
    # Restore the originally escaped braces
    text = text.replace('!!DOUBLE_OPEN!!', '{{')
    text = text.replace('!!DOUBLE_CLOSE!!', '}}')
    
    return text

def escape_math_expressions(text: Optional[str]) -> str:
    """
    Escape mathematical expressions to prevent string formatting errors.
    
    Args:
        text: The input text that might contain mathematical expressions
        
    Returns:
        Text with mathematical expressions safely escaped
    """
    if not text or not isinstance(text, str):
        return "" if text is None else str(text)
    
    # Escape curly braces that aren't already escaped
    text = re.sub(r'(?<!{){(?!{)', '{{', text)
    text = re.sub(r'(?<!})}(?!})', '}}', text)
    
    # Handle common math notations with carets (^) - super/subscripts
    math_patterns = [
        # Match patterns like n^2, x^i, etc. 
        r'(\b[a-zA-Z0-9]+\^[a-zA-Z0-9]+\b)',
        # Match patterns like 2^n, 10^6, etc.
        r'(\b[0-9]+\^[a-zA-Z0-9]+\b)',
        # Match O(n^2), Θ(n^2), etc.
        r'([OΘΩoθω]\([a-zA-Z0-9\s]*\^[a-zA-Z0-9\s]*\))',
        # Match log expressions like log^2(n)
        r'(log\^[0-9]+\([a-zA-Z0-9]+\))',
        # Match complex subscript patterns like a_i, x_{i,j}
        r'([a-zA-Z0-9]\_\{[^\}]+\})',
        # Match simple subscript patterns like a_i
        r'([a-zA-Z0-9]\_[a-zA-Z0-9])'
    ]
    
    # Process each pattern
    for pattern in math_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Create a version of the match with all { and } escaped
            escaped_match = match.replace('{', '{{').replace('}', '}}')
            # Replace the original with the escaped version
            text = text.replace(match, escaped_match)
    
    return text

def escape_template_variables(text: Optional[str]) -> str:
    """
    Escape any text that looks like a template variable.
    Particularly useful for mathematical expressions like n^2, log^2 n, etc.
    
    Args:
        text: The input text that might contain template variable-like patterns
        
    Returns:
        Text with potential template variables escaped
    """
    if text is None:
        return ""
        
    # Find patterns that look like template variables: {word} or {word^word}
    # This regex looks for common mathematical notation patterns
    math_patterns = [
        r'(\b[a-zA-Z]+\^[0-9]+\b)',         # Matches n^2, d^2, etc.
        r'(\b[a-zA-Z]+\^[0-9]+\s*\\log[^\s]*\b)',  # Matches n^2 \log, etc.
        r'(\b\\log\^[0-9]+\s*[a-zA-Z]+\b)',  # Matches \log^2 n, etc.
        r'(\b[a-zA-Z]+\s*\\cdot\s*[a-zA-Z]+\b)'  # Matches n \cdot m, etc.
    ]
    
    result = text
    for pattern in math_patterns:
        # Find all matches
        matches = re.findall(pattern, result)
        
        # Replace each match with the escaped version
        for match in matches:
            escaped = match.replace('{', '{{').replace('}', '}}')
            result = result.replace(match, escaped)
    
    # Additional check for explicit template variables that need escaping
    potential_vars = re.findall(r'{([^{}]+)}', result)
    for var in potential_vars:
        if re.search(r'[a-zA-Z]+[\^\\]', var):  # Likely a math expression
            result = result.replace(f"{{{var}}}", f"{{{{{var}}}}}")
    
    return result

def safe_process_text(text: Optional[str]) -> str:
    """
    Safely process text by removing thinking tags and escaping curly braces.
    
    Args:
        text: The input text
        
    Returns:
        Safely processed text
    """
    if text is None:
        return ""
    
    # First remove thinking tags
    text = remove_thinking_tags(text)
    
    # Then escape curly braces
    text = escape_curly_braces(text)
    
    return text

def chunk_text(text: str, chunk_size: int = 24000, chunk_overlap: int = 2000) -> List[str]:
    """
    Split text into manageable chunks with specified size and overlap.
    
    Args:
        text: Text content to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    
    return chunks

def clean_web_text(text: str) -> str:
    """
    Clean extracted text from web pages by removing extra whitespace and empty lines.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Replace multiple newlines with double newline
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Strip whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    
    # Remove empty lines
    lines = [line for line in lines if line]
    
    return '\n'.join(lines)