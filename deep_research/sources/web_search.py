"""
Web search integration for the Deep Research Agent.
"""
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import logging
import requests
from bs4 import BeautifulSoup
import time
import re

from ..utils.error_handling import with_retry, SourceError
from ..utils.text_processing import clean_web_text

logger = logging.getLogger(__name__)

@with_retry(max_attempts=3, backoff_factor=2.0)
def search_web(
    query: str, 
    max_results: int = 3, 
    fetch_content: bool = True
) -> List[Document]:
    """
    Search the web using DuckDuckGo with enhanced content retrieval and rate limit handling.
    
    Args:
        query: The search query
        max_results: Maximum number of results to retrieve
        fetch_content: Whether to attempt fetching full content from URLs
        
    Returns:
        List of Document objects containing web content and metadata
    """
    try:
        logger.info(f"Searching web for: '{query}', max results: {max_results}")
        search = DuckDuckGoSearchAPIWrapper()
        results = search.results(query, max_results=max_results)
        
        if not results:
            logger.warning(f"No web results found for query: {query}")
            return []
            
        # Convert to Document objects for consistent handling
        documents = []
        for result in results:
            # Prepare to enhance content
            full_text = ""
            text_length = 0
            
            # Try to fetch the actual webpage content if enabled
            if fetch_content:
                try:
                    full_text, text_length = fetch_webpage_content(result['link'])
                except Exception as e:
                    logger.warning(f"Failed to fetch content from {result['link']}: {e}")
            
            # Create more detailed content string
            content = f"Title: {result['title']}\n\n"
            content += f"Link: {result['link']}\n\n"
            content += f"Snippet: {result['snippet']}\n\n"
            
            # Add any additional fields if available
            if 'published' in result:
                content += f"Published: {result['published']}\n\n"
            
            # Add extracted full content if available
            if full_text:
                content += f"Full Content:\n{full_text}\n\n"
                
            metadata = {
                "title": result['title'],
                "link": result['link'],
                "snippet": result['snippet'],
                "source_type": "Web",  # Explicitly set the source type
                "content_length": text_length  # Add content length
            }
            
            # Add any additional metadata that might be available
            for key in ['published', 'source', 'description']:
                if key in result:
                    metadata[key] = result[key]
                    
            documents.append(Document(page_content=content, metadata=metadata))
            
        logger.info(f"Found {len(documents)} web results for query: '{query}'")
        return documents
        
    except Exception as e:
        if "Ratelimit" in str(e):
            logger.warning(f"Rate limit hit in web search: {e}")
            raise SourceError(f"Web search rate limit encountered: {str(e)}", 
                             {"query": query, "rate_limited": True})
        else:
            logger.error(f"Error searching the web: {e}")
            raise SourceError(f"Failed to search the web: {str(e)}", 
                             {"query": query})

def fetch_webpage_content(url: str, max_content_length: int = 24000) -> tuple:
    """
    Fetch and extract main content from a webpage.
    
    Args:
        url: The URL to fetch content from
        max_content_length: Maximum length of content to return
        
    Returns:
        Tuple of (extracted_text, original_text_length)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        logger.debug(f"Fetching content from: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Use BeautifulSoup to parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
        
        # Extract main content - try various methods
        main_content = extract_main_content(soup)
        
        # Get text and clean it
        text = main_content.get_text(separator='\n')
        text = clean_web_text(text)
        
        # Store original length before truncation
        original_length = len(text)
        
        # Truncate if too long
        if len(text) > max_content_length:
            # Take beginning, middle and end for better coverage
            begin = text[:max_content_length//3]
            middle_start = original_length//2 - max_content_length//6
            middle = text[middle_start:middle_start + max_content_length//3]
            end = text[-(max_content_length//3):]
            text = f"{begin}\n\n[...content truncated...]\n\n{middle}\n\n[...content truncated...]\n\n{end}"
        
        logger.debug(f"Successfully fetched content from {url} ({original_length} chars)")
        return text, original_length
        
    except Exception as e:
        logger.warning(f"Error fetching webpage {url}: {e}")
        return "", 0

def extract_main_content(soup):
    """
    Extract the main content from a webpage using heuristics.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        BeautifulSoup object with main content
    """
    # Try to find main content by common identifiers
    main_candidates = [
        soup.find('main'),
        soup.find(id='content'),
        soup.find(id='main'),
        soup.find(class_='content'),
        soup.find(class_='main'),
        soup.find('article'),
        soup.find(id='article'),
        soup.find(class_='article'),
        soup.find(id='post'),
        soup.find(class_='post')
    ]
    
    # Use the first valid candidate
    for candidate in main_candidates:
        if candidate:
            return candidate
    
    # If no candidate is found, return the body or the whole document
    return soup.body or soup

def process_web_results(web_results: List[Document]) -> List[Dict[str, Any]]:
    """
    Extract and format web search results for display with enhanced metadata.
    
    Args:
        web_results: List of Document objects from web search
        
    Returns:
        List of dictionaries containing formatted web result metadata
    """
    source_info = []
    
    for result in web_results:
        source_id = result.metadata.get("link", "")
        title = result.metadata.get("title", "Untitled")
        
        # Get content length from metadata if available
        content_length = result.metadata.get("content_length", 0)
        
        # Extract a better preview from the page content
        preview = ""
        if "Full Content:" in result.page_content:
            # Try to get a meaningful preview from the full content
            full_content_part = result.page_content.split("Full Content:")[1].strip()
            # Get first 300 characters for preview
            preview = full_content_part[:300] + "..." if len(full_content_part) > 300 else full_content_part
        else:
            # Fall back to snippet
            preview = result.metadata.get("snippet", "No preview available")
        
        # Save enhanced source info for display
        source_info.append({
            "type": "Web",
            "id": source_id,
            "title": title,
            "url": source_id,
            "snippet": result.metadata.get("snippet", "No snippet available"),
            "source_type": "Web",  # Explicitly set as "Web"
            "preview": preview,  # Add a better preview
            "content_length": content_length,  # Add content length
            "published": result.metadata.get("published", "Unknown date")  # Add published date if available
        })
    
    return source_info

def extract_key_web_sections(content: str) -> Dict[str, str]:
    """
    Attempt to extract key sections from web content for better analysis.
    
    Args:
        content: The web page content text
        
    Returns:
        Dictionary with key sections
    """
    sections = {}
    
    # Look for header/title area (first few paragraphs)
    paragraphs = content.split('\n\n')
    if len(paragraphs) >= 2:
        sections["header"] = '\n\n'.join(paragraphs[:2])
    
    # Try to identify introduction (early paragraphs)
    if len(paragraphs) >= 4:
        sections["introduction"] = '\n\n'.join(paragraphs[2:4])
    
    # Try to identify main content (middle paragraphs)
    if len(paragraphs) >= 6:
        mid_point = len(paragraphs) // 2
        sections["main_content"] = '\n\n'.join(paragraphs[mid_point-1:mid_point+1])
    
    # Try to identify conclusion (last paragraphs)
    if len(paragraphs) >= 2:
        sections["conclusion"] = '\n\n'.join(paragraphs[-2:])
    
    return sections