"""
ArXiv integration for the Deep Research Agent.
"""
from langchain_community.retrievers import ArxivRetriever
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import logging
import time

from ..utils.error_handling import with_retry, SourceError
from .source_processor import SourceProcessor  # Import the source processor

logger = logging.getLogger(__name__)

@with_retry(max_attempts=3, backoff_factor=2.0)
def search_arxiv(query: str, max_papers: int = 2) -> List[Document]:
    """
    Search ArXiv for papers matching the query with enhanced content extraction
    and rate limit handling.
    
    Args:
        query: The search query
        max_papers: Maximum number of papers to retrieve
        
    Returns:
        List of Document objects containing paper content and metadata
    """
    try:
        logger.info(f"Searching ArXiv for: '{query}', max papers: {max_papers}")
        retriever = ArxivRetriever(
            doc_content_chars_max=40000,  # Increased for larger context window
            load_max_docs=max_papers,
            load_all_available_meta=True,
            get_full_documents=True  # Added parameter to get the full document metadata
        )
        results = retriever.get_relevant_documents(query)
        
        if not results:
            logger.warning(f"No ArXiv papers found for query: {query}")
            
        logger.info(f"Found {len(results)} papers for query: '{query}'")
        return results
        
    except Exception as e:
        # Check if error might be related to rate limiting
        rate_limit_indicators = ["429", "too many requests", "rate limit", "timeout", "connection"]
        is_rate_limit = any(indicator in str(e).lower() for indicator in rate_limit_indicators)
        
        if is_rate_limit:
            logger.warning(f"Rate limit detected in ArXiv API: {e}")
            raise SourceError(f"ArXiv rate limit encountered: {str(e)}", 
                             {"query": query, "rate_limited": True})
        else:
            logger.error(f"Error retrieving papers from ArXiv: {e}")
            raise SourceError(f"Failed to retrieve ArXiv papers: {str(e)}", 
                             {"query": query})