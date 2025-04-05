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

def process_papers(papers: List[Document]) -> List[Dict[str, Any]]:
    """
    Extract and format paper information for display with enhanced metadata.
    
    Args:
        papers: List of Document objects from ArXiv
        
    Returns:
        List of dictionaries containing formatted paper metadata
    """
    paper_info = []
    
    for paper in papers:
        # Extract more comprehensive metadata
        paper_id = paper.metadata.get("entry_id", "")
        if "abs/" in paper_id:
            paper_id = paper_id.split("abs/")[-1]
            
        title = paper.metadata.get("Title", "Untitled")
        authors = paper.metadata.get("Authors", "Unknown")
        published = paper.metadata.get("Published", "Unknown date")
        summary = paper.metadata.get("Summary", "No abstract available")
        categories = paper.metadata.get("Categories", "No categories")
        
        # Save enhanced paper info for display
        paper_info.append({
            "type": "Paper",  # Using "Paper" instead of "ArXiv"
            "id": paper_id,
            "title": title,
            "authors": authors,
            "published": published,
            "url": f"https://arxiv.org/abs/{paper_id}",
            "source_type": "Paper",  # Explicitly set as "Paper"
            "summary": summary[:10000] + "..." if len(summary) > 10000 else summary,
            "categories": categories,  # Add categories for better classification
            "content_length": len(paper.page_content)  # Track content size
        })
    
    return paper_info

# Use the chunk_content function from SourceProcessor instead of duplicating
def chunk_paper_content(paper: Document, chunk_size: int = 24000, chunk_overlap: int = 2000) -> List[str]:
    """
    Split paper content into manageable chunks with larger size and more overlap.
    
    Args:
        paper: Document object containing paper content
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = SourceProcessor.chunk_content(paper.page_content, chunk_size, chunk_overlap)
    
    # Process chunks in a smarter way
    if len(chunks) > 2:
        # If we have many chunks, return first, middle and last for better coverage
        return [chunks[0], chunks[len(chunks)//2], chunks[-1]]
    else:
        # If we have just a few chunks, return all but limit size
        return [chunk[:8000] for chunk in chunks]