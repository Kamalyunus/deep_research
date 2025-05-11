<<<<<<< Updated upstream
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
=======
"""
ArXiv integration for the Deep Research Agent.
"""
from langchain_community.retrievers import ArxivRetriever
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import logging
import time
import re

from ..utils.error_handling import with_retry, SourceError

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
            load_all_available_meta=True
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
    Includes improved academic citation information.

    Args:
        papers: List of Document objects from ArXiv

    Returns:
        List of dictionaries containing formatted paper metadata with citation info
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

        # Extract publication year for citation
        pub_year = "Unknown"
        if published:
            # Try to extract year from published date
            try:
                import re
                year_match = re.search(r'(19|20)\d{2}', published)
                if year_match:
                    pub_year = year_match.group(0)
            except:
                pass

        # Extract first author for citation
        first_author = "Unknown"
        if authors:
            try:
                # Try to get first author's last name
                if ',' in authors:
                    first_author = authors.split(',')[0].strip()
                elif ' and ' in authors:
                    first_author = authors.split(' and ')[0].strip()
                else:
                    # Just take the first author from the list
                    authors_list = authors.split()
                    if authors_list:
                        first_author = authors_list[-1]  # Assume last name is last
            except:
                pass

        # Create formatted citation
        citation = f"{first_author} ({pub_year})"

        # Extract sections if possible
        sections = extract_key_paper_sections(paper.page_content)

        # Save enhanced paper info for display
        paper_info.append({
            "type": "Paper",
            "id": paper_id,
            "title": title,
            "authors": authors,
            "published": published,
            "url": f"https://arxiv.org/abs/{paper_id}",
            "source_type": "Paper",
            "summary": summary[:500] + "..." if len(summary) > 500 else summary,
            "categories": categories,
            "content_length": len(paper.page_content),
            "citation": citation,
            "pub_year": pub_year,
            "first_author": first_author,
            "has_sections": bool(sections),
            "sections": list(sections.keys()) if sections else []
        })

    return paper_info

def extract_key_paper_sections(content: str) -> Dict[str, str]:
    """
    Attempt to extract key sections from paper content for better analysis.
    
    Args:
        content: The paper content text
        
    Returns:
        Dictionary with key sections (intro, methods, results, conclusion)
    """
    sections = {}
    
    # Look for introduction
    intro_markers = ["introduction", "background", "1. introduction", "i. introduction"]
    for marker in intro_markers:
        if marker in content.lower():
            pos = content.lower().find(marker)
            intro_end = find_next_section(content, pos + len(marker))
            if intro_end > 0:
                sections["introduction"] = content[pos:intro_end].strip()
                break
    
    # Look for methods/methodology
    method_markers = ["method", "methodology", "approach", "experimental setup"]
    for marker in method_markers:
        if marker in content.lower():
            pos = content.lower().find(marker)
            method_end = find_next_section(content, pos + len(marker))
            if method_end > 0:
                sections["methods"] = content[pos:method_end].strip()
                break
    
    # Look for results
    result_markers = ["result", "results", "findings", "evaluation"]
    for marker in result_markers:
        if marker in content.lower():
            pos = content.lower().find(marker)
            result_end = find_next_section(content, pos + len(marker))
            if result_end > 0:
                sections["results"] = content[pos:result_end].strip()
                break
    
    # Look for conclusion
    conclusion_markers = ["conclusion", "conclusions", "discussion", "summary", "future work"]
    for marker in conclusion_markers:
        if marker in content.lower():
            pos = content.lower().find(marker)
            # For conclusion, we'll take text until the end or the references/bibliography
            end_markers = ["references", "bibliography", "acknowledgments"]
            conclusion_end = len(content)
            for end_marker in end_markers:
                if end_marker in content.lower()[pos:]:
                    end_pos = content.lower()[pos:].find(end_marker) + pos
                    if end_pos < conclusion_end:
                        conclusion_end = end_pos
            
            sections["conclusion"] = content[pos:conclusion_end].strip()
            break
    
    return sections

def find_next_section(content: str, start_pos: int) -> int:
    """
    Find the position where the next section begins.
    
    Args:
        content: The content string
        start_pos: Starting position
        
    Returns:
        Position of next section or end of string
    """
    section_markers = [
        "\n1.", "\n2.", "\n3.", "\n4.", "\n5.",
        "\nI.", "\nII.", "\nIII.", "\nIV.", "\nV.",
        "\nsection", "\nchapter",
        "introduction", "background", "method", "approach", "result", 
        "discussion", "conclusion", "reference"
    ]
    
    next_pos = len(content)
    for marker in section_markers:
        # Find the marker after our starting position
        pos = content.lower().find(marker, start_pos + 100)  # Skip at least 100 chars
        if pos != -1 and pos < next_pos:
            next_pos = pos
    
    return next_pos if next_pos < len(content) else len(content)

def chunk_paper_content(paper: Document, chunk_size: int = 16000, chunk_overlap: int = 1500) -> List[str]:
    """
    Split paper content into manageable chunks with larger size and more overlap.
    Uses improved chunking strategy for better context preservation.

    Args:
        paper: Document object containing paper content
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks (increased for better context)

    Returns:
        List of text chunks with focus on important sections
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # First check if we can extract key paper sections
    sections = extract_key_paper_sections(paper.page_content)

    # If we successfully extracted key sections, prioritize them
    if sections and len(sections) >= 2:
        # Prioritize introduction, methods, results, and conclusion sections
        key_chunks = []
        for section_name, section_text in sections.items():
            # Include section header in chunk for better context
            formatted_section = f"--- {section_name.upper()} ---\n{section_text}"
            key_chunks.append(formatted_section)

        return key_chunks

    # Fallback to enhanced chunking when sections can't be extracted
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(paper.page_content)

    # Enhanced chunk selection strategy
    if len(chunks) > 3:
        # If we have many chunks, select strategically: beginning (intro),
        # first quarter, middle (methods/results), third quarter, and end (conclusion)
        return [
            chunks[0],
            chunks[len(chunks)//4],
            chunks[len(chunks)//2],
            chunks[3*len(chunks)//4],
            chunks[-1]
        ]
    else:
        # If we have just a few chunks, return all with higher limit
        return [chunk[:12000] for chunk in chunks]
>>>>>>> Stashed changes
