<<<<<<< Updated upstream
"""
Unified source processing for research documents.
"""
import logging
import re
from typing import List, Dict, Any, Tuple, Optional, Union
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..utils.text_processing import clean_web_text

logger = logging.getLogger(__name__)

class SourceProcessor:
    """
    Processor for research sources that handles different source types consistently.
    """
    
    @staticmethod
    def process_research_sources(
        sources: List[Document], 
        source_type: str
    ) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
        """
        Process research sources into message context and display information.
        
        Args:
            sources: List of Document objects from ArXiv or web search
            source_type: Type of sources ("arxiv" or "web")
            
        Returns:
            Tuple containing:
            - List of tuples (role, content) for the message context
            - List of dictionaries with source information for display
        """
        if source_type.lower() == "arxiv":
            return SourceProcessor._process_arxiv_sources(sources)
        elif source_type.lower() == "web":
            return SourceProcessor._process_web_sources(sources)
        else:
            logger.warning(f"Unknown source type: {source_type}")
            return [], []
    
    @staticmethod
    def _process_arxiv_sources(
        papers: List[Document]
    ) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
        """
        Process ArXiv papers into context messages and display info.
        
        Args:
            papers: List of Document objects from ArXiv
            
        Returns:
            Tuple of context messages and source info
        """
        context_messages = []
        source_info = []
        
        for i, paper in enumerate(papers):
            # Extract paper metadata
            paper_id = paper.metadata.get("entry_id", "").split("abs/")[-1]
            title = paper.metadata.get("Title", "Untitled")
            authors = paper.metadata.get("Authors", "Unknown")
            published = paper.metadata.get("Published", "Unknown date")
            summary = paper.metadata.get("Summary", "No abstract available")
            
            # Extract content and sections
            content = paper.page_content
            paper_sections = SourceProcessor.extract_key_paper_sections(content)
            
            # Format content with section markers if available
            content_display = ""
            if paper_sections:
                for section_name, section_text in paper_sections.items():
                    content_display += f"\n--- {section_name.upper()} ---\n"
                    # Limit each section to reasonable length
                    if len(section_text) > 10000:
                        content_display += f"{section_text[:10000]}...\n"
                    else:
                        content_display += f"{section_text}\n"
            else:
                # If sections couldn't be identified, use beginning and end approach
                if len(content) > 10000:
                    # Take the first 5000 and last 5000 chars to capture both intro and conclusions
                    content_display = f"{content[:5000]}...\n\n[Content truncated]...\n\n{content[-5000:]}"
                else:
                    content_display = content
            
            # Create context message with more detailed structure
            paper_content = f"""
            Paper {i+1}: {title}
            Authors: {authors}
            Published: {published}
            ID: {paper_id}
            
            Abstract:
            {summary}
            
            Content:
            {content_display}
            """
            context_messages.append(("user", paper_content))
            
            # Save paper info for display, adding more metadata
            source_info.append({
                "id": paper_id,
                "title": title,
                "authors": authors,
                "published": published,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "source_type": "Paper",  # Explicitly set as "Paper" not "ArXiv"
                "summary": summary[:300] + "..." if len(summary) > 300 else summary,
                "content_length": len(content)  # Track content size
            })
        
        return context_messages, source_info
    
    @staticmethod
    def _process_web_sources(
        web_results: List[Document]
    ) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
        """
        Process web search results into context messages and display info.
        
        Args:
            web_results: List of Document objects from web search
            
        Returns:
            Tuple of context messages and source info
        """
        context_messages = []
        source_info = []
        
        for i, result in enumerate(web_results):
            # Extract result metadata
            title = result.metadata.get("title", "Untitled")
            url = result.metadata.get("link", "No URL")
            snippet = result.metadata.get("snippet", "No snippet available")
            
            # Extract content and try to identify key sections
            content = result.page_content
            start_idx = content.find("Full Content:")
            
            if start_idx > 0:
                full_content = content[start_idx + len("Full Content:"):]
                web_sections = SourceProcessor.extract_key_web_sections(full_content)
                
                # Format content with section markers if available
                content_display = ""
                if web_sections:
                    for section_name, section_text in web_sections.items():
                        content_display += f"\n--- {section_name.upper()} ---\n"
                        # Limit each section to reasonable length
                        if len(section_text) > 5000:
                            content_display += f"{section_text[:5000]}...\n"
                        else:
                            content_display += f"{section_text}\n"
                else:
                    # If sections couldn't be identified, use a preview approach
                    if len(full_content) > 12000:
                        # Take beginning, middle, and end for better coverage
                        content_display = f"{full_content[:3000]}...\n\n"
                        middle_start = len(full_content) // 2 - 1500
                        content_display += f"[Middle content]\n{full_content[middle_start:middle_start+3000]}...\n\n"
                        content_display += f"[End content]\n{full_content[-3000:]}"
                    else:
                        content_display = full_content
            else:
                content_display = content
            
            # Create context message with more structured information
            web_content = f"""
            Web Result {i+1}: {title}
            URL: {url}
            
            Snippet:
            {snippet}
            
            Content:
            {content_display}
            """
            context_messages.append(("user", web_content))
            
            # Save web info for display with additional data
            source_info.append({
                "id": url,
                "title": title,
                "url": url,
                "snippet": snippet,
                "source_type": "Web",  # Explicitly set as "Web"
                "content_length": len(content)  # Track content size
            })
        
        return context_messages, source_info
    
    @staticmethod
    def chunk_content(
        content: str, 
        chunk_size: int = 24000, 
        chunk_overlap: int = 2000
    ) -> List[str]:
        """
        Split content into manageable chunks with larger size and overlap.
        
        Args:
            content: Text content to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(content)
        
        return chunks
    
    @staticmethod
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
                intro_end = SourceProcessor._find_next_section(content, pos + len(marker))
                if intro_end > 0:
                    sections["introduction"] = content[pos:intro_end].strip()
                    break
        
        # Look for methodology
        method_markers = ["methodology", "approach", "experimental setup","experiments"]
        for marker in method_markers:
            if marker in content.lower():
                pos = content.lower().find(marker)
                method_end = SourceProcessor._find_next_section(content, pos + len(marker))
                if method_end > 0:
                    sections["methods"] = content[pos:method_end].strip()
                    break
        
        # Look for results
        result_markers = ["result", "results", "findings", "evaluation"]
        for marker in result_markers:
            if marker in content.lower():
                pos = content.lower().find(marker)
                result_end = SourceProcessor._find_next_section(content, pos + len(marker))
                if result_end > 0:
                    sections["results"] = content[pos:result_end].strip()
                    break
        
        # Look for conclusion
        conclusion_markers = ["conclusion", "conclusions", "discussion", "future work"]
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
    
    @staticmethod
    def _find_next_section(content: str, start_pos: int) -> int:
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
            pos = content.lower().find(marker, start_pos + 100)  # Skip at least 100 chars to avoid finding the current section
            if pos != -1 and pos < next_pos:
                next_pos = pos
        
        return next_pos if next_pos < len(content) else len(content)
    
    @staticmethod
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
=======
"""
Source processing for ArXiv research documents.
"""
import logging
import re
from typing import List, Dict, Any, Tuple, Optional, Union
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class SourceProcessor:
    """
    Processor for ArXiv research sources.
    """
    
    @staticmethod
    def process_research_sources(
        sources: List[Document], 
        source_type: str = "arxiv"
    ) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
        """
        Process research sources into message context and display information.
        
        Args:
            sources: List of Document objects from ArXiv
            source_type: Type of sources (defaults to "arxiv")
            
        Returns:
            Tuple containing:
            - List of tuples (role, content) for the message context
            - List of dictionaries with source information for display
        """
        if source_type.lower() == "arxiv":
            return SourceProcessor._process_arxiv_sources(sources)
        else:
            logger.warning(f"Unknown source type: {source_type}")
            return [], []
    
    @staticmethod
    def _process_arxiv_sources(
        papers: List[Document]
    ) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
        """
        Process ArXiv papers into context messages and display info.
        
        Args:
            papers: List of Document objects from ArXiv
            
        Returns:
            Tuple of context messages and source info
        """
        context_messages = []
        source_info = []
        
        for i, paper in enumerate(papers):
            # Extract paper metadata
            paper_id = paper.metadata.get("entry_id", "").split("abs/")[-1]
            title = paper.metadata.get("Title", "Untitled")
            authors = paper.metadata.get("Authors", "Unknown")
            published = paper.metadata.get("Published", "Unknown date")
            summary = paper.metadata.get("Summary", "No abstract available")
            
            # Extract content and sections
            content = paper.page_content
            paper_sections = SourceProcessor.extract_key_paper_sections(content)
            
            # Format content with section markers if available
            content_display = ""
            if paper_sections:
                for section_name, section_text in paper_sections.items():
                    content_display += f"\n--- {section_name.upper()} ---\n"
                    # Limit each section to reasonable length
                    if len(section_text) > 3000:
                        content_display += f"{section_text[:3000]}...\n"
                    else:
                        content_display += f"{section_text}\n"
            else:
                # If sections couldn't be identified, use beginning and end approach
                if len(content) > 10000:
                    # Take the first 5000 and last 5000 chars to capture both intro and conclusions
                    content_display = f"{content[:5000]}...\n\n[Content truncated]...\n\n{content[-5000:]}"
                else:
                    content_display = content
            
            # Create context message with more detailed structure
            paper_content = f"""
            Paper {i+1}: {title}
            Authors: {authors}
            Published: {published}
            ID: {paper_id}
            
            Abstract:
            {summary}
            
            Content:
            {content_display}
            """
            context_messages.append(("user", paper_content))
            
            # Extract year and create academic citation
            pub_year = "Unknown"
            if published:
                try:
                    import re
                    year_match = re.search(r'(19|20)\d{2}', published)
                    if year_match:
                        pub_year = year_match.group(0)
                except:
                    pass

            # Get first author's last name for citation
            first_author = "Unknown"
            if authors:
                try:
                    if ',' in authors:
                        first_author = authors.split(',')[0].strip()
                    elif ' and ' in authors:
                        first_author = authors.split(' and ')[0].strip()
                    else:
                        author_parts = authors.split()
                        if author_parts:
                            first_author = author_parts[-1]  # Last name
                except:
                    pass

            citation = f"{first_author} ({pub_year})"

            # Extract sections data
            extracted_sections = SourceProcessor.extract_key_paper_sections(content)

            # Save paper info for display with enhanced academic metadata
            source_info.append({
                "id": paper_id,
                "title": title,
                "authors": authors,
                "published": published,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "source_type": "Paper",
                "summary": summary[:300] + "..." if len(summary) > 300 else summary,
                "content_length": len(content),
                "citation": citation,
                "pub_year": pub_year,
                "first_author": first_author,
                "has_sections": bool(extracted_sections),
                "sections": list(extracted_sections.keys()) if extracted_sections else []
            })
        
        return context_messages, source_info
    
    @staticmethod
    def chunk_content(
        content: str, 
        chunk_size: int = 16000, 
        chunk_overlap: int = 1000
    ) -> List[str]:
        """
        Split content into manageable chunks with larger size and overlap.
        
        Args:
            content: Text content to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(content)
        
        return chunks
    
    @staticmethod
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
                intro_end = SourceProcessor._find_next_section(content, pos + len(marker))
                if intro_end > 0:
                    sections["introduction"] = content[pos:intro_end].strip()
                    break
        
        # Look for methods/methodology
        method_markers = ["method", "methodology", "approach", "experimental setup"]
        for marker in method_markers:
            if marker in content.lower():
                pos = content.lower().find(marker)
                method_end = SourceProcessor._find_next_section(content, pos + len(marker))
                if method_end > 0:
                    sections["methods"] = content[pos:method_end].strip()
                    break
        
        # Look for results
        result_markers = ["result", "results", "findings", "evaluation"]
        for marker in result_markers:
            if marker in content.lower():
                pos = content.lower().find(marker)
                result_end = SourceProcessor._find_next_section(content, pos + len(marker))
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
    
    @staticmethod
    def _find_next_section(content: str, start_pos: int) -> int:
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
            pos = content.lower().find(marker, start_pos + 100)  # Skip at least 100 chars to avoid finding the current section
            if pos != -1 and pos < next_pos:
                next_pos = pos
        
        return next_pos if next_pos < len(content) else len(content)
>>>>>>> Stashed changes
