<<<<<<< Updated upstream
"""
Output formatting utilities for the Deep Research Agent.
"""
from typing import Dict, List, Any, Optional
import re
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def format_citation(
    source_type: str, 
    source_data: Dict[str, Any], 
    citation_index: int
) -> str:
    """
    Format a source citation in a consistent way.
    
    Args:
        source_type: Type of source ("Paper" or "Web")
        source_data: Source data
        citation_index: Citation index number
        
    Returns:
        Formatted citation text
    """
    if source_type.lower() == "paper" or source_type.lower() == "arxiv":
        # Format paper citation
        authors = source_data.get("authors", "Unknown")
        title = source_data.get("title", "Untitled")
        paper_id = source_data.get("id", "")
        
        # Fix for ArXiv links - ensure proper URL format
        if not paper_id:
            url = "#"
        else:
            # Clean the ID by removing any 'arxiv:' prefix
            clean_id = str(paper_id).lower().replace('arxiv:', '').strip()
            url = f"https://arxiv.org/abs/{clean_id}"
        
        published = source_data.get("published", "")
        published_text = f" ({published})" if published else ""
        
        return f"[{citation_index}] {authors}. \"{title}\"{published_text}. [ArXiv: {paper_id}]({url})"
        
    elif source_type.lower() == "web":
        # Format web citation
        title = source_data.get("title", "Untitled")
        url = source_data.get("url", "#")
        date = source_data.get("published", datetime.now().strftime("%Y-%m-%d"))
        
        return f"[{citation_index}] \"{title}\". {url} (Accessed: {date})"
    
    else:
        # Generic citation for unknown types
        return f"[{citation_index}] {source_data.get('title', 'Unknown source')}"

def format_inline_citation(citation_index: int) -> str:
    """
    Format an inline citation reference.
    
    Args:
        citation_index: Citation index number
        
    Returns:
        Formatted inline citation
    """
    return f"[{citation_index}]"

def format_sources_section(papers: List[Dict[str, Any]], web_results: List[Dict[str, Any]]) -> str:
    """
    Generate a sources section for the report with formatted citations.
    
    Args:
        papers: List of papers used in research
        web_results: List of web results used in research
        
    Returns:
        Formatted sources section
    """
    sources_section = "# Sources\n\n"
    
    # Add paper sources
    if papers:
        sources_section += "## ArXiv Papers\n\n"
        for i, paper in enumerate(papers):
            try:
                citation = format_citation("Paper", paper, i+1)
                sources_section += f"{citation}\n\n"
            except Exception as e:
                logger.warning(f"Error formatting paper citation: {e}")
                sources_section += f"[{i+1}] Paper citation unavailable due to formatting error\n\n"
    
    # Add web sources
    if web_results:
        sources_section += "## Web Sources\n\n"
        for i, result in enumerate(web_results):
            try:
                citation = format_citation("Web", result, i+1+len(papers))
                sources_section += f"{citation}\n\n"
            except Exception as e:
                logger.warning(f"Error formatting web citation: {e}")
                sources_section += f"[{i+1+len(papers)}] Web citation unavailable due to formatting error\n\n"
    
    return sources_section

def format_research_findings(findings: List[Dict[str, Any]]) -> str:
    """
    Format research findings in a structured way.
    
    Args:
        findings: List of research findings
        
    Returns:
        Formatted findings text
    """
    if not findings:
        return "No specific findings were recorded."
    
    formatted_text = ""
    
    # Group findings by relevance score
    high_relevance = []
    medium_relevance = []
    low_relevance = []
    
    for finding in findings:
        relevance = finding.get("relevance_score", 5)
        if relevance >= 8:
            high_relevance.append(finding)
        elif relevance >= 5:
            medium_relevance.append(finding)
        else:
            low_relevance.append(finding)
    
    # Format high relevance findings
    if high_relevance:
        formatted_text += "### High Relevance Findings\n\n"
        for i, finding in enumerate(high_relevance):
            formatted_text += f"**{finding.get('title', 'Finding')}** (Relevance: {finding.get('relevance_score', '?')}/10)\n\n"
            formatted_text += f"{finding.get('summary', 'No summary available')}\n\n"
            formatted_text += f"*Source: {finding.get('source_type', 'Unknown')} - {finding.get('source_id', 'Unknown')}*\n\n"
    
    # Format medium relevance findings
    if medium_relevance:
        formatted_text += "### Medium Relevance Findings\n\n"
        for i, finding in enumerate(medium_relevance):
            formatted_text += f"**{finding.get('title', 'Finding')}** (Relevance: {finding.get('relevance_score', '?')}/10)\n\n"
            formatted_text += f"{finding.get('summary', 'No summary available')}\n\n"
            formatted_text += f"*Source: {finding.get('source_type', 'Unknown')} - {finding.get('source_id', 'Unknown')}*\n\n"
    
    # Format low relevance findings
    if low_relevance:
        formatted_text += "### Lower Relevance Findings\n\n"
        for i, finding in enumerate(low_relevance):
            formatted_text += f"**{finding.get('title', 'Finding')}** (Relevance: {finding.get('relevance_score', '?')}/10)\n\n"
            formatted_text += f"{finding.get('summary', 'No summary available')}\n\n"
            formatted_text += f"*Source: {finding.get('source_type', 'Unknown')} - {finding.get('source_id', 'Unknown')}*\n\n"
    
    return formatted_text

def format_json_for_display(data: Any, indent: int = 2) -> str:
    """
    Format JSON data for display with pretty printing.
    
    Args:
        data: Data to format as JSON
        indent: Indentation level
        
    Returns:
        Pretty formatted JSON string
    """
    try:
        if isinstance(data, str):
            # Try to parse string as JSON first
            parsed_data = json.loads(data)
            return json.dumps(parsed_data, indent=indent)
        else:
            # Directly dump the object
            return json.dumps(data, indent=indent)
    except Exception as e:
        logger.warning(f"Error formatting JSON for display: {e}")
        # Return original data if parsing fails
        return str(data)

def format_metadata_table(metadata: Dict[str, Any]) -> str:
    """
    Format a metadata dictionary as a markdown table.
    
    Args:
        metadata: Dictionary of metadata
        
    Returns:
        Formatted markdown table
    """
    if not metadata:
        return "No metadata available."
    
    table = "| Property | Value |\n| --- | --- |\n"
    for key, value in metadata.items():
        # Format the value based on type
        if isinstance(value, dict):
            formatted_value = "Complex object"
        elif isinstance(value, list):
            if len(value) > 3:
                formatted_value = f"List with {len(value)} items"
            else:
                formatted_value = ", ".join(str(item) for item in value)
        else:
            formatted_value = str(value)
            
        # Truncate long values
        if len(formatted_value) > 50:
            formatted_value = formatted_value[:47] + "..."
            
        # Escape pipe characters in markdown tables
        formatted_value = formatted_value.replace("|", "\\|")
        key = key.replace("|", "\\|")
        
        table += f"| {key} | {formatted_value} |\n"
    
    return table

def format_research_summary(
    topic: str,
    objective: str,
    key_findings: List[str],
    synthesis: str,
    limitations: List[str] = None
) -> str:
    """
    Format a comprehensive research summary.
    
    Args:
        topic: Research topic
        objective: Research objective
        key_findings: List of key findings
        synthesis: Synthesized insights text
        limitations: Optional list of research limitations
        
    Returns:
        Formatted research summary
    """
    summary = f"# Research Summary: {topic}\n\n"
    summary += f"## Objective\n\n{objective}\n\n"
    
    # Key findings section
    summary += "## Key Findings\n\n"
    for i, finding in enumerate(key_findings):
        summary += f"{i+1}. {finding}\n"
    summary += "\n"
    
    # Synthesis section
    summary += "## Synthesis\n\n"
    summary += synthesis + "\n\n"
    
    # Limitations section (if provided)
    if limitations:
        summary += "## Limitations\n\n"
        for i, limitation in enumerate(limitations):
            summary += f"- {limitation}\n"
        summary += "\n"
    
=======
"""
Output formatting utilities for the Deep Research Agent.
"""
from typing import Dict, List, Any, Optional
import re
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def format_academic_citation(
    source_data: Dict[str, Any],
    citation_index: int,
    style: str = "numeric"
) -> Dict[str, str]:
    """
    Format academic citations in various styles.

    Args:
        source_data: Source data for the paper
        citation_index: Citation index number
        style: Citation style ("numeric", "author-date", or "apa")

    Returns:
        Dictionary with different citation formats
    """
    # Extract basic information
    authors = source_data.get("authors", "Unknown")
    title = source_data.get("title", "Untitled")
    paper_id = source_data.get("id", "")
    published = source_data.get("published", "")

    # Extract or format additional information
    pub_year = source_data.get("pub_year", "")
    if not pub_year and published:
        try:
            import re
            year_match = re.search(r'(19|20)\d{2}', published)
            if year_match:
                pub_year = year_match.group(0)
        except:
            pub_year = ""

    # Get first author for author-date citation
    first_author = source_data.get("first_author", "")
    if not first_author and authors:
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
            first_author = "Unknown"

    # Fix for ArXiv links - ensure proper URL format
    if not paper_id:
        url = "#"
    else:
        # Clean the ID by removing any 'arxiv:' prefix
        clean_id = str(paper_id).lower().replace('arxiv:', '').strip()
        url = f"https://arxiv.org/abs/{clean_id}"

    # Format author list for citations
    formatted_authors = authors
    if ',' in authors:
        # If multiple authors separated by commas
        author_list = [a.strip() for a in authors.split(',')]
        if len(author_list) > 3:
            formatted_authors = f"{author_list[0]} et al."

    # Create different citation formats
    numeric_citation = f"[{citation_index}]"
    author_date = f"{first_author} ({pub_year})"

    # Full bibliography reference formats
    bibliography_numeric = (
        f"[{citation_index}] {formatted_authors} ({pub_year}). \"{title}\". "
        f"*arXiv:{paper_id}*. [Available online: {url}]({url})"
    )

    bibliography_apa = (
        f"{formatted_authors} ({pub_year}). {title}. *arXiv preprint "
        f"arXiv:{paper_id}*. Retrieved from {url}"
    )

    return {
        "numeric": numeric_citation,
        "author_date": author_date,
        "bibliography_numeric": bibliography_numeric,
        "bibliography_apa": bibliography_apa,
        "url": url,
        "paper_id": paper_id,
        "year": pub_year,
        "first_author": first_author
    }

def format_citation(
    source_type: str,
    source_data: Dict[str, Any],
    citation_index: int
) -> str:
    """
    Format a source citation in a consistent way.

    Args:
        source_type: Type of source ("Paper" or "Web")
        source_data: Source data
        citation_index: Citation index number

    Returns:
        Formatted citation text
    """
    if source_type.lower() == "paper" or source_type.lower() == "arxiv":
        # Use academic citation formatting
        citation_info = format_academic_citation(source_data, citation_index)
        return citation_info["bibliography_numeric"]

    elif source_type.lower() == "web":
        # Format web citation
        title = source_data.get("title", "Untitled")
        url = source_data.get("url", "#")
        date = source_data.get("published", datetime.now().strftime("%Y-%m-%d"))

        return f"[{citation_index}] \"{title}\". {url} (Accessed: {date})"

    else:
        # Generic citation for unknown types
        return f"[{citation_index}] {source_data.get('title', 'Unknown source')}"

def format_inline_citation(citation_index: int) -> str:
    """
    Format an inline citation reference.
    
    Args:
        citation_index: Citation index number
        
    Returns:
        Formatted inline citation
    """
    return f"[{citation_index}]"

def format_sources_section(papers: List[Dict[str, Any]], web_results: List[Dict[str, Any]]) -> str:
    """
    Generate an academic bibliography section for the report with proper citation formatting.

    Args:
        papers: List of papers used in research
        web_results: List of web results used in research (will be ignored)

    Returns:
        Formatted academic bibliography section
    """
    sources_section = "# References\n\n"

    # Add paper sources with academic formatting
    if papers:
        for i, paper in enumerate(papers):
            try:
                # Get comprehensive citation info
                citation_info = format_academic_citation(paper, i+1)

                # Use the bibliography numeric format
                bibliography_entry = citation_info["bibliography_numeric"]
                sources_section += f"{bibliography_entry}\n\n"

                # Add additional paper details if available
                if paper.get("has_sections", False) and paper.get("sections", []):
                    sections = paper.get("sections", [])
                    if sections:
                        sources_section += f"*Paper contains sections: {', '.join(sections)}*\n\n"

            except Exception as e:
                logger.warning(f"Error formatting paper citation: {e}")
                sources_section += f"[{i+1}] {paper.get('title', 'Untitled')} by {paper.get('authors', 'Unknown')}. arXiv:{paper.get('id', 'Unknown')}\n\n"
    else:
        sources_section += "*No academic papers were referenced in this research.*\n\n"

    return sources_section

def format_research_findings(findings: List[Dict[str, Any]]) -> str:
    """
    Format research findings in a structured way.
    
    Args:
        findings: List of research findings
        
    Returns:
        Formatted findings text
    """
    if not findings:
        return "No specific findings were recorded."
    
    formatted_text = ""
    
    # Group findings by relevance score
    high_relevance = []
    medium_relevance = []
    low_relevance = []
    
    for finding in findings:
        relevance = finding.get("relevance_score", 5)
        if relevance >= 8:
            high_relevance.append(finding)
        elif relevance >= 5:
            medium_relevance.append(finding)
        else:
            low_relevance.append(finding)
    
    # Format high relevance findings
    if high_relevance:
        formatted_text += "### High Relevance Findings\n\n"
        for i, finding in enumerate(high_relevance):
            formatted_text += f"**{finding.get('title', 'Finding')}** (Relevance: {finding.get('relevance_score', '?')}/10)\n\n"
            formatted_text += f"{finding.get('summary', 'No summary available')}\n\n"
            formatted_text += f"*Source: {finding.get('source_type', 'Unknown')} - {finding.get('source_id', 'Unknown')}*\n\n"
    
    # Format medium relevance findings
    if medium_relevance:
        formatted_text += "### Medium Relevance Findings\n\n"
        for i, finding in enumerate(medium_relevance):
            formatted_text += f"**{finding.get('title', 'Finding')}** (Relevance: {finding.get('relevance_score', '?')}/10)\n\n"
            formatted_text += f"{finding.get('summary', 'No summary available')}\n\n"
            formatted_text += f"*Source: {finding.get('source_type', 'Unknown')} - {finding.get('source_id', 'Unknown')}*\n\n"
    
    # Format low relevance findings
    if low_relevance:
        formatted_text += "### Lower Relevance Findings\n\n"
        for i, finding in enumerate(low_relevance):
            formatted_text += f"**{finding.get('title', 'Finding')}** (Relevance: {finding.get('relevance_score', '?')}/10)\n\n"
            formatted_text += f"{finding.get('summary', 'No summary available')}\n\n"
            formatted_text += f"*Source: {finding.get('source_type', 'Unknown')} - {finding.get('source_id', 'Unknown')}*\n\n"
    
    return formatted_text

def format_json_for_display(data: Any, indent: int = 2) -> str:
    """
    Format JSON data for display with pretty printing.
    
    Args:
        data: Data to format as JSON
        indent: Indentation level
        
    Returns:
        Pretty formatted JSON string
    """
    try:
        if isinstance(data, str):
            # Try to parse string as JSON first
            parsed_data = json.loads(data)
            return json.dumps(parsed_data, indent=indent)
        else:
            # Directly dump the object
            return json.dumps(data, indent=indent)
    except Exception as e:
        logger.warning(f"Error formatting JSON for display: {e}")
        # Return original data if parsing fails
        return str(data)

def format_metadata_table(metadata: Dict[str, Any]) -> str:
    """
    Format a metadata dictionary as a markdown table.
    
    Args:
        metadata: Dictionary of metadata
        
    Returns:
        Formatted markdown table
    """
    if not metadata:
        return "No metadata available."
    
    table = "| Property | Value |\n| --- | --- |\n"
    for key, value in metadata.items():
        # Format the value based on type
        if isinstance(value, dict):
            formatted_value = "Complex object"
        elif isinstance(value, list):
            if len(value) > 3:
                formatted_value = f"List with {len(value)} items"
            else:
                formatted_value = ", ".join(str(item) for item in value)
        else:
            formatted_value = str(value)
            
        # Truncate long values
        if len(formatted_value) > 50:
            formatted_value = formatted_value[:47] + "..."
            
        # Escape pipe characters in markdown tables
        formatted_value = formatted_value.replace("|", "\\|")
        key = key.replace("|", "\\|")
        
        table += f"| {key} | {formatted_value} |\n"
    
    return table

def format_filename(topic: str, suffix: str = "") -> str:
    """
    Format a filename from a topic, ensuring it's valid for file systems.
    
    Args:
        topic: The topic to use for the filename
        suffix: Optional suffix to add (e.g., timestamp)
        
    Returns:
        Sanitized filename
    """
    # Replace invalid filename characters
    sanitized_topic = re.sub(r'[\\/*?:"<>|]', "_", topic)
    sanitized_topic = re.sub(r'\s+', "_", sanitized_topic)
    
    # Truncate if too long
    if len(sanitized_topic) > 50:
        sanitized_topic = sanitized_topic[:50]
    
    # Add suffix if provided
    if suffix:
        return f"{sanitized_topic}_{suffix}"
    else:
        return sanitized_topic

def format_timestamp() -> str:
    """
    Get a formatted timestamp for filenames.
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def format_research_summary(
    topic: str,
    objective: str,
    key_findings: List[str],
    synthesis: str,
    limitations: List[str] = None
) -> str:
    """
    Format a comprehensive research summary.
    
    Args:
        topic: Research topic
        objective: Research objective
        key_findings: List of key findings
        synthesis: Synthesized insights text
        limitations: Optional list of research limitations
        
    Returns:
        Formatted research summary
    """
    summary = f"# Research Summary: {topic}\n\n"
    summary += f"## Objective\n\n{objective}\n\n"
    
    # Key findings section
    summary += "## Key Findings\n\n"
    for i, finding in enumerate(key_findings):
        summary += f"{i+1}. {finding}\n"
    summary += "\n"
    
    # Synthesis section
    summary += "## Synthesis\n\n"
    summary += synthesis + "\n\n"
    
    # Limitations section (if provided)
    if limitations:
        summary += "## Limitations\n\n"
        for i, limitation in enumerate(limitations):
            summary += f"- {limitation}\n"
        summary += "\n"
    
>>>>>>> Stashed changes
    return summary