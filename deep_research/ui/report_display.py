<<<<<<< Updated upstream
"""
Report display utilities for the Deep Research Agent.
"""
import streamlit as st
import os
import time
from typing import Dict, List, Any, Optional
import logging

from ..utils.text_processing import remove_thinking_tags

logger = logging.getLogger(__name__)

def format_research_plan(plan: Any, modified: bool = False) -> str:
    """
    Format a research plan for display.
    
    Args:
        plan: The research plan to format
        modified: Whether this is a modified plan
        
    Returns:
        Formatted plan display text
    """
    prefix = "Modified " if modified else ""
    plan_display = f"## {prefix}Research Plan for: {plan.topic}\n\n"
    plan_display += f"**Objective:** {plan.objective}\n\n"
    plan_display += "### Research Tasks:\n\n"
    
    for task in plan.tasks:
        plan_display += f"**Task {task.task_id}:** {task.description}\n\n"
        plan_display += "Queries:\n"
        for query in task.queries:
            plan_display += f"- {query}\n"
        plan_display += "\n"
    
    return plan_display

def format_configuration_summary(task_iterations: Dict[int, Dict[str, Any]], tasks: List[Any]) -> str:
    """
    Format a summary of task iterations configuration.
    
    Args:
        task_iterations: Dict of task iteration configurations
        tasks: List of research tasks
        
    Returns:
        Formatted configuration summary
    """
    config_summary = "## Research Iteration Configuration\n\n"
    config_summary += "I'll run the following iterations for each task:\n\n"
    
    for task in tasks:
        task_config = task_iterations.get(task.task_id, {"iterations": 1})
        iterations = task_config.get("iterations", 1)
        detail_level = task_config.get("detail_level", "Standard")
        config_summary += f"- **Task {task.task_id}**: {iterations} iteration(s) with {detail_level} detail level\n"
    
    config_summary += "\nStarting research execution now...\n\n"
    config_summary += "For each task, I'll conduct multiple iterations, building knowledge with each iteration. This allows for increasingly focused and in-depth research."
    
    return config_summary

def format_iteration_results(
    iteration_result: Any, 
    task_id: int, 
    current_papers: List[Dict[str, Any]], 
    current_web_results: List[Dict[str, Any]]
) -> str:
    """
    Format the results of a research iteration for display.
    
    Args:
        iteration_result: The research iteration result
        task_id: The task ID
        current_papers: Current papers from this iteration
        current_web_results: Current web results from this iteration
        
    Returns:
        Formatted result display text
    """
    result_display = f"### Research Iteration {iteration_result.iteration_number} for Task {task_id}\n\n"
    result_display += "#### Sources Analyzed:\n\n"
    
    # Display papers
    if current_papers:
        result_display += "##### ArXiv Papers:\n\n"
        for paper in current_papers:
            result_display += f"📄 **{paper['title']}**\n"
            result_display += f"Authors: {paper['authors']}\n"
            result_display += f"ID: {paper['id']}\n"
            result_display += f"Published: {paper['published']}\n"
            
            # Fix for ArXiv links - create proper URL
            paper_id = str(paper['id']).lower().replace('arxiv:', '').strip()
            fixed_url = f"https://arxiv.org/abs/{paper_id}"
            
            result_display += f"Link: [ArXiv:{paper['id']}]({fixed_url})\n"
            if 'summary' in paper:
                result_display += f"Summary: {paper['summary']}\n"
            result_display += "\n"
    
    # Display web results
    if current_web_results:
        result_display += "##### Web Sources:\n\n"
        for result in current_web_results:
            result_display += f"🌐 **{result['title']}**\n"
            result_display += f"URL: [{result['url']}]({result['url']})\n"
            result_display += f"Snippet: {result['snippet']}\n"
            if 'preview' in result:
                result_display += f"Preview: {result['preview']}\n"
            result_display += "\n"
    
    # Display findings
    result_display += "#### Key Findings:\n\n"
    for finding in iteration_result.findings:
        result_display += f"**{finding.title}** (Relevance: {finding.relevance_score}/10)\n"
        result_display += f"{finding.summary}\n\n"
    
    # Display insights
    cleaned_insights = remove_thinking_tags(iteration_result.insights)
    result_display += f"#### Synthesized Insights:\n\n{cleaned_insights}\n\n"
    
    # Display questions for further research
    result_display += "#### Questions for Further Research:\n\n"
    for question in iteration_result.next_questions:
        result_display += f"- {question}\n"
    
    return result_display

def format_task_summary(
    task_id: int, 
    task: Any, 
    detail_level: str, 
    task_knowledge: str, 
    task_papers: List[Dict[str, Any]], 
    task_web_results: List[Dict[str, Any]]
) -> str:
    """
    Format a task summary after all iterations.
    
    Args:
        task_id: The task ID
        task: The research task
        detail_level: The detail level used
        task_knowledge: Accumulated knowledge text
        task_papers: Papers used for this task
        task_web_results: Web results used for this task
        
    Returns:
        Formatted task summary
    """
    # Clean knowledge text
    cleaned_task_knowledge = remove_thinking_tags(task_knowledge)
    
    # Create the summary text
    task_summary = f"### Task {task_id} Summary\n\n"
    task_summary += f"**Task:** {task.description}\n\n"
    task_summary += f"**Detail Level:** {detail_level}\n\n"

    # Include the cleaned accumulated knowledge synthesis
    task_summary += f"**Accumulated Knowledge:**\n\n{cleaned_task_knowledge}\n\n"

    # Add sources used in this task
    task_summary += "**Sources Used:**\n\n"

    # Display papers used in this task
    if task_papers:
        task_summary += "ArXiv Papers:\n"
        for i, paper in enumerate(task_papers):
            # Fix for ArXiv links
            paper_id = str(paper['id']).lower().replace('arxiv:', '').strip()
            fixed_url = f"https://arxiv.org/abs/{paper_id}"
            
            task_summary += f"{i+1}. [{paper['title']}]({fixed_url}) by {paper['authors']}\n"
    
    # Display web results used in this task
    if task_web_results:
        task_summary += "\nWeb Sources:\n"
        for i, result in enumerate(task_web_results):
            task_summary += f"{i+1}. [{result['title']}]({result['url']})\n"
    
    return task_summary

def save_research_report(
    plan: Any, 
    report_display: str, 
    task_knowledge: Dict[int, str], 
    output_dir: str = "research_outputs"
) -> str:
    """
    Save the research report to a file.
    
    Args:
        plan: The research plan
        report_display: The report display text
        task_knowledge: Dict of task-specific knowledge
        output_dir: Directory to save the report
        
    Returns:
        Path to the saved file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a filename based on the research topic
        sanitized_topic = "".join([c if c.isalnum() else "_" for c in plan.topic])
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{output_dir}/{sanitized_topic}_{timestamp}.md"
        
        # Compile comprehensive report with task summaries
        comprehensive_report = report_display + "\n\n"
        
        # Add task-specific summaries
        comprehensive_report += "# Task Summaries\n\n"
        for task_id, knowledge in task_knowledge.items():
            if knowledge:
                task = next((t for t in plan.tasks if t.task_id == task_id), None)
                if task:
                    comprehensive_report += f"## Task {task_id}: {task.description}\n\n"
                    comprehensive_report += remove_thinking_tags(knowledge) + "\n\n"
        
        # Save to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(comprehensive_report)
            
        logger.info(f"Research report saved to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error saving research report: {e}")
        return ""

def format_error_message(error: Exception, context: str, debug_mode: bool = False) -> str:
    """
    Format an error message for display.
    
    Args:
        error: The exception
        context: Description of what was happening when the error occurred
        debug_mode: Whether to include detailed debug info
        
    Returns:
        Formatted error message
    """
    import traceback
    
    error_msg = f"Error in {context}: {str(error)}"
    
    if debug_mode:
        error_msg += f"\n\nStacktrace:\n```\n{traceback.format_exc()}\n```"
    
    return error_msg

def display_final_report(
    report: str, 
    filename: Optional[str] = None, 
    allow_download: bool = True
) -> None:
    """
    Display the final research report with download option.
    
    Args:
        report: The research report text
        filename: Path to the saved report file (if any)
        allow_download: Whether to allow downloading the report
    """
    # Display the report
    st.markdown(report)
    
    # Add download button if filename is provided
    if allow_download and filename and os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            report_content = f.read()
            
        st.download_button(
            label="Download Research Report",
            data=report_content,
            file_name=os.path.basename(filename),
            mime="text/markdown",
            help="Download the full research report as a Markdown file"
=======
"""
Report display utilities for the Deep Research Agent.
"""
import streamlit as st
import os
import time
from typing import Dict, List, Any, Optional
import logging

from ..utils.text_processing import remove_thinking_tags

logger = logging.getLogger(__name__)

def format_research_plan(plan: Any, modified: bool = False) -> str:
    """
    Format a research plan for display.
    
    Args:
        plan: The research plan to format
        modified: Whether this is a modified plan
        
    Returns:
        Formatted plan display text
    """
    prefix = "Modified " if modified else ""
    plan_display = f"## {prefix}Research Plan for: {plan.topic}\n\n"
    plan_display += f"**Objective:** {plan.objective}\n\n"
    plan_display += "### Research Tasks:\n\n"
    
    for task in plan.tasks:
        plan_display += f"**Task {task.task_id}:** {task.description}\n\n"
        plan_display += "Queries:\n"
        for query in task.queries:
            plan_display += f"- {query}\n"
        plan_display += "\n"
    
    return plan_display

def format_configuration_summary(task_iterations: Dict[int, Dict[str, Any]], tasks: List[Any]) -> str:
    """
    Format a summary of task iterations configuration.
    
    Args:
        task_iterations: Dict of task iteration configurations
        tasks: List of research tasks
        
    Returns:
        Formatted configuration summary
    """
    config_summary = "## Research Iteration Configuration\n\n"
    config_summary += "I'll run the following iterations for each task:\n\n"
    
    for task in tasks:
        task_config = task_iterations.get(task.task_id, {"iterations": 1})
        iterations = task_config.get("iterations", 1)
        detail_level = task_config.get("detail_level", "Standard")
        config_summary += f"- **Task {task.task_id}**: {iterations} iteration(s) with {detail_level} detail level\n"
    
    config_summary += "\nStarting research execution now...\n\n"
    config_summary += "For each task, I'll conduct multiple iterations, building knowledge with each iteration. This allows for increasingly focused and in-depth research."
    
    return config_summary

def format_iteration_results(
    iteration_result: Any,
    task_id: int,
    current_papers: List[Dict[str, Any]]
) -> str:
    """
    Format the results of a research iteration for display.

    Args:
        iteration_result: The research iteration result
        task_id: The task ID
        current_papers: Current papers from this iteration

    Returns:
        Formatted result display text
    """
    result_display = f"### Research Iteration {iteration_result.iteration_number} for Task {task_id}\n\n"
    result_display += "#### Sources Analyzed:\n\n"

    # Display papers
    if current_papers:
        result_display += "##### ArXiv Papers:\n\n"
        for paper in current_papers:
            result_display += f"📄 **{paper['title']}**\n"
            result_display += f"Authors: {paper['authors']}\n"
            result_display += f"ID: {paper['id']}\n"
            result_display += f"Published: {paper['published']}\n"

            # Fix for ArXiv links - create proper URL
            paper_id = str(paper['id']).lower().replace('arxiv:', '').strip()
            fixed_url = f"https://arxiv.org/abs/{paper_id}"

            result_display += f"Link: [ArXiv:{paper['id']}]({fixed_url})\n"
            if 'summary' in paper:
                result_display += f"Summary: {paper['summary']}\n"
            result_display += "\n"

    # Display findings
    result_display += "#### Key Findings:\n\n"
    for finding in iteration_result.findings:
        result_display += f"**{finding.title}** (Relevance: {finding.relevance_score}/10)\n"
        result_display += f"{finding.summary}\n\n"

    # Display insights
    cleaned_insights = remove_thinking_tags(iteration_result.insights)
    result_display += f"#### Synthesized Insights:\n\n{cleaned_insights}\n\n"

    # Display questions for further research
    result_display += "#### Questions for Further Research:\n\n"
    for question in iteration_result.next_questions:
        result_display += f"- {question}\n"

    return result_display

def format_task_summary(
    task_id: int,
    task: Any,
    detail_level: str,
    task_knowledge: str,
    task_papers: List[Dict[str, Any]]
) -> str:
    """
    Format a task summary after all iterations.

    Args:
        task_id: The task ID
        task: The research task
        detail_level: The detail level used
        task_knowledge: Accumulated knowledge text
        task_papers: Papers used for this task

    Returns:
        Formatted task summary
    """
    # Clean knowledge text
    cleaned_task_knowledge = remove_thinking_tags(task_knowledge)

    # Create the summary text
    task_summary = f"### Task {task_id} Summary\n\n"
    task_summary += f"**Task:** {task.description}\n\n"
    task_summary += f"**Detail Level:** {detail_level}\n\n"

    # Include the cleaned accumulated knowledge synthesis
    task_summary += f"**Accumulated Knowledge:**\n\n{cleaned_task_knowledge}\n\n"

    # Add sources used in this task
    task_summary += "**Sources Used:**\n\n"

    # Display papers used in this task
    if task_papers:
        task_summary += "ArXiv Papers:\n"
        for i, paper in enumerate(task_papers):
            # Fix for ArXiv links
            paper_id = str(paper['id']).lower().replace('arxiv:', '').strip()
            fixed_url = f"https://arxiv.org/abs/{paper_id}"

            task_summary += f"{i+1}. [{paper['title']}]({fixed_url}) by {paper['authors']}\n"

    return task_summary

def save_research_report(
    plan: Any,
    report_display: str,
    task_knowledge: Dict[int, str],
    output_dir: str = "research_outputs"
) -> str:
    """
    Save the research report to a file with academic formatting.

    Args:
        plan: The research plan
        report_display: The report display text
        task_knowledge: Dict of task-specific knowledge
        output_dir: Directory to save the report

    Returns:
        Path to the saved file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create a filename based on the research topic
        sanitized_topic = "".join([c if c.isalnum() else "_" for c in plan.topic])
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{output_dir}/{sanitized_topic}_{timestamp}.md"

        # Add metadata header for academic report
        current_date = time.strftime("%B %d, %Y")
        academic_header = f"""---
title: "Research Report: {plan.topic}"
date: "{current_date}"
author: "Deep Research LLM Agent"
abstract: "This report presents a comprehensive academic analysis on {plan.topic}. Research was conducted using ArXiv papers as primary sources. The objective was to {plan.objective}."
keywords: [research, academic, arxiv, {plan.topic.replace(' ', ', ')}]
---

"""

        # Compile comprehensive report with academic structure
        comprehensive_report = academic_header + report_display + "\n\n"

        # Add appendix with task-specific summaries
        comprehensive_report += "# Appendix: Detailed Task Analysis\n\n"
        for task_id, knowledge in task_knowledge.items():
            if knowledge:
                task = next((t for t in plan.tasks if t.task_id == task_id), None)
                if task:
                    comprehensive_report += f"## Task {task_id}: {task.description}\n\n"
                    comprehensive_report += remove_thinking_tags(knowledge) + "\n\n"

        # Save to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(comprehensive_report)

        logger.info(f"Academic research report saved to {filename}")
        return filename

    except Exception as e:
        logger.error(f"Error saving research report: {e}")
        return ""

def format_error_message(error: Exception, context: str, debug_mode: bool = False) -> str:
    """
    Format an error message for display.
    
    Args:
        error: The exception
        context: Description of what was happening when the error occurred
        debug_mode: Whether to include detailed debug info
        
    Returns:
        Formatted error message
    """
    import traceback
    
    error_msg = f"Error in {context}: {str(error)}"
    
    if debug_mode:
        error_msg += f"\n\nStacktrace:\n```\n{traceback.format_exc()}\n```"
    
    return error_msg

def display_final_report(
    report: str, 
    filename: Optional[str] = None, 
    allow_download: bool = True
) -> None:
    """
    Display the final research report with download option.
    
    Args:
        report: The research report text
        filename: Path to the saved report file (if any)
        allow_download: Whether to allow downloading the report
    """
    # Display the report
    st.markdown(report)
    
    # Add download button if filename is provided
    if allow_download and filename and os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            report_content = f.read()
            
        st.download_button(
            label="Download Research Report",
            data=report_content,
            file_name=os.path.basename(filename),
            mime="text/markdown",
            help="Download the full research report as a Markdown file"
>>>>>>> Stashed changes
        )