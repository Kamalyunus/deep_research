"""
Reusable UI components for the Deep Research Agent.
"""
import streamlit as st
import time
from typing import Dict, List, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

def display_research_plan(
    plan: Dict[str, Any], 
    modified: bool = False
) -> None:
    """
    Display a research plan in the UI.
    
    Args:
        plan: Research plan data
        modified: Whether this is a modified plan
    """
    prefix = "Modified " if modified else ""
    plan_display = f"## {prefix}Research Plan for: {plan['topic']}\n\n"
    plan_display += f"**Objective:** {plan['objective']}\n\n"
    plan_display += "### Research Tasks:\n\n"
    
    for task in plan['tasks']:
        plan_display += f"**Task {task['task_id']}:** {task['description']}\n\n"
        plan_display += "Queries:\n"
        for query in task['queries']:
            plan_display += f"- {query}\n"
        plan_display += "\n"
    
    st.markdown(plan_display)

def display_research_iteration_results(
    iteration_result: Dict[str, Any],
    papers: List[Dict[str, Any]],
    web_results: List[Dict[str, Any]]
) -> None:
    """
    Display the results of a research iteration.
    
    Args:
        iteration_result: Research iteration result
        papers: Papers used in this iteration
        web_results: Web results used in this iteration
    """
    task_id = iteration_result.get("task_id", "Unknown")
    iteration_number = iteration_result.get("iteration_number", 1)
    
    st.markdown(f"### Research Iteration {iteration_number} for Task {task_id}\n\n")
    
    # Create tabs for different sections
    tabs = st.tabs(["Sources", "Findings", "Insights", "Next Questions"])
    
    # Sources tab
    with tabs[0]:
        if papers:
            st.markdown("#### ArXiv Papers")
            for i, paper in enumerate(papers):
                with st.expander(f"{i+1}. {paper.get('title', 'Untitled')}"):
                    st.markdown(f"**Authors:** {paper.get('authors', 'Unknown')}")
                    st.markdown(f"**Published:** {paper.get('published', 'Unknown date')}")
                    st.markdown(f"**ID:** {paper.get('id', 'Unknown')}")
                    
                    # Fix for ArXiv links
                    paper_id = str(paper.get('id', '')).lower().replace('arxiv:', '').strip()
                    st.markdown(f"**Link:** [ArXiv:{paper_id}](https://arxiv.org/abs/{paper_id})")
                    
                    if 'summary' in paper:
                        st.markdown("**Summary:**")
                        st.markdown(paper['summary'])
        else:
            st.info("No ArXiv papers were used in this iteration.")
            
        if web_results:
            st.markdown("#### Web Results")
            for i, result in enumerate(web_results):
                with st.expander(f"{i+1}. {result.get('title', 'Untitled')}"):
                    st.markdown(f"**URL:** [{result.get('url', '#')}]({result.get('url', '#')})")
                    st.markdown(f"**Snippet:** {result.get('snippet', 'No snippet available')}")
                    
                    if 'preview' in result:
                        st.markdown("**Preview:**")
                        st.markdown(result['preview'])
        else:
            st.info("No web results were used in this iteration.")
    
    # Findings tab
    with tabs[1]:
        findings = iteration_result.get("findings", [])
        if findings:
            for i, finding in enumerate(findings):
                with st.expander(f"{i+1}. {finding.get('title', 'Finding')} (Relevance: {finding.get('relevance_score', '?')}/10)"):
                    st.markdown(finding.get('summary', 'No summary available'))
                    st.markdown(f"*Source: {finding.get('source_type', 'Unknown')} - {finding.get('source_id', 'Unknown')}*")
        else:
            st.info("No specific findings were recorded in this iteration.")
    
    # Insights tab
    with tabs[2]:
        insights = iteration_result.get("insights", "No insights provided")
        st.markdown(insights)
    
    # Next Questions tab
    with tabs[3]:
        next_questions = iteration_result.get("next_questions", [])
        if next_questions:
            for i, question in enumerate(next_questions):
                st.markdown(f"{i+1}. {question}")
        else:
            st.info("No further questions were generated in this iteration.")

def display_configuration_form(
    tasks: List[Dict[str, Any]],
    max_iterations: int = 5,
    default_iterations: int = 2,
    detail_levels: List[str] = ["Basic", "Standard", "Comprehensive"],
    default_level: str = "Standard"
) -> Dict[int, Dict[str, Any]]:
    """
    Display a form for configuring task iterations.
    
    Args:
        tasks: List of research tasks
        max_iterations: Maximum number of iterations allowed
        default_iterations: Default number of iterations
        detail_levels: Available detail levels
        default_level: Default detail level
        
    Returns:
        Dictionary mapping task_id to configuration or None if form not submitted
    """
    st.subheader("Configure Iterations Per Task")
    st.write("Specify how many iterations to run for each task. More iterations will result in deeper, more thorough research.")
    st.write("💡 **Tip**: Tasks that are broader or more complex usually benefit from more iterations.")
    
    task_config = {}
    
    # Create a form for more organized input
    with st.form("task_iterations_form"):
        for task in tasks:
            task_id = task.get("task_id", 0)
            st.markdown(f"**Task {task_id}:** {task.get('description', 'No description')}")
            
            # Show queries
            queries = task.get("queries", [])
            if queries:
                st.markdown("*Queries:*")
                for query in queries:
                    st.markdown(f"- {query}")
            
            # Configure iterations
            col1, col2 = st.columns(2)
            with col1:
                iterations = st.number_input(
                    f"Iterations for Task {task_id}",
                    min_value=1,
                    max_value=max_iterations,
                    value=default_iterations,
                    key=f"task_{task_id}_iterations"
                )
            
            # Configure detail level
            with col2:
                detail_level = st.select_slider(
                    f"Detail level for Task {task_id}",
                    options=detail_levels,
                    value=default_level,
                    key=f"task_{task_id}_detail_level"
                )
            
            task_config[task_id] = {
                "iterations": iterations,
                "detail_level": detail_level
            }
            
            st.markdown("---")
        
        # Submit button for the form
        submit_button = st.form_submit_button("Confirm Iterations")
        
        if submit_button:
            return task_config
        else:
            return None

def display_error_message(
    error: Exception,
    context: str,
    debug_mode: bool = False
) -> None:
    """
    Display an error message with appropriate detail based on debug mode.
    
    Args:
        error: The exception
        context: Description of what was happening when the error occurred
        debug_mode: Whether to include detailed debug info
    """
    import traceback
    
    # Base error message
    error_msg = f"Error during {context}: {str(error)}"
    
    # Show with appropriate styling
    st.error(error_msg)
    
    # Show debug info if enabled
    if debug_mode:
        with st.expander("Debug Information"):
            st.code(traceback.format_exc(), language="python")
            
            # Show error type and attributes
            st.markdown("**Error Type:** " + type(error).__name__)
            st.markdown("**Error Attributes:**")
            for attr in dir(error):
                if not attr.startswith('_') and not callable(getattr(error, attr)):
                    try:
                        value = getattr(error, attr)
                        st.markdown(f"- **{attr}:** {value}")
                    except:
                        st.markdown(f"- **{attr}:** [Unable to retrieve value]")