<<<<<<< Updated upstream
"""
UI components for configuring task iterations.
"""
import streamlit as st
from typing import Dict, List, Optional, Any
import logging

from ..agent.models import ResearchTask, TaskExecutionConfig
from ..config.app_config import (
    MAX_ITERATIONS_PER_TASK,
    DEFAULT_ITERATIONS_PER_TASK,
    DETAIL_LEVEL_OPTIONS,
    DEFAULT_DETAIL_LEVEL,
    ITERATION_CONFIG_HELP,
    DETAIL_LEVEL_ADJUSTMENTS
)
from ..utils.text_processing import remove_thinking_tags

logger = logging.getLogger(__name__)

def configure_task_iterations(tasks: List[ResearchTask]) -> Optional[Dict[int, Dict[str, Any]]]:
    """
    Let the user configure iterations per task.
    
    Args:
        tasks: List of ResearchTask objects
        
    Returns:
        Dict mapping task_id to configuration or None if form not submitted
    """
    st.subheader("Configure Iterations Per Task")
    st.write("Specify how many iterations to run for each task. More iterations will result in deeper, more thorough research.")
    st.write("💡 **Tip**: Tasks that are broader or more complex usually benefit from more iterations.")
    
    # Create a form for more organized input
    with st.form("task_iterations_form"):
        for task in tasks:
            st.markdown(f"**Task {task.task_id}:** {task.description}")
            st.markdown("*Queries:*")
            for query in task.queries:
                st.markdown(f"- {query}")
            
            # Configure iterations
            task_iterations = st.number_input(
                f"Number of iterations for Task {task.task_id}",
                min_value=1,
                max_value=MAX_ITERATIONS_PER_TASK,
                value=DEFAULT_ITERATIONS_PER_TASK,
                key=f"task_{task.task_id}_iterations",
                help=ITERATION_CONFIG_HELP["iterations"]
            )
            
            # Configure detail level
            fetch_level = st.select_slider(
                f"Content detail level for Task {task.task_id}",
                options=DETAIL_LEVEL_OPTIONS,
                value=DEFAULT_DETAIL_LEVEL,
                key=f"task_{task.task_id}_detail_level",
                help=ITERATION_CONFIG_HELP["detail_level"]
            )
            
            st.markdown("---")
        
        # Submit button for the form
        submit_button = st.form_submit_button("Confirm Iterations")
        
        # Only return task_iterations if the form was submitted
        if submit_button:
            # Create a dictionary with task_id -> configuration mapping
            task_config = {}
            for task in tasks:
                iterations = st.session_state[f"task_{task.task_id}_iterations"]
                detail_level = st.session_state[f"task_{task.task_id}_detail_level"]
                
                # Store both iterations and detail level
                task_config[task.task_id] = {
                    "iterations": iterations,
                    "detail_level": detail_level
                }
            
            logger.info(f"Task iterations configured: {task_config}")
            return task_config
        else:
            # Return None if the form hasn't been submitted yet
            return None

def execute_tasks_with_iterations(
    research_plan, 
    task_iterations, 
    agent, 
    search_sources, 
    max_papers, 
    max_web_results
):
    """
    Execute each task with the specified number of iterations.
    Enhanced to support detail level configuration.
    
    Args:
        research_plan: The research plan
        task_iterations: Dict mapping task_id to iteration configuration
        agent: The research agent
        search_sources: List of search sources to use
        max_papers: Maximum papers per query
        max_web_results: Maximum web results per query
        
    Returns:
        Tuple of (all_papers, all_web_results)
    """
    # Collect all papers and web results across iterations
    all_papers = []
    all_web_results = []
    total_iterations_completed = 0
    
    # Create a progress bar
    total_iterations = sum([config.get("iterations", 1) for config in task_iterations.values()])
    progress_bar = st.progress(0)
    
    # Execute iterations for each task independently
    for task in research_plan.tasks:
        task_id = task.task_id
        task_config = task_iterations.get(task_id, {"iterations": 1, "detail_level": "Standard"})
        iterations_for_task = task_config.get("iterations", 1)
        detail_level = task_config.get("detail_level", "Standard")
        
        # Adjust papers and web results based on detail level
        task_max_papers = max_papers
        task_max_web_results = max_web_results
        
        # Apply detail level adjustments
        if detail_level in DETAIL_LEVEL_ADJUSTMENTS:
            adjustment = DETAIL_LEVEL_ADJUSTMENTS[detail_level]
            task_max_papers = max(1, max_papers + adjustment["papers_adjustment"])
            task_max_web_results = max(1, max_web_results + adjustment["web_results_adjustment"])
        
        # Display task header
        task_header = f"## Task {task_id}: {task.description}\n\n"
        task_header += f"Running {iterations_for_task} iterations with {detail_level} detail level for this task...\n\n"
        
        with st.chat_message("assistant"):
            st.markdown(task_header)
            
        st.session_state.messages.append({"role": "assistant", "content": task_header})
        
        # Run the specified number of iterations for this task
        task_papers = []
        task_web_results = []
        
        for i in range(iterations_for_task):
            try:
                iteration_msg = f"Executing iteration {i + 1}/{iterations_for_task} for task {task_id} with {detail_level} detail level..."
                with st.chat_message("assistant"):
                    st.markdown(iteration_msg)
                
                st.session_state.messages.append({"role": "assistant", "content": iteration_msg})
                
                # Execute the iteration with adjusted parameters
                iteration_result = agent.execute_research_iteration(
                    task, 
                    i + 1,  # Iteration number within this task
                    sources=search_sources,
                    max_papers=task_max_papers,
                    max_web_results=task_max_web_results
                )
                
                # Collect papers and web results for this task
                if agent.current_papers:
                    task_papers.extend(agent.current_papers)
                    all_papers.extend(agent.current_papers)
                
                if agent.current_web_results:
                    task_web_results.extend(agent.current_web_results)
                    all_web_results.extend(agent.current_web_results)
                
                # Format results for display - WITHOUT LLM thinking part
                result_display = format_iteration_results(
                    iteration_result, 
                    task_id,
                    agent.current_papers,
                    agent.current_web_results
                )
                
                # Add to chat
                with st.chat_message("assistant"):
                    st.markdown(result_display)
                
                st.session_state.messages.append({"role": "assistant", "content": result_display})
                
                # Update progress bar
                total_iterations_completed += 1
                progress_bar.progress(total_iterations_completed / total_iterations)
                
            except Exception as e:
                logger.error(f"Error in iteration {i + 1} for task {task_id}: {e}")
                
                # Add error message to chat
                error_chat_msg = f"I encountered an error during iteration {i + 1} for task {task_id}. Moving to the next iteration."
                with st.chat_message("assistant"):
                    st.markdown(error_chat_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_chat_msg})
                
                # Update progress bar despite error
                total_iterations_completed += 1
                progress_bar.progress(total_iterations_completed / total_iterations)
        
        # After completing all iterations for this task, display a task summary
        if iterations_for_task > 1:
            display_task_summary(task_id, task, detail_level, agent.task_knowledge, task_papers, task_web_results)
    
    # Mark progress as complete
    progress_bar.progress(1.0)
    
    return all_papers, all_web_results

def format_iteration_results(
    iteration_result, 
    task_id, 
    current_papers, 
    current_web_results
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

def display_task_summary(
    task_id, 
    task, 
    detail_level, 
    task_knowledge, 
    task_papers, 
    task_web_results
):
    """
    Display a summary of task findings after all iterations.
    
    Args:
        task_id: The task ID
        task: The research task
        detail_level: The detail level used
        task_knowledge: Dict of task knowledge
        task_papers: Papers used for this task
        task_web_results: Web results used for this task
    """
    try:
        # Get the task-specific accumulated knowledge
        task_knowledge_text = task_knowledge.get(task_id, "")
        
        # Remove thinking tags from task knowledge
        cleaned_task_knowledge = remove_thinking_tags(task_knowledge_text)
        
        # Display task summary with synthesis
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
                # Fix for ArXiv links - create proper URL
                paper_id = str(paper['id']).lower().replace('arxiv:', '').strip()
                fixed_url = f"https://arxiv.org/abs/{paper_id}"
                
                task_summary += f"{i+1}. [{paper['title']}]({fixed_url}) by {paper['authors']}\n"
        
        # Display web results used in this task
        if task_web_results:
            task_summary += "\nWeb Sources:\n"
            for i, result in enumerate(task_web_results):
                task_summary += f"{i+1}. [{result['title']}]({result['url']})\n"
        
        with st.chat_message("assistant"):
            st.markdown(task_summary)
        
        st.session_state.messages.append({"role": "assistant", "content": task_summary})
        
    except Exception as e:
        logger.error(f"Error generating task summary for task {task_id}: {e}")
        error_msg = f"Error generating summary for task {task_id}. The task was completed successfully, but a summary could not be created."
        with st.chat_message("assistant"):
            st.markdown(error_msg)
=======
"""
UI components for configuring task iterations.
"""
import streamlit as st
from typing import Dict, List, Optional, Any
import logging

from ..agent.models import ResearchTask, TaskExecutionConfig
from ..config.app_config import (
    MAX_ITERATIONS_PER_TASK,
    DEFAULT_ITERATIONS_PER_TASK,
    DETAIL_LEVEL_OPTIONS,
    DEFAULT_DETAIL_LEVEL,
    ITERATION_CONFIG_HELP,
    DETAIL_LEVEL_ADJUSTMENTS
)
from ..utils.text_processing import remove_thinking_tags

logger = logging.getLogger(__name__)

def configure_task_iterations(tasks: List[ResearchTask]) -> Optional[Dict[int, Dict[str, Any]]]:
    """
    Let the user configure iterations per task.
    
    Args:
        tasks: List of ResearchTask objects
        
    Returns:
        Dict mapping task_id to configuration or None if form not submitted
    """
    st.subheader("Configure Iterations Per Task")
    st.write("Specify how many iterations to run for each task. More iterations will result in deeper, more thorough research.")
    st.write("💡 **Tip**: Tasks that are broader or more complex usually benefit from more iterations.")
    
    # Create a form for more organized input
    with st.form("task_iterations_form"):
        for task in tasks:
            st.markdown(f"**Task {task.task_id}:** {task.description}")
            st.markdown("*Queries:*")
            for query in task.queries:
                st.markdown(f"- {query}")
            
            # Configure iterations
            task_iterations = st.number_input(
                f"Number of iterations for Task {task.task_id}",
                min_value=1,
                max_value=MAX_ITERATIONS_PER_TASK,
                value=DEFAULT_ITERATIONS_PER_TASK,
                key=f"task_{task.task_id}_iterations",
                help=ITERATION_CONFIG_HELP["iterations"]
            )
            
            # Configure detail level
            fetch_level = st.select_slider(
                f"Content detail level for Task {task.task_id}",
                options=DETAIL_LEVEL_OPTIONS,
                value=DEFAULT_DETAIL_LEVEL,
                key=f"task_{task.task_id}_detail_level",
                help=ITERATION_CONFIG_HELP["detail_level"]
            )
            
            st.markdown("---")
        
        # Submit button for the form
        submit_button = st.form_submit_button("Confirm Iterations")
        
        # Only return task_iterations if the form was submitted
        if submit_button:
            # Create a dictionary with task_id -> configuration mapping
            task_config = {}
            for task in tasks:
                iterations = st.session_state[f"task_{task.task_id}_iterations"]
                detail_level = st.session_state[f"task_{task.task_id}_detail_level"]
                
                # Store both iterations and detail level
                task_config[task.task_id] = {
                    "iterations": iterations,
                    "detail_level": detail_level
                }
            
            logger.info(f"Task iterations configured: {task_config}")
            return task_config
        else:
            # Return None if the form hasn't been submitted yet
            return None

def execute_tasks_with_iterations(
    research_plan,
    task_iterations,
    agent,
    search_sources,
    max_papers
):
    """
    Execute each task with the specified number of iterations.
    Enhanced to support detail level configuration.

    Args:
        research_plan: The research plan
        task_iterations: Dict mapping task_id to iteration configuration
        agent: The research agent
        search_sources: List of search sources to use
        max_papers: Maximum papers per query

    Returns:
        List of all papers used in research
    """
    # Collect all papers across iterations
    all_papers = []
    total_iterations_completed = 0
    
    # Create a progress bar
    total_iterations = sum([config.get("iterations", 1) for config in task_iterations.values()])
    progress_bar = st.progress(0)
    
    # Execute iterations for each task independently
    for task in research_plan.tasks:
        task_id = task.task_id
        task_config = task_iterations.get(task_id, {"iterations": 1, "detail_level": "Standard"})
        iterations_for_task = task_config.get("iterations", 1)
        detail_level = task_config.get("detail_level", "Standard")
        
        # Adjust papers based on detail level
        task_max_papers = max_papers
        
        # Apply detail level adjustments
        if detail_level in DETAIL_LEVEL_ADJUSTMENTS:
            adjustment = DETAIL_LEVEL_ADJUSTMENTS[detail_level]
            task_max_papers = max(1, max_papers + adjustment["papers_adjustment"])
        
        # Display task header
        task_header = f"## Task {task_id}: {task.description}\n\n"
        task_header += f"Running {iterations_for_task} iterations with {detail_level} detail level for this task...\n\n"
        
        with st.chat_message("assistant"):
            st.markdown(task_header)
            
        st.session_state.messages.append({"role": "assistant", "content": task_header})
        
        # Run the specified number of iterations for this task
        task_papers = []
        
        for i in range(iterations_for_task):
            try:
                iteration_msg = f"Executing iteration {i + 1}/{iterations_for_task} for task {task_id} with {detail_level} detail level..."
                with st.chat_message("assistant"):
                    st.markdown(iteration_msg)
                
                st.session_state.messages.append({"role": "assistant", "content": iteration_msg})
                
                # Execute the iteration with adjusted parameters
                iteration_result = agent.execute_research_iteration(
                    task, 
                    i + 1,  # Iteration number within this task
                    sources=search_sources,
                    max_papers=task_max_papers
                )
                
                # Collect papers for this task
                if agent.current_papers:
                    task_papers.extend(agent.current_papers)
                    all_papers.extend(agent.current_papers)
                
                # Format results for display - WITHOUT LLM thinking part
                result_display = format_iteration_results(
                    iteration_result, 
                    task_id,
                    agent.current_papers
                )
                
                # Add to chat
                with st.chat_message("assistant"):
                    st.markdown(result_display)
                
                st.session_state.messages.append({"role": "assistant", "content": result_display})
                
                # Update progress bar
                total_iterations_completed += 1
                progress_bar.progress(total_iterations_completed / total_iterations)
                
            except Exception as e:
                logger.error(f"Error in iteration {i + 1} for task {task_id}: {e}")
                
                # Add error message to chat
                error_chat_msg = f"I encountered an error during iteration {i + 1} for task {task_id}. Moving to the next iteration."
                with st.chat_message("assistant"):
                    st.markdown(error_chat_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_chat_msg})
                
                # Update progress bar despite error
                total_iterations_completed += 1
                progress_bar.progress(total_iterations_completed / total_iterations)
        
        # After completing all iterations for this task, display a task summary
        if iterations_for_task > 1:
            display_task_summary(task_id, task, detail_level, agent.task_knowledge, task_papers)
    
    # Mark progress as complete
    progress_bar.progress(1.0)

    return all_papers

def format_iteration_results(
    iteration_result, 
    task_id, 
    current_papers
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

def display_task_summary(
    task_id, 
    task, 
    detail_level, 
    task_knowledge, 
    task_papers
):
    """
    Display a summary of task findings after all iterations.
    
    Args:
        task_id: The task ID
        task: The research task
        detail_level: The detail level used
        task_knowledge: Dict of task knowledge
        task_papers: Papers used for this task
    """
    try:
        # Get the task-specific accumulated knowledge
        task_knowledge_text = task_knowledge.get(task_id, "")
        
        # Remove thinking tags from task knowledge
        cleaned_task_knowledge = remove_thinking_tags(task_knowledge_text)
        
        # Display task summary with synthesis
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
                # Fix for ArXiv links - create proper URL
                paper_id = str(paper['id']).lower().replace('arxiv:', '').strip()
                fixed_url = f"https://arxiv.org/abs/{paper_id}"
                
                task_summary += f"{i+1}. [{paper['title']}]({fixed_url}) by {paper['authors']}\n"
        
        with st.chat_message("assistant"):
            st.markdown(task_summary)
        
        st.session_state.messages.append({"role": "assistant", "content": task_summary})
        
    except Exception as e:
        logger.error(f"Error generating task summary for task {task_id}: {e}")
        error_msg = f"Error generating summary for task {task_id}. The task was completed successfully, but a summary could not be created."
        with st.chat_message("assistant"):
            st.markdown(error_msg)
>>>>>>> Stashed changes
        st.session_state.messages.append({"role": "assistant", "content": error_msg})