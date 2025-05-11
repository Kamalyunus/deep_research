"""
Main Streamlit application entry point for the Deep Research Agent.
"""
import streamlit as st
import json
import time
import traceback
import os
from typing import List, Dict, Any

from deep_research.agent.research_agent import ResearchAgent
from deep_research.agent.models import ResearchPlan, ResearchTask
from deep_research.ui.task_config import configure_task_iterations, execute_tasks_with_iterations
from deep_research.ui.report_display import (
    format_research_plan,
    format_configuration_summary,
    save_research_report,
    format_error_message
)
from deep_research.config.app_config import (
    RESEARCH_OUTPUT_DIR,
    MODEL_OPTIONS,
    DEFAULT_MAX_PAPERS,
    DEFAULT_MAX_WEB_RESULTS,
    DEFAULT_SEARCH_SOURCES,
    get_sanitized_filename,
    reset_session_state,
    SESSION_KEYS
)

# Configure application-wide logging
from deep_research.utils.logger import configure_root_logger, get_logger

# Initialize root logger
configure_root_logger(level="info", log_to_file=True)

# Get a logger for this module
logger = get_logger(__name__)

# Set up page configuration
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide"
)

# Log application startup
logger.info("Deep Research Agent application starting up")

# Initialize session state variables
if SESSION_KEYS["messages"] not in st.session_state:
    st.session_state[SESSION_KEYS["messages"]] = []
if SESSION_KEYS["research_plan"] not in st.session_state:
    st.session_state[SESSION_KEYS["research_plan"]] = None
if SESSION_KEYS["plan_approved"] not in st.session_state:
    st.session_state[SESSION_KEYS["plan_approved"]] = False
if SESSION_KEYS["research_iterations"] not in st.session_state:
    st.session_state[SESSION_KEYS["research_iterations"]] = 0
if SESSION_KEYS["agent"] not in st.session_state:
    st.session_state[SESSION_KEYS["agent"]] = None
if SESSION_KEYS["all_papers"] not in st.session_state:
    st.session_state[SESSION_KEYS["all_papers"]] = []
if SESSION_KEYS["all_web_results"] not in st.session_state:
    st.session_state[SESSION_KEYS["all_web_results"]] = []
if SESSION_KEYS["plan_generated"] not in st.session_state:
    st.session_state[SESSION_KEYS["plan_generated"]] = False
if SESSION_KEYS["task_iterations"] not in st.session_state:
    st.session_state[SESSION_KEYS["task_iterations"]] = {}
if SESSION_KEYS["iterations_configured"] not in st.session_state:
    st.session_state[SESSION_KEYS["iterations_configured"]] = False
if SESSION_KEYS["research_complete"] not in st.session_state:
    st.session_state[SESSION_KEYS["research_complete"]] = False
if SESSION_KEYS["debug_mode"] not in st.session_state:
    st.session_state[SESSION_KEYS["debug_mode"]] = False

# UI Components
st.title("📚 Deep Research LLM Agent")
st.markdown("""
This application uses LangChain, LangGraph, and Ollama to:
1. Generate a comprehensive research plan based on your topic
2. Configure iterations for each research task
3. Execute research in iterative loops using ArXiv and web search
4. Build task-specific knowledge across iterations
5. Consolidate findings into a comprehensive final report with citations
""")

# Create research_outputs directory if it doesn't exist
if not os.path.exists(RESEARCH_OUTPUT_DIR):
    os.makedirs(RESEARCH_OUTPUT_DIR)

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    llm_model = st.selectbox(
        "Select LLM Model",
        MODEL_OPTIONS,
        index=0,
        help="Choose a more powerful model for deeper analysis but slower processing"
    )
    
    max_papers = st.slider(
        "Papers per Query",
        min_value=1,
        max_value=5,
        value=DEFAULT_MAX_PAPERS,
        help="How many papers to retrieve for each search query"
    )
    
    max_web_results = st.slider(
        "Web Search Results per Query",
        min_value=1,
        max_value=10,
        value=DEFAULT_MAX_WEB_RESULTS,
        help="How many web results to retrieve for each search query"
    )
    
    search_sources = st.multiselect(
        "Research Sources",
        DEFAULT_SEARCH_SOURCES,
        default=DEFAULT_SEARCH_SOURCES,
        help="Select which sources to use for research"
    )
    
    st.session_state[SESSION_KEYS["debug_mode"]] = st.checkbox(
        "Debug Mode", 
        value=False,
        help="Show detailed error messages and logs"
    )
    
    if st.button("Reset Conversation"):
        reset_session_state(st.session_state)
        st.rerun()

# Display chat messages
for message in st.session_state[SESSION_KEYS["messages"]]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize or update the research agent if needed
if (st.session_state[SESSION_KEYS["agent"]] is None or 
    (hasattr(st.session_state[SESSION_KEYS["agent"]], 'model_name') and 
     st.session_state[SESSION_KEYS["agent"]].model_name != llm_model)):
    
    with st.spinner("Initializing research agent..."):
        try:
            st.session_state[SESSION_KEYS["agent"]] = ResearchAgent(llm_model=llm_model)
        except Exception as e:
            error_msg = format_error_message(e, "initializing research agent", st.session_state[SESSION_KEYS["debug_mode"]])
            st.error(error_msg)

# Main application flow
if not st.session_state[SESSION_KEYS["plan_generated"]]:
    research_topic = st.text_input("Enter your research topic:", 
                                  help="Be specific about what you want to research. Complex topics work best.")

    # Research Plan Generation
    if research_topic:
        with st.spinner("Generating comprehensive research plan..."):
            # Generate research plan
            try:
                agent = st.session_state[SESSION_KEYS["agent"]]
                research_plan = agent.generate_research_plan(research_topic)
                st.session_state[SESSION_KEYS["research_plan"]] = research_plan
                st.session_state[SESSION_KEYS["plan_generated"]] = True  # Mark plan as generated
                
                # Format plan for display
                plan_display = format_research_plan(research_plan)
                
                # Display the plan immediately
                with st.chat_message("assistant"):
                    st.markdown(plan_display)
                    st.markdown("Do you approve this research plan? Type 'yes' to proceed or suggest modifications to make it more relevant to your needs.")
                
                # Add to chat history
                st.session_state[SESSION_KEYS["messages"]].append({"role": "assistant", "content": plan_display})
                st.session_state[SESSION_KEYS["messages"]].append({"role": "assistant", "content": "Do you approve this research plan? Type 'yes' to proceed or suggest modifications to make it more relevant to your needs."})
                
            except Exception as e:
                error_msg = format_error_message(e, "generating research plan", st.session_state[SESSION_KEYS["debug_mode"]])
                st.error(error_msg)
                
                # Add error message to chat
                error_chat_msg = "I encountered an error while generating the research plan. Please try again with a different topic."
                with st.chat_message("assistant"):
                    st.markdown(error_chat_msg)
                st.session_state[SESSION_KEYS["messages"]].append({"role": "assistant", "content": error_chat_msg})
                st.session_state[SESSION_KEYS["plan_generated"]] = False  # Reset plan status
else:
    # Show the current research topic when plan is already generated
    if st.session_state[SESSION_KEYS["research_plan"]]:
        st.info(f"Current research topic: **{st.session_state[SESSION_KEYS['research_plan']].topic}**")

# Configure iterations for each task if plan is approved but iterations not yet configured
if (st.session_state[SESSION_KEYS["plan_approved"]] and 
    not st.session_state[SESSION_KEYS["iterations_configured"]] and 
    st.session_state[SESSION_KEYS["research_plan"]]):
    
    task_iterations = configure_task_iterations(st.session_state[SESSION_KEYS["research_plan"]].tasks)
    
    # Only proceed if the user has submitted the form and we have task iterations
    if task_iterations is not None:
        st.session_state[SESSION_KEYS["task_iterations"]] = task_iterations
        st.session_state[SESSION_KEYS["iterations_configured"]] = True
        
        # Create a summary of the iteration configuration
        config_summary = format_configuration_summary(task_iterations, st.session_state[SESSION_KEYS["research_plan"]].tasks)
        
        with st.chat_message("assistant"):
            st.markdown(config_summary)
        
        st.session_state[SESSION_KEYS["messages"]].append({"role": "assistant", "content": config_summary})
        
        # Execute all tasks with their configured iterations
        with st.spinner("Executing research tasks..."):
            try:
                all_papers, all_web_results = execute_tasks_with_iterations(
                    st.session_state[SESSION_KEYS["research_plan"]],
                    st.session_state[SESSION_KEYS["task_iterations"]],
                    st.session_state[SESSION_KEYS["agent"]],
                    search_sources,
                    max_papers,
                    max_web_results
                )
                
                # Save collected sources
                st.session_state[SESSION_KEYS["all_papers"]] = all_papers
                st.session_state[SESSION_KEYS["all_web_results"]] = all_web_results
                
                # Generate final report
                with st.spinner("Generating comprehensive final research report..."):
                    try:
                        agent = st.session_state[SESSION_KEYS["agent"]]
                        final_report = agent.generate_final_report_with_sources(
                            st.session_state[SESSION_KEYS["research_plan"]].topic,
                            st.session_state[SESSION_KEYS["research_plan"]].objective,
                            all_papers,
                            all_web_results
                        )
                        
                        report_display = "# Final Research Report\n\n"
                        report_display += final_report
                        
                        with st.chat_message("assistant"):
                            st.markdown(report_display)
                        
                        st.session_state[SESSION_KEYS["messages"]].append({"role": "assistant", "content": report_display})
                        st.session_state[SESSION_KEYS["research_complete"]] = True
                        
                        # Save the research report to a file
                        save_research_report(
                            st.session_state[SESSION_KEYS["research_plan"]],
                            report_display,
                            agent.task_knowledge
                        )
                        
                    except Exception as e:
                        error_msg = format_error_message(e, "generating final report", st.session_state[SESSION_KEYS["debug_mode"]])
                        st.error(error_msg)
                        
                        # Add error message to chat
                        error_chat_msg = "I encountered an error while generating the final report. The research iterations were completed successfully, but I couldn't create a comprehensive summary."
                        with st.chat_message("assistant"):
                            st.markdown(error_chat_msg)
                        st.session_state[SESSION_KEYS["messages"]].append({"role": "assistant", "content": error_chat_msg})
                        st.session_state[SESSION_KEYS["research_complete"]] = True
            
            except Exception as e:
                error_msg = format_error_message(e, "executing research", st.session_state[SESSION_KEYS["debug_mode"]])
                st.error(error_msg)
                
                # Add error message to chat
                error_chat_msg = "I encountered an error while conducting the research. Please try again with different settings or a different topic."
                with st.chat_message("assistant"):
                    st.markdown(error_chat_msg)
                st.session_state[SESSION_KEYS["messages"]].append({"role": "assistant", "content": error_chat_msg})

# User input handling
if prompt := st.chat_input("Your response:"):
    st.session_state[SESSION_KEYS["messages"]].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Handle plan approval or modification
    if not st.session_state[SESSION_KEYS["plan_approved"]] and st.session_state[SESSION_KEYS["research_plan"]]:
        if prompt.lower() in ["yes", "yes.", "approved", "i approve", "proceed", "looks good", "sure", "go ahead"]:
            st.session_state[SESSION_KEYS["plan_approved"]] = True
            
            with st.chat_message("assistant"):
                st.markdown("Plan approved! Now you can configure the number of iterations for each task. More iterations will provide deeper, more thorough research.")
                st.session_state[SESSION_KEYS["messages"]].append({"role": "assistant", "content": "Plan approved! Now you can configure the number of iterations for each task. More iterations will provide deeper, more thorough research."})
            
            # The configuration form will appear on the next refresh
            st.rerun()
            
        else:
            # Handle plan modification request
            with st.spinner("Modifying research plan based on your feedback..."):
                try:
                    agent = st.session_state[SESSION_KEYS["agent"]]
                    modified_plan = agent.modify_research_plan(
                        st.session_state[SESSION_KEYS["research_plan"]], 
                        prompt
                    )
                    st.session_state[SESSION_KEYS["research_plan"]] = modified_plan
                    
                    # Format plan for display
                    plan_display = format_research_plan(modified_plan, modified=True)
                    
                    with st.chat_message("assistant"):
                        st.markdown(plan_display)
                        st.markdown("Do you approve this modified research plan? Type 'yes' to proceed or suggest further modifications.")
                    
                    st.session_state[SESSION_KEYS["messages"]].append({"role": "assistant", "content": plan_display})
                    st.session_state[SESSION_KEYS["messages"]].append({"role": "assistant", "content": "Do you approve this modified research plan? Type 'yes' to proceed or suggest further modifications."})
                    
                except Exception as e:
                    error_msg = format_error_message(e, "modifying research plan", st.session_state[SESSION_KEYS["debug_mode"]])
                    st.error(error_msg)
                    
                    # Add error message to chat
                    error_chat_msg = "I encountered an error while modifying the research plan. Please try again with different suggestions."
                    with st.chat_message("assistant"):
                        st.markdown(error_chat_msg)
                    st.session_state[SESSION_KEYS["messages"]].append({"role": "assistant", "content": error_chat_msg})
    
    # Handle questions after research is complete
    elif st.session_state[SESSION_KEYS["research_complete"]]:
        # Process follow-up question
        with st.spinner("Processing your question using the research findings..."):
            try:
                agent = st.session_state[SESSION_KEYS["agent"]]
                response = agent.answer_followup_question(
                    prompt,
                    st.session_state[SESSION_KEYS["research_plan"]].topic
                )
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                st.session_state[SESSION_KEYS["messages"]].append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = format_error_message(e, "processing follow-up question", st.session_state[SESSION_KEYS["debug_mode"]])
                st.error(error_msg)
                
                error_msg = "I'm sorry, I couldn't process your question. Please try asking in a different way."
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                st.session_state[SESSION_KEYS["messages"]].append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    # This will allow running the app with `streamlit run app.py`
    pass