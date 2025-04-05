"""
Core research agent that orchestrates the research process.
"""
import logging
from typing import Dict, Any, List, Optional, Union

from deep_research.agent.models import ResearchPlan, ResearchTask, ResearchIteration, ResearchFindings
from deep_research.agent.plan_generator import PlanGenerator
from deep_research.agent.research_executor import ResearchExecutor
from deep_research.agent.knowledge_synthesizer import KnowledgeSynthesizer
from deep_research.config.llm_config import get_ollama_llm, test_ollama_connection
from deep_research.utils.text_processing import remove_thinking_tags
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class ResearchAgent:
    """
    Main research agent class that orchestrates the research process.
    This class acts as a facade to coordinate the specialized components.
    """
    
    def __init__(self, llm_model: str = "phi4:latest"):
        """
        Initialize the research agent.
        
        Args:
            llm_model: Name of the Ollama model to use
        """
        try:
            # Test Ollama connection
            if not test_ollama_connection(llm_model):
                raise RuntimeError(f"Failed to connect to Ollama with model {llm_model}")
            
            # Store the model name
            self.model_name = llm_model
            
            # Initialize a basic LLM instance for general queries
            self.llm = get_ollama_llm(model_name=llm_model)
            
            # Initialize specialized components
            self.plan_generator = PlanGenerator(model_name=llm_model)
            self.research_executor = ResearchExecutor(model_name=llm_model)
            self.knowledge_synthesizer = KnowledgeSynthesizer(model_name=llm_model)
            
            # Initialize state variables
            self.current_papers = []
            self.current_web_results = []
            self.accumulated_knowledge = ""
            self.task_knowledge = {}  # Dictionary to track knowledge per task
            
            logger.info(f"Research agent initialized with model: {llm_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize research agent: {e}")
            raise RuntimeError(
                f"Failed to initialize research agent: {e}. Please ensure Ollama is running "
                f"and the specified model is installed."
            )
    
    def generate_research_plan(self, topic: str) -> ResearchPlan:
        """
        Generate a research plan for the given topic.
        
        Args:
            topic: Research topic
            
        Returns:
            Research plan as a ResearchPlan object
        """
        # Delegate to the plan generator
        plan = self.plan_generator.generate_research_plan(topic)
        
        # Initialize task knowledge dictionary for each task
        for task in plan.tasks:
            self.task_knowledge[task.task_id] = ""
        
        return plan
    
    def modify_research_plan(self, original_plan: ResearchPlan, feedback: str) -> ResearchPlan:
        """
        Modify a research plan based on user feedback.
        
        Args:
            original_plan: Original research plan
            feedback: User feedback
            
        Returns:
            Modified research plan
        """
        # Delegate to the plan generator
        modified_plan = self.plan_generator.modify_research_plan(original_plan, feedback)
        
        # Reset task knowledge for the modified plan
        self.task_knowledge = {}
        for task in modified_plan.tasks:
            self.task_knowledge[task.task_id] = ""
        
        return modified_plan
    
    def execute_research_iteration(
        self, 
        task: ResearchTask, 
        iteration_number: int, 
        sources: List[str] = ["ArXiv", "Web Search"],
        max_papers: int = 2, 
        max_web_results: int = 3
    ) -> ResearchIteration:
        """
        Execute a single research iteration for a specific task.
        
        Args:
            task: Research task to execute
            iteration_number: Current iteration number
            sources: List of research sources to use
            max_papers: Maximum number of papers to retrieve
            max_web_results: Maximum number of web results to retrieve
            
        Returns:
            Results of the research iteration
        """
        # Get the current task knowledge
        task_specific_knowledge = self.task_knowledge.get(task.task_id, "")
        
        # Delegate to the research executor
        iteration_result = self.research_executor.execute_research_iteration(
            task, 
            iteration_number,
            sources=sources,
            max_papers=max_papers,
            max_web_results=max_web_results,
            task_knowledge=task_specific_knowledge
        )
        
        # Update current sources (for display)
        self.current_papers = self.research_executor.current_papers
        self.current_web_results = self.research_executor.current_web_results
        
        # Update task-specific knowledge
        task_knowledge = self.knowledge_synthesizer.update_task_knowledge(
            task.task_id, 
            iteration_result,
            task_specific_knowledge
        )
        self.task_knowledge[task.task_id] = task_knowledge
        
        return iteration_result
    
    def synthesize_knowledge(self) -> str:
        """
        Synthesize knowledge from all tasks.
        
        Returns:
            Synthesized knowledge
        """
        # Delegate to the knowledge synthesizer
        synthesized_knowledge = self.knowledge_synthesizer.synthesize_global_knowledge(self.task_knowledge)
        
        # Update accumulated knowledge
        self.accumulated_knowledge = synthesized_knowledge
        
        return synthesized_knowledge
    
    def generate_final_report_with_sources(
        self, 
        topic: str, 
        objective: str, 
        papers: List[Dict[str, Any]], 
        web_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a comprehensive final research report.
        
        Args:
            topic: Research topic
            objective: Research objective
            papers: List of papers used in research
            web_results: List of web results used in research
            
        Returns:
            Final research report with citations
        """
        # Delegate to the knowledge synthesizer
        report = self.knowledge_synthesizer.generate_final_report(
            topic, 
            objective, 
            self.task_knowledge,
            papers, 
            web_results
        )
        
        return report
    
    def answer_followup_question(self, question: str, topic: str) -> str:
        """
        Answer a follow-up question using the accumulated knowledge.
        
        Args:
            question: User's follow-up question
            topic: Research topic
            
        Returns:
            Response to the question
        """
        follow_up_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant providing information based on previous research findings.
            Use the accumulated knowledge to answer the user's question in detail.
            If the question requires information not covered in the research, acknowledge the limitations
            and suggest what additional research might be needed."""),
            ("user", f"Research topic: {topic}"),
            ("user", f"Accumulated research knowledge: {self.accumulated_knowledge}"),
            ("user", f"User question: {question}"),
            ("user", "Provide a detailed, informative answer with all relevant information from the research.")
        ])
        
        follow_up_chain = follow_up_prompt | self.llm
        response = follow_up_chain.invoke({})
        response = remove_thinking_tags(response)
        return response