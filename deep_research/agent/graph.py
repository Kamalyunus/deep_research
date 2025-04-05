"""
LangGraph workflow definition for the Deep Research Agent.
"""
from langgraph.graph import Graph as LangGraph
from langgraph.graph import END
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

def create_research_graph(plan_chain, execute_iteration_fn, synthesize_knowledge_fn):
    """
    Create a LangGraph workflow for the research process.
    
    Args:
        plan_chain: Chain for generating research plans
        execute_iteration_fn: Function for executing a research iteration
        synthesize_knowledge_fn: Function for synthesizing knowledge
        
    Returns:
        LangGraph workflow for research
    """
    # Define nodes
    def generate_plan(state):
        """
        Node to generate a research plan from the topic.
        """
        topic = state["topic"]
        try:
            plan = plan_chain.invoke({"topic": topic})
            return {"plan": plan, "current_task_index": 0}
        except Exception as e:
            logger.error(f"Error generating plan: {e}")
            return {"error": str(e)}
    
    def execute_task(state):
        """
        Node to execute the current research task.
        """
        plan = state["plan"]
        task_index = state["current_task_index"]
        iteration = state.get("iteration", 1)
        
        if task_index >= len(plan.tasks):
            return {"status": "completed"}
        
        try:
            current_task = plan.tasks[task_index]
            result = execute_iteration_fn(current_task, iteration)
            
            # Update state
            previous_iterations = state.get("iterations", [])
            previous_iterations.append(result.dict())
            
            # Synthesize knowledge
            knowledge = synthesize_knowledge_fn(previous_iterations)
            
            return {
                "current_result": result,
                "iterations": previous_iterations,
                "accumulated_knowledge": knowledge,
                "current_task_index": task_index + 1,
                "status": "in_progress"
            }
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return {"error": str(e), "status": "error"}
    
    def should_continue(state):
        """
        Decision node to determine if the research process should continue.
        """
        if "error" in state:
            return "complete"
            
        max_iter = state.get("max_iterations", 3)
        current_iter = state.get("iteration", 1)
        status = state.get("status", "in_progress")
        
        if status == "completed" or status == "error":
            return "complete"
        elif current_iter >= max_iter:
            return "complete"
        else:
            return "continue"
    
    def prepare_next_iteration(state):
        """
        Node to prepare for the next iteration.
        """
        current_iter = state.get("iteration", 1)
        return {"iteration": current_iter + 1}
    
    # Define graph
    workflow = LangGraph()
    
    # Add nodes
    workflow.add_node("generate_plan", generate_plan)
    workflow.add_node("execute_task", execute_task)
    workflow.add_node("prepare_next_iteration", prepare_next_iteration)
    
    # Add edges
    workflow.add_edge("generate_plan", "execute_task")
    workflow.add_conditional_edges(
        "execute_task",
        should_continue,
        {
            "continue": "prepare_next_iteration",
            "complete": END
        }
    )
    workflow.add_edge("prepare_next_iteration", "execute_task")
    
    # Set entry point
    workflow.set_entry_point("generate_plan")
    
    return workflow

class ResearchGraphState:
    """
    State class for managing the research graph execution state.
    """
    def __init__(self, topic, max_iterations=3):
        self.state = {
            "topic": topic,
            "max_iterations": max_iterations,
            "iteration": 1,
            "current_task_index": 0,
            "status": "not_started",
            "iterations": [],
            "accumulated_knowledge": ""
        }
    
    def update(self, new_state):
        """
        Update the state with new values.
        """
        self.state.update(new_state)
    
    def get(self):
        """
        Get the current state.
        """
        return self.state.copy()