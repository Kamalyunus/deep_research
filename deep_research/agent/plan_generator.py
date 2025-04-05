from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import json
import logging
from typing import Dict, Any, Optional

from .models import ResearchPlan, ResearchTask
from ..config.llm_config import get_ollama_llm

logger = logging.getLogger(__name__)

class PlanGenerator:
    """
    Responsible for generating and modifying research plans.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the plan generator.
        
        Args:
            model_name: Optional model name to use for plan generation
        """
        self.model_name = model_name
        self.llm = get_ollama_llm(model_name, preset="plan_generation")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(Exception)
    )
    def get_structured_response(self, model_class, system_prompt, user_prompt, temperature=0.5):
        """
        Get a structured response from the LLM using Pydantic parser.
        
        Args:
            model_class: Pydantic model class to structure the response
            system_prompt: System prompt to send to the LLM
            user_prompt: User prompt to send to the LLM
            temperature: Temperature for generation
            
        Returns:
            Structured response as a Pydantic model instance
        """
        try:
            # Create a Pydantic output parser
            parser = PydanticOutputParser(pydantic_object=model_class)
            
            # Get the format instructions and escape curly braces
            format_instructions = parser.get_format_instructions()
            escaped_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
        
            # Combine system prompt with format instructions
            full_system_prompt = f"{system_prompt}\n\n{escaped_instructions}"

            # Format prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", full_system_prompt),
                ("user", user_prompt)
            ])
            
            # Generate response
            chain = prompt | self.llm
            response = chain.invoke({})
            
            # Parse the response
            parsed_response = parser.parse(response)
            
            return parsed_response
        except Exception as e:
            logger.error(f"Error getting structured response: {e}")
            # Create a minimal fallback instance with error information
            try:
                # Try to extract JSON from the response using simple heuristics
                if isinstance(response, str) and '{' in response and '}' in response:
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    json_str = response[start_idx:end_idx]
                    
                    try:
                        # Try to parse as JSON and create an instance
                        data = json.loads(json_str)
                        instance = model_class(**data)
                        logger.info(f"Recovered partial structured output via JSON extraction")
                        return instance
                    except:
                        pass
            except:
                pass
                
            # If all parsing fails, raise the original error
            raise
    
    def generate_research_plan(self, topic: str) -> ResearchPlan:
        """
        Generate a detailed research plan for the given topic.
        
        Args:
            topic: Research topic
            
        Returns:
            Research plan as a ResearchPlan object
        """
        try:
            # Use the original, detailed system prompt
            system_prompt = """You are an expert research agent designing a comprehensive research plan. 
            Your task is to create a detailed research plan based on the user's topic.
            
            Break down the research into 3-5 specific tasks that cover different dimensions or aspects of the topic. 
            Each task should:
            1. Have a clear objective and focus
            2. Be distinct from other tasks to avoid overlap
            3. Together, provide comprehensive coverage of the topic
            
            For each task, generate 3-5 specific search queries that:
            1. Are highly specific and targeted to yield relevant results
            2. Use technical terminology appropriate for academic papers and scholarly sources
            3. Include variations to capture different aspects of the same subtopic
            4. Avoid overly general terms that would return too many irrelevant results
            
            The overall research plan should systematically explore the topic from multiple angles, ensuring depth and breadth."""
            
            # Use the original user prompt
            user_prompt = f"""Create a detailed, comprehensive research plan for the topic: {topic}
            
            Think carefully about the different dimensions of this topic and how to break it into distinct research tasks.
            Your research plan should be thorough enough that executing it would result in a comprehensive understanding of {topic}."""
            
            # Get structured response
            plan = self.get_structured_response(ResearchPlan, system_prompt, user_prompt)
            
            logger.info(f"Successfully generated research plan for topic: {topic}")
            return plan
            
        except Exception as e:
            logger.error(f"Error generating research plan: {e}")
            return self._generate_fallback_plan(topic)
    
    def modify_research_plan(self, original_plan: ResearchPlan, feedback: str) -> ResearchPlan:
        """
        Modify a research plan based on user feedback.
        
        Args:
            original_plan: Original research plan
            feedback: User feedback
            
        Returns:
            Modified research plan
        """
        try:
            # Create a Pydantic output parser
            parser = PydanticOutputParser(pydantic_object=ResearchPlan)
            
            # Get the format instructions
            format_instructions = parser.get_format_instructions()
            
            # Create system prompt
            system_prompt = """You are an expert research agent. Modify the research plan based on user feedback.
            Pay careful attention to:
            
            1. Adding, removing, or modifying tasks according to user feedback
            2. Refining queries to be more specific and targeted
            3. Addressing any concerns about comprehensiveness or focus
            4. Ensuring the plan covers all aspects mentioned by the user
            
            The modified plan should fully address the user's feedback while maintaining a coherent, comprehensive structure."""
            
            # Convert tasks to a simpler text format to avoid JSON formatting issues
            tasks_text = ""
            for task in original_plan.tasks:
                tasks_text += f"Task {task.task_id}: {task.description}\n"
                tasks_text += f"Queries: {', '.join(task.queries)}\n\n"
            
            # Create user prompt
            user_prompt = f"""
            Original plan:
            Topic: {original_plan.topic}
            Objective: {original_plan.objective}
            
            Tasks:
            {tasks_text}
            
            User feedback: {feedback}
            
            Please provide a modified research plan that addresses the user's feedback.
            """
            
            # Combine prompts and format instructions directly instead of using ChatPromptTemplate
            combined_prompt = f"{system_prompt}\n\n{format_instructions}\n\n{user_prompt}"
            
            # Invoke the LLM directly
            response = self.llm.invoke(combined_prompt)
            
            # Parse the response
            try:
                parsed_response = parser.parse(response)
                logger.info("Successfully modified research plan based on user feedback")
                return parsed_response
            except Exception as parse_error:
                logger.warning(f"Error parsing response: {parse_error}")
                
                # Try to extract JSON from the response as a fallback
                try:
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = response[start_idx:end_idx]
                        data = json.loads(json_str)
                        
                        # Manually create ResearchPlan
                        topic = data.get("topic", original_plan.topic)
                        objective = data.get("objective", original_plan.objective)
                        
                        tasks = []
                        for task_data in data.get("tasks", []):
                            task = ResearchTask(
                                task_id=task_data.get("task_id", len(tasks) + 1),
                                description=task_data.get("description", "Research task"),
                                queries=task_data.get("queries", ["Research query"])
                            )
                            tasks.append(task)
                        
                        if not tasks:
                            tasks = original_plan.tasks
                        
                        return ResearchPlan(
                            topic=topic,
                            objective=objective,
                            tasks=tasks
                        )
                except Exception as json_error:
                    logger.warning(f"Error extracting JSON: {json_error}")
                
                # Return original plan if all parsing attempts fail
                return original_plan
        except Exception as e:
            logger.error(f"Error modifying research plan: {e}")
            logger.info("Returning original plan due to modification error")
            return original_plan
    
    def _generate_fallback_plan(self, topic: str) -> ResearchPlan:
        """
        Generate a minimal fallback plan when plan generation fails.
        
        Args:
            topic: Research topic
            
        Returns:
            Minimal research plan
        """
        logger.warning(f"Using fallback plan for topic: {topic}")
        
        # Create a single generic task
        fallback_task = ResearchTask(
            task_id=1,
            description=f"Research on {topic}",
            queries=[
                f"{topic} research", 
                f"{topic} latest developments", 
                f"{topic} key concepts"
            ]
        )
        
        # Create a minimal plan
        fallback_plan = ResearchPlan(
            topic=topic,
            objective=f"Research and gather information about {topic}",
            tasks=[fallback_task]
        )
        
        return fallback_plan