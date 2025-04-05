"""
Research execution functionality for the Deep Research Agent.
"""
from langchain_core.prompts import ChatPromptTemplate
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import time

from .models import ResearchTask, ResearchIteration, ResearchFindings
from ..config.llm_config import get_ollama_llm
from ..utils.error_handling import (
    parse_pydantic_from_llm,
    ExecutionError,
    extract_json_from_text,
    with_retry,
)
from ..utils.text_processing import remove_thinking_tags
from ..sources.source_processor import SourceProcessor
from ..sources.arxiv import search_arxiv
from ..sources.web_search import search_web

logger = logging.getLogger(__name__)

class ResearchExecutor:
    """
    Responsible for executing research tasks and iterations.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the research executor.
        
        Args:
            model_name: Optional model name to use for research execution
        """
        self.model_name = model_name
        self.current_papers = []
        self.current_web_results = []
    
    def _refine_search_queries(
        self, 
        task: ResearchTask, 
        task_knowledge: str, 
        iteration: int
    ) -> List[str]:
        """
        Refine search queries based on accumulated task knowledge.
        
        Args:
            task: The current research task
            task_knowledge: Accumulated knowledge for this task
            iteration: Current iteration number
            
        Returns:
            List of refined search queries
        """
        system_prompt = """You are a research query specialist. Based on the accumulated knowledge from previous iterations, 
        generate refined search queries that will help uncover new information and fill knowledge gaps.
        
        Your queries should:
        1. Target specific areas identified as knowledge gaps in previous iterations
        2. Use precise terminology from the domain
        3. Avoid repeating earlier queries that have already been explored
        4. Focus on areas where contradictions or debates were identified
        5. Explore emerging trends or future directions mentioned in previous findings"""
        
        user_prompt = f"""
        Research Task: {task.description}
        Current Iteration: {iteration}
        
        Original queries from task:
        {", ".join(task.queries)}
        
        Accumulated knowledge from previous iterations:
        {task_knowledge}
        
        Please generate 3-5 refined search queries that will help fill knowledge gaps and uncover new information.
        Focus on aspects that haven't been thoroughly explored yet.
        """
        
        try:
            # Get LLM
            llm = get_ollama_llm(self.model_name)
            
            refined_queries_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", user_prompt)
            ])
            
            refined_queries_chain = refined_queries_prompt | llm
            result = refined_queries_chain.invoke({})
            
            # Parse the response to extract queries
            queries = []
            for line in result.strip().split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('*') or line.startswith('1.') or 
                            line.startswith('2.') or line.startswith('3.')):
                    # Extract the query by removing list markers
                    query = line.lstrip('-* 0123456789.').strip()
                    if query:
                        queries.append(query)
            
            # If we couldn't parse any queries, create some based on the response
            if not queries and result:
                # Try to generate queries from the text
                chunks = [chunk.strip() for chunk in result.split('.') if len(chunk.strip()) > 10]
                queries = chunks[:5]  # Take up to 5 chunks as queries
            
            # Fall back to original queries if we still don't have any
            if not queries:
                logger.warning("Could not extract refined queries, using original queries")
                queries = task.queries
                
            return queries[:5]  # Return at most 5 queries
            
        except Exception as e:
            logger.error(f"Error refining search queries: {e}")
            # Fall back to original queries
            return task.queries
    
    @with_retry(max_attempts=3)
    def execute_research_iteration(
        self, 
        task: ResearchTask, 
        iteration_number: int, 
        sources: List[str] = ["ArXiv", "Web Search"],
        max_papers: int = 2, 
        max_web_results: int = 3,
        task_knowledge: str = ""
    ) -> ResearchIteration:
        """
        Execute a single research iteration for a specific task.
        
        Args:
            task: Research task to execute
            iteration_number: Current iteration number
            sources: List of research sources to use
            max_papers: Maximum number of papers to retrieve
            max_web_results: Maximum number of web results to retrieve
            task_knowledge: Current task-specific knowledge
            
        Returns:
            Results of the research iteration
        """
        all_research_sources = []
        
        # Reset current sources
        self.current_papers = []
        self.current_web_results = []
        
        # Generate enhanced queries for this iteration based on previous knowledge
        if iteration_number > 1 and task_knowledge:
            # If this is not the first iteration, enhance queries based on accumulated knowledge
            enhanced_queries = self._refine_search_queries(task, task_knowledge, iteration_number)
        else:
            # For the first iteration, use the original task queries
            enhanced_queries = task.queries
        
        # Execute searches for each query
        for query in enhanced_queries:
            # ArXiv search if selected
            if "ArXiv" in sources:
                try:
                    papers = search_arxiv(query, max_papers=max_papers)
                    if papers:
                        paper_context, paper_info = SourceProcessor.process_research_sources(papers, "arxiv")
                        all_research_sources.append({
                            "source_type": "ArXiv",
                            "query": query,
                            "results": papers
                        })
                        self.current_papers.extend(paper_info)
                except Exception as e:
                    logger.error(f"Error in ArXiv search for query '{query}': {e}")
            
            # Web search if selected
            if "Web Search" in sources:
                try:
                    web_results = search_web(query, max_results=max_web_results)
                    if web_results:
                        web_context, web_info = SourceProcessor.process_research_sources(web_results, "web")
                        all_research_sources.append({
                            "source_type": "Web",
                            "query": query,
                            "results": web_results
                        })
                        self.current_web_results.extend(web_info)
                except Exception as e:
                    logger.error(f"Error in web search for query '{query}': {e}")
        
        # Format the research sources for the prompt
        sources_text = self._format_sources_for_analysis(all_research_sources)
        
        # Analyze the sources
        return self._analyze_research_sources(
            task, 
            iteration_number, 
            sources_text, 
            task_knowledge
        )
    
    def _format_sources_for_analysis(self, all_research_sources: List[Dict[str, Any]]) -> str:
        """
        Format research sources for analysis.
        
        Args:
            all_research_sources: List of source groups
            
        Returns:
            Formatted sources text
        """
        sources_text = ""
        for source_group in all_research_sources:
            sources_text += f"Source Type: {source_group['source_type']}\n"
            sources_text += f"Query: {source_group['query']}\n"
            sources_text += "Results:\n"
            
            for result in source_group['results']:
                if source_group['source_type'] == "ArXiv":
                    sources_text += f"Title: {result.metadata.get('Title', 'Untitled')}\n"
                    sources_text += f"Authors: {result.metadata.get('Authors', 'Unknown')}\n"
                    sources_text += f"Published: {result.metadata.get('Published', 'Unknown date')}\n"
                    sources_text += f"Abstract: {result.metadata.get('Summary', 'No abstract')}\n"
                    
                    # Include more content for analysis - up to 3000 chars per paper
                    content_preview = result.page_content
                    if len(content_preview) > 9000:
                        # Take beginning, middle and end to get a better overview
                        content_len = len(content_preview)
                        begin = content_preview[:3000]
                        middle = content_preview[content_len//2-1500:content_len//2+1500]
                        end = content_preview[-3000:]
                        sources_text += f"Content Beginning: {begin}...\n\n"
                        sources_text += f"Content Middle: {middle}...\n\n"
                        sources_text += f"Content End: {end}...\n\n"
                    else:
                        sources_text += f"Content: {content_preview}\n\n"
                else:
                    sources_text += f"Title: {result.metadata.get('title', 'Untitled')}\n"
                    sources_text += f"URL: {result.metadata.get('link', 'No URL')}\n"
                    sources_text += f"Snippet: {result.metadata.get('snippet', 'No snippet')}\n"
                    
                    # Include more content for analysis
                    content_preview = result.page_content
                    if len(content_preview) > 5000:
                        sources_text += f"Content Preview: {content_preview[:5000]}...\n\n"
                    else:
                        sources_text += f"Content: {content_preview}\n\n"
        
        if not sources_text:
            sources_text = "No research sources found for the given queries."
        
        return sources_text
    
    def _analyze_research_sources(
        self, 
        task: ResearchTask, 
        iteration_number: int, 
        sources_text: str,
        task_knowledge: str
    ) -> ResearchIteration:
        """
        Analyze research sources to extract findings and insights.
        
        Args:
            task: Research task
            iteration_number: Current iteration number
            sources_text: Formatted sources text
            task_knowledge: Current task-specific knowledge
            
        Returns:
            Research iteration results
        """
        # Enhanced system prompt for comprehensive analysis
        system_prompt = """You are an expert research analyst conducting deep, comprehensive analysis on a specific research task. 
        Your objective is to produce an extremely detailed and thorough analysis that:
        
        1. Extracts ALL relevant information from the provided sources as it relates to the research task
        2. Analyzes the sources in significant depth, going beyond surface-level observations
        3. Identifies key findings with extensive supporting evidence and context
        4. Synthesizes connections between sources that others might miss
        5. Examines methodological approaches in detail when present
        6. Identifies limitations, contradictions, and gaps in the current research
        7. Generates specific, targeted questions for further investigation
        
        Your analysis should be exceptionally thorough. For each key finding:
        - Provide extensive detail and context
        - Include relevant quotes or data points from the sources
        - Analyze implications in depth
        - Connect to broader themes in the research task
        
        IMPORTANT: When reporting findings, use "Paper" as the source_type for ArXiv papers and "Web" for web search results.
        The source_type field must be exactly "Paper" or "Web" - not "ArXiv" or any other value.
        
        CRITICAL: Your response MUST include ALL of the following fields:
        - iteration_number: The current iteration number (provided to you)
        - task_id: The ID of the current task (provided to you)
        - task_description: Description of the current task (provided to you)
        - findings: A comprehensive list of research findings (each with source_type, source_id, title, summary, and relevance_score)
        - insights: Extensively synthesized insights from analyzing the sources
        - next_questions: Specific questions for further research
        
        For the insights section, aim for at least 400-600 words of deep analysis that synthesizes all the findings.
        For each finding, provide a detailed summary of at least 150-200 words that captures the key information.
        
        Missing ANY of these fields will cause a validation error."""
        
        # Enhanced user prompt
        user_prompt = f"""
        Research Task: {task.description}
        Iteration: {iteration_number}
        Task ID: {task.task_id}
        
        Previous Knowledge from Task Iterations: 
        {task_knowledge}
        
        Research Sources:
        {sources_text}
        
        Please conduct an extremely thorough and comprehensive analysis of these research sources for this specific task.
        Your analysis should be much more detailed and in-depth than a standard summary.
        
        For each key finding:
        1. Assess its relevance to the task on a scale of 1-10
        2. Provide a detailed summary (150-200 words minimum) that captures key information, methodologies, and implications
        3. Include specific quotes or data points from the source when relevant
        
        In your insights section:
        1. Synthesize all findings into a comprehensive analysis (minimum 400-600 words)
        2. Identify patterns, contradictions, and connections across sources
        3. Analyze how these findings build upon or challenge previous knowledge
        4. Discuss implications for the broader research topic
        5. Identify methodological strengths and limitations when present
        
        For your next questions, generate at least 5 specific, targeted questions that would advance the research.
        
        IMPORTANT: Your response MUST follow this exact format with all required fields:
        
        ```json
        {{
          "iteration_number": {iteration_number},
          "task_id": {task.task_id},
          "task_description": "{task.description}",
          "findings": [
            {{
              "source_type": "Paper",
              "source_id": "PAPER_ID_OR_URL",
              "title": "FINDING_TITLE",
              "summary": "DETAILED_FINDING_SUMMARY_OF_AT_LEAST_150_WORDS",
              "relevance_score": 8
            }}
          ],
          "insights": "COMPREHENSIVE_SYNTHESIZED_INSIGHTS_OF_AT_LEAST_400_WORDS",
          "next_questions": [
            "SPECIFIC_QUESTION_1",
            "SPECIFIC_QUESTION_2",
            "SPECIFIC_QUESTION_3",
            "SPECIFIC_QUESTION_4",
            "SPECIFIC_QUESTION_5"
          ]
        }}
        ```
        
        Note: Use exactly "Paper" for ArXiv papers, "Web" for web results. Relevance score should be between 1-10.
        Make sure your JSON response includes ALL required fields and is valid JSON format.
        """
        
        # Get specialized LLM for research analysis
        llm = get_ollama_llm(self.model_name, preset="research_analysis")
        
        try:
            # Try first with direct approach to get JSON response
            response = llm.invoke(f"{system_prompt}\n\n{user_prompt}")
            
            # Try to parse JSON from the response
            try:
                # Extract JSON from response text
                json_str = None
                if '```json' in response:
                    json_str = response.split('```json')[1].split('```')[0].strip()
                elif '```' in response:
                    json_str = response.split('```')[1].split('```')[0].strip()
                else:
                    # Find the first { and last }
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = response[start_idx:end_idx]
                    else:
                        raise ValueError("Could not extract JSON from response")
                
                # Parse the JSON
                result_dict = json.loads(json_str)
                
                # Parse findings
                findings = []
                for finding_dict in result_dict.get("findings", []):
                    try:
                        # Ensure source_type is valid
                        if finding_dict.get("source_type") not in ["Paper", "Web"]:
                            finding_dict["source_type"] = "Paper"  # Default if invalid
                        
                        findings.append(ResearchFindings(**finding_dict))
                    except Exception as finding_error:
                        logger.warning(f"Error parsing finding: {finding_error}. Skipping.")
                        continue
                
                # Add default finding if none were parsed
                if not findings and result_dict.get("findings"):
                    findings.append(ResearchFindings(
                        source_type="Paper",
                        source_id="unknown",
                        title="Parsed Finding",
                        summary="This finding was extracted from the analysis but couldn't be fully parsed.",
                        relevance_score=5
                    ))
                
                # Create the ResearchIteration
                result = ResearchIteration(
                    iteration_number=result_dict.get("iteration_number", iteration_number),
                    task_id=result_dict.get("task_id", task.task_id),
                    task_description=result_dict.get("task_description", task.description),
                    findings=findings,
                    insights=result_dict.get("insights", "No insights provided"),
                    next_questions=result_dict.get("next_questions", [])
                )
                
                # Set task_id from the task if not set
                if not result.task_id:
                    result.task_id = task.task_id
                
                return result
                
            except Exception as json_parse_error:
                logger.warning(f"JSON parsing failed: {json_parse_error}. Trying Pydantic parsing.")
                
                # Try with Pydantic parsing approach
                result = parse_pydantic_from_llm(
                    response,
                    ResearchIteration,
                    lambda: self._generate_fallback_iteration(task, iteration_number)
                )
                
                # Set task_id from the task
                result.task_id = task.task_id
                
                return result
                
        except Exception as e:
            logger.error(f"Error analyzing research sources: {e}")
            return self._generate_fallback_iteration(task, iteration_number)
    
    def _generate_fallback_iteration(
        self, 
        task: ResearchTask, 
        iteration_number: int
    ) -> ResearchIteration:
        """
        Generate a fallback iteration result when analysis fails.
        
        Args:
            task: Research task
            iteration_number: Current iteration number
            
        Returns:
            Minimal research iteration
        """
        minimal_finding = ResearchFindings(
            source_type="Paper", 
            source_id="error",
            title="Limited results",
            summary="Could not fully analyze sources due to technical issues. The analysis process encountered errors when trying to extract and synthesize information from the research sources. This may be due to parsing errors, connection issues, or limitations in the available content.",
            relevance_score=5
        )
        
        return ResearchIteration(
            iteration_number=iteration_number,
            task_id=task.task_id,
            task_description=task.description,
            findings=[minimal_finding],
            insights="Analysis of available research sources indicates limited information was able to be extracted. This may be due to technical issues in the analysis process rather than a lack of relevant information. Consider reviewing the sources manually or trying different search queries to explore this topic further.",
            next_questions=[
                "What alternative approaches can be used to research this topic?", 
                "Are there related fields or concepts that might provide useful insights?",
                "Would more specialized or technical search terms yield better results?",
                "Are there specific journals or publications that focus on this area?",
                "What experts or research groups are leaders in this domain?"
            ]
        )