"""
Knowledge synthesis and reporting functionality for the Deep Research Agent.
"""
from langchain_core.prompts import ChatPromptTemplate
import logging
import time
import json
import re
from typing import Dict, Any, List, Optional, Tuple

from .models import AccumulatedKnowledge, ResearchIteration, FinalReport, SourceReference
from ..config.llm_config import get_ollama_llm
from ..utils.text_processing import (
    remove_thinking_tags,
    escape_math_expressions,
    escape_template_variables,
    safe_process_text
)
from ..utils.error_handling import (
    parse_pydantic_from_llm,
    PlanError,
    with_retry,
)

logger = logging.getLogger(__name__)

class KnowledgeSynthesizer:
    """
    Responsible for synthesizing knowledge and generating reports.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the knowledge synthesizer.
        
        Args:
            model_name: Optional model name to use for knowledge synthesis
        """
        self.model_name = model_name
    
    def update_task_knowledge(
        self, 
        task_id: int, 
        iteration_result: ResearchIteration,
        current_knowledge: Optional[str] = ""
    ) -> str:
        """
        Update task-specific knowledge with new iteration results.
        
        Args:
            task_id: ID of the task
            iteration_result: Results from the current iteration
            current_knowledge: Existing knowledge for this task (if any)
            
        Returns:
            Updated task knowledge
        """
        # Format new knowledge from the current iteration
        new_knowledge = self._format_iteration_knowledge(iteration_result)
        
        # If this is the first iteration, just use the new knowledge
        if not current_knowledge:
            return new_knowledge
        
        # Otherwise, synthesize the knowledge
        return self._synthesize_task_knowledge(current_knowledge, new_knowledge)
    
    def _format_iteration_knowledge(self, iteration_result: ResearchIteration) -> str:
        """
        Format iteration result into structured knowledge text.
        
        Args:
            iteration_result: Results from a research iteration
            
        Returns:
            Formatted knowledge text
        """
        # Format insights and findings
        new_knowledge = f"## Iteration {iteration_result.iteration_number} Insights\n\n"
        new_knowledge += safe_process_text(iteration_result.insights) + "\n\n"
        new_knowledge += "## Key Findings\n\n"
        
        for i, finding in enumerate(iteration_result.findings):
            new_knowledge += f"### Finding {i+1}: {finding.title} (Relevance: {finding.relevance_score}/10)\n\n"
            new_knowledge += f"{finding.summary}\n\n"
            # Add source information for better tracking
            new_knowledge += f"*Source: {finding.source_type} - {finding.source_id}*\n\n"
        
        # Add questions for future research
        new_knowledge += "## Questions for Future Research\n\n"
        for i, question in enumerate(iteration_result.next_questions):
            new_knowledge += f"{i+1}. {question}\n"
        
        return new_knowledge
    
    def _synthesize_task_knowledge(self, current_knowledge: str, new_knowledge: str) -> str:
        """
        Synthesize existing knowledge with new findings.
        
        Args:
            current_knowledge: Existing knowledge text
            new_knowledge: New knowledge text
            
        Returns:
            Synthesized knowledge
        """
        system_prompt = """You are a research knowledge synthesizer specializing in creating comprehensive research summaries. 
        Combine the existing knowledge with new findings to create a coherent, detailed understanding of the research task. 
        
        Focus on:
        1. Integrating new information with existing knowledge
        2. Resolving contradictions or confirming patterns
        3. Highlighting the most important insights across all iterations
        4. Creating a logical structure that organizes information by subtopics
        5. Maintaining a comprehensive record of all important findings
        6. Preserving specific details, quotes, data points, and methodological information
        7. Identifying the most promising questions for future research
        
        Your synthesis should be extremely thorough (800-1000+ words) and maintain the rich detail from all iterations.
        
        IMPORTANT: When writing mathematical expressions like n^2, log^2 n, etc., 
        write them plainly without using special formatting or template variables."""
        
        # Escape potential template variables in both current and new knowledge
        current_knowledge_escaped = escape_template_variables(current_knowledge)
        new_knowledge_escaped = escape_template_variables(new_knowledge)
        
        user_prompt = f"""
        # Existing Knowledge:
        
        {current_knowledge_escaped}
        
        # New Knowledge from Current Iteration:
        
        {new_knowledge_escaped}
        
        Please synthesize these into a comprehensive, detailed understanding of the research task.
        Create a logical structure organized by subtopics or themes rather than iterations.
        Maintain the rich detail, preserving specific findings, quotes, data points, and methodological information.
        The synthesis should be very thorough (800-1000+ words) and cover all important aspects of the research.
        """
        
        try:
            # Use specialized LLM for knowledge synthesis
            llm = get_ollama_llm(self.model_name, preset="knowledge_synthesis")
            
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", user_prompt)
            ])
            
            synthesis_chain = synthesis_prompt | llm
            synthesized_knowledge = synthesis_chain.invoke({})
            
            # Clean any thinking tags from the result
            synthesized_knowledge = remove_thinking_tags(synthesized_knowledge)
            
            return synthesized_knowledge
        except Exception as e:
            logger.error(f"Error synthesizing task knowledge: {e}")
            # If synthesis fails, combine knowledge in a simpler way
            return f"# Previous Knowledge\n\n{current_knowledge}\n\n# New Knowledge\n\n{new_knowledge}"
    
    def synthesize_global_knowledge(self, task_knowledge: Dict[int, str]) -> str:
        """
        Synthesize global knowledge from all task-specific knowledge.
        
        Args:
            task_knowledge: Dictionary mapping task IDs to task-specific knowledge
            
        Returns:
            Global synthesized knowledge
        """
        # Collect all task-specific knowledge
        all_task_knowledge = []
        for task_id, knowledge in task_knowledge.items():
            if knowledge:
                all_task_knowledge.append(f"# Task {task_id} Knowledge\n\n{knowledge}")
        
        if not all_task_knowledge:
            return ""
            
        task_knowledge_text = "\n\n".join(all_task_knowledge)
        
        system_prompt = """You are a research knowledge synthesizer creating a comprehensive synthesis of research findings.
        Analyze and integrate knowledge from all research tasks to create a cohesive understanding of the overall topic.
        
        Your synthesis should:
        1. Identify the most important insights across all tasks
        2. Connect related findings from different tasks
        3. Highlight patterns, trends, and contradictions
        4. Organize information in a logical structure by themes rather than tasks
        5. Maintain rich detail and specificity while creating a coherent narrative
        6. Preserve specific methodologies, data points, and evidence
        
        This synthesis must be extremely detailed (1000-1500+ words) and will form the basis for the final research report."""
        
        user_prompt = f"# Knowledge from all research tasks:\n\n{task_knowledge_text}"
        
        try:
            # Use specialized LLM for comprehensive synthesis
            llm = get_ollama_llm(self.model_name, preset="knowledge_synthesis")
            
            try:
                # First try direct synthesis without structured output
                synthesis_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("user", user_prompt)
                ])
                
                synthesis_chain = synthesis_prompt | llm
                result = synthesis_chain.invoke({})
                
                # Clean any thinking tags
                result = remove_thinking_tags(result)
                
                return result
                
            except Exception as direct_error:
                logger.warning(f"Error in direct synthesis approach: {direct_error}. Trying structured approach.")
                
                # Fall back to structured approach
                try:
                    # Get LLM response
                    response = llm.invoke(f"{system_prompt}\n\n{user_prompt}")
                    
                    # Parse into AccumulatedKnowledge model
                    knowledge = parse_pydantic_from_llm(
                        response,
                        AccumulatedKnowledge,
                        lambda: self._generate_fallback_knowledge()
                    )
                    
                    # Format the knowledge
                    formatted_knowledge = f"# Research Summary\n\n{knowledge.summary}\n\n## Key Points\n\n"
                    for point in knowledge.key_points:
                        formatted_knowledge += f"- {point}\n"
                    
                    if knowledge.patterns:
                        formatted_knowledge += "\n## Patterns and Trends\n\n"
                        for pattern in knowledge.patterns:
                            formatted_knowledge += f"- {pattern}\n"
                            
                    if knowledge.controversies:
                        formatted_knowledge += "\n## Controversies and Debates\n\n"
                        for controversy in knowledge.controversies:
                            formatted_knowledge += f"- {controversy}\n"
                    
                    return formatted_knowledge
                except Exception as structured_error:
                    logger.error(f"Error in structured synthesis approach: {structured_error}")
                    raise
                
        except Exception as e:
            logger.error(f"Error synthesizing global knowledge: {e}")
            # Fall back to combining task knowledge with minimal processing
            combined_knowledge = "# Combined Research Knowledge\n\n"
            for task_id, knowledge in task_knowledge.items():
                if knowledge:
                    combined_knowledge += f"## Task {task_id}\n\n{knowledge}\n\n"
            
            return combined_knowledge
    
    def _generate_fallback_knowledge(self) -> AccumulatedKnowledge:
        """
        Generate fallback accumulated knowledge when parsing fails.
        
        Returns:
            Minimal AccumulatedKnowledge instance
        """
        return AccumulatedKnowledge(
            summary="Research was conducted but could not be fully synthesized due to technical issues.",
            key_points=["Multiple sources were analyzed", "See task-specific knowledge for details"],
            patterns=[],
            controversies=[]
        )
    
    def generate_final_report(
        self, 
        topic: str, 
        objective: str, 
        task_knowledge: Dict[int, str],
        papers: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a comprehensive final research report.
        
        Args:
            topic: Research topic
            objective: Research objective
            task_knowledge: Dictionary mapping task IDs to task-specific knowledge
            papers: List of papers used in research
            web_results: List of web results used in research
            
        Returns:
            Final research report
        """
        try:
            # Create citation mapping for sources
            citation_map = self._create_citation_map(papers, web_results)
            
            # Clean all task knowledge
            cleaned_task_knowledge = {}
            for task_id, knowledge in task_knowledge.items():
                if knowledge:
                    cleaned_task_knowledge[task_id] = remove_thinking_tags(knowledge)
            
            # If no task knowledge is available, return a basic report
            if not cleaned_task_knowledge:
                return self._generate_basic_report(topic, objective, papers, web_results)
            
            # Generate comprehensive report with citations
            comprehensive_report = self._generate_comprehensive_report(
                topic, objective, cleaned_task_knowledge, citation_map, papers, web_results
            )
            
            # Add detailed task summaries
            comprehensive_report += "\n\n# Detailed Task Summaries\n\n"
            for task_id, knowledge in cleaned_task_knowledge.items():
                comprehensive_report += f"## Task {task_id} Summary\n\n"
                comprehensive_report += knowledge + "\n\n"
            
            # Add sources section
            comprehensive_report += self._generate_sources_section(papers, web_results)
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"Error in final report generation: {e}")
            # Fallback to basic report
            return self._generate_basic_report(topic, objective, papers, web_results)
    
    def _create_citation_map(
        self, 
        papers: List[Dict[str, Any]], 
        web_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create a mapping of source IDs to citation information.
        
        Args:
            papers: List of papers
            web_results: List of web results
            
        Returns:
            Dictionary mapping source IDs to citation info
        """
        citation_map = {}
        
        try:
            # Process papers
            for i, paper in enumerate(papers or []):
                try:
                    paper_id = str(paper.get('id', f'unknown-{i}'))
                    citation_map[paper_id] = {
                        'index': i + 1,
                        'type': 'paper',
                        'title': str(paper.get('title', 'Untitled')),
                        'authors': str(paper.get('authors', 'Unknown')),
                        'url': str(paper.get('url', '#'))
                    }
                except Exception as paper_error:
                    logger.warning(f"Error processing paper for citation: {paper_error}")
                    continue
            
            # Process web results
            for i, result in enumerate(web_results or []):
                try:
                    url = str(result.get('url', f'unknown-url-{i}'))
                    citation_map[url] = {
                        'index': i + 1 + len(papers or []),
                        'type': 'web',
                        'title': str(result.get('title', 'Untitled')),
                        'url': url
                    }
                except Exception as web_error:
                    logger.warning(f"Error processing web result for citation: {web_error}")
                    continue
        
        except Exception as e:
            logger.error(f"Error creating citation map: {e}")
        
        return citation_map
    
    def _generate_comprehensive_report(
        self, 
        topic: str, 
        objective: str, 
        task_knowledge: Dict[int, str],
        citation_map: Dict[str, Dict[str, Any]],
        papers: List[Dict[str, Any]], 
        web_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a coherent comprehensive report from task summaries with inline citations.
        
        Args:
            topic: Research topic
            objective: Research objective
            task_knowledge: Dictionary of cleaned task knowledge
            citation_map: Mapping of source IDs to citation information
            papers: List of papers
            web_results: List of web results
            
        Returns:
            Comprehensive research report with inline citations
        """
        # Prepare the prompt for the LLM
        system_prompt = """You are creating a comprehensive, detailed research report based on summaries from multiple research tasks.
        
        Your goal is to synthesize the task summaries into a coherent, flowing narrative (4000-5000 words) with:
        1. A detailed executive summary (350-450 words)
        2. A thorough introduction explaining the research topic, objectives, and methodology
        3. Core analysis sections that integrate findings across all tasks
        4. Cross-cutting themes, patterns, and connections
        5. Implications and applications
        6. Limitations and future directions
        
        IMPORTANT REQUIREMENTS:
        1. Add inline citations in the format [X] when referencing specific information
        2. For each key point, add a reference like [See Task X for more details] to link to the detailed task summaries
        3. Make the report a complete, standalone document that flows naturally
        4. DO NOT include the <think> tags or text between them in your report
        5. Include specific details, examples, and findings from the source materials
        
        The report should read as a cohesive whole rather than a collection of task summaries."""
        
        # Create a consolidated task summary
        task_summaries_text = ""
        for task_id, knowledge in task_knowledge.items():
            task_summaries_text += f"# Task {task_id} Summary\n\n{knowledge}\n\n"
        
        # Create a source listing
        sources_text = self._format_sources_for_prompt(papers, web_results)
        
        # Create the user prompt
        user_prompt = f"""
        # Research Topic: {topic}
        # Research Objective: {objective}
        
        # Task Summaries for Integration:
        
        {task_summaries_text}
        
        # Sources Available for Citation:
        
        {sources_text}
        
        Please create a comprehensive, coherent research report (4000-5000 words) that synthesizes all task summaries.
        
        Requirements for your report:
        1. Include inline citations in [X] format when referencing specific information
        2. Add references like [See Task X for more details] to link to task summaries
        3. Create a flowing narrative that integrates all task findings
        4. Include all significant insights from the task summaries
        5. Organize by themes rather than by tasks
        6. Ensure this reads as a polished, standalone academic document
        
        The report will be followed by detailed task summaries, so you can refer readers to those sections for in-depth information.
        """
        
        try:
            # Get LLM with report generation preset
            llm = get_ollama_llm(self.model_name, preset="report_generation")
            
            # Safely process the prompt
            safe_user_prompt = escape_math_expressions(user_prompt)
            
            report_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", safe_user_prompt)
            ])
            
            # Try to generate the report
            try:
                report_chain = report_prompt | llm
                report = report_chain.invoke({})
                report = remove_thinking_tags(report)
            except ValueError as format_error:
                # If we encounter a formatting error, try a direct approach
                logger.warning(f"String formatting error in report generation: {format_error}")
                direct_prompt = f"{system_prompt}\n\n{safe_user_prompt}"
                report = llm.invoke(direct_prompt)
                report = remove_thinking_tags(report)
            
            # Validate we got something meaningful
            if not report or not isinstance(report, str) or len(report) < 500:
                raise ValueError("Failed to generate a substantial comprehensive report")
            
            return report
            
        except Exception as e:
            logger.error(f"Error in comprehensive report generation: {e}")
            return self._generate_fallback_integrated_report(topic, objective, task_knowledge)
    
    def _format_sources_for_prompt(
        self, 
        papers: List[Dict[str, Any]], 
        web_results: List[Dict[str, Any]]
    ) -> str:
        """
        Format source information for the report generation prompt.
        
        Args:
            papers: List of papers
            web_results: List of web results
            
        Returns:
            Formatted source text
        """
        sources_text = "# Available Sources for Citation\n\n"
        
        # Format papers
        try:
            if papers:
                sources_text += "## ArXiv Papers\n\n"
                for i, paper in enumerate(papers):
                    try:
                        title = str(paper.get('title', 'Untitled'))
                        authors = str(paper.get('authors', 'Unknown'))
                        paper_id = str(paper.get('id', 'Unknown'))
                        url = str(paper.get('url', '#'))
                        published = str(paper.get('published', 'Unknown date'))
                        
                        sources_text += f"{i+1}. Title: {title}\n"
                        sources_text += f"   Authors: {authors}\n"
                        sources_text += f"   ID: {paper_id}\n"
                        sources_text += f"   URL: {url}\n"
                        sources_text += f"   Published: {published}\n\n"
                    except Exception as e:
                        logger.warning(f"Error formatting paper for source text: {e}")
                        sources_text += f"{i+1}. Paper entry (formatting error)\n\n"
            else:
                sources_text += "## ArXiv Papers\n\nNo academic papers were retrieved for this research.\n\n"
        except Exception as e:
            logger.error(f"Error creating papers section: {e}")
            sources_text += "## ArXiv Papers\n\nUnable to format paper sources due to an error.\n\n"
        
        # Format web results
        try:
            if web_results:
                sources_text += "## Web Sources\n\n"
                for i, result in enumerate(web_results):
                    try:
                        title = str(result.get('title', 'Untitled'))
                        url = str(result.get('url', '#'))
                        snippet = str(result.get('snippet', 'No snippet'))
                        
                        sources_text += f"{i+1+len(papers or [])}. Title: {title}\n"
                        sources_text += f"   URL: {url}\n"
                        sources_text += f"   Snippet: {snippet}\n\n"
                    except Exception as e:
                        logger.warning(f"Error formatting web result for source text: {e}")
                        sources_text += f"{i+1+len(papers or [])}. Web entry (formatting error)\n\n"
            else:
                sources_text += "## Web Sources\n\nNo web sources were retrieved for this research.\n\n"
        except Exception as e:
            logger.error(f"Error creating web sources section: {e}")
            sources_text += "## Web Sources\n\nUnable to format web sources due to an error.\n\n"
        
        return sources_text
    
    def _generate_fallback_integrated_report(
        self, 
        topic: str, 
        objective: str, 
        task_knowledge: Dict[int, str]
    ) -> str:
        """
        Generate a simpler integrated report when comprehensive report generation fails.
        
        Args:
            topic: Research topic
            objective: Research objective
            task_knowledge: Dictionary of task knowledge
            
        Returns:
            Basic integrated report
        """
        integrated_report = f"# Comprehensive Research Report: {topic}\n\n"
        integrated_report += f"## Executive Summary\n\n"
        integrated_report += f"This report synthesizes research on {topic} conducted to {objective}. "
        integrated_report += f"Multiple research tasks were performed to investigate different aspects of the topic, "
        integrated_report += f"analyzing information from academic papers and web sources.\n\n"
        
        # Add a section for each task
        integrated_report += "## Integrated Research Findings\n\n"
        for task_id, knowledge in task_knowledge.items():
            # Extract first paragraph as a summary
            paragraphs = knowledge.split('\n\n')
            first_para = paragraphs[0] if paragraphs else "No summary available."
            
            integrated_report += f"### Key Findings from Task {task_id}\n\n"
            integrated_report += first_para + "\n\n"
            integrated_report += f"[See Task {task_id} Summary for more details]\n\n"
            
        integrated_report += "## Conclusions\n\n"
        integrated_report += f"The research conducted on {topic} provides valuable insights into various aspects of the topic. "
        integrated_report += f"For detailed information on specific aspects, please refer to the individual task summaries.\n\n"
        
        return integrated_report
    
    def _generate_sources_section(
        self, 
        papers: List[Dict[str, Any]], 
        web_results: List[Dict[str, Any]]
    ) -> str:
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
                    authors = str(paper.get('authors', 'Unknown'))
                    title = str(paper.get('title', 'Untitled'))
                    paper_id = str(paper.get('id', 'Unknown'))
                    
                    # Fix for ArXiv links
                    url = str(paper.get('url', ''))
                    if not url or 'arxiv.org' not in url:
                        clean_id = paper_id.lower().replace('arxiv:', '').strip()
                        url = f"https://arxiv.org/abs/{clean_id}"
                    
                    sources_section += f"[{i+1}] {authors}. \"{title}\". [ArXiv: {paper_id}]({url})\n\n"
                except Exception as e:
                    logger.warning(f"Error formatting paper citation: {e}")
                    sources_section += f"[{i+1}] Paper citation unavailable due to formatting error\n\n"
        
        # Add web sources
        if web_results:
            sources_section += "## Web Sources\n\n"
            for i, result in enumerate(web_results):
                try:
                    title = str(result.get('title', 'Untitled'))
                    url = str(result.get('url', '#'))
                    sources_section += f"[{i+1+len(papers)}] \"{title}\". [{url}]({url})\n\n"
                except Exception as e:
                    logger.warning(f"Error formatting web citation: {e}")
                    sources_section += f"[{i+1+len(papers)}] Web citation unavailable due to formatting error\n\n"
        
        return sources_section
    
    def _generate_basic_report(
        self, 
        topic: str, 
        objective: str, 
        papers: List[Dict[str, Any]], 
        web_results: List[Dict[str, Any]],
        accumulated_knowledge: str = ""
    ) -> str:
        """
        Generate a basic report when normal generation fails.
        
        Args:
            topic: Research topic
            objective: Research objective
            papers: List of papers used in research
            web_results: List of web results used in research
            accumulated_knowledge: Optional accumulated knowledge text
            
        Returns:
            Basic research report
        """
        basic_report = f"# Research Report: {topic}\n\n"
        basic_report += f"## Overview\n\n"
        basic_report += f"This report summarizes research on {topic} conducted to {objective}.\n\n"
        
        # Add accumulated knowledge if available
        if accumulated_knowledge:
            cleaned_knowledge = remove_thinking_tags(accumulated_knowledge)
            basic_report += "## Research Summary\n\n"
            basic_report += cleaned_knowledge + "\n\n"
        
        # Add sources
        basic_report += self._generate_sources_section(papers, web_results)
        
        return basic_report