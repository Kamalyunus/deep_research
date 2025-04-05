"""
Pydantic data models for the Deep Research Agent.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union

class ResearchTask(BaseModel):
    """Model representing a research task in the plan."""
    task_id: int = Field(description="Unique identifier for the task")
    description: str = Field(description="Detailed description of the research task")
    queries: List[str] = Field(description="List of specific queries to search on ArXiv and web")

class ResearchPlan(BaseModel):
    """Model representing the overall research plan."""
    topic: str = Field(description="The main research topic")
    objective: str = Field(description="The overall objective of the research")
    tasks: List[ResearchTask] = Field(description="List of research tasks to accomplish")

class ResearchFindings(BaseModel):
    """Model representing findings from a single source."""
    source_type: str = Field(description="Type of source (Paper or Web)")
    source_id: str = Field(description="ID or URL of the source")
    title: str = Field(description="Title of the finding")
    summary: str = Field(description="Summary of key findings relevant to the research task")
    relevance_score: int = Field(description="Relevance score from 1-10")
    implications: Optional[str] = Field(default=None, description="Implications or significance of this finding")
    methodology: Optional[str] = Field(default=None, description="Research methodology used (if applicable)")

    @validator('source_type')
    def validate_source_type(cls, v):
        """Validate that source type is either 'Paper' or 'Web'"""
        if v not in ["Paper", "Web", "ArXiv"]:
            raise ValueError('source_type must be either "Paper", "Web", or "ArXiv"')
        return v

    @validator('relevance_score')
    def validate_relevance_score(cls, v):
        """Validate and normalize relevance score to 1-10 range"""
        # Handle float values by converting to int
        if isinstance(v, float):
            v = int(round(v))  # Round to nearest integer
        
        # Validate the range
        if not 1 <= v <= 10:
            # If outside range, clamp to valid range
            v = max(1, min(10, v))
            
        return v

class ResearchIteration(BaseModel):
    """Model representing a single research iteration."""
    iteration_number: int = Field(description="Current iteration number")
    task_id: int = Field(description="ID of the task being researched")
    task_description: str = Field(description="Description of the task")
    findings: List[ResearchFindings] = Field(default_factory=list, description="List of findings from papers and web sources")
    insights: str = Field(description="Synthesized insights from this iteration")
    next_questions: List[str] = Field(default_factory=list, description="Questions to explore in next iteration")
    knowledge_gaps: Optional[List[str]] = Field(default_factory=list, description="Identified gaps in current knowledge")
    contradictions: Optional[List[str]] = Field(default_factory=list, description="Contradictory findings or debates in the literature")

class AccumulatedKnowledge(BaseModel):
    """Model representing synthesized knowledge from multiple iterations."""
    summary: str = Field(description="Concise summary of key findings and insights")
    key_points: List[str] = Field(description="List of key points extracted from research")
    patterns: Optional[List[str]] = Field(default_factory=list, description="Recurring patterns identified across sources")
    controversies: Optional[List[str]] = Field(default_factory=list, description="Areas of debate or contradiction")
    
class FinalReport(BaseModel):
    """Model representing the final research report."""
    topic: str = Field(description="Research topic")
    objective: str = Field(description="Research objective")
    summary: str = Field(description="Executive summary")
    key_findings: List[str] = Field(description="Key findings of the research")
    analysis: str = Field(description="In-depth analysis of findings")
    implications: str = Field(description="Practical and theoretical implications")
    limitations: List[str] = Field(description="Limitations of the research")
    conclusions: str = Field(description="Conclusions drawn from the research")
    future_directions: List[str] = Field(description="Suggestions for future research")

class TaskExecutionConfig(BaseModel):
    """Model for task execution configuration."""
    iterations: int = Field(default=2, ge=1, le=5, description="Number of iterations to perform")
    detail_level: str = Field(default="Standard", description="Level of detail for research")

    @validator('detail_level')
    def validate_detail_level(cls, v):
        """Validate that detail level is valid"""
        valid_levels = ["Basic", "Standard", "Comprehensive"]
        if v not in valid_levels:
            raise ValueError(f"detail_level must be one of {valid_levels}")
        return v

class SourceReference(BaseModel):
    """Model for source reference in reports."""
    id: str = Field(description="Unique identifier for the source")
    title: str = Field(description="Title of the source")
    url: str = Field(description="URL of the source")
    authors: Optional[str] = Field(default=None, description="Authors of the source (for papers)")
    published: Optional[str] = Field(default=None, description="Publication date (if available)")
    source_type: str = Field(description="Type of source (Paper or Web)")
    
    @validator('source_type')
    def validate_source_type(cls, v):
        """Validate that source type is valid"""
        if v not in ["Paper", "Web"]:
            return "Paper" if v == "ArXiv" else "Web"
        return v