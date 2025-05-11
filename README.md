<<<<<<< Updated upstream
# Deep Research LLM Agent

A powerful, iterative research assistant that conducts comprehensive, multi-source research on any topic using LangChain, LangGraph, and local LLMs via Ollama.

## Overview

Deep Research is a Streamlit application that leverages Large Language Models to automate the research process. It follows a structured, iterative approach:

1. **Research Planning**: Generates a comprehensive research plan with specific tasks
2. **Iterative Research**: Executes each task by searching academic papers (ArXiv) and web sources
3. **Knowledge Building**: Synthesizes findings across iterations to build deeper understanding
4. **Comprehensive Reporting**: Consolidates all findings into a final report with proper citations

## Features

- **Intelligent Planning**: Breaks research topics into focused, non-overlapping tasks
- **Multi-source Research**: Searches both ArXiv papers and web sources
- **Iterative Approach**: Builds knowledge progressively across iterations
- **Configurable Detail**: Adjust research depth for each task 
- **Knowledge Synthesis**: Combines findings into coherent knowledge representations
- **Structured Reporting**: Generates comprehensive reports with inline citations
- **Local Execution**: Runs entirely using local LLMs via Ollama
- **Follow-up Questions**: Ask questions about the research after completion

## Architecture

Deep Research uses a modular architecture with specialized components:

- **ResearchAgent**: Main coordinator that orchestrates the research process
- **PlanGenerator**: Creates detailed research plans with tasks and queries
- **ResearchExecutor**: Searches sources and analyzes content for each task
- **KnowledgeSynthesizer**: Combines findings into coherent knowledge and reports
- **SourceProcessor**: Handles different source types (ArXiv, web) consistently

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- Compatible LLM models pulled into Ollama (phi4, qwq, etc.)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deep-research.git
cd deep-research
```

2. Install dependencies:
```bash
pip install -e .
```

3. Ensure Ollama is installed and running:
```bash
# Install from https://ollama.ai/
# Start the service
ollama serve
```

4. Pull the required LLM models:
```bash
# Pull a compatible model
ollama pull phi4
# or
ollama pull qwq:32b
```

## Usage

1. Start the application:
```bash
streamlit run deep_research/app.py
```

2. Enter your research topic in the text field
3. Review the generated research plan (or suggest modifications)
4. Configure iterations for each research task
5. Wait for the research to complete
6. Review the comprehensive research report
7. Ask follow-up questions based on the research

## Configuration Options

- **LLM Model**: Select from available Ollama models
- **Papers per Query**: Control how many academic papers to retrieve per query
- **Web Results per Query**: Control how many web results to analyze per query 
- **Research Sources**: Select which sources to use (ArXiv, Web Search)
- **Detail Level**: For each task, choose between Basic, Standard, or Comprehensive
- **Iterations**: Configure how many iterations to run for each task

## How It Works

### Research Planning

The agent uses the LLM to generate a structured research plan for any topic, breaking it down into:
- An overall research objective
- 3-5 specialized research tasks
- Specific search queries for each task

### Iterative Research

For each task, the agent:
1. Searches ArXiv for academic papers and the web for other sources
2. Analyzes the retrieved content for relevant information
3. Extracts key findings and synthesizes insights
4. Identifies questions for further investigation
5. Uses these insights to refine queries for the next iteration

### Knowledge Synthesis

The agent builds knowledge across iterations by:
1. Updating task-specific knowledge with new findings
2. Resolving contradictions or confirming patterns
3. Organizing information into a coherent structure
4. Preserving specific details, quotes, and data points

### Report Generation

The final report includes:
1. Executive summary
2. Comprehensive analysis of the topic
3. Key findings from all research tasks
4. Cross-cutting themes and insights
5. Detailed task summaries
6. Complete source citations

## Troubleshooting

### "Using fallback plan for topic"

This message appears when the agent can't generate a detailed plan and uses a simpler one. Possible causes:

1. **Ollama Connection Issues**:
   - Ensure Ollama is running (`ollama serve`)
   - Check that the models are installed (`ollama list`)
   - Verify the API is accessible (default: http://localhost:11434)

2. **LLM Response Issues**:
   - The LLM may be returning responses that can't be parsed correctly
   - Try using a different model
   - Check your prompt format and ensure it matches what the model expects

3. **Topic Complexity**:
   - Very complex or specialized topics might be challenging for the LLM
   - Try reformulating your topic to be more specific

### Other Common Issues

- **Missing Search Results**: The agent might encounter rate limiting or connectivity issues when searching sources. Try again later.
- **Long Processing Times**: Research on complex topics with many iterations can take time, especially with larger models.

## Project Structure

- **deep_research/agent/**: Core agent components (research_agent.py, plan_generator.py, etc.)
- **deep_research/sources/**: Source integration (arxiv.py, web_search.py, source_processor.py)
- **deep_research/utils/**: Utility functions (error_handling.py, formatting.py, etc.)
- **deep_research/ui/**: Streamlit UI components (task_config.py, report_display.py)
- **deep_research/config/**: Configuration settings (app_config.py, llm_config.py)

## Limitations

- Research quality depends on the capabilities of the LLM model used
- Source retrieval is limited to ArXiv papers and web search
- Processing time increases with more iterations and higher detail levels
- Limited to text-based sources (no image or video analysis)

## Future Improvements

- Integration with additional research sources
- Support for custom document uploads
- Enhanced visualization of research findings
- Collaborative research with multiple agents
- Support for RAG (Retrieval Augmented Generation)
- Customizable report templates
- Export options for different formats (PDF, DOCX, etc.)
- Citation management with multiple formatting styles
=======
# Deep Research LLM Agent

A powerful, iterative research assistant that conducts comprehensive, multi-source research on any topic using LangChain, LangGraph, and local LLMs via Ollama.

## Overview

Deep Research is a Streamlit application that leverages Large Language Models to automate the research process. It follows a structured, iterative approach:

1. **Research Planning**: Generates a comprehensive research plan with specific tasks
2. **Iterative Research**: Executes each task by searching academic papers (ArXiv) and web sources
3. **Knowledge Building**: Synthesizes findings across iterations to build deeper understanding
4. **Comprehensive Reporting**: Consolidates all findings into a final report with proper citations

## Features

- **Intelligent Planning**: Breaks research topics into focused, non-overlapping tasks
- **Multi-source Research**: Searches both ArXiv papers and web sources
- **Iterative Approach**: Builds knowledge progressively across iterations
- **Configurable Detail**: Adjust research depth for each task 
- **Knowledge Synthesis**: Combines findings into coherent knowledge representations
- **Structured Reporting**: Generates comprehensive reports with inline citations
- **Local Execution**: Runs entirely using local LLMs via Ollama

## Architecture

Deep Research uses a modular architecture with specialized components:

- **ResearchAgent**: Main coordinator that orchestrates the research process
- **PlanGenerator**: Creates detailed research plans with tasks and queries
- **ResearchExecutor**: Searches sources and analyzes content for each task
- **KnowledgeSynthesizer**: Combines findings into coherent knowledge and reports
- **SourceProcessor**: Handles different source types (ArXiv, web) consistently

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- Phi-4 or Gemma-3 models pulled into Ollama

## Installation

1. Clone the repository:
```bash
git clone https://github.com/user/deep-research.git
cd deep-research
```

2. Install dependencies:
```bash
pip install -e .
```

3. Ensure Ollama is installed and running:
```bash
# Install from https://ollama.ai/
# Start the service
ollama serve
```

4. Pull the required LLM models:
```bash
# Pull either of these models
ollama pull phi4
# or
ollama pull gemma3:12b
```

## Usage

1. Start the application:
```bash
streamlit run deep_research/app.py
```

2. Enter your research topic in the text field
3. Review the generated research plan (or suggest modifications)
4. Configure iterations for each research task
5. Wait for the research to complete
6. Review the comprehensive research report
7. Ask follow-up questions based on the research

## Configuration Options

- **LLM Model**: Select between phi4 and gemma3 models
- **Papers per Query**: Control how many academic papers to retrieve per query
- **Web Results per Query**: Control how many web results to analyze per query 
- **Research Sources**: Select which sources to use (ArXiv, Web Search)
- **Detail Level**: For each task, choose between Basic, Standard, or Comprehensive
- **Iterations**: Configure how many iterations to run for each task

## Troubleshooting

### "Using fallback plan for topic"

This message appears when the agent can't generate a detailed plan and uses a simpler one. Possible causes:

1. **Ollama Connection Issues**:
   - Ensure Ollama is running (`ollama serve`)
   - Check that the models are installed (`ollama list`)
   - Verify the API is accessible (default: http://localhost:11434)

2. **LLM Response Issues**:
   - The LLM may be returning responses that can't be parsed correctly
   - Try using a different model (phi4 vs gemma3)
   - Check your prompt format and ensure it matches what the model expects

3. **Topic Complexity**:
   - Very complex or specialized topics might be challenging for the LLM
   - Try reformulating your topic to be more specific

### Other Common Issues

- **Missing Search Results**: The agent might encounter rate limiting or connectivity issues when searching sources. Try again later.
- **Long Processing Times**: Research on complex topics with many iterations can take time, especially with larger models.

## Project Structure

- **deep_research/agent/**: Core agent components (research_agent.py, plan_generator.py, etc.)
- **deep_research/sources/**: Source integration (arxiv.py, web_search.py, source_processor.py)
- **deep_research/utils/**: Utility functions (error_handling.py, formatting.py, etc.)
- **deep_research/ui/**: Streamlit UI components (task_config.py, report_display.py)
- **deep_research/config/**: Configuration settings (app_config.py, llm_config.py)

## How It Works

### Research Planning

The agent uses the LLM to generate a structured research plan for any topic, breaking it down into:
- An overall research objective
- 3-5 specialized research tasks
- Specific search queries for each task

### Iterative Research

For each task, the agent:
1. Searches ArXiv for academic papers and the web for other sources
2. Analyzes the retrieved content for relevant information
3. Extracts key findings and synthesizes insights
4. Identifies questions for further investigation
5. Uses these insights to refine queries for the next iteration

### Knowledge Synthesis

The agent builds knowledge across iterations by:
1. Updating task-specific knowledge with new findings
2. Resolving contradictions or confirming patterns
3. Organizing information into a coherent structure
4. Preserving specific details, quotes, and data points

### Report Generation

The final report includes:
1. Executive summary
2. Comprehensive analysis of the topic
3. Key findings from all research tasks
4. Cross-cutting themes and insights
5. Detailed task summaries
6. Complete source citations

## Limitations

- Research quality depends on the capabilities of the LLM model used
- Source retrieval is limited to ArXiv papers and web search
- Processing time increases with more iterations and higher detail levels
- Limited to text-based sources (no image or video analysis)

## Future Improvements

- Integration with additional research sources
- Support for custom document uploads
- Enhanced visualization of research findings
- Collaborative research with multiple agents
- Support for RAG (Retrieval Augmented Generation)
>>>>>>> Stashed changes
