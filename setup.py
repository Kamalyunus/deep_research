from setuptools import setup, find_packages

setup(
    name="deep_research",
    version="0.1.0",
    packages=find_packages(),
    description="Deep Research LLM Agent",
    author="Yunus Kamal",
    install_requires=[
        "streamlit>=1.25.0",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langchain-community>=0.0.13",
        "langchain-text-splitters>=0.0.1",
        "langchain-ollama>=0.0.1",
        "pydantic>=2.0.0",
        "langgraph>=0.0.10",
        "requests>=2.31.0",
        "duckduckgo-search>=4.1.1",
        "beautifulsoup4>=4.12.0",
        "tenacity>=8.2.3",
        "arxiv>=1.4.7",
        "pymupdf>=1.25.0"
    ],
)