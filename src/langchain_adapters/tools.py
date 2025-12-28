"""
Tools for the personal assistant.
Provides web search and other utility tools using LangChain integrations.
"""

from typing import List, Optional, Dict, Any
from langchain.tools import tool
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.tools import BaseTool
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class WebSearchTools:
    """Collection of web search tools."""

    def __init__(self, tavily_api_key: Optional[str] = None,
                 exa_api_key: Optional[str] = None,
                 jina_api_key: Optional[str] = None):
        """
        Initialize web search tools with API keys.

        Args:
            tavily_api_key: API key for Tavily search
            exa_api_key: API key for Exa search
            jina_api_key: API key for Jina search
        """
        self.tavily_api_key = tavily_api_key
        self.exa_api_key = exa_api_key
        self.jina_api_key = jina_api_key

    def get_available_tools(self) -> List[BaseTool]:
        """
        Get list of available web search tools based on configured API keys.

        Returns:
            List of tool instances
        """
        tools = []

        if self.tavily_api_key:
            tool = self._create_tavily_search_tool()
            if tool:
                tools.append(tool)
                logger.info("Tavily search tool enabled")
            else:
                logger.warning("Tavily search tool failed to initialize")

        if self.exa_api_key:
            tool = self._create_exa_search_tool()
            if tool:
                tools.append(tool)
                logger.info("Exa search tool enabled")
            else:
                logger.warning("Exa search tool failed to initialize")

        if self.jina_api_key:
            tool = self._create_jina_search_tool()
            if tool:
                tools.append(tool)
                logger.info("Jina search tool enabled")
            else:
                logger.warning("Jina search tool failed to initialize")

        return tools

    def _create_tavily_search_tool(self) -> Optional[BaseTool]:
        """Create Tavily search tool."""
        try:
            @tool("tavily_search")
            def tavily_search(query: str, k: int = 3) -> str:
                """
                Search the web using Tavily for up-to-date information.

                Args:
                    query: Search query
                    k: Number of results to return (max 10)

                Returns:
                    Search results as formatted text with embedded image gallery
                """
                try:
                    # Import requests to make direct API call for full response
                    import requests

                    # Make direct API call to get full response with images
                    headers = {
                        'Authorization': f'Bearer {self.tavily_api_key}',
                        'Content-Type': 'application/json'
                    }

                    data = {
                        'query': query,
                        'search_depth': 'advanced',
                        'include_images': True,
                        'include_answer': True,
                        'max_results': min(k, 10)
                    }

                    response = requests.post('https://api.tavily.com/search', headers=headers, json=data)
                    response.raise_for_status()

                    result_data = response.json()

                    # Extract text results and images
                    results = result_data.get('results', [])
                    images = result_data.get('images', [])

                    if not results:
                        return "No search results found."

                    # Format text results
                    text_results = []
                    for i, result in enumerate(results, 1):
                        title = result.get('title', 'No title')
                        url = result.get('url', 'Unknown source')
                        content = result.get('content', '')[:1000] + "..." if len(result.get('content', '')) > 1000 else result.get('content', '')

                        text_results.append(f"[{i}] {title}\nSource: {url}\n{content}\n")

                    # Combine text and images into structured response
                    if images:
                        # Create image gallery HTML
                        image_gallery = "\n\n**Image Gallery:**\n" + "\n".join([f"![Image]({img})" for img in images[:10]])  # Limit to 10 images
                        return "\n".join(text_results) + image_gallery
                    else:
                        return "\n".join(text_results)

                except Exception as e:
                    logger.error(f"Tavily search failed: {e}")
                    return f"Search failed: {str(e)}"

            return tavily_search

        except Exception as e:
            logger.error(f"Failed to create Tavily search tool: {e}")
            return None

    def _create_exa_search_tool(self) -> Optional[BaseTool]:
        """Create Exa search tool."""
        try:
            # Try import paths for Exa
            try:
                from langchain_community.retrievers import ExaSearchRetriever
            except ImportError:
                try:
                    from langchain_exa import ExaSearchRetriever
                except ImportError:
                    logger.warning("Exa search not available. Install with: pip install langchain-exa")
                    return None

            @tool("exa_search")
            def exa_search(query: str, k: int = 3) -> str:
                """
                Search the web using Exa for comprehensive web search.

                Args:
                    query: Search query
                    k: Number of results to return (max 10)

                Returns:
                    Search results as formatted text with embedded image gallery
                """
                try:
                    # Import requests to make direct API call for full response
                    import requests

                    # Make direct API call to get full response with images
                    headers = {
                        'Authorization': f'Bearer {self.exa_api_key}',
                        'Content-Type': 'application/json'
                    }

                    data = {
                        'query': query,
                        'numResults': min(k, 10),
                        'includeDomains': [],
                        'excludeDomains': [],
                        'startCrawlDate': None,
                        'endCrawlDate': None,
                        'startPublishedDate': None,
                        'endPublishedDate': None,
                        'useAutoprompt': True,
                        'type': 'neural'
                    }

                    response = requests.post('https://api.exa.ai/search', headers=headers, json=data)
                    response.raise_for_status()

                    result_data = response.json()

                    # Extract text results and images
                    results = result_data.get('results', [])
                    if not results:
                        return "No search results found."

                    # Collect all images
                    images = []
                    for result in results:
                        image_url = result.get('image')
                        if image_url:
                            images.append(image_url)

                    # Format text results
                    text_results = []
                    for i, result in enumerate(results, 1):
                        title = result.get('title', 'No title')
                        url = result.get('url', 'Unknown source')
                        text = result.get('text', '')[:1000] + "..." if len(result.get('text', '')) > 1000 else result.get('text', '')

                        text_results.append(f"[{i}] {title}\nSource: {url}\n{text}\n")

                    # Combine text and images into structured response
                    if images:
                        # Create image gallery HTML
                        image_gallery = "\n\n**Image Gallery:**\n" + "\n".join([f"![Image]({img})" for img in images[:10]])  # Limit to 10 images
                        return "\n".join(text_results) + image_gallery
                    else:
                        return "\n".join(text_results)

                except Exception as e:
                    logger.error(f"Exa search failed: {e}")
                    return f"Search failed: {str(e)}"

            return exa_search

        except Exception as e:
            logger.error(f"Failed to create Exa search tool: {e}")
            return None

    def _create_jina_search_tool(self) -> Optional[BaseTool]:
        """Create Jina search tool."""
        try:
            # Import here to avoid dependency issues if not installed
            try:
                from langchain_community.retrievers import JinaSearchRetriever
            except ImportError:
                logger.warning("Jina search retriever not available. Install langchain-community with Jina support.")
                return None

            @tool("jina_search")
            def jina_search(query: str, k: int = 3) -> str:
                """
                Search the web using Jina for AI-powered web search.

                Args:
                    query: Search query
                    k: Number of results to return (max 10)

                Returns:
                    Search results as formatted text
                """
                try:
                    retriever = JinaSearchRetriever(
                        api_key=self.jina_api_key,
                        k=min(k, 10)
                    )

                    docs = retriever.invoke(query)
                    if not docs:
                        return "No search results found."

                    results = []
                    for i, doc in enumerate(docs, 1):
                        title = doc.metadata.get('title', 'No title')
                        source = doc.metadata.get('source', 'Unknown source')
                        content = doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content

                        results.append(f"[{i}] {title}\nSource: {source}\n{content}\n")

                    return "\n".join(results)

                except Exception as e:
                    logger.error(f"Jina search failed: {e}")
                    return f"Search failed: {str(e)}"

            return jina_search

        except Exception as e:
            logger.error(f"Failed to create Jina search tool: {e}")
            return None


class AssistantTools:
    """Main tools collection for the assistant."""

    def __init__(self, web_search_config: Optional[Dict[str, str]] = None):
        """
        Initialize assistant tools.

        Args:
            web_search_config: Dictionary with API keys for web search services
                Keys: 'tavily_api_key', 'exa_api_key', 'jina_api_key'
        """
        self.web_search_config = web_search_config or {}
        self.web_tools = WebSearchTools(
            tavily_api_key=self.web_search_config.get('tavily_api_key'),
            exa_api_key=self.web_search_config.get('exa_api_key'),
            jina_api_key=self.web_search_config.get('jina_api_key')
        )

    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all available tools.

        Returns:
            List of all tool instances
        """
        tools = []

        # Add web search tools
        tools.extend(self.web_tools.get_available_tools())

        # Add other utility tools here in the future
        # tools.extend(self.utility_tools.get_tools())

        logger.info(f"Loaded {len(tools)} tools")
        return tools

    def get_tool_names(self) -> List[str]:
        """
        Get names of all available tools.

        Returns:
            List of tool names
        """
        return [tool.name for tool in self.get_all_tools()]

    def update_web_search_config(self, config: Dict[str, str]):
        """
        Update web search configuration.

        Args:
            config: New configuration dictionary
        """
        self.web_search_config.update(config)
        self.web_tools = WebSearchTools(
            tavily_api_key=self.web_search_config.get('tavily_api_key'),
            exa_api_key=self.web_search_config.get('exa_api_key'),
            jina_api_key=self.web_search_config.get('jina_api_key')
        )
        logger.info("Web search configuration updated")
