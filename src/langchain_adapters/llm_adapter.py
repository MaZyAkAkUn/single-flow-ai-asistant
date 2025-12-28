"""
LangChain LLM adapter for provider abstraction.
Handles OpenRouter integration and other LLM providers.
"""

from typing import Optional, Dict, Any, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, AIMessageChunk
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
import asyncio
import time

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.logging_config import get_logger, get_tools_logger
from .tools import AssistantTools
from ..core.tool_set_manager import ToolSetManager, ToolSet

logger = get_logger(__name__)


class LLMProvider:
    """Enum-like class for LLM providers."""

    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    OPENAI = "openai"


class LLMAdapter:
    """
    Adapter for different LLM providers.
    Provides a unified interface for LangChain LLM interactions.
    """

    def __init__(self, provider: str = LLMProvider.OPENROUTER, **kwargs):
        """
        Initialize the LLM adapter.

        Args:
            provider: The LLM provider to use
            **kwargs: Provider-specific configuration
        """
        self.provider = provider
        self.config = kwargs
        self._llm: Optional[BaseLanguageModel] = None
        self._llm_with_tools: Optional[BaseLanguageModel] = None
        self._tools: List[BaseTool] = []
        self._assistant_tools: Optional[AssistantTools] = None
        # Token streaming configuration
        self._streaming_buffer_timeout = 0.1  # seconds (reduced for faster UI response)
        self._max_buffer_size = 10  # max tokens before flushing

        # Reasoning configuration
        self.reasoning_config = {
            'enabled': False,
            'effort': 'medium',  # 'high', 'medium', 'low', 'minimal', 'none'
            'max_tokens': None,  # Specific token limit for reasoning
            'exclude': False,    # Whether to exclude reasoning from response
        }

        # Tool set management
        self.tool_set_manager = ToolSetManager()
        self._current_tool_set_name: Optional[str] = None

    def get_llm(self) -> BaseLanguageModel:
        """
        Get the configured LLM instance.

        Returns:
            Configured LangChain LLM instance
        """
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self) -> BaseLanguageModel:
        """
        Create the LLM instance based on provider configuration.

        Returns:
            Configured LLM instance
        """
        if self.provider == LLMProvider.OPENROUTER:
            return self._create_openrouter_llm()
        elif self.provider == LLMProvider.OLLAMA:
            return self._create_ollama_llm()
        elif self.provider == LLMProvider.OPENAI:
            return self._create_openai_llm()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _create_openrouter_llm(self) -> ChatOpenAI:
        """
        Create OpenRouter LLM instance.

        OpenRouter uses OpenAI-compatible API, so we use ChatOpenAI with custom base URL.
        """
        api_key = self.config.get('api_key')
        if not api_key:
            raise ValueError("OpenRouter API key is required")

        model = self.config.get('model', 'anthropic/claude-3-haiku')
        temperature = self.config.get('temperature', 0.7)
        # reasoning models need more tokens for thinking process
        max_tokens = self.config.get('max_tokens', 4096)

        logger.info(f"Creating OpenRouter LLM with model: {model}")

        # Create LLM instance
        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True  # Enable streaming for token-by-token responses
        )

        # Add reasoning parameters if enabled
        reasoning_params = self._get_reasoning_params()
        if reasoning_params:
            # Bind reasoning parameters to the LLM
            llm = llm.bind(extra_body={"reasoning": reasoning_params})
            logger.info(f"Bound reasoning parameters to OpenRouter LLM: {reasoning_params}")

        return llm

    def _create_ollama_llm(self) -> Ollama:
        """
        Create Ollama LLM instance for local models.
        """
        model = self.config.get('model', 'llama2')
        temperature = self.config.get('temperature', 0.7)

        logger.info(f"Creating Ollama LLM with model: {model}")

        return Ollama(
            model=model,
            temperature=temperature,
        )

    def _create_openai_llm(self) -> ChatOpenAI:
        """
        Create OpenAI LLM instance.
        """
        api_key = self.config.get('api_key')
        if not api_key:
            raise ValueError("OpenAI API key is required")

        model = self.config.get('model', 'gpt-3.5-turbo')
        temperature = self.config.get('temperature', 0.7)
        max_tokens = self.config.get('max_tokens', 1000)

        logger.info(f"Creating OpenAI LLM with model: {model}")

        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True  # Enable streaming for real-time responses
        )

    def create_chain(self, prompt_template: PromptTemplate) -> Runnable:
        """
        Create a basic LLM chain with the given prompt template.

        Args:
            prompt_template: The prompt template to use

        Returns:
            Runnable chain
        """
        llm = self.get_llm()
        return prompt_template | llm

    async def agenerate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response asynchronously.

        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated response text
        """
        llm = self.get_llm()

        # Log request details
        prompt_length = len(prompt)
        logger.info(f"LLM request - provider: {self.provider}, prompt_length: {prompt_length}")

        try:
            response = await llm.ainvoke(prompt, **kwargs)

            # Log response details
            logger.info("LLM response received", extra={
                "response_type": type(response).__name__,
                "has_content_attr": hasattr(response, 'content'),
                "content_is_none": hasattr(response, 'content') and response.content is None,
                "response_str_length": len(str(response)) if response else 0
            })

            if hasattr(response, 'content'):
                content = response.content
                content_length = len(content) if content else 0
                logger.info(f"LLM response content - length: {content_length}")
                if not content:
                    logger.warning("LLM returned empty content string")
                return content
            else:
                logger.warning(f"LLM response has no content attribute, using str() representation")
                response_str = str(response)
                logger.info(f"LLM response str representation - length: {len(response_str)}")
                return response_str

        except Exception as e:
            logger.error(f"Error generating response: {e}", extra={
                "error_type": type(e).__name__,
                "provider": self.provider,
                "prompt_length": prompt_length
            })
            raise

    async def astream_llm_response(self, prompt_or_messages, streaming=True, **kwargs):
        """
        Unified LLM response method that automatically uses tools when available.
        Handles both LangChain messages and plain text prompts.

        Args:
            prompt_or_messages: Either a string prompt or list of LangChain messages
            streaming: Whether to stream the response (default True)
            **kwargs: Additional generation parameters

        Yields:
            Streaming events with tool execution support when tools are available
        """
        # Auto-detect: Use tools if available (default behavior)
        has_tools = bool(self._tools)

        if not has_tools:
            # No tools configured, use simple streaming path
            logger.info("No tools configured, using simple streaming")
            async for event in self._astream_without_tools(prompt_or_messages, **kwargs):
                yield event
            return

        # Tools available - use optimized streaming with minimal buffering
        logger.info(f"Tools configured ({len(self._tools)}), using optimized streaming")
        async for event in self._astream_with_tools_optimized(prompt_or_messages, **kwargs):
            yield event

    async def _astream_without_tools(self, prompt_or_messages, **kwargs):
        """
        Stream response without tools - basic LLM call with optimized buffering.
        Handles reasoning tokens if present in the response.
        """
        # Convert input to appropriate format for LLM
        if isinstance(prompt_or_messages, str):
            # Plain text prompt
            prompt = prompt_or_messages
        else:
            # LangChain messages - extract text from last user message
            prompt = "Hello"  # Default fallback
            if prompt_or_messages:
                for msg in reversed(prompt_or_messages):
                    if hasattr(msg, 'content') and msg.content:
                        prompt = msg.content
                        break

        llm = self.get_llm()
        logger.info(f"Simple streaming - provider: {self.provider}, prompt_length: {len(prompt)}")

        try:
            # Use optimized streaming with token buffering
            token_buffer = []
            reasoning_buffer = []
            last_flush_time = time.time()

            async for chunk in llm.astream(prompt, **kwargs):
                # Handle reasoning tokens (OpenRouter format)
                reasoning_content = self._extract_reasoning_from_chunk(chunk)
                if reasoning_content:
                    reasoning_buffer.append(reasoning_content)
                    # Yield reasoning immediately for real-time display
                    yield {"type": "reasoning", "content": reasoning_content}

                # Handle regular content tokens
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    if content:
                        token_buffer.append(content)
                        current_time = time.time()

                        # Flush buffer conditions:
                        # 1. Timeout reached
                        # 2. Buffer size limit reached
                        # 3. Chunk contains significant content
                        should_flush = (
                            current_time - last_flush_time >= self._streaming_buffer_timeout or
                            len(token_buffer) >= self._max_buffer_size or
                            len(content) > 20  # Flush on larger chunks
                        )

                        if should_flush:
                            combined_tokens = ''.join(token_buffer)
                            if combined_tokens:
                                yield {"type": "token", "content": combined_tokens}
                            token_buffer.clear()
                            last_flush_time = current_time

            # Flush any remaining tokens
            if token_buffer:
                combined_tokens = ''.join(token_buffer)
                if combined_tokens:
                    yield {"type": "token", "content": combined_tokens}

            # Flush any remaining reasoning (shouldn't happen in normal flow)
            if reasoning_buffer:
                combined_reasoning = ''.join(reasoning_buffer)
                if combined_reasoning:
                    yield {"type": "reasoning", "content": combined_reasoning}

        except Exception as e:
            logger.error(f"Error in simple streaming: {e}", extra={
                "error_type": type(e).__name__,
                "provider": self.provider
            })
            yield {"type": "error", "content": f"Error during streaming: {str(e)}"}

    async def _astream_with_tools_optimized(self, prompt_or_messages, max_iterations: int = 3, **kwargs):
        """
        Optimized streaming with tools - yields tokens immediately without blocking on tool execution.
        """
        # Convert to LangChain messages if needed
        if isinstance(prompt_or_messages, str):
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt_or_messages)]
        else:
            messages = prompt_or_messages.copy() if prompt_or_messages else []

        logger.info(f"Optimized tool streaming - messages: {len(messages)}, max_iterations: {max_iterations}")

        llm_with_tools = self.get_llm_with_tools()
        iteration = 0
        token_buffer = []
        last_flush_time = time.time()

        while iteration < max_iterations:
            try:
                # Use astream instead of ainvoke to enable true streaming
                accumulated_msg = None
                
                async for chunk in llm_with_tools.astream(messages):
                    # Handle reasoning tokens (OpenRouter format)
                    reasoning_content = self._extract_reasoning_from_chunk(chunk)
                    if reasoning_content:
                        # Yield reasoning immediately for real-time display
                        yield {"type": "reasoning", "content": reasoning_content}

                    # Accumulate chunks to build full response for tool checking
                    if accumulated_msg is None:
                        accumulated_msg = chunk
                    else:
                        accumulated_msg += chunk
                    
                    # Stream content tokens immediately
                    if hasattr(chunk, 'content') and chunk.content:
                        token_buffer.append(chunk.content)
                        current_time = time.time()
                        
                        # Flush buffer conditions:
                        # 1. Timeout reached
                        # 2. Buffer size limit reached
                        should_flush = (
                            current_time - last_flush_time >= self._streaming_buffer_timeout or
                            len(token_buffer) >= self._max_buffer_size
                        )
                        
                        if should_flush:
                            combined_tokens = ''.join(token_buffer)
                            if combined_tokens:
                                yield {"type": "token", "content": combined_tokens}
                            token_buffer.clear()
                            last_flush_time = current_time

                # Flush any remaining tokens after stream completion
                if token_buffer:
                    combined_tokens = ''.join(token_buffer)
                    if combined_tokens:
                        yield {"type": "token", "content": combined_tokens}
                    token_buffer.clear()

                # Process the full accumulated response
                response = accumulated_msg
                if response is None:
                    # Should not happen unless empty stream
                    yield {"type": "error", "content": "LLM returned empty stream"}
                    break
                
                logger.info(f"Iteration {iteration + 1}: LLM response received", extra={
                    "has_tool_calls": hasattr(response, 'tool_calls') and bool(getattr(response, 'tool_calls', None)),
                    "tool_calls_count": len(getattr(response, 'tool_calls', [])) if hasattr(response, 'tool_calls') else 0
                })

                # Check if there are tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    logger.info(f"Iteration {iteration + 1}: LLM requested {len(response.tool_calls)} tool calls")

                    # Validate tool calls
                    available_tool_names = {tool.name for tool in self._tools}
                    valid_tool_calls = []
                    invalid_tool_calls = []

                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get("name", "")
                        if tool_name in available_tool_names:
                            valid_tool_calls.append(tool_call)
                        else:
                            invalid_tool_calls.append(tool_call)
                            logger.warning(f"LLM requested unknown tool: {tool_name}")

                    # Add tool error messages for invalid tools
                    for invalid_call in invalid_tool_calls:
                        tool_call_id = invalid_call.get("id")
                        tool_name = invalid_call.get("name", "unknown")
                        from langchain_core.messages import ToolMessage
                        error_message = ToolMessage(
                            content=f"Error: Tool '{tool_name}' is not available. Available tools: {', '.join(available_tool_names)}",
                            tool_call_id=tool_call_id
                        )
                        messages.append(error_message)

                    if valid_tool_calls:
                        messages.append(response)

                        # Execute tools asynchronously without blocking token streaming
                        try:
                            # Add status update for tool execution
                            tool_names = [call.get("name", "unknown") for call in valid_tool_calls]
                            tool_name_str = ", ".join(tool_names)
                            yield {"type": "status", "content": f"Using tool(s): {tool_name_str}...", "tool_names": tool_names}
                            
                            # Execute tools - we'll do this in a loop to yield events for each
                            tool_results = []
                            
                            for tool_call in valid_tool_calls:
                                tool_name = tool_call.get("name")
                                tool_args = tool_call.get("args")
                                
                                # Emit tool start event
                                yield {
                                    "type": "tool_start",
                                    "tool": tool_name,
                                    "input": tool_args
                                }
                                
                                # Execute single tool in executor
                                # We need to use self._execute_single_tool instead of execute_tool_calls to get individual results
                                # checking if we have helper for single tool or need to adapt
                                loop = asyncio.get_event_loop()
                                
                                # Helper wrapper for single tool execution to run in executor
                                def exec_single():
                                    return self.execute_tool_calls([tool_call])[0]
                                
                                result_message = await loop.run_in_executor(None, exec_single)
                                tool_results.append(result_message)
                                
                                # Emit tool end event
                                yield {
                                    "type": "tool_end",
                                    "tool": tool_name,
                                    "output": result_message.content
                                }
                                
                            # Add tool results to messages
                            messages.extend(tool_results)
                            
                            # Continue to next iteration without waiting
                            iteration += 1
                            
                        except Exception as tool_error:
                            logger.error(f"Tool execution failed: {tool_error}")
                            yield {"type": "error", "content": f"Tool execution error: {str(tool_error)}"}
                            break
                    else:
                        # No valid tools, continue or finish
                        iteration += 1
                else:
                    # No more tool calls, we are done
                    yield {"type": "done"}
                    break

            except Exception as e:
                logger.error(f"Error in tool iteration {iteration + 1}: {e}", extra={
                    "error_type": type(e).__name__,
                    "iteration": iteration + 1,
                    "max_iterations": max_iterations
                })
                yield {"type": "error", "content": f"Error during tool execution: {str(e)}"}
                break

        if iteration >= max_iterations:
            logger.warning(f"Reached maximum iterations ({max_iterations}) without final response")
            yield {"type": "error", "content": f"Maximum tool iterations ({max_iterations}) reached"}

    async def _aexecute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List:
        """
        Asynchronously execute tool calls for future async tool support.
        """
        return await asyncio.get_event_loop().run_in_executor(None, self.execute_tool_calls, tool_calls)

    # Legacy method for backward compatibility
    async def astream_response(self, prompt: str, **kwargs):
        """
        Legacy streaming method - now delegates to unified interface.
        """
        async for event in self.astream_llm_response(prompt, **kwargs):
            yield event

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response synchronously.

        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated response text
        """
        llm = self.get_llm()

        try:
            response = llm.invoke(prompt, **kwargs)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """
        Get list of available models for the current provider.

        Returns:
            List of model names
        """
        if self.provider == LLMProvider.OPENROUTER:
            # OpenRouter supports many models, return some popular ones
            return [
                'anthropic/claude-3-haiku',
                'anthropic/claude-3-sonnet',
                'openai/gpt-4o-mini',
                'openai/gpt-4o',
                'meta-llama/llama-3.1-8b-instruct',
                'google/gemini-pro'
            ]
        elif self.provider == LLMProvider.OLLAMA:
            # For Ollama, we'd need to query available models
            # For now, return common ones
            return ['llama2', 'mistral', 'codellama']
        elif self.provider == LLMProvider.OPENAI:
            return ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo-preview']
        else:
            return []

    def validate_config(self) -> bool:
        """
        Validate the current configuration.

        Returns:
            True if configuration is valid
        """
        try:
            if self.provider == LLMProvider.OPENROUTER:
                api_key = self.config.get('api_key')
                if not api_key or not isinstance(api_key, str) or len(api_key.strip()) == 0:
                    logger.warning("OpenRouter API key is missing or invalid")
                    return False
                # Basic format validation for OpenRouter keys (they typically start with 'sk-or-v1-')
                if not api_key.strip().startswith('sk-or-v1-'):
                    logger.warning("OpenRouter API key format appears invalid")
                    return False
                return True
            elif self.provider == LLMProvider.OLLAMA:
                # Ollama doesn't require API key, but we can validate model exists
                model = self.config.get('model', 'llama2')
                if not model or not isinstance(model, str):
                    logger.warning("Ollama model name is missing or invalid")
                    return False
                return True
            elif self.provider == LLMProvider.OPENAI:
                api_key = self.config.get('api_key')
                if not api_key or not isinstance(api_key, str) or len(api_key.strip()) == 0:
                    logger.warning("OpenAI API key is missing or invalid")
                    return False
                # Basic format validation for OpenAI keys (they typically start with 'sk-')
                if not api_key.strip().startswith('sk-'):
                    logger.warning("OpenAI API key format appears invalid")
                    return False
                return True
            else:
                logger.error(f"Unknown provider: {self.provider}")
                return False
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

    def test_connection(self) -> bool:
        """
        Test the connection to the LLM provider.

        Returns:
            True if connection is successful
        """
        try:
            # Try to create the LLM instance and make a simple request
            llm = self.get_llm()
            # For testing, we could make a minimal request, but for now just validate config
            return self.validate_config()
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    # Tool-related methods

    def configure_tools(self, web_search_config: Optional[Dict[str, str]] = None):
        """
        Configure tools for the default tool set.

        Args:
            web_search_config: Configuration for web search tools
        """
        # Create default tool set if it doesn't exist
        if not self.tool_set_manager.get_tool_set("default"):
            self.create_tool_set("default")
            self.set_tool_set("default")

        # Configure assistant tools
        self._assistant_tools = AssistantTools(web_search_config)
        new_tools = self._assistant_tools.get_all_tools()

        # Get current tool set (should be default at this point)
        current_tool_set = self.tool_set_manager.get_default_tool_set()
        if not current_tool_set:
            logger.error("No default tool set available")
            return

        # Clear existing tools in current set
        current_tool_set.clear()

        # Add new tools to current set
        for tool in new_tools:
            current_tool_set.add_tool(tool)

        # Update internal tools list
        self._tools = current_tool_set.tools.copy()

        self._llm_with_tools = None  # Reset cached LLM with tools
        logger.info(f"Configured {len(new_tools)} web search tools in default tool set "
                   f"(total: {len(self._tools)} tools)")

    def create_tool_set(self, name: str, tools: List[BaseTool] = None) -> ToolSet:
        """
        Create a new tool set.

        Args:
            name: Name of the tool set
            tools: Initial list of tools (optional)

        Returns:
            Created ToolSet instance
        """
        return self.tool_set_manager.create_tool_set(name, tools)

    def set_tool_set(self, name: str) -> bool:
        """
        Switch to a specific tool set.

        Args:
            name: Name of tool set to switch to

        Returns:
            True if successful, False if tool set not found
        """
        tool_set = self.tool_set_manager.get_tool_set(name)
        if not tool_set:
            logger.warning(f"Tool set '{name}' not found")
            return False

        self._current_tool_set_name = name
        self._tools = tool_set.tools.copy()
        self._llm_with_tools = None  # Reset cached LLM with tools

        logger.info(f"Switched to tool set '{name}' with {len(self._tools)} tools")
        return True

    def get_current_tool_set(self) -> Optional[ToolSet]:
        """
        Get the currently active tool set.

        Returns:
            Current ToolSet instance or None if not set
        """
        if self._current_tool_set_name:
            return self.tool_set_manager.get_tool_set(self._current_tool_set_name)
        return None

    def get_tool_set(self, name: str) -> Optional[ToolSet]:
        """
        Get a tool set by name.

        Args:
            name: Name of tool set to get

        Returns:
            ToolSet instance or None if not found
        """
        return self.tool_set_manager.get_tool_set(name)

    def list_tool_sets(self) -> List[str]:
        """
        List all available tool set names.

        Returns:
            List of tool set names
        """
        return self.tool_set_manager.list_tool_sets()

    def set_default_tool_set(self, name: str) -> bool:
        """
        Set the default tool set.

        Args:
            name: Name of tool set to set as default

        Returns:
            True if successful, False if tool set doesn't exist
        """
        return self.tool_set_manager.set_default_tool_set(name)

    def get_default_tool_set(self) -> Optional[ToolSet]:
        """
        Get the default tool set.

        Returns:
            Default ToolSet instance or None if not set
        """
        return self.tool_set_manager.get_default_tool_set()

    def get_tools(self) -> List[BaseTool]:
        """
        Get the configured tools.

        Returns:
            List of available tools
        """
        logger.info(f"Available tools in get_tools() (LLMAdapter): {self._tools}")
        return self._tools.copy()

    def get_tool_names(self) -> List[str]:
        """
        Get names of available tools.

        Returns:
            List of tool names
        """
        return [tool.name for tool in self._tools]

    def get_llm_with_tools(self) -> BaseLanguageModel:
        """
        Get the LLM instance with tools bound.

        Returns:
            LLM instance with tools
        """
        if self._llm_with_tools is None:
            llm = self.get_llm()
            if self._tools:
                self._llm_with_tools = llm.bind_tools(self._tools)
                logger.info(f"Bound {len(self._tools)} tools to LLM")
            else:
                self._llm_with_tools = llm
        return self._llm_with_tools

    def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[ToolMessage]:
        """
        Execute tool calls and return results as ToolMessages.

        Args:
            tool_calls: List of tool call dictionaries from LLM response

        Returns:
            List of ToolMessage objects with results
        """
        results = []
        tools_by_name = {tool.name: tool for tool in self._tools}

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            try:
                if tool_name in tools_by_name:
                    tool = tools_by_name[tool_name]
                    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                    # Execute the tool
                    tool_result = tool.invoke(tool_args)

                    # Log tool execution and result
                    tools_logger = get_tools_logger()
                    tools_logger.info(f"Tool executed: {tool_name}", extra={
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_result": str(tool_result)[:1000]  # Limit result length for logging
                    })

                    # Create ToolMessage
                    result_message = ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id
                    )
                    results.append(result_message)
                    logger.info(f"Tool {tool_name} executed successfully")
                else:
                    logger.error(f"Unknown tool: {tool_name}")
                    results.append(ToolMessage(
                        content=f"Error: Unknown tool '{tool_name}'",
                        tool_call_id=tool_call_id
                    ))

            except Exception as e:
                logger.error(f"Tool execution failed for {tool_name}: {e}")
                results.append(ToolMessage(
                    content=f"Error executing tool '{tool_name}': {str(e)}",
                    tool_call_id=tool_call_id
                ))

        return results

    def generate_response_with_tools(self, messages: List[Dict[str, Any]], max_iterations: int = 3) -> str:
        """
        Generate a response using tools if needed.

        Args:
            messages: List of message dictionaries
            max_iterations: Maximum number of tool call iterations

        Returns:
            Final response text
        """
        if not self._tools:
            # No tools configured, use regular generation
            logger.info("No tools available, falling back to regular generation")
            # Convert messages to a simple prompt for regular generation
            if messages and len(messages) > 0:
                last_message = messages[-1]
                if last_message.get("role") == "user":
                    return self.generate_response(last_message.get("content", ""))
            return self.generate_response("Hello")

        llm_with_tools = self.get_llm_with_tools()

        # Convert messages to LangChain format
        langchain_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                # Handle tool calls if present
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    # Create AIMessage with tool calls
                    ai_msg = AIMessage(content=content)
                    ai_msg.tool_calls = tool_calls
                    langchain_messages.append(ai_msg)
                else:
                    langchain_messages.append(AIMessage(content=content))
            elif role == "tool":
                # Tool result message
                tool_call_id = msg.get("tool_call_id")
                langchain_messages.append(ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id
                ))

        iteration = 0
        while iteration < max_iterations:
            try:
                # Get LLM response
                response = llm_with_tools.invoke(langchain_messages)

                # Check if there are tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    logger.info(f"Iteration {iteration + 1}: LLM requested {len(response.tool_calls)} tool calls")

                    # Validate that requested tools actually exist
                    available_tool_names = {tool.name for tool in self._tools}
                    valid_tool_calls = []
                    invalid_tool_calls = []

                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get("name", "")
                        if tool_name in available_tool_names:
                            valid_tool_calls.append(tool_call)
                        else:
                            invalid_tool_calls.append(tool_call)
                            logger.warning(f"LLM requested unknown tool: {tool_name}")

                    if invalid_tool_calls:
                        # Add error messages for invalid tools
                        for invalid_call in invalid_tool_calls:
                            tool_call_id = invalid_call.get("id")
                            tool_name = invalid_call.get("name", "unknown")
                            error_message = ToolMessage(
                                content=f"Error: Tool '{tool_name}' is not available. Available tools: {', '.join(available_tool_names)}",
                                tool_call_id=tool_call_id
                            )
                            langchain_messages.append(error_message)

                    if valid_tool_calls:
                        # Add the LLM response to messages
                        langchain_messages.append(response)

                        # Execute valid tools
                        tool_results = self.execute_tool_calls(valid_tool_calls)

                        # Add tool results to messages
                        langchain_messages.extend(tool_results)

                        iteration += 1
                    else:
                        # No valid tools, continue to next iteration or return response
                        iteration += 1
                else:
                    # No more tool calls, return the final response
                    logger.info(f"Final response reached at iteration {iteration + 1}", extra={
                        "iterations_used": iteration + 1,
                        "max_iterations": max_iterations,
                        "message_count": len(langchain_messages)
                    })

                    # Log response details
                    logger.info("Tool-based generation final response", extra={
                        "response_type": type(response).__name__,
                        "has_content_attr": hasattr(response, 'content'),
                        "content_is_none": hasattr(response, 'content') and response.content is None,
                        "response_str_length": len(str(response)) if response else 0
                    })

                    if hasattr(response, 'content'):
                        content = response.content
                        content_length = len(content) if content else 0
                        logger.info(f"Final response content - length: {content_length}")
                        if not content:
                            logger.warning("LLM returned empty content string in tool-based generation")
                        return content
                    else:
                        logger.warning("Final response has no content attribute, using str() representation")
                        response_str = str(response)
                        logger.info(f"Final response str representation - length: {len(response_str)}")
                        return response_str

            except Exception as e:
                logger.error(f"Error in tool iteration {iteration + 1}: {e}", extra={
                    "error_type": type(e).__name__,
                    "iteration": iteration + 1,
                    "max_iterations": max_iterations,
                    "message_count": len(langchain_messages)
                })
                return f"I'm sorry, I encountered an error while processing your request: {str(e)}"

        # Max iterations reached
        logger.warning(f"Reached maximum iterations ({max_iterations}) without final response", extra={
            "iterations_completed": iteration,
            "max_iterations": max_iterations,
            "final_message_count": len(langchain_messages)
        })
        return "I'm sorry, I couldn't complete the request within the allowed number of tool calls."

    def update_tool_config(self, web_search_config: Dict[str, str]):
        """
        Update tool configuration for the current tool set.

        Args:
            web_search_config: New web search configuration
        """
        if self._assistant_tools:
            self._assistant_tools.update_web_search_config(web_search_config)
            new_tools = self._assistant_tools.get_all_tools()

            # Get current tool set
            current_tool_set = self.get_current_tool_set()
            if not current_tool_set:
                logger.error("No current tool set available")
                return

            # Clear existing tools in current set
            current_tool_set.clear()

            # Add new tools to current set
            for tool in new_tools:
                current_tool_set.add_tool(tool)

            # Update internal tools list
            self._tools = current_tool_set.tools.copy()

            self._llm_with_tools = None  # Reset cached LLM with tools
            logger.info(f"Tool configuration updated: {len(new_tools)} tools in current tool set "
                       f"(total: {len(self._tools)} tools)")
        else:
            self.configure_tools(web_search_config)

    def set_streaming_buffer_config(self, timeout: float = 0.5, max_buffer_size: int = 10):
        """
        Configure streaming buffer settings for optimized token delivery.

        Args:
            timeout: Buffer flush timeout in seconds (default 0.5)
            max_buffer_size: Maximum tokens to buffer before flushing (default 10)
        """
        self._streaming_buffer_timeout = timeout
        self._max_buffer_size = max_buffer_size
        logger.info(f"Streaming buffer configured: timeout={timeout}s, max_size={max_buffer_size}")

    def configure_reasoning(self, enabled: bool = False, effort: str = 'medium',
                           max_tokens: Optional[int] = None, exclude: bool = False):
        """
        Configure reasoning tokens for models that support it.

        Args:
            enabled: Whether to enable reasoning
            effort: Reasoning effort level ('high', 'medium', 'low', 'minimal', 'none')
            max_tokens: Specific token limit for reasoning (overrides effort)
            exclude: Whether to exclude reasoning from response
        """
        self.reasoning_config = {
            'enabled': enabled,
            'effort': effort,
            'max_tokens': max_tokens,
            'exclude': exclude,
        }

        # Reset LLM instances to apply new reasoning config
        self._llm = None
        self._llm_with_tools = None

        logger.info(f"Reasoning configured: enabled={enabled}, effort={effort}, "
                   f"max_tokens={max_tokens}, exclude={exclude}")

    def get_reasoning_config(self) -> Dict[str, Any]:
        """
        Get current reasoning configuration.

        Returns:
            Dictionary with reasoning settings
        """
        return self.reasoning_config.copy()

    def _get_reasoning_params(self) -> Optional[Dict[str, Any]]:
        """
        Get reasoning parameters for OpenRouter API calls.

        Returns:
            Reasoning parameters dict or None if disabled
        """
        if not self.reasoning_config.get('enabled', False):
            return None

        reasoning_params = {}

        # Set effort or max_tokens (but not both)
        if self.reasoning_config.get('max_tokens'):
            reasoning_params['max_tokens'] = self.reasoning_config['max_tokens']
        else:
            effort = self.reasoning_config.get('effort', 'medium')
            reasoning_params['effort'] = effort

        # Set exclude flag
        if self.reasoning_config.get('exclude', False):
            reasoning_params['exclude'] = True

        return reasoning_params

    def _extract_reasoning_from_chunk(self, chunk) -> Optional[str]:
        """
        Extract reasoning content from a streaming chunk.

        Handles different reasoning formats from OpenRouter API:
        - Legacy: chunk.reasoning (plain text)
        - New: chunk.reasoning_details (structured array)
        - LangChain extra fields: chunk.additional_kwargs.get('reasoning')

        Args:
            chunk: Streaming chunk from LLM

        Returns:
            Reasoning text content or None if no reasoning
        """
        try:
            # Check direct attribute (if supported by library version)
            if hasattr(chunk, 'reasoning') and chunk.reasoning:
                return str(chunk.reasoning)

            # Check additional_kwargs (Standard LangChain location for extra fields)
            if hasattr(chunk, 'additional_kwargs'):
                # Check for direct 'reasoning' field
                if 'reasoning' in chunk.additional_kwargs and chunk.additional_kwargs['reasoning']:
                    return str(chunk.additional_kwargs['reasoning'])
                
                # Check for 'reasoning_details' in additional_kwargs (if passed as raw object)
                # Note: complex objects might not stream well this way, but we check just in case
                if 'reasoning_details' in chunk.additional_kwargs:
                    # Logic to parse reasoning_details object if needed
                    pass

            # Check for structured reasoning_details format (direct attribute)
            if hasattr(chunk, 'reasoning_details') and chunk.reasoning_details:
                reasoning_parts = []

                # reasoning_details is an array of reasoning objects
                for reasoning_item in chunk.reasoning_details:
                    if isinstance(reasoning_item, dict):
                        # Handle different reasoning types
                        reasoning_type = reasoning_item.get('type', '')

                        if reasoning_type == 'reasoning.text':
                            # Plain text reasoning
                            text = reasoning_item.get('text', '')
                            if text:
                                reasoning_parts.append(text)

                        elif reasoning_type == 'reasoning.summary':
                            # Summary reasoning
                            summary = reasoning_item.get('summary', '')
                            if summary:
                                reasoning_parts.append(f"Summary: {summary}")

                        elif reasoning_type == 'reasoning.encrypted':
                            # Encrypted reasoning (usually redacted)
                            data = reasoning_item.get('data', '')
                            if data and data != '[REDACTED]':
                                reasoning_parts.append(f"[Encrypted reasoning: {data[:50]}...]")

                        # Skip reasoning.signature and reasoning.unknown types
                        # as they don't contain displayable content

                if reasoning_parts:
                    return '\n'.join(reasoning_parts)

            return None

        except Exception as e:
            logger.warning(f"Error extracting reasoning from chunk: {e}")
            return None
