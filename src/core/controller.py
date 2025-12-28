"""
AI Controller - Central orchestrator for the personal assistant.
Handles user input, coordinates LangChain chains, and manages conversation flow.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import os
import json
import re
import hashlib
import urllib.request
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage

from .logging_config import get_logger
from .file_ingestor import FileIngestor
from .memory_adapter import MemoryAdapter
from .audio_provider import AudioProvider
from .project_manager import ProjectContextManager
from ..langchain_adapters import LLMAdapter, LLMProvider, EnhancedLLMAdapter
from ..langchain_adapters.tools import AssistantTools

logger = get_logger(__name__)


class AIController:
    """
    Central controller that orchestrates AI interactions.
    Manages conversation flow, prompt building, and LLM responses.
    """

    def __init__(self, llm_adapter: Optional[LLMAdapter] = None, file_ingestor: Optional[FileIngestor] = None, memory_adapter: Optional[MemoryAdapter] = None, openai_api_key: Optional[str] = None, settings_manager: Optional['PlainSettings'] = None):
        """
        Initialize the AI controller.

        Args:
            llm_adapter: Pre-configured LLM adapter, or None to create default
            file_ingestor: Pre-configured file ingestor, or None to create default
            memory_adapter: Pre-configured memory adapter, or None to create default
            openai_api_key: OpenAI API key for TTS and memory operations
            settings_manager: Settings manager for loading tool configurations
        """
        self.openai_api_key = openai_api_key
        self.settings_manager = settings_manager
        self.llm_adapter = llm_adapter or self._create_default_llm_adapter()
        self.file_ingestor = file_ingestor or self._create_default_file_ingestor()
        self.memory_adapter = memory_adapter or self._create_default_memory_adapter()
        self.audio_provider = self._create_default_audio_provider()

        # Initialize Project Manager
        self.project_manager = ProjectContextManager()

        # Initialize basic attributes that are always needed
        self.conversation_messages: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, Any]] = []
        self.message_prompts: Dict[str, str] = {}
        self.max_history_length = 50
        self.retrieval_enabled = True
        self.max_retrieved_docs = 3
        self.hybrid_retrieval_enabled = False
        self.semantic_weight = 0.7
        self.keyword_weight = 0.3
        self.memory_enabled = True
        self.max_retrieved_memories = 3
        self._providers_need_update = False

        # Initialize draft management
        self.drafts: List[Dict[str, Any]] = []
        self.archived_drafts: List[Dict[str, Any]] = []
        self.max_drafts = 5
        self.current_draft_id: Optional[str] = None

        # Load settings if manager is available
        if self.settings_manager:
            memory_settings = self.settings_manager.get_setting('memory') or {}
            self.memory_enabled = memory_settings.get('enabled', True)
            self.max_retrieved_memories = memory_settings.get('max_memories', 3)
            
            self._load_tool_configurations()
        else:
            # Initialize system prompt with no tools
            self.system_prompt = self._get_default_system_prompt([])

    def _get_tool_config_from_settings(self) -> Optional[Dict[str, str]]:
        """
        Get and map tool configuration from settings manager.
        """
        if not self.settings_manager:
            return None
            
        tool_api_keys = self.settings_manager.get_all_tool_api_keys()
        if not tool_api_keys:
            return None
            
        return {
            'tavily_api_key': tool_api_keys.get('tavily', ''),
            'exa_api_key': tool_api_keys.get('exa', ''),
            'jina_api_key': tool_api_keys.get('jina', '')
        }

    def _load_tool_configurations(self):
        """
        Load tool configurations from settings manager.
        """
        try:
            tool_api_keys = self.settings_manager.get_all_tool_api_keys()
            if tool_api_keys:
                self.configure_tools(tool_api_keys)
                logger.info(f"Loaded tool configurations for: {list(tool_api_keys.keys())}")

            # Load tool set configurations if available
            self._load_tool_set_configurations()
        except Exception as e:
            logger.error(f"Failed to load tool configurations: {e}")

        # Update system prompt with available tools
        available_tools = self.get_available_tools()
        self.system_prompt = self._get_default_system_prompt(available_tools)

    def _load_tool_set_configurations(self):
        """
        Load tool set configurations from settings manager.
        """
        try:
            # Get tool set configurations from settings
            tool_set_configs = self.settings_manager.get_setting('tool_sets') or {}

            # Configure each tool set
            for set_name, config in tool_set_configs.items():
                web_search_config = config.get('web_search')
                custom_tools = config.get('custom_tools')

                # Configure the tool set
                if web_search_config or custom_tools:
                    success = self.configure_tool_set(set_name, web_search_config, custom_tools)
                    if success:
                        logger.info(f"Configured tool set '{set_name}' from settings")
                else:
                    # Create empty tool set
                    success = self.create_tool_set(set_name)
                    if success:
                        logger.info(f"Created empty tool set '{set_name}' from settings")

            # Set active tool set if specified
            active_set = self.settings_manager.get_setting('active_tool_set')
            if active_set:
                success = self.set_active_tool_set(active_set)
                if success:
                    logger.info(f"Activated tool set '{active_set}' from settings")

        except Exception as e:
            logger.error(f"Failed to load tool set configurations: {e}")

    def _create_default_llm_adapter(self) -> LLMAdapter:
        """
        Create a default LLM adapter.
        In production, this should load from configuration.
        """
        # Determine correct API key for Memori (OpenAI key)
        # If we have settings manager, try to get explicit OpenAI key
        # otherwise fallback to provided openai_api_key (which might be OpenRouter key in some contexts)
        memori_api_key = self.openai_api_key

        if self.settings_manager:
            # Try to get specific OpenAI key from settings
            openai_key_from_settings = self.settings_manager.get_api_key("openai")
            if openai_key_from_settings:
                memori_api_key = openai_key_from_settings

        # Get current settings for system information and combo resolution
        settings = None
        resolved_provider = LLMProvider.OPENROUTER
        resolved_api_key = self.openai_api_key
        resolved_model = None

        if self.settings_manager:
            settings = self.settings_manager.load_settings()

            # Resolve LLM combo from settings
            llm_settings = settings.get('llm', {})
            combo_id = llm_settings.get('combo')
            if combo_id:
                combo_data = self.settings_manager.resolve_combo(combo_id)
                if combo_data:
                    resolved_provider = combo_data['provider']
                    resolved_model = combo_data['model']
                    # Get the correct API key for the resolved provider
                    resolved_api_key = self.settings_manager.get_api_key(resolved_provider) or self.openai_api_key
                    logger.info(f"Resolved LLM combo '{combo_id}' to provider: {resolved_provider}, model: {resolved_model}")

        # Use EnhancedLLMAdapter for structured prompt support
        return EnhancedLLMAdapter(
            provider=resolved_provider,
            api_key=resolved_api_key,
            model=resolved_model,
            openai_api_key=memori_api_key,
            settings=settings
        )

    def _create_default_file_ingestor(self) -> FileIngestor:
        """
        Create a default file ingestor.
        In production, this should load from configuration.
        """
        return FileIngestor()

    def _create_default_memory_adapter(self) -> Optional[MemoryAdapter]:
        """
        Create a default memory adapter.
        Only creates adapter if API key is available.
        In production, this should load from configuration.
        """
        if self.openai_api_key:
            return MemoryAdapter(openai_api_key=self.openai_api_key)
        return None

    def _create_default_audio_provider(self) -> AudioProvider:
        """
        Create a default audio provider.
        In production, this should load from configuration.
        """
        return AudioProvider(openai_api_key=self.openai_api_key)

    def _get_default_system_prompt(self, available_tools: Optional[List[str]] = None) -> str:
        """
        Get the default system prompt for the assistant.

        Args:
            available_tools: List of available tool names

        Returns:
            System prompt string
        """
        base_prompt = """You are a helpful, intelligent personal assistant. You help users with various tasks, answer questions, and engage in meaningful conversations. Be concise but informative in your responses."""

        if available_tools:
            tool_names = [tool.replace('_search', '') for tool in available_tools if tool.endswith('_search')]
            if tool_names:
                base_prompt += f"""

You have access to web search tools: {', '.join(tool_names)}. When answering questions that require current information, recent events, or data that might have changed since your last training, use these tools to get up-to-date information. Always cite your sources when using search results."""

        base_prompt += """

If you don't know something and cannot find it through available tools, admit it rather than making up information."""

        return base_prompt

    async def stream_message(self, user_message: str, combo_id: Optional[str] = None):
        """
        Process a user message with streaming support.

        Args:
            user_message: The user's input message
            combo_id: Optional combo ID to override the default LLM combo for this message

        Yields:
            Events like {"type": "status", "content": "..."} and {"type": "token", "content": "..."}
        """
        try:
            logger.info(f"Streaming message: {user_message[:100]}... (combo: {combo_id})")

            # Input validation
            if not isinstance(user_message, str):
                yield {"type": "error", "content": "Invalid input: message must be a string"}
                return

            if not user_message.strip():
                yield {"type": "error", "content": "Empty message received"}
                return

            if len(user_message) > 10000:
                yield {"type": "error", "content": f"Message too long: {len(user_message)} characters (max 10000)"}
                return

            # Add user message to conversation messages
            self.conversation_messages.append({
                'role': 'user',
                'content': user_message,
                'timestamp': datetime.now().isoformat(),
                'metadata': {}  # NEW: Add metadata field (will be filled by adapter)
            })

            # Add to legacy history
            self._add_to_history('user', user_message)

            response_text = ""

            # Handle combo override for this message
            temp_adapter = None
            if combo_id and self.settings_manager:
                combo_data = self.settings_manager.resolve_combo(combo_id)
                if combo_data:
                    logger.info(f"Using combo override '{combo_id}' for this message: {combo_data}")
                    # Create temporary adapter with combo settings
                    temp_adapter = EnhancedLLMAdapter(
                        provider=combo_data['provider'],
                        api_key=self.settings_manager.get_api_key(combo_data['provider']),
                        model=combo_data['model'],
                        settings=self.settings_manager.load_settings() if self.settings_manager else None
                    )
                    
                    # Configure tools for the temporary adapter
                    tool_config = self._get_tool_config_from_settings()
                    if tool_config:
                        temp_adapter.configure_tools(tool_config)
                        logger.info(f"Configured tools for temporary adapter (combo: {combo_id})")
                else:
                    logger.warning(f"Could not resolve combo '{combo_id}', using default adapter")

            # Use the temporary adapter if available, otherwise use the default adapter
            active_adapter = temp_adapter or self.llm_adapter

            # Use EnhancedLLMAdapter for structured streaming processing
            if isinstance(active_adapter, EnhancedLLMAdapter):
                try:
                    project_contexts = []  # TODO: Get from somewhere like self.project_manager.get_active_projects()

                    async for event in active_adapter.astream_message_structured(
                        user_message=user_message,
                        conversation_history=self.conversation_history,
                        project_contexts=project_contexts,
                        memory_enabled=self.memory_enabled,
                        retrieval_enabled=self.retrieval_enabled,
                        max_retrieved_memories=self.max_retrieved_memories,
                        max_retrieved_docs=self.max_retrieved_docs
                    ):
                        # Filter out 'done' events to prevent premature termination by UI
                        # We will send 'done' explicitly after saving history
                        if event.get("type") == "done":
                            continue
                            
                        if event.get("type") == "prompt":
                            # Enhance adapter yielded the full prompt context
                            structured_prompt = event.get("content")
                            # We can also get the hash if available
                            metadata = event.get("metadata", {})
                            message_hash = metadata.get("message_hash")
                            
                            if not message_hash:
                                # Fallback hash generation
                                import hashlib
                                message_hash = hashlib.md5(user_message.encode()).hexdigest()
                            
                            # Store in controller persistence
                            self.message_prompts[message_hash] = structured_prompt
                            logger.info(f"Captured structured prompt for message hash: {message_hash[:8]}...")
                            
                            # Forward event to UI
                            yield event
                            
                        else:
                            yield event
                            
                        if event.get("type") == "token":
                            response_text += event.get("content", "")

                except Exception as e:
                    logger.error(f"Enhanced LLM adapter streaming failed: {e}")
                    # Fallback to basic streaming
                    yield {"type": "status", "content": "Processing with basic generation...", "stage": "fallback"}
                    async for event in self.llm_adapter.astream_response(user_message):
                        if event.get("type") == "done":
                            continue
                        yield event
                        if event.get("type") == "token":
                            response_text += event.get("content", "")
            else:
                # Use basic LLM streaming
                yield {"type": "status", "content": "Processing with basic generation...", "stage": "basic"}
                async for event in self.llm_adapter.astream_response(user_message):
                    if event.get("type") == "done":
                        continue
                    yield event
                    if event.get("type") == "token":
                        response_text += event.get("content", "")

            # Save assistant response when done
            if response_text:
                # Process HTML content to download and replace external images
                processed_content = self.process_html_images(response_text)

                # NEW: Get intent and topic metadata from last events
                message_metadata = {
                    'intent': None,  # Will be filled from adapter metadata
                    'topic': None    # Will be filled from adapter metadata
                }

                self.conversation_messages.append({
                    'role': 'assistant',
                    'content': processed_content,
                    'timestamp': datetime.now().isoformat(),
                    'metadata': message_metadata  # NEW: Add metadata field
                })

                # Add to legacy history (use original response_text for history)
                self._add_to_history('assistant', response_text)

                # Record conversation in memory
                if self.memory_adapter and self.memory_adapter.is_available():
                    try:
                        await self.memory_adapter.arecord_conversation(user_message, response_text)
                    except Exception as e:
                        logger.warning(f"Failed to record conversation in legacy memory: {e}")

                logger.info(f"Streaming response completed: {response_text[:100]}...")

                # Autosave conversation after each message
                try:
                    self.autosave_conversation("./data/conversation.json")
                except Exception as e:
                    logger.warning(f"Failed to autosave conversation: {e}")

            # Explicitly signal completion to the UI after everything is saved
            yield {"type": "done"}

        except Exception as e:
            logger.error(f"Unexpected error in streaming: {e}")
            yield {"type": "error", "content": f"Unexpected error: {str(e)}"}

    async def process_message(self, user_message: str) -> str:
        """
        Process a user message and return the complete response.
        This method maintains backward compatibility by collecting the streaming response.
        """
        tokens = []
        async for event in self.stream_message(user_message):
            if event.get("type") == "token":
                tokens.append(event.get("content", ""))
            elif event.get("type") == "error":
                return f"Error: {event.get('content', 'Unknown error')}"
        return "".join(tokens)

    def process_message_sync(self, user_message: str) -> str:
        """
        Synchronous version of process_message.
        Used for compatibility with UI components that don't support async.
        """
        # Wrapper for async process_message running in sync
        return asyncio.run(self.process_message(user_message))

    def _build_prompt(self, current_message: str) -> tuple[str, str]:
        """
        Build the complete prompt from system prompt, history, retrieved documents, memories, and current message.

        Args:
            current_message: The current user message

        Returns:
            Tuple of (complete_prompt_string, structured_prompt_with_xml_tags)
        """
        prompt_parts = []
        structured_parts = []

        # Add system prompt
        prompt_parts.append(f"System: {self.system_prompt}")
        structured_parts.append(f"<system_prompt>\n{self.system_prompt}\n</system_prompt>")

        # Retrieve relevant memories if enabled
        retrieved_memories = []
        if self.memory_enabled and self.memory_adapter and self.memory_adapter.is_available():
            try:
                retrieved_memories = self.memory_adapter.search_memory(
                    current_message,
                    limit=self.max_retrieved_memories
                )
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")

        # Retrieve relevant documents if enabled
        retrieved_docs = []
        if self.retrieval_enabled:
            try:
                # Use hybrid search if enabled, otherwise semantic search
                if hasattr(self, 'hybrid_retrieval_enabled') and self.hybrid_retrieval_enabled:
                    retrieved_docs = self.file_ingestor.hybrid_search(
                        current_message,
                        k=self.max_retrieved_docs,
                        semantic_weight=getattr(self, 'semantic_weight', 0.7),
                        keyword_weight=getattr(self, 'keyword_weight', 0.3)
                    )
                else:
                    retrieved_docs = self.file_ingestor.search_similar(
                        current_message,
                        k=self.max_retrieved_docs,
                        score_threshold=0.1  # Only include reasonably relevant docs
                    )
            except ValueError as e:
                if "OpenAI API key" in str(e):
                    logger.warning("Document retrieval disabled: OpenAI API key not configured")
                    self.retrieval_enabled = False  # Temporarily disable until configured
                else:
                    logger.warning(f"Document retrieval failed: {e}")
            except Exception as e:
                logger.warning(f"Document retrieval failed: {e}")

        # Add retrieved memories if any
        if retrieved_memories:
            prompt_parts.append("\nRelevant memories from our past conversations:")
            memory_lines = []
            for i, memory in enumerate(retrieved_memories, 1):
                content = memory.get('content', '')
                mem_type = memory.get('metadata', {}).get('type', 'unknown')
                score = memory.get('score', 0.0)
                memory_line = f"[{i}] ({mem_type}, relevance: {score:.2f}): {content}"
                prompt_parts.append(memory_line)
                memory_lines.append(memory_line)
            prompt_parts.append("")  # Empty line after memories
            structured_parts.append(f"<relevant_memories>\n{chr(10).join(memory_lines)}\n</relevant_memories>")

        # Add retrieved documents if any
        if retrieved_docs:
            prompt_parts.append("\nRelevant Information from your documents:")
            doc_lines = []
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.get('file_name', 'Unknown source')
                content = doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content']
                doc_line = f"[{i}] From {source}: {content}"
                prompt_parts.append(doc_line)
                doc_lines.append(doc_line)
            prompt_parts.append("")  # Empty line after documents
            structured_parts.append(f"<relevant_documents>\n{chr(10).join(doc_lines)}\n</relevant_documents>")

        # Add recent conversation history (excluding current message)
        recent_history = self.conversation_history[-self.max_history_length:]
        history_lines = []
        for msg in recent_history[:-1]:  # Exclude the current message we just added
            role = "User" if msg['role'] == 'user' else "Assistant"
            timestamp = msg.get('timestamp', '')
            content = msg['content']
            history_line = f"{role} ({timestamp}): {content}"
            prompt_parts.append(history_line)
            history_lines.append(history_line)

        if history_lines:
            structured_parts.append(f"<message_history>\n{chr(10).join(history_lines)}\n</message_history>")

        # Add current message
        prompt_parts.append(f"User: {current_message}")
        structured_parts.append(f"<user_input>\n{current_message}\n</user_input>")

        # Add instruction for assistant
        instruction_parts = []
        if retrieved_memories:
            instruction_parts.append("Use relevant memories from past conversations when appropriate")
        if retrieved_docs:
            instruction_parts.append("Use the relevant information from documents when appropriate. Cite sources using [number] notation when referencing specific information")

        instructions = '; '.join(instruction_parts) if instruction_parts else ""
        prompt_parts.append(f"Assistant: {instructions}.")
        structured_parts.append(f"<assistant_instructions>\n{instructions}\n</assistant_instructions>")

        # Build final strings
        final_prompt = "\n\n".join(prompt_parts)
        structured_prompt = "\n\n".join(structured_parts)

        return final_prompt, structured_prompt

    async def _process_legacy_message(self, user_message: str) -> str:
        """
        Process a message using the legacy (non-structured) prompt building.

        Args:
            user_message: User's message

        Returns:
            Generated response
        """
        try:
            prompt, structured_prompt = self._build_prompt(user_message)
            import hashlib
            message_hash = hashlib.md5(user_message.encode()).hexdigest()
            self.message_prompts[message_hash] = structured_prompt
            response = await self.llm_adapter.agenerate_response(prompt)
            return response
        except Exception as e:
             logger.error(f"Legacy processing failed: {e}")
             return "I'm sorry, I encountered an error processing your request."

    def _add_to_history(self, role: str, content: str):
        """
        Add a message to the conversation history.

        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        self.conversation_history.append(message)

        # Trim history if it gets too long
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the current conversation history.

        Returns:
            List of message dictionaries
        """
        return self.conversation_history.copy()

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
        # Clear conversation messages
        self.conversation_messages.clear()
        # Clear message prompts
        self.message_prompts.clear()
        # Clear drafts
        self.drafts.clear()
        self.archived_drafts.clear()
        self.current_draft_id = None
        logger.info("Conversation history cleared")

    def save_conversation_history(self, file_path: str):
        """
        Save conversation history to a file.

        Args:
            file_path: Path to save the conversation
        """
        try:
            # Use conversation messages for saving
            messages = self.conversation_messages.copy()

            conversation_data = {
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'message_count': len(messages),
                    'version': '0.2.0'
                },
                'messages': messages,
                'message_prompts': self.message_prompts.copy(),  # Include prompts for persistence
                'drafts': self.drafts.copy(),
                'archived_drafts': self.archived_drafts.copy()
            }

            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Conversation history saved to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")
            raise

    def load_conversation_history(self, file_path: str):
        """
        Load conversation history from a file.

        Args:
            file_path: Path to load the conversation from
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            messages = data.get('messages', [])
            if not messages:
                logger.warning("No messages found in conversation file")
                return

            # Clear existing history
            self.conversation_history.clear()
            self.conversation_messages.clear()
            self.message_prompts.clear()  # Clear existing prompts
            self.drafts.clear()
            self.archived_drafts.clear()
            self.current_draft_id = None

            # Load messages into both systems
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')

                # Add to conversation messages
                self.conversation_messages.append({
                    'role': role,
                    'content': content,
                    'timestamp': msg.get('timestamp', datetime.now().isoformat())
                })

                # Add to legacy history
                self._add_to_history(role, content)

            # Restore message prompts if available
            message_prompts = data.get('message_prompts', {})
            if message_prompts:
                self.message_prompts.update(message_prompts)
                logger.info(f"Restored {len(message_prompts)} message prompts")

            # Restore drafts if available
            drafts = data.get('drafts', [])
            if drafts:
                self.drafts = drafts.copy()
                # Set current draft to the first one if it exists
                if self.drafts:
                    self.current_draft_id = self.drafts[0].get('id')
                logger.info(f"Restored {len(drafts)} active drafts")

            archived_drafts = data.get('archived_drafts', [])
            if archived_drafts:
                self.archived_drafts = archived_drafts.copy()
                logger.info(f"Restored {len(archived_drafts)} archived drafts")

            logger.info(f"Conversation history loaded from {file_path} ({len(messages)} messages)")

        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")
            raise

    def get_conversation_history_from_memory(self) -> List[Dict[str, Any]]:
        """
        Get conversation history from conversation messages in our format.

        Returns:
            List of message dictionaries
        """
        return self.conversation_messages.copy()

    def get_messages_page(self, page: int, limit: int) -> List[Dict[str, Any]]:
        """
        Get a page of conversation messages for lazy loading.

        Args:
            page: Page number (0-based, where 0 is the most recent messages)
            limit: Number of messages per page

        Returns:
            List of message dictionaries for the requested page
        """
        if not self.conversation_messages:
            return []

        # Messages are stored in chronological order (oldest first)
        # Return messages in chronological order for correct UI display
        # This ensures newest messages appear at the bottom of the chat
        total_messages = len(self.conversation_messages)
        start_index = total_messages - (page + 1) * limit
        end_index = total_messages - page * limit

        # Handle edge cases
        if start_index < 0:
            start_index = 0
        if end_index < 0:
            end_index = 0

        # Return messages in chronological order (oldest first)
        messages = self.conversation_messages[start_index:end_index]
        return messages

    def get_messages_from_index(self, start_index: int, count: int) -> List[Dict[str, Any]]:
        """
        Get messages starting from a specific index.

        Args:
            start_index: Starting index in conversation_messages (0 = oldest)
            count: Number of messages to retrieve

        Returns:
            List of message dictionaries
        """
        if not self.conversation_messages or start_index >= len(self.conversation_messages):
            return []

        end_index = min(start_index + count, len(self.conversation_messages))
        return self.conversation_messages[start_index:end_index]

    def get_total_message_count(self) -> int:
        """
        Get the total number of messages in conversation history.

        Returns:
            Total message count
        """
        return len(self.conversation_messages)

    def set_system_prompt(self, prompt: str):
        """
        Update the system prompt.

        Args:
            prompt: New system prompt
        """
        self.system_prompt = prompt
        logger.info("System prompt updated")

    def get_system_prompt(self) -> str:
        """Get the current system prompt."""
        return self.system_prompt

    def configure_llm(self, provider: str, **config):
        """
        Reconfigure the LLM adapter.

        Args:
            provider: LLM provider name
            **config: Provider-specific configuration
        """
        # Get intent analysis settings if available
        intent_config = config.pop('intent_config', None)

        # Resolve combo if combo_id is provided instead of direct provider/model
        combo_id = config.get('combo')
        if combo_id and self.settings_manager:
            combo_data = self.settings_manager.resolve_combo(combo_id)
            if combo_data:
                provider = combo_data['provider']
                config['model'] = combo_data['model']
                logger.info(f"Resolved combo '{combo_id}' to provider: {provider}, model: {combo_data['model']}")
            else:
                logger.warning(f"Could not resolve combo '{combo_id}', using fallback configuration")

        # If settings manager is available, we can fetch missing API keys
        if self.settings_manager:
            # Get API key for the provider
            api_key = self.settings_manager.get_api_key(provider)
            if api_key:
                config['api_key'] = api_key

        # Handle intent config combo resolution
        if intent_config and intent_config.get('enabled'):
            logger.info(f"Processing intent_config: {intent_config}")
            intent_combo_id = intent_config.get('combo')
            if intent_combo_id and self.settings_manager:
                intent_combo_data = self.settings_manager.resolve_combo(intent_combo_id)
                if intent_combo_data:
                    intent_config['provider'] = intent_combo_data['provider']
                    intent_config['model'] = intent_combo_data['model']
                    # Get API key for intent provider
                    intent_api_key = self.settings_manager.get_api_key(intent_combo_data['provider'])
                    if intent_api_key:
                        intent_config['api_key'] = intent_api_key
                    logger.info(f"Resolved intent combo '{intent_combo_id}' to provider: {intent_combo_data['provider']}, model: {intent_combo_data['model']}")
                else:
                    logger.warning(f"Could not resolve intent combo '{intent_combo_id}'")

        # Check if reconfiguration is actually needed
        if (hasattr(self, '_current_llm_provider') and
            hasattr(self, '_current_llm_config') and
            self._current_llm_provider == provider and
            self._current_llm_config == config and
            getattr(self, '_current_intent_config', None) == intent_config):
            logger.info(f"LLM configuration unchanged (provider: {provider})")
            return

        # Preserve tool configuration before recreating adapter
        tool_config = None
        if hasattr(self.llm_adapter, '_assistant_tools') and self.llm_adapter._assistant_tools:
            # Extract the web search config from the assistant tools and convert back to settings format
            web_search_config = self.llm_adapter._assistant_tools.web_search_config
            if web_search_config:
                tool_config = {
                    'tavily': web_search_config.get('tavily_api_key', ''),
                    'exa': web_search_config.get('exa_api_key', ''),
                    'jina': web_search_config.get('jina_api_key', '')
                }
                # Only keep non-empty keys
                tool_config = {k: v for k, v in tool_config.items() if v}

        try:
            # Use EnhancedLLMAdapter with intent config
            self.llm_adapter = EnhancedLLMAdapter(
                provider=provider,
                intent_config=intent_config,
                **config
            )

            # Restore tool configuration if it existed
            if tool_config:
                self.configure_tools(tool_config)
                logger.info("Restored tool configuration after LLM reconfiguration")

            # Store current configuration for future comparison
            self._current_llm_provider = provider
            self._current_llm_config = config.copy() if config else {}
            self._current_intent_config = intent_config.copy() if intent_config else None
            logger.info(f"LLM reconfigured to provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to configure LLM: {e}")
            raise

    def validate_llm_config(self) -> bool:
        """
        Validate the current LLM configuration.

        Returns:
            True if configuration is valid
        """
        return self.llm_adapter.validate_config()

    def get_available_models(self) -> List[str]:
        """
        Get available models for the current LLM provider.

        Returns:
            List of model names
        """
        return self.llm_adapter.get_available_models()

    def create_custom_chain(self, prompt_template: PromptTemplate) -> Runnable:
        """
        Create a custom LangChain runnable with the given prompt template.

        Args:
            prompt_template: Prompt template to use

        Returns:
            Runnable chain
        """
        return self.llm_adapter.create_chain(prompt_template)

    # File management methods

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a file into the vector store.

        Args:
            file_path: Path to the file to ingest

        Returns:
            Ingestion result
        """
        try:
            result = self.file_ingestor.ingest_file(file_path)
            logger.info(f"File ingestion completed: {result}")
            return result
        except Exception as e:
            logger.error(f"File ingestion failed: {e}")
            return {
                'status': 'error',
                'file_path': file_path,
                'error': str(e)
            }

    def get_ingested_files(self) -> List[Dict[str, Any]]:
        """
        Get list of all ingested files.

        Returns:
            List of file metadata
        """
        return self.file_ingestor.get_ingested_files()

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from the vector store.

        Args:
            file_path: Path of file to delete

        Returns:
            True if deletion was successful
        """
        return self.file_ingestor.delete_file(file_path)

    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents in the vector store.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of search results
        """
        return self.file_ingestor.search_similar(query, k=k)

    def toggle_retrieval(self, enabled: bool):
        """
        Enable or disable document retrieval in prompts.

        Args:
            enabled: Whether to enable retrieval
        """
        self.retrieval_enabled = enabled
        logger.info(f"Document retrieval {'enabled' if enabled else 'disabled'}")

    def set_max_retrieved_docs(self, max_docs: int):
        """
        Set the maximum number of documents to retrieve per query.

        Args:
            max_docs: Maximum number of documents
        """
        self.max_retrieved_docs = max(max_docs, 1)
        logger.info(f"Max retrieved documents set to {max_docs}")

    def set_hybrid_retrieval(self, enabled: bool, semantic_weight: float = 0.7, keyword_weight: float = 0.3):
        """
        Configure hybrid retrieval settings.

        Args:
            enabled: Whether to enable hybrid retrieval
            semantic_weight: Weight for semantic similarity (0.0 to 1.0)
            keyword_weight: Weight for keyword matching (0.0 to 1.0)
        """
        self.hybrid_retrieval_enabled = enabled
        self.semantic_weight = max(0.0, min(1.0, semantic_weight))
        self.keyword_weight = max(0.0, min(1.0, keyword_weight))

        # Ensure weights sum to 1.0
        total_weight = self.semantic_weight + self.keyword_weight
        if total_weight > 0:
            self.semantic_weight /= total_weight
            self.keyword_weight /= total_weight

        logger.info(f"Hybrid retrieval {'enabled' if enabled else 'disabled'} "
                   f"(semantic: {self.semantic_weight:.2f}, keyword: {self.keyword_weight:.2f})")

    # Memory management methods

    def toggle_memory(self, enabled: bool):
        """
        Enable or disable memory in prompts.

        Args:
            enabled: Whether to enable memory
        """
        self.memory_enabled = enabled
        
        # Persist setting if manager is available
        if self.settings_manager:
             try:
                 self.settings_manager.set_setting(enabled, 'memory', 'enabled')
             except Exception as e:
                 logger.error(f"Failed to save memory enabled setting: {e}")
                 
        logger.info(f"Memory {'enabled' if enabled else 'disabled'}")

    def set_max_retrieved_memories(self, max_memories: int):
        """
        Set the maximum number of memories to retrieve per query.

        Args:
            max_memories: Maximum number of memories
        """
        self.max_retrieved_memories = max(max_memories, 1)
        logger.info(f"Max retrieved memories set to {max_memories}")

    def search_memory(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for memories.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of relevant memories
        """
        if not self.memory_adapter or not self.memory_adapter.is_available():
            return []

        return self.memory_adapter.search_memory(query, limit=limit or self.max_retrieved_memories)

    def store_memory_fact(self, fact: str, fact_type: str = "semantic", importance: str = "normal",
                          tags: Optional[List[str]] = None):
        """
        Store a fact in memory.

        Args:
            fact: The fact to store
            fact_type: Type of fact (semantic, episodic, procedural)
            importance: Importance level (low, normal, high, critical)
            tags: List of tags for categorization
        """
        if not self.memory_adapter or not self.memory_adapter.is_available():
            logger.warning("Memory system not available")
            return

        self.memory_adapter.store_fact(fact, fact_type, importance, tags)

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.

        Returns:
            Memory statistics
        """
        stats = self.memory_adapter.get_memory_stats()
        # Override enabled state with controller's state (which is the effective state)
        # The adapter might be 'enabled' (initialized), but we might have disabled using it in prompts
        stats['enabled'] = self.memory_enabled
        return stats

    def clear_memory(self, memory_types: Optional[List[str]] = None,
                    tags: Optional[List[str]] = None):
        """
        Clear memories based on filters.

        Args:
            memory_types: Types of memories to clear
            tags: Tags to filter by
        """
        if not self.memory_adapter or not self.memory_adapter.is_available():
            logger.warning("Memory system not available")
            return

        self.memory_adapter.clear_memory(memory_types, tags)

    def export_memory(self, file_path: str):
        """
        Export memory data to a file.

        Args:
            file_path: Path to export file
        """
        if not self.memory_adapter or not self.memory_adapter.is_available():
            logger.warning("Memory system not available")
            return

        self.memory_adapter.export_memory(file_path)

    def import_memory(self, file_path: str):
        """
        Import memory data from a file.

        Args:
            file_path: Path to import file
        """
        if not self.memory_adapter or not self.memory_adapter.is_available():
            logger.warning("Memory system not available")
            return

        self.memory_adapter.import_memory(file_path)

    # Audio management methods

    def speak_text(self, text: str, save_to_file: bool = False) -> Optional[str]:
        """
        Convert text to speech.

        Args:
            text: Text to convert to speech
            save_to_file: Whether to save audio to a temporary file

        Returns:
            Path to audio file if save_to_file is True, None otherwise
        """
        return self.audio_provider.speak_text(text, save_to_file)

    def play_audio_file(self, file_path: str):
        """
        Play an audio file asynchronously.
        
        Args:
            file_path: Path to the audio file
        """
        if hasattr(self.audio_provider, 'play_audio_file'):
            self.audio_provider.play_audio_file(file_path)

    def pause_audio(self):
        """Pause audio playback."""
        if hasattr(self.audio_provider, 'pause_audio'):
            self.audio_provider.pause_audio()

    def resume_audio(self):
        """Resume audio playback."""
        if hasattr(self.audio_provider, 'resume_audio'):
            self.audio_provider.resume_audio()

    def stop_audio(self):
        """Stop audio playback."""
        if hasattr(self.audio_provider, 'stop_audio'):
            self.audio_provider.stop_audio()

    def seek_audio(self, offset_ms: int):
        """Seek audio playback by offset."""
        if hasattr(self.audio_provider, 'seek_audio'):
            self.audio_provider.seek_audio(offset_ms)
            
    def listen_for_speech(self, timeout: Optional[float] = 10.0) -> Optional[str]:
        """
        Listen for speech and convert to text.

        Args:
            timeout: Maximum time to listen in seconds

        Returns:
            Transcribed text or None if failed
        """
        return self.audio_provider.listen_for_speech(timeout)

    def is_asr_listening(self) -> bool:
        """
        Check if ASR is currently listening.

        Returns:
            True if listening, False otherwise
        """
        return self.audio_provider.is_asr_listening()

    def get_available_tts_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of available TTS voices.

        Returns:
            List of voice information dictionaries
        """
        return self.audio_provider.get_available_voices()

    def set_tts_voice(self, voice_name: str):
        """
        Set the TTS voice.

        Args:
            voice_name: Name of the voice to use
        """
        self.audio_provider.set_tts_voice(voice_name)

    def set_tts_speed(self, speed: float):
        """
        Set the TTS speaking speed.

        Args:
            speed: Speed multiplier (0.5 to 2.0)
        """
        self.audio_provider.set_tts_speed(speed)

    def configure_audio(self, tts_enabled: bool = True, asr_enabled: bool = True,
                       voice: str = "alloy", speed: float = 1.0, language: str = "en-US"):
        """
        Configure audio settings.

        Args:
            tts_enabled: Whether TTS is enabled
            asr_enabled: Whether ASR is enabled
            voice: Voice name for TTS
            speed: Speaking speed for TTS
            language: Language for ASR
        """
        self.audio_provider.configure_tts(tts_enabled, voice, speed)
        self.audio_provider.configure_asr(asr_enabled, language)

    def get_audio_settings(self) -> Dict[str, Any]:
        """
        Get current audio settings.

        Returns:
            Dictionary of audio settings
        """
        return self.audio_provider.get_audio_settings()

    def cleanup_audio_files(self, max_age_hours: int = 24):
        """
        Clean up old temporary audio files.

        Args:
            max_age_hours: Maximum age of files to keep
        """
        self.audio_provider.cleanup_temp_files(max_age_hours)

    def update_api_keys(self, openai_api_key: Optional[str] = None, openrouter_api_key: Optional[str] = None):
        """
        Update API keys for providers that need them.

        Args:
            openai_api_key: OpenAI API key
            openrouter_api_key: OpenRouter API key
        """
        updated = False

        # Update OpenAI API key
        if openai_api_key is not None:
            self.openai_api_key = openai_api_key
            # Set environment variable for components that expect it
            if openai_api_key:
                import os
                os.environ['OPENAI_API_KEY'] = openai_api_key
                logger.info("OPENAI_API_KEY environment variable set")
            updated = True

        # Update memory adapter if API key is available and different from current
        if self.openai_api_key and (not self.memory_adapter or not hasattr(self.memory_adapter, 'openai_api_key') or self.memory_adapter.openai_api_key != self.openai_api_key):
            try:
                # Create or re-initialize memory adapter with current API key
                self.memory_adapter = MemoryAdapter(openai_api_key=self.openai_api_key)
                logger.info("Memory adapter initialized/re-initialized with API key")
                updated = True
            except Exception as e:
                logger.warning(f"Failed to initialize memory adapter: {e}")

        # Update audio provider if API key changed
        if self.audio_provider and self.openai_api_key:
            try:
                # Always re-initialize audio provider with current API key
                self.audio_provider = AudioProvider(openai_api_key=self.openai_api_key)
                logger.info("Audio provider re-initialized with API key")
                updated = True
            except Exception as e:
                logger.warning(f"Failed to re-initialize audio provider: {e}")

        if updated:
            logger.info("API keys updated for providers")
        return updated

    def reconfigure_file_ingestor(self, embedding_provider: str = "openrouter",
                                 embedding_model: str = "text-embedding-3-small",
                                 api_key: Optional[str] = None,
                                 vector_provider: str = "faiss"):
        """
        Reconfigure the file ingestor with new settings.

        Args:
            embedding_provider: Embedding provider ('openai' or 'openrouter')
            embedding_model: Embedding model to use
            api_key: API key for the embedding provider
            vector_provider: Vector store provider ('faiss' or 'chroma')
        """
        try:
            vector_store_path = "./data/vector_store"
            self.file_ingestor = FileIngestor(
                vector_store_path=vector_store_path,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                api_key=api_key,
                vector_provider=vector_provider
            )
            logger.info(f"File ingestor reconfigured with {embedding_provider} embeddings and {vector_provider} vector store")
        except Exception as e:
            logger.error(f"Failed to reconfigure file ingestor: {e}")
            raise

    def get_prompt_for_message(self, message_data: Dict[str, Any]) -> Optional[str]:
        logger.debug(f"Calling get_prompt_for_message with message_data: {message_data}")
        """
        Get the structured prompt that was used for a specific user message.

        Args:
            message_data: Message data dictionary containing the user message

        Returns:
            Structured prompt string with XML tags, or None if not found
        """
        if message_data.get('role') != 'user':
            logger.warning("get_prompt_for_message called with non-user message")
            return None

        message_content = message_data.get('content', '')
        if not message_content:
            logger.warning("get_prompt_for_message called with empty message content")
            return None

        import hashlib
        message_hash = hashlib.md5(message_content.encode()).hexdigest()

        # First check controller's prompt storage
        prompt = self.message_prompts.get(message_hash)
        if prompt:
            logger.info(f"Found prompt for message hash: {message_hash[:8]}... (in controller)")
            return prompt

        # If not found, check adapter's prompt storage if it's EnhancedLLMAdapter
        if isinstance(self.llm_adapter, EnhancedLLMAdapter):
            adapter_prompt = self.llm_adapter.message_prompts.get(message_hash)
            if adapter_prompt:
                logger.info(f"Found prompt for message hash: {message_hash[:8]}... (in adapter)")
                return adapter_prompt

        logger.warning(f"No prompt found for message hash: {message_hash[:8]}... (checked both controller and adapter)")
        return None

    # Tool configuration methods

    def configure_tools(self, web_search_config: Optional[Dict[str, str]] = None):
        """
        Configure tools for the assistant.

        Args:
            web_search_config: Configuration for web search tools
                Keys: 'tavily', 'exa', 'jina' (will be mapped to *_api_key format)
        """
        try:
            # Map the keys from settings format to AssistantTools expected format
            if web_search_config:
                mapped_config = {
                    'tavily_api_key': web_search_config.get('tavily', ''),
                    'exa_api_key': web_search_config.get('exa', ''),
                    'jina_api_key': web_search_config.get('jina', '')
                }
            else:
                mapped_config = None

            self.llm_adapter.configure_tools(mapped_config)
            logger.info("Tools configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure tools: {e}")
            raise

    def create_tool_set(self, name: str, tool_names: Optional[List[str]] = None) -> bool:
        """
        Create a new tool set with specified tools.

        Args:
            name: Name of the tool set
            tool_names: List of tool names to include (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create the tool set
            tool_set = self.llm_adapter.create_tool_set(name)

            # Add specified tools if provided
            if tool_names:
                available_tools = self.get_available_tools()
                for tool_name in tool_names:
                    if tool_name in available_tools:
                        # Find the tool and add to set
                        tool = next((t for t in self.llm_adapter._tools if t.name == tool_name), None)
                        if tool:
                            tool_set.add_tool(tool)

            logger.info(f"Created tool set '{name}' with {len(tool_names or [])} tools")
            return True
        except Exception as e:
            logger.error(f"Failed to create tool set: {e}")
            return False

    def configure_tool_set(self, name: str, web_search_config: Optional[Dict[str, str]] = None,
                          custom_tools: Optional[List[str]] = None) -> bool:
        """
        Configure a specific tool set.

        Args:
            name: Name of the tool set to configure
            web_search_config: Web search configuration (optional)
            custom_tools: List of custom tool names to add (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get or create the tool set
            tool_set = self.llm_adapter.get_tool_set(name)
            if not tool_set:
                tool_set = self.llm_adapter.create_tool_set(name)

            # Clear existing tools
            tool_set.clear()

            # Add web search tools if configured
            if web_search_config:
                web_tools = self._create_web_search_tools(web_search_config)
                for tool in web_tools:
                    tool_set.add_tool(tool)

            # Add custom tools
            if custom_tools:
                available_tools = self.get_available_tools()
                for tool_name in custom_tools:
                    if tool_name in available_tools:
                        tool = next((t for t in self.llm_adapter._tools if t.name == tool_name), None)
                        if tool:
                            tool_set.add_tool(tool)

            logger.info(f"Configured tool set '{name}' with {len(tool_set.tools)} tools")
            return True
        except Exception as e:
            logger.error(f"Failed to configure tool set: {e}")
            return False

    def set_active_tool_set(self, name: str) -> bool:
        """
        Set the active tool set for the current LLM adapter.

        Args:
            name: Name of the tool set to activate

        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.llm_adapter.set_tool_set(name)
            if success:
                logger.info(f"Activated tool set: {name}")
            return success
        except Exception as e:
            logger.error(f"Failed to activate tool set: {e}")
            return False

    def get_active_tool_set(self) -> Optional[str]:
        """
        Get the name of the currently active tool set.

        Returns:
            Name of active tool set or None if not set
        """
        try:
            current_set = self.llm_adapter.get_current_tool_set()
            return current_set.name if current_set else None
        except Exception as e:
            logger.error(f"Failed to get active tool set: {e}")
            return None

    def list_tool_sets(self) -> List[str]:
        """
        List all available tool sets.

        Returns:
            List of tool set names
        """
        try:
            return self.llm_adapter.list_tool_sets()
        except Exception as e:
            logger.error(f"Failed to list tool sets: {e}")
            return []

    def get_tool_set_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific tool set.

        Args:
            name: Name of the tool set

        Returns:
            Dictionary with tool set information or None if not found
        """
        try:
            tool_set = self.llm_adapter.get_tool_set(name)
            if not tool_set:
                return None

            return {
                'name': tool_set.name,
                'tool_count': len(tool_set.tools),
                'tool_names': list(tool_set.tool_names),
                'is_active': self.get_active_tool_set() == name
            }
        except Exception as e:
            logger.error(f"Failed to get tool set info: {e}")
            return None

    def _create_web_search_tools(self, web_search_config: Dict[str, str]) -> List:
        """
        Create web search tools from configuration.

        Args:
            web_search_config: Web search configuration

        Returns:
            List of web search tool instances
        """
        try:
            from ..langchain_adapters.tools import AssistantTools
            assistant_tools = AssistantTools(web_search_config)
            return assistant_tools.get_all_tools()
        except Exception as e:
            logger.error(f"Failed to create web search tools: {e}")
            return []

    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names.

        Returns:
            List of tool names
        """
        return self.llm_adapter.get_tool_names()

    def update_tool_config(self, web_search_config: Dict[str, str]):
        """
        Update tool configuration.

        Args:
            web_search_config: New web search configuration
        """
        try:
            self.llm_adapter.update_tool_config(web_search_config)
            # Regenerate system prompt with updated tools
            available_tools = self.get_available_tools()
            self.system_prompt = self._get_default_system_prompt(available_tools)
            logger.info("Tool configuration updated")
        except Exception as e:
            logger.error(f"Failed to update tool configuration: {e}")
            raise

    def process_html_images(self, html_content: str) -> str:
        """
        Process HTML content to download external images, generate thumbnails,
        and replace URLs with local paths for gallery display.

        Args:
            html_content: HTML content containing potentially external image URLs

        Returns:
            HTML content with local image URLs and data attributes for full images
        """
        try:
            # Find all img tags with src attributes
            img_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
            matches = re.findall(img_pattern, html_content, re.IGNORECASE)

            if not matches:
                return html_content

            # Create images directory if it doesn't exist
            images_dir = Path("src/ui/assets/images")
            images_dir.mkdir(parents=True, exist_ok=True)

            # Import PIL for thumbnail generation
            try:
                from PIL import Image
            except ImportError:
                logger.warning("PIL not available, falling back to basic image processing")
                return self._fallback_process_html_images(html_content)

            # Process each image URL
            processed_urls = {}
            for url in matches:
                if url in processed_urls:
                    continue  # Already processed this URL

                try:
                    # Check if it's an external URL (not already local)
                    if url.startswith(('http://', 'https://')):
                        # Generate filename from URL hash to avoid conflicts
                        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                        # Try to get file extension from URL
                        url_path = urllib.parse.urlparse(url).path
                        ext = Path(url_path).suffix.lower()
                        if not ext or ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
                            ext = '.jpg'  # Default extension

                        filename = f"{url_hash}{ext}"
                        local_path = images_dir / filename
                        thumbnail_filename = f"{url_hash}_thumb.jpg"
                        thumbnail_path = images_dir / thumbnail_filename

                        # Check if we already downloaded this image
                        if not local_path.exists():
                            logger.info(f"Downloading image: {url}")
                            # Download the image
                            with urllib.request.urlopen(url, timeout=10) as response:
                                if response.status == 200:
                                    image_data = response.read()
                                    with open(local_path, 'wb') as f:
                                        f.write(image_data)
                                    logger.info(f"Downloaded image to: {local_path}")

                                    # Generate thumbnail
                                    if self._generate_image_thumbnail(local_path, thumbnail_path):
                                        logger.debug(f"Generated thumbnail: {thumbnail_path}")
                                    else:
                                        logger.warning(f"Failed to generate thumbnail for: {local_path}")
                                else:
                                    logger.warning(f"Failed to download image {url}: HTTP {response.status}")
                                    processed_urls[url] = url  # Keep original URL
                                    continue
                        else:
                            logger.debug(f"Image already exists: {local_path}")
                            # Generate thumbnail if it doesn't exist
                            if not thumbnail_path.exists():
                                if self._generate_image_thumbnail(local_path, thumbnail_path):
                                    logger.debug(f"Generated missing thumbnail: {thumbnail_path}")

                        # Use thumbnail URL for display, full image for modal
                        thumbnail_url = f"file://{thumbnail_path.resolve()}"
                        full_image_url = f"file://{local_path.resolve()}"

                        # Store both URLs (thumbnail for display, full for modal)
                        processed_urls[url] = {
                            'thumbnail': thumbnail_url,
                            'full': full_image_url
                        }
                    else:
                        # Already local or relative URL, keep as is
                        processed_urls[url] = url

                except Exception as e:
                    logger.warning(f"Failed to process image URL {url}: {e}")
                    processed_urls[url] = url  # Keep original URL on error

            # Replace URLs in HTML - now we use thumbnail URLs for display
            processed_html = html_content
            for original_url, url_data in processed_urls.items():
                if isinstance(url_data, dict):
                    # Use thumbnail URL for display, store full URL in data attribute
                    thumbnail_url = url_data['thumbnail']
                    full_url = url_data['full']
                    escaped_url = re.escape(original_url)
                    processed_html = re.sub(
                        rf'<img([^>]+)src=["\']{escaped_url}["\']([^>]*)>',
                        f'<img\\1src="{thumbnail_url}" data-full-src="{full_url}"\\2>',
                        processed_html,
                        flags=re.IGNORECASE
                    )
                else:
                    # Fallback for non-processed URLs
                    escaped_url = re.escape(original_url)
                    processed_html = re.sub(
                        rf'src=["\']{escaped_url}["\']',
                        f'src="{url_data}"',
                        processed_html,
                        flags=re.IGNORECASE
                    )

            return processed_html

        except Exception as e:
            logger.error(f"Error processing HTML images: {e}")
            return html_content  # Return original content on error

    def _generate_image_thumbnail(self, image_path: Path, thumbnail_path: Path, size: tuple = (200, 200)) -> bool:
        """
        Generate a thumbnail for an image.

        Args:
            image_path: Path to the original image
            thumbnail_path: Path where to save the thumbnail
            size: Maximum size for the thumbnail (width, height)

        Returns:
            True if thumbnail was generated successfully, False otherwise
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (for JPEG compatibility)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')

                # Create thumbnail maintaining aspect ratio
                img.thumbnail(size, Image.Resampling.LANCZOS)

                # Save thumbnail
                thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
                return True
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail for {image_path}: {e}")
            return False

    def _fallback_process_html_images(self, html_content: str) -> str:
        """
        Fallback image processing without PIL (no thumbnails).

        Args:
            html_content: HTML content containing image URLs

        Returns:
            Processed HTML content
        """
        try:
            # Find all img tags with src attributes
            img_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
            matches = re.findall(img_pattern, html_content, re.IGNORECASE)

            if not matches:
                return html_content

            # Create images directory if it doesn't exist
            images_dir = Path("src/ui/assets/images")
            images_dir.mkdir(parents=True, exist_ok=True)

            # Process each image URL (basic version without thumbnails)
            processed_urls = {}
            for url in matches:
                if url in processed_urls:
                    continue

                try:
                    if url.startswith(('http://', 'https://')):
                        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                        url_path = urllib.parse.urlparse(url).path
                        ext = Path(url_path).suffix.lower()
                        if not ext or ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
                            ext = '.jpg'

                        filename = f"{url_hash}{ext}"
                        local_path = images_dir / filename

                        if not local_path.exists():
                            logger.info(f"Downloading image: {url}")
                            with urllib.request.urlopen(url, timeout=10) as response:
                                if response.status == 200:
                                    with open(local_path, 'wb') as f:
                                        f.write(response.read())
                                    logger.info(f"Downloaded image to: {local_path}")
                                else:
                                    processed_urls[url] = url
                                    continue
                        else:
                            logger.debug(f"Image already exists: {local_path}")

                        file_url = f"file://{local_path.resolve()}"
                        processed_urls[url] = file_url
                    else:
                        processed_urls[url] = url

                except Exception as e:
                    logger.warning(f"Failed to process image URL {url}: {e}")
                    processed_urls[url] = url

            # Replace URLs in HTML
            processed_html = html_content
            for original_url, local_url in processed_urls.items():
                escaped_url = re.escape(original_url)
                processed_html = re.sub(
                    rf'src=["\']{escaped_url}["\']',
                    f'src="{local_url}"',
                    processed_html,
                    flags=re.IGNORECASE
                )

            return processed_html

        except Exception as e:
            logger.error(f"Error in fallback image processing: {e}")
            return html_content

    # Draft management methods

    def save_draft(self, content: str):
        """
        Save or update the current draft.

        Args:
            content: The draft content to save
        """
        import uuid

        if not content.strip():
            # Clear current draft if empty
            if self.current_draft_id:
                # Remove from drafts list
                self.drafts = [d for d in self.drafts if d.get('id') != self.current_draft_id]
                self.current_draft_id = None
            return

        # Check if we have a current draft
        if self.current_draft_id:
            # Update existing draft
            for draft in self.drafts:
                if draft.get('id') == self.current_draft_id:
                    draft['content'] = content
                    draft['timestamp'] = datetime.now().isoformat()
                    logger.info(f"Updated existing draft: {self.current_draft_id[:8]}...")
                    break
        else:
            # Create new draft
            draft_id = str(uuid.uuid4())
            draft = {
                'id': draft_id,
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'status': 'active'
            }
            self.drafts.insert(0, draft)  # Add to beginning
            self.current_draft_id = draft_id
            logger.info(f"Created new draft: {draft_id[:8]}...")

        # Keep only max_drafts
        if len(self.drafts) > self.max_drafts:
            self.drafts = self.drafts[:self.max_drafts]

    def get_current_draft(self) -> Optional[str]:
        """
        Get the current draft content.

        Returns:
            Current draft content or None if no draft
        """
        if self.current_draft_id:
            for draft in self.drafts:
                if draft.get('id') == self.current_draft_id:
                    return draft.get('content', '')
        return None

    def archive_current_draft(self):
        """
        Archive the current draft (mark as sent).
        """
        if self.current_draft_id:
            # Find and archive the current draft
            for draft in self.drafts:
                if draft.get('id') == self.current_draft_id:
                    draft['status'] = 'sent'
                    draft['sent_at'] = datetime.now().isoformat()
                    # Move to archived drafts
                    self.archived_drafts.insert(0, draft)
                    # Remove from active drafts
                    self.drafts.remove(draft)
                    logger.info(f"Archived draft: {self.current_draft_id[:8]}...")
                    break

            self.current_draft_id = None

    def get_drafts(self) -> List[Dict[str, Any]]:
        """
        Get all active drafts.

        Returns:
            List of active draft dictionaries
        """
        return self.drafts.copy()

    def get_archived_drafts(self) -> List[Dict[str, Any]]:
        """
        Get all archived drafts.

        Returns:
            List of archived draft dictionaries
        """
        return self.archived_drafts.copy()

    def clear_old_drafts(self):
        """
        Clear old drafts, keeping only the most recent max_drafts.
        """
        if len(self.drafts) > self.max_drafts:
            self.drafts = self.drafts[:self.max_drafts]
            logger.info(f"Cleared old drafts, keeping {self.max_drafts}")

    def autosave_conversation(self, file_path: str):
        """
        Autosave conversation with drafts to file.

        Args:
            file_path: Path to save the conversation
        """
        try:
            messages = self.conversation_messages.copy()

            conversation_data = {
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'message_count': len(messages),
                    'version': '0.2.0'
                },
                'messages': messages,
                'message_prompts': self.message_prompts.copy(),
                'drafts': self.drafts.copy(),
                'archived_drafts': self.archived_drafts.copy()
            }

            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Conversation autosaved to {file_path}")

        except Exception as e:
            logger.error(f"Failed to autosave conversation: {e}")
            raise
