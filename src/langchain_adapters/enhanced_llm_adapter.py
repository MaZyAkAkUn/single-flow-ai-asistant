"""
Enhanced LLM adapter with structured prompt support.
Extends the existing LLM adapter to work with the structured prompt building system.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from datetime import datetime

from .llm_adapter import LLMAdapter
from ..core.prompt_builder import StructuredPromptBuilder
from ..core.context_aggregator import ContextAggregator
from ..core.memory_management import MemoryManager
from ..core.intent_analyzer import IntentAnalyzer
# Import topic tracking components (moved from TYPE_CHECKING since they're used at runtime)
from ..core.topic_tracker import TopicTracker
from ..core.llm_topic_analyzer import LLMTopicAnalyzer
from ..core.topic_storage import TopicStorage
from ..data.schemas import (
    StructuredPromptConfig, RetrievedContext, UserIntent,
    AgentState, ConversationFrame, MemoryRules, UserPersonalization,
    ProjectContext, TopicInfo, IntentType
)
from ..core.logging_config import get_logger
from .tools import AssistantTools

logger = get_logger(__name__)


class EnhancedLLMAdapter(LLMAdapter):
    """
    Enhanced LLM adapter that integrates structured prompt building.
    Provides unified interface for structured prompts while maintaining backward compatibility.
    """
    
    def __init__(self, provider: str = "openrouter", intent_config: Optional[Dict[str, Any]] = None, openai_api_key: Optional[str] = None, settings: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize enhanced LLM adapter.

        Args:
            provider: LLM provider name
            intent_config: Configuration for intent analysis
            openai_api_key: OpenAI API key for TTS and memory operations
            settings: Application settings for system information
            **kwargs: Additional arguments for LLMAdapter
        """
        super().__init__(provider, **kwargs)

        # Store settings for system information
        self.settings = settings

        # Structured prompt components
        self.prompt_config = StructuredPromptConfig()
        self.prompt_builder = StructuredPromptBuilder(self.prompt_config)
        self.context_aggregator = ContextAggregator(self.prompt_config.token_limits)
        
        # Use provided openai_api_key, or fallback to config api_key if provider is openai, 
        # otherwise try to use available key but it might fail if it is not openai-compatible
        memory_key = openai_api_key or (self.config.get('api_key') if provider == 'openai' else None)
        self.memory_manager = MemoryManager(openai_api_key=memory_key)
        
        # Initialize intent analyzer (reusing self or creating dedicated adapter)
        if intent_config and intent_config.get('enabled'):
            logger.info(f"Initializing separate intent analyzer with config: {intent_config}")
            try:
                intent_provider = intent_config.get('provider', 'openrouter')
                # Create generic LLM adapter for intent analysis
                intent_adapter = LLMAdapter(
                    provider=intent_provider,
                    api_key=intent_config.get('api_key'),
                    model=intent_config.get('model'),
                    temperature=0.0  # Low temperature for classification tasks
                )
                self.intent_analyzer = IntentAnalyzer(llm_adapter=intent_adapter)
                logger.info(f"Initialized separate intent analyzer using {intent_provider}")
            except Exception as e:
                logger.error(f"Failed to initialize separate intent analyzer: {e}")
                self.intent_analyzer = IntentAnalyzer(llm_adapter=self)
                logger.info("Falling back to main adapter for intent analysis")
        else:
            logger.info("Intent analysis disabled, using main adapter for IntentAnalyzer")
            self.intent_analyzer = IntentAnalyzer(llm_adapter=self)
        
        # Initialize LLM-based topic tracking components
        self.topic_tracker = None  # Initialize to None first
        try:
            # Resolve LLM combo from settings for topic analysis
            topic_llm_adapter = self.intent_analyzer.llm_adapter  # Use same adapter as intent analyzer

            # Initialize topic components
            self.topic_storage = TopicStorage()
            self.llm_topic_analyzer = LLMTopicAnalyzer(topic_llm_adapter, self.topic_storage)
            self.topic_tracker = TopicTracker(self.topic_storage, self.llm_topic_analyzer)

            # Now that topic tracker is initialized, populate topic tools in topic_tracking tool set
            self._populate_topic_tools()

            logger.info("✅ LLM-based topic tracking components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM-based topic tracking: {e}")  # Changed to error level
            logger.debug(f"Topic tracking initialization error details: {e}", exc_info=True)
            self.topic_tracker = None
            # NEW: Log what will happen without topic tracker
            logger.warning("⚠️ Topic tracking will show 'not_detected_yet' status in prompts")

        # State management
        self.conversation_frame = ConversationFrame()
        self.memory_rules = MemoryRules()
        self.user_personalization = UserPersonalization()
        self.agent_state = None
        self.current_project = None

        # Message prompt storage for UI
        self.message_prompts = {}

        # Initialize tool sets for different use cases (must be after topic tracker)
        self._initialize_tool_sets()

        logger.info("Enhanced LLM adapter initialized with structured prompt support")

    def configure_tools(self, web_search_config: Optional[Dict[str, str]] = None):
        """
        Configure tools for the main_chat tool set (overridden from base class).

        Args:
            web_search_config: Configuration for web search tools
        """
        # Ensure main_chat tool set exists
        if not self.tool_set_manager.get_tool_set("main_chat"):
            self.create_tool_set("main_chat")

        # Switch to main_chat tool set for configuration
        self.set_tool_set("main_chat")

        # Configure assistant tools
        self._assistant_tools = AssistantTools(web_search_config)
        new_tools = self._assistant_tools.get_all_tools()

        # Get current tool set (should be main_chat now)
        current_tool_set = self.tool_set_manager.get_tool_set("main_chat")
        if  current_tool_set is None:
            logger.error("No main_chat tool set available")
            return

        # Clear existing tools in main_chat set
        current_tool_set.clear()

        # Add new tools to main_chat set
        for tool in new_tools:
            current_tool_set.add_tool(tool)

        # Update internal tools list to match main_chat tool set
        self._tools = current_tool_set.tools.copy()

        self._llm_with_tools = None  # Reset cached LLM with tools
        logger.info(f"Configured {len(new_tools)} web search tools in main_chat tool set "
                   f"(total: {len(self._tools)} tools)")

    def _initialize_tool_sets(self):
        """
        Initialize tool sets for different use cases.
        Creates default tool sets for main chat, intent analysis, and topic tracking.
        This method is idempotent - it ensures tool sets exist without failing if they already do.
        """
        # Ensure tool sets exist for different use cases (idempotent)
        if not self.tool_set_manager.get_tool_set("main_chat"):
            self.create_tool_set("main_chat")
        if not self.tool_set_manager.get_tool_set("intent_analysis"):
            self.create_tool_set("intent_analysis")  # Intent analysis uses no tools
        if not self.tool_set_manager.get_tool_set("topic_tracking"):
            self.create_tool_set("topic_tracking")   # Topic tracking will get topic tools later

        # Set main_chat as the default tool set
        self.set_tool_set("main_chat")
        self.set_default_tool_set("main_chat")

        logger.info("Ensured tool sets exist: main_chat, intent_analysis, topic_tracking")

    def _populate_topic_tools(self):
        """
        Populate the topic_tracking tool set with topic management tools.
        This is called after the topic tracker is initialized.
        """
        if not self.topic_tracker or not hasattr(self.topic_tracker, 'llm_topic_analyzer'):
            logger.warning("Topic tracker or LLM topic analyzer not available, cannot populate topic tools")
            return

        try:
            # Get topic tools from the LLM topic analyzer
            topic_tools = self.topic_tracker.llm_topic_analyzer.topic_tools.get_all_tools()

            # Get or create the topic_tracking tool set
            topic_tool_set = self.tool_set_manager.get_tool_set("topic_tracking")
            if not topic_tool_set:
                logger.error("topic_tracking tool set not found")
                return

            # Clear existing tools and add topic tools
            topic_tool_set.clear()
            for tool in topic_tools:
                topic_tool_set.add_tool(tool)

            logger.info(f"Populated topic_tracking tool set with {len(topic_tools)} topic tools")

        except Exception as e:
            logger.error(f"Failed to populate topic tools: {e}")

    def process_message_structured(
        self,
        user_message: str,
        conversation_history: List[Dict[str, Any]] = None,
        project_contexts: List[ProjectContext] = None,
        agent_state: AgentState = None,
        user_profile: UserPersonalization = None,
        memory_enabled: bool = True,
        retrieval_enabled: bool = True,
        max_retrieved_memories: int = 3,
        max_retrieved_docs: int = 3
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Process message using structured prompt building.
        
        Args:
            user_message: User's message
            conversation_history: Recent conversation history
            project_contexts: Available project contexts
            agent_state: Current agent state
            user_profile: User personalization profile
            
        Returns:
            Tuple of (structured_prompt, flattened_prompt, metadata)
        """
        try:
            conversation_history = conversation_history or []
            project_contexts = project_contexts or []
            
            # 1. Add user message to memory
            self.memory_manager.add_to_stm({
                'role': 'user',
                'content': user_message,
                'timestamp': datetime.now().isoformat()
            })
            
            # 2. Analyze user intent
            user_intent = self.intent_analyzer.analyze_intent(user_message, conversation_history)

            # 2.5. Analyze topic (if topic tracker is available)
            topic_info = None
            if self.topic_tracker:
                try:
                    # Get message index for tracking (use length of conversation_history as approximation)
                    message_index = len(conversation_history) if conversation_history else 0
                    topic_info = self.topic_tracker.analyze_message_topic(
                        user_message, message_index, conversation_context=conversation_history
                    )
                    logger.info(f"Topic analyzed: '{topic_info.topic_name}' (domain: {topic_info.domain})")

                    # Store the user message in the current topic for future context
                    if topic_info:
                        user_message_dict = {
                            'role': 'user',
                            'content': user_message,
                            'timestamp': datetime.now().isoformat()
                        }
                        success = self.topic_tracker.add_message_to_current_topic(user_message_dict)
                        if success:
                            logger.info(f"Stored user message in topic '{topic_info.topic_name}'")
                        else:
                            logger.warning(f"Failed to store user message in topic '{topic_info.topic_name}'")

                except Exception as topic_error:
                    logger.warning(f"Topic analysis failed: {topic_error}")
                    topic_info = None

            # 3. Update agent state based on intent
            if agent_state:
                recommended_mode = self.intent_analyzer.get_recommended_agent_mode(
                    user_intent, conversation_history
                )
                if recommended_mode != agent_state.current_mode:
                    agent_state.current_mode = recommended_mode
                    agent_state.current_goal = f"Processing {user_intent.intent_type.value} request"
                    
            # 4. Retrieve memories (only if enabled)
            memories = []
            if memory_enabled:
                try:
                    memories = self.memory_manager.retrieve_from_ltm(
                        query=user_message,
                        memory_types=user_intent.context_requirements,
                        limit=max_retrieved_memories
                    )
                    
                    # Log collected memories
                    if memories:
                        memory_contents = [m.content if hasattr(m, 'content') else str(m) for m in memories]
                        logger.info(f"Retrieved {len(memories)} memories from LTM: {memory_contents}")
                    else:
                        logger.info(f"Retrieved {len(memories)} memories from LTM")
                        
                except Exception as memory_error:
                    logger.warning(f"Memory retrieval failed: {memory_error}")
            else:
                logger.info("Memory retrieval disabled, skipping LTM retrieval")

            # 5. Retrieve documents (only if enabled)
            retrieved_docs = []
            if retrieval_enabled:
                try:
                    # Import file_ingestor here to avoid circular imports
                    from ..core.file_ingestor import FileIngestor
                    # Create a temporary file ingestor instance (this could be optimized)
                    temp_ingestor = FileIngestor()
                    retrieved_docs = temp_ingestor.search_similar(
                        query=user_message,
                        k=max_retrieved_docs,
                        score_threshold=0.1
                    )
                    logger.info(f"Retrieved {len(retrieved_docs)} documents from vector store")
                except Exception as doc_error:
                    logger.warning(f"Document retrieval failed: {doc_error}")
            else:
                logger.info("Document retrieval disabled, skipping vector search")

            # 5.5. Get topic-related context if topic tracking is available
            topic_messages = []
            if self.topic_tracker and topic_info:
                try:
                    # Get messages related to the current topic
                    topic_messages = self.topic_tracker.get_topic_conversation_history(
                        topic_info.topic_id,
                        limit=10  # Get recent topic messages
                    )
                    logger.info(f"Retrieved {len(topic_messages)} topic-related messages for context")
                except Exception as topic_context_error:
                    logger.warning(f"Failed to retrieve topic context: {topic_context_error}")

            # 6. Aggregate context with topic awareness
            retrieved_context = self.context_aggregator.aggregate_context(
                user_intent=user_intent,
                conversation_history=conversation_history,
                memories=memories,
                documents=retrieved_docs,
                project_contexts=project_contexts,
                topic_info=topic_info,
                topic_messages=topic_messages
            )
            
            # 6. Build structured prompt
            current_project = self.current_project
            other_projects = [p for p in project_contexts if p != current_project]
            
            structured_prompt, flattened_prompt = self.prompt_builder.build_structured_prompt(
                user_message=user_message,
                conversation_history=conversation_history,
                retrieved_context=retrieved_context,
                agent_state=agent_state or self.agent_state or AgentState(),
                conversation_frame=self.conversation_frame,
                memory_rules=self.memory_rules,
                user_personalization=user_profile or self.user_personalization,
                user_intent=user_intent,
                settings=self.settings,
                current_project=current_project,
                other_projects=other_projects,
                topic_info=topic_info
            )
            
            # 7. Generate response using existing LLM adapter
            # Use structured prompt for generation to ensure AI follows the XML structure
            response = self.generate_response(structured_prompt)
            
            # 8. Add assistant response to memory
            self.memory_manager.add_to_stm({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # 9. Build metadata
            metadata = {
                'intent': user_intent.intent_type.value,
                'confidence': user_intent.confidence_score,
                'agent_mode': (agent_state or self.agent_state or AgentState()).current_mode.value,
                'memories_retrieved': len(memories),
                'projects_in_context': len(project_contexts),
                'context_summary': self.context_aggregator.get_context_summary(retrieved_context)
            }
            
            logger.info(f"Structured processing completed: {user_intent.intent_type.value} "
                       f"(confidence: {user_intent.confidence_score:.2f})")
            
            return structured_prompt, flattened_prompt, metadata
            
        except Exception as e:
            logger.error(f"Failed to process message with structured prompt: {e}")
            # Fallback to regular processing
            fallback_response = self.generate_response(user_message)
            return user_message, fallback_response, {'error': str(e)}
            
    async def aprocess_message_structured(
        self,
        user_message: str,
        conversation_history: List[Dict[str, Any]] = None,
        project_contexts: List[ProjectContext] = None,
        agent_state: AgentState = None,
        user_profile: UserPersonalization = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Async version of process_message_structured."""
        try:
            # For now, delegate to sync version (can be optimized later)
            return self.process_message_structured(
                user_message, conversation_history, project_contexts, agent_state, user_profile
            )
        except Exception as e:
            logger.error(f"Failed to process message asynchronously: {e}")
            raise

    async def astream_message_structured(
        self,
        user_message: str,
        conversation_history: List[Dict[str, Any]] = None,
        project_contexts: List[ProjectContext] = None,
        agent_state: AgentState = None,
        user_profile: UserPersonalization = None,
        memory_enabled: bool = True,
        retrieval_enabled: bool = True,
        max_retrieved_memories: int = 3,
        max_retrieved_docs: int = 3
    ):
        """
        Process message using structured prompt building with optimized streaming support.
        
        Provides real-time token streaming with minimal buffering (0.5s intervals).

        Args:
            user_message: User's message
            conversation_history: Recent conversation history
            project_contexts: Available project contexts
            agent_state: Current agent state
            user_profile: User personalization profile

        Yields:
            Structured events (status, token, error) with optimized streaming
        """
        import time
        
        try:
            conversation_history = conversation_history or []
            project_contexts = project_contexts or []

            # 1. Status: Initializing (immediate response)
            yield {"type": "status", "content": "Initializing processing...", "stage": "init"}

            # 2. Add user message to memory
            self.memory_manager.add_to_stm({
                'role': 'user',
                'content': user_message,
                'timestamp': datetime.now().isoformat()
            })

            # 3. Status: Analyzing intent (with streaming indicator)
            yield {"type": "status", "content": "Analyzing user intent...", "stage": "intent"}

            logger.info(f"Starting intent analysis for message: '{user_message[:50]}...'")
            try:
                user_intent = self.intent_analyzer.analyze_intent(user_message, conversation_history)
                logger.info(f"Intent analysis completed: {user_intent.intent_type.value} (confidence: {user_intent.confidence_score:.2f})")
                # NEW: Yield intent result as status
                yield {
                    "type": "status",
                    "content": f"Intent detected: {user_intent.intent_type.value} (confidence: {user_intent.confidence_score:.2f})",
                    "stage": "intent_complete"
                }
            except Exception as intent_error:
                logger.error(f"Intent analysis failed: {intent_error}")
                # NEW: Yield failure status
                yield {
                    "type": "status",
                    "content": "Intent analysis failed, using default",
                    "stage": "intent_fallback"
                }
                # Fallback to basic intent
                from ..data.schemas import IntentType
                user_intent = type('Intent', (), {
                    'intent_type': IntentType.GENERAL,
                    'confidence_score': 0.5,
                    'context_requirements': []
                })()

            # 3.5. Analyze topic (if topic tracker is available)
            topic_info = None
            if self.topic_tracker:
                # NEW: Yield status before analysis
                yield {"type": "status", "content": "Analyzing conversation topic...", "stage": "topic"}
                try:
                    # Get message index for tracking (use length of conversation_history as approximation)
                    message_index = len(conversation_history) if conversation_history else 0
                    topic_info = self.topic_tracker.analyze_message_topic(
                        user_message, message_index, conversation_context=conversation_history
                    )
                    # NEW: Yield topic result as status
                    topic_name = topic_info.topic_name if topic_info else "Unknown"
                    yield {
                        "type": "status",
                        "content": f"Topic: '{topic_name}' (domain: {topic_info.domain if topic_info else 'N/A'})",
                        "stage": "topic_complete"
                    }
                    logger.info(f"Topic analyzed: '{topic_info.topic_name}' (domain: {topic_info.domain})")

                    # Store the user message in the current topic for future context
                    if topic_info:
                        user_message_dict = {
                            'role': 'user',
                            'content': user_message,
                            'timestamp': datetime.now().isoformat()
                        }
                        success = self.topic_tracker.add_message_to_current_topic(user_message_dict)
                        if success:
                            logger.info(f"Stored user message in topic '{topic_info.topic_name}'")
                        else:
                            logger.warning(f"Failed to store user message in topic '{topic_info.topic_name}'")
                except Exception as topic_error:
                    logger.warning(f"Topic analysis failed: {topic_error}")
                    # NEW: Yield failure status
                    yield {
                        "type": "status",
                        "content": "Topic analysis failed, will continue without topic context",
                        "stage": "topic_fallback"
                    }
                    topic_info = None
            else:
                # NEW: Yield status when tracker not available
                yield {
                    "type": "status",
                    "content": "Topic tracker not initialized",
                    "stage": "topic_unavailable"
                }
                logger.debug("Topic tracker not available, skipping topic analysis")

            # 4. Update agent state based on intent
            if agent_state:
                try:
                    recommended_mode = self.intent_analyzer.get_recommended_agent_mode(
                        user_intent, conversation_history
                    )
                    if recommended_mode != agent_state.current_mode:
                        agent_state.current_mode = recommended_mode
                        agent_state.current_goal = f"Processing {user_intent.intent_type.value} request"
                except Exception as state_error:
                    logger.warning(f"Agent state update failed: {state_error}")

            # 5. Status: Retrieving context
            yield {"type": "status", "content": "Retrieving relevant context...", "stage": "context"}

            # 5a. Retrieve memories with timeout protection (only if enabled)
            memories = []
            if memory_enabled:
                try:
                    memories = self.memory_manager.retrieve_from_ltm(
                        query=user_message,
                        memory_types=user_intent.context_requirements,
                        limit=max_retrieved_memories
                    )
                    
                    # Log collected memories
                    if memories:
                        memory_contents = [m.content if hasattr(m, 'content') else str(m) for m in memories]
                        logger.info(f"Retrieved {len(memories)} memories from LTM: {memory_contents}")
                    else:
                        logger.info(f"Retrieved {len(memories)} memories from LTM")

                except Exception as memory_error:
                    logger.warning(f"Memory retrieval failed: {memory_error}")
            else:
                logger.info("Memory retrieval disabled, skipping LTM retrieval")

            # 5b. Retrieve relevant documents (only if enabled)
            retrieved_docs = []
            if retrieval_enabled:
                try:
                    # Import file_ingestor here to avoid circular imports
                    from ..core.file_ingestor import FileIngestor
                    # Create a temporary file ingestor instance (this could be optimized)
                    temp_ingestor = FileIngestor()
                    retrieved_docs = temp_ingestor.search_similar(
                        query=user_message,
                        k=max_retrieved_docs,
                        score_threshold=0.1
                    )
                    logger.info(f"Retrieved {len(retrieved_docs)} documents from vector store")
                except Exception as doc_error:
                    logger.warning(f"Document retrieval failed: {doc_error}")
            else:
                logger.info("Document retrieval disabled, skipping vector search")

            # 5c. Get topic-related context if topic tracking is available
            topic_messages = []
            if self.topic_tracker and topic_info:
                try:
                    # Get messages related to the current topic
                    topic_messages = self.topic_tracker.get_topic_conversation_history(
                        topic_info.topic_id,
                        limit=10  # Get recent topic messages
                    )
                    logger.info(f"Retrieved {len(topic_messages)} topic-related messages for context")
                except Exception as topic_context_error:
                    logger.warning(f"Failed to retrieve topic context: {topic_context_error}")

            # 5d. Aggregate context with topic awareness and error handling
            retrieved_context = None
            try:
                retrieved_context = self.context_aggregator.aggregate_context(
                    user_intent=user_intent,
                    conversation_history=conversation_history,
                    memories=memories,
                    documents=retrieved_docs,
                    project_contexts=project_contexts,
                    topic_info=topic_info,
                    topic_messages=topic_messages
                )
            except Exception as context_error:
                logger.warning(f"Context aggregation failed: {context_error}")
                # Create minimal context
                retrieved_context = type('Context', (), {
                    'memories': memories,
                    'documents': retrieved_docs,
                    'project_contexts': project_contexts,
                    'summary': 'Context aggregation failed'
                })()

            # 6. Status: Building prompt
            yield {"type": "status", "content": "Building optimized prompt...", "stage": "prompt_build"}

            # 7. Build structured prompt with error handling
            current_project = self.current_project
            other_projects = [p for p in project_contexts if p != current_project]

            structured_prompt = ""
            flattened_prompt = ""
            try:
                structured_prompt, flattened_prompt = self.prompt_builder.build_structured_prompt(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    retrieved_context=retrieved_context,
                    agent_state=agent_state or self.agent_state or AgentState(),
                    conversation_frame=self.conversation_frame,
                    memory_rules=self.memory_rules,
                    user_personalization=user_profile or self.user_personalization,
                    user_intent=user_intent,
                    settings=self.settings,
                    current_project=current_project,
                    other_projects=other_projects,
                    topic_info=topic_info
                )
            except Exception as prompt_error:
                logger.warning(f"Structured prompt building failed, using fallback: {prompt_error}")
                # Fallback to simple prompt
                structured_prompt = f"User: {user_message}\n\nAssistant:"
                flattened_prompt = f"User: {user_message}\n\nAssistant:"

            # Store structured prompt for UI
            try:
                import hashlib
                message_hash = hashlib.md5(user_message.encode()).hexdigest()
                self.message_prompts[message_hash] = structured_prompt
                
                # Yield prompt event for Controller to capture
                yield {
                    "type": "prompt", 
                    "content": structured_prompt, 
                    "flattened": flattened_prompt,
                    "metadata": {
                        "message_hash": message_hash,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            except Exception as hash_error:
                logger.warning(f"Failed to store prompt hash: {hash_error}")

            # 8. Ensure we're using the main_chat tool set for response generation
            current_tool_set = self.get_current_tool_set()
            if current_tool_set and current_tool_set.name != "main_chat":
                logger.info(f"Switching back to main_chat tool set from {current_tool_set.name} for response generation")
                self.set_tool_set("main_chat")

            # 8. Status: Generating response
            yield {"type": "status", "content": "Generating response...", "stage": "generation"}

            # 9. Generate response using optimized streaming LLM with minimal buffering
            response = ""
            token_count = 0
            start_time = time.time()
            
            try:
                # Use structured prompt for streaming generation
                async for chunk in self.astream_llm_response(structured_prompt):
                    # Check for done signal first
                    if chunk.get("type") == "done":
                        # Don't yield done signal yet, wait for post-processing
                        break
                    
                    # Always yield chunks immediately for real-time streaming
                    yield chunk
                    
                    if chunk.get("type") == "token":
                        response += chunk.get("content", "")
                        token_count += 1
                    elif chunk.get("type") == "error":
                        # Log error and stop processing
                        logger.error(f"Streaming error: {chunk.get('content')}")
                        return

                # Log streaming performance
                elapsed_time = time.time() - start_time
                if token_count > 0:
                    logger.info(f"Streaming completed: {token_count} tokens in {elapsed_time:.2f}s "
                               f"({token_count/max(0.001, elapsed_time):.1f} tokens/sec)")
                
            except Exception as stream_error:
                logger.error(f"Streaming failed: {stream_error}")
                yield {"type": "error", "content": f"Streaming error: {str(stream_error)}"}
                return

            # 10. Status: Finalizing
            yield {"type": "status", "content": "Finalizing response...", "stage": "finalize"}

            # 11. Add assistant response to memory with error handling
            try:
                self.memory_manager.add_to_stm({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as memory_error:
                logger.warning(f"Failed to store assistant response in memory: {memory_error}")

            # 12. Build final metadata
            try:
                metadata = {
                    'intent': user_intent.intent_type.value,
                    'confidence': user_intent.confidence_score,
                    'agent_mode': (agent_state or self.agent_state or AgentState()).current_mode.value,
                    'memories_retrieved': len(memories),
                    'projects_in_context': len(project_contexts),
                    'tokens_generated': token_count,
                    'streaming_time': elapsed_time if 'elapsed_time' in locals() else 0,
                    'context_summary': getattr(retrieved_context, 'summary', 'N/A') if retrieved_context else 'N/A',
                    'topic': {
                        'name': topic_info.topic_name if topic_info else None,
                        'domain': topic_info.domain if topic_info else None,
                        'id': topic_info.topic_id if topic_info else None
                    } if topic_info else None
                }
            except Exception as metadata_error:
                logger.warning(f"Metadata generation failed: {metadata_error}")
                metadata = {
                    'intent': 'general',
                    'confidence': 0.5,
                    'tokens_generated': token_count if 'token_count' in locals() else 0,
                    'streaming_time': elapsed_time if 'elapsed_time' in locals() else 0,
                }

            # 13. Status: Done (final yield with performance info)
            yield {"type": "status", "content": "Response completed", "stage": "done", "metadata": metadata}

            logger.info(f"Optimized streaming structured processing completed: {user_intent.intent_type.value} "
                       f"(confidence: {user_intent.confidence_score:.2f}, tokens: {token_count if 'token_count' in locals() else 0})")

        except Exception as e:
            logger.error(f"Failed to stream process message with structured prompt: {e}")
            # Yield error event with detailed information
            yield {"type": "error", "content": f"Processing failed: {str(e)}", "stage": "error", "details": str(e)}
            
    def generate_structured_response_with_tools(
        self,
        user_message: str,
        messages: List[Dict[str, Any]] = None,
        max_iterations: int = 3
    ) -> str:
        """
        Generate response using tools with structured prompt context.

        Args:
            user_message: User's message
            messages: Message history for tools
            max_iterations: Maximum tool call iterations

        Returns:
            Generated response
        """
        try:
            logger.info("Starting enhanced structured response with tools", extra={
                "user_message_length": len(user_message),
                "conversation_history_length": len(messages or [])
            })

            # Use structured prompt for tool-enabled generation
            structured_prompt, flattened_prompt, metadata = self.process_message_structured(
                user_message=user_message,
                conversation_history=messages or []
            )

            logger.info("Structured prompt generated", extra={
                "structured_prompt_length": len(structured_prompt),
                "flattened_prompt_length": len(flattened_prompt),
                "intent_type": metadata.get('intent'),
                "confidence": metadata.get('confidence')
            })

            # Convert to LangChain message format for tools
            langchain_messages = []
            for msg in messages or []:
                if msg.get('role') == 'user':
                    from langchain_core.messages import HumanMessage
                    langchain_messages.append(HumanMessage(content=msg.get('content', '')))
                elif msg.get('role') == 'assistant':
                    # Handle tool calls if present
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        from langchain_core.messages import AIMessage
                        ai_msg = AIMessage(content=msg.get('content', ''))
                        ai_msg.tool_calls = tool_calls
                        langchain_messages.append(ai_msg)
                    else:
                        from langchain_core.messages import AIMessage
                        langchain_messages.append(AIMessage(content=msg.get('content', '')))
                elif msg.get('role') == 'tool':
                    # Tool result message
                    tool_call_id = msg.get("tool_call_id")
                    from langchain_core.messages import ToolMessage
                    langchain_messages.append(ToolMessage(
                        content=msg.get('content', ''),
                        tool_call_id=tool_call_id
                    ))

            # Add current user message
            from langchain_core.messages import HumanMessage
            langchain_messages.append(HumanMessage(content=user_message))

            logger.info(f"Prepared {len(langchain_messages)} LangChain messages for tool-enabled generation")

            # Use the enhanced generation with tool execution loop
            llm_with_tools = self.get_llm_with_tools()

            iteration = 0
            while iteration < max_iterations:
                try:
                    # Get LLM response
                    response = llm_with_tools.invoke(langchain_messages)

                    # Log response details
                    logger.info(f"Tool iteration {iteration + 1}: LLM response received", extra={
                        "response_type": type(response).__name__,
                        "has_content_attr": hasattr(response, 'content'),
                        "has_tool_calls": hasattr(response, 'tool_calls') and bool(getattr(response, 'tool_calls', None)),
                        "tool_calls_count": len(getattr(response, 'tool_calls', [])) if hasattr(response, 'tool_calls') else 0
                    })

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
                                from langchain_core.messages import ToolMessage
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

                        if hasattr(response, 'content'):
                            content = response.content
                            content_length = len(content) if content else 0
                            logger.info(f"Final response content - length: {content_length}")
                            if not content:
                                logger.warning("LLM returned empty content string in enhanced structured generation")
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

        except Exception as e:
            logger.error(f"Failed to generate structured response with tools: {e}", extra={
                "error_type": type(e).__name__,
                "user_message_length": len(user_message),
                "fallback_available": True
            })
            # Fallback to regular tool generation
            logger.info("Falling back to regular tool generation")
            return self.generate_response_with_tools(messages or [{'role': 'user', 'content': user_message}], max_iterations)
            
    def store_memory_from_interaction(
        self,
        user_message: str,
        assistant_response: str,
        memory_type: str = "interaction",
        importance: str = "normal"
    ) -> Optional[str]:
        """
        Store important interaction in long-term memory.
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
            memory_type: Type of memory to store
            importance: Importance level
            
        Returns:
            Memory ID if successful
        """
        try:
            # Create memory content
            memory_content = f"User: {user_message[:100]}... | Assistant: {assistant_response[:100]}..."
            
            # Store in memory
            memory_id = self.memory_manager.store_in_ltm(
                content=memory_content,
                memory_type=memory_type,
                importance=importance,
                source_context="User interaction",
                tags=["interaction", "conversation"]
            )
            
            # Update user profile based on interaction
            self.user_personalization = self.user_personalization or UserPersonalization()
            self.user_profile_manager.learn_from_interaction(user_message, assistant_response)
            
            logger.info(f"Stored interaction memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store interaction memory: {e}")
            return None
            
    def get_context_for_tools(self, query: str) -> Dict[str, Any]:
        """
        Get relevant context for tool execution.
        
        Args:
            query: Tool execution query
            
        Returns:
            Context dictionary for tools
        """
        try:
            # Retrieve relevant memories
            memories = self.memory_manager.retrieve_from_ltm(
                query=query,
                importance_threshold="low",
                limit=3
            )
            
            # Get current STM
            stm_messages = self.memory_manager.get_stm(limit=5)
            
            # Get project context
            project_context = None
            if self.current_project:
                project_context = {
                    'project_id': self.current_project.project_id,
                    'project_name': self.current_project.project_name,
                    'context_summary': self.current_project.context_summary
                }
                
            return {
                'memories': [memory.content for memory in memories],
                'recent_conversation': [msg['content'] for msg in stm_messages],
                'current_project': project_context,
                'agent_state': {
                    'mode': (self.agent_state or AgentState()).current_mode.value,
                    'goal': (self.agent_state or AgentState()).current_goal
                } if self.agent_state else None,
                'user_preferences': {
                    'detail_level': self.user_personalization.detail_preference,
                    'communication_style': self.user_personalization.communication_style
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get context for tools: {e}")
            return {}
            
    def update_configuration(self, config_updates: Dict[str, Any]):
        """
        Update adapter configuration.
        
        Args:
            config_updates: Configuration updates
        """
        try:
            # Update prompt configuration
            if 'structured_format_enabled' in config_updates:
                self.prompt_config.structured_format_enabled = config_updates['structured_format_enabled']
            if 'token_limits' in config_updates:
                self.prompt_config.token_limits.update(config_updates['token_limits'])
                
            # Update memory rules
            if 'memory_rules' in config_updates:
                for key, value in config_updates['memory_rules'].items():
                    if hasattr(self.memory_rules, key):
                        setattr(self.memory_rules, key, value)
                        
            # Update user personalization
            if 'user_personalization' in config_updates:
                for key, value in config_updates['user_personalization'].items():
                    if hasattr(self.user_personalization, key):
                        setattr(self.user_personalization, key, value)
                        
            logger.info("Enhanced LLM adapter configuration updated")
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            memory_stats = self.memory_manager.get_memory_statistics()
            intent_stats = self.intent_analyzer.get_intent_statistics()
            
            return {
                'adapter_type': 'enhanced',
                'structured_prompt_enabled': self.prompt_config.structured_format_enabled,
                'memory_system': memory_stats,
                'intent_analysis': intent_stats,
                'current_project': {
                    'id': self.current_project.project_id,
                    'name': self.current_project.project_name
                } if self.current_project else None,
                'agent_state': {
                    'mode': (self.agent_state or AgentState()).current_mode.value,
                    'goal': (self.agent_state or AgentState()).current_goal
                } if self.agent_state else None,
                'user_preferences': {
                    'communication_style': self.user_personalization.communication_style,
                    'detail_preference': self.user_personalization.detail_preference
                },
                'conversation_frame': {
                    'type': self.conversation_frame.conversation_type,
                    'maintain_tone': self.conversation_frame.maintain_tone
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
            
    def set_current_project(self, project: ProjectContext):
        """Set current active project."""
        self.current_project = project
        logger.info(f"Current project set to: {project.project_name}")
        
    def set_agent_state(self, agent_state: AgentState):
        """Set current agent state."""
        self.agent_state = agent_state
        logger.info(f"Agent state updated: {agent_state.current_mode.value}")
        
    def set_user_profile(self, user_profile: UserPersonalization):
        """Set user personalization profile."""
        self.user_personalization = user_profile
        logger.info("User profile updated")
        
    def export_conversation_with_context(self, file_path: str) -> bool:
        """Export conversation with full context for analysis."""
        try:
            export_data = {
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'adapter_type': 'enhanced_structured',
                    'system_status': self.get_system_status()
                },
                'conversation': {
                    'stm': self.memory_manager.get_stm(),
                    'ltm_count': len(self.memory_manager._ltm)
                },
                'current_state': {
                    'agent_state': {
                        'current_mode': (self.agent_state or AgentState()).current_mode.value,
                        'current_goal': (self.agent_state or AgentState()).current_goal
                    },
                    'current_project': {
                        'id': self.current_project.project_id,
                        'name': self.current_project.project_name
                    } if self.current_project else None,
                    'user_personalization': {
                        'communication_style': self.user_personalization.communication_style,
                        'detail_preference': self.user_personalization.detail_preference
                    }
                }
            }
            
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
                
            logger.info(f"Conversation exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export conversation: {e}")
            return False
