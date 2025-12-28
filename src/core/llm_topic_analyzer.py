"""
LLM-based topic analyzer using tool calling.
Replaces embedding-based topic matching with LLM tool calling for intelligent topic management.
"""
import json
from typing import List, Dict, Optional, Any, Tuple, TYPE_CHECKING
from datetime import datetime

# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    from ..langchain_adapters.llm_adapter import LLMAdapter
from .topic_tools import TopicTools
from .topic_storage import TopicStorage
from ..data.schemas import TopicInfo
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class LLMTopicAnalyzer:
    """
    LLM-based topic analyzer that uses tool calling for intelligent topic management.
    Replaces the embedding-based TopicMatcher with LLM-driven analysis.
    """

    def __init__(self, llm_adapter, topic_storage: TopicStorage):
        """
        Initialize LLM topic analyzer.

        Args:
            llm_adapter: LLM adapter for tool calling
            topic_storage: Topic storage backend
        """
        self.llm_adapter = llm_adapter
        self.topic_tools = TopicTools(topic_storage)
        self.topic_storage = topic_storage

        # Configure LLM with topic management tools
        self._configure_llm_tools()

        logger.info("LLMTopicAnalyzer initialized with tool-calling capabilities")

    def _configure_llm_tools(self):
        """Configure the LLM with topic management tools in the topic_tracking tool set."""
        try:
            topic_tools = self.topic_tools.get_all_tools()

            # Ensure topic_tracking tool set exists
            topic_tool_set = self.llm_adapter.get_tool_set("topic_tracking")
            if not topic_tool_set:
                topic_tool_set = self.llm_adapter.create_tool_set("topic_tracking")

            # Clear existing tools in topic_tracking set and add topic tools
            topic_tool_set.clear()
            for tool in topic_tools:
                topic_tool_set.add_tool(tool)

            logger.info(f"Configured topic_tracking tool set with {len(topic_tools)} topic tools")
        except Exception as e:
            logger.error(f"Failed to configure LLM tools: {e}")
            raise

    def analyze_message_topic(self, user_message: str, message_index: int,
                            conversation_context: List[Dict[str, Any]] = None) -> TopicInfo:
        """
        Analyze message and determine/create appropriate topic using LLM tool calling.

        Args:
            user_message: User's message text
            message_index: Index of message in conversation
            conversation_context: Recent conversation history

        Returns:
            TopicInfo for the message
        """
        conversation_context = conversation_context or []

        # Switch to topic tracking tool set
        original_tool_set = None
        if hasattr(self.llm_adapter, 'get_current_tool_set'):
            original_tool_set = self.llm_adapter.get_current_tool_set()
            self.llm_adapter.set_tool_set("topic_tracking")

        try:
            # Format conversation context for LLM
            context_str = self._format_conversation_context(conversation_context)

            # Create analysis prompt
            prompt = f"""You are a conversation topic analyzer. Analyze this user message and use the available tools to manage conversation topics.

CONVERSATION CONTEXT (Recent Messages):
{context_str}

CURRENT USER MESSAGE:
"{user_message}"

TASK:
1. Analyze if this message starts a new topic or continues an existing one
2. Use tools to check available topics and make appropriate decisions
3. Create new topics, update existing ones, or merge topics as needed
4. Always provide reasoning for your decisions

AVAILABLE TOOLS:
- get_available_topics: Check existing topics
- create_new_topic: Create a new topic
- update_topic_summary: Update existing topic
- merge_topics: Merge similar topics
- get_topic_details: Get detailed topic info

INSTRUCTIONS:
- Use get_available_topics first to understand existing topics
- If this is a new topic, use create_new_topic
- If continuing an existing topic, use update_topic_summary
- If topics are similar, consider using merge_topics
- Provide clear reasoning for all actions

Start by checking available topics, then make your decision."""

            try:
                # Use LLM with tools to analyze and manage topics
                messages = [{"role": "user", "content": prompt}]

                # Get LLM response with tool calling
                response = self.llm_adapter.generate_response_with_tools(messages, max_iterations=5)

                # Parse the final response to extract topic information
                topic_info = self._extract_topic_from_response(response, user_message, message_index)

                logger.info(f"LLM topic analysis completed: {topic_info.topic_name} (confidence: {topic_info.llm_confidence})")
                return topic_info

            except Exception as e:
                logger.error(f"LLM topic analysis failed: {e}")
                # Fallback to simple topic creation
                return self._create_fallback_topic(user_message, message_index)
        finally:
            # Restore original tool set
            if original_tool_set and hasattr(self.llm_adapter, 'set_tool_set'):
                self.llm_adapter.set_tool_set(original_tool_set.name)

    def _format_conversation_context(self, conversation_context: List[Dict[str, Any]]) -> str:
        """Format conversation context for LLM consumption."""
        if not conversation_context:
            return "No recent conversation context"

        # Format last 5 messages
        formatted_messages = []
        for i, msg in enumerate(conversation_context[-5:]):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:200]  # Truncate long messages
            formatted_messages.append(f"{i+1}. {role}: {content}")

        return "\n".join(formatted_messages)

    def _extract_topic_from_response(self, llm_response: str, user_message: str, message_index: int) -> TopicInfo:
        """
        Extract topic information from LLM response.
        Since the LLM uses tools, we need to check what actions were taken.
        """
        try:
            # The LLM response should contain information about what tools were called
            # For now, we'll create a topic based on the response content
            # In a more sophisticated implementation, we'd track tool call results

            # Parse JSON if present
            try:
                response_data = json.loads(llm_response)
                if isinstance(response_data, dict) and 'topic_name' in response_data:
                    # LLM provided structured topic info
                    return TopicInfo(
                        topic_id=response_data.get('topic_id', f"topic_{message_index}"),
                        topic_name=response_data.get('topic_name', 'General Conversation'),
                        domain=response_data.get('domain', ''),
                        message_count=1,
                        summary=response_data.get('summary', f'Discussion about {response_data.get("topic_name", "general topic")}'),
                        start_message_index=message_index,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        llm_confidence=response_data.get('confidence', 0.5),
                        llm_reasoning=response_data.get('reasoning', ''),
                        llm_last_analyzed=datetime.now()
                    )
            except json.JSONDecodeError:
                pass

            # Fallback: extract topic info from text response
            return self._create_topic_from_text_response(llm_response, user_message, message_index)

        except Exception as e:
            logger.error(f"Failed to extract topic from LLM response: {e}")
            return self._create_fallback_topic(user_message, message_index)

    def _create_topic_from_text_response(self, response_text: str, user_message: str, message_index: int) -> TopicInfo:
        """Create topic from LLM text response."""
        # Simple extraction - look for topic mentions in response
        import re

        # Look for topic name patterns
        topic_patterns = [
            r'topic["\s]+([^"]+)["\s]',
            r'["\']([^"\']+)["\']',
            r'topic:?\s*([^\n,.]+)'
        ]

        topic_name = "General Conversation"
        for pattern in topic_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 3 and len(extracted) < 50:
                    topic_name = extracted
                    break

        # Extract domain if mentioned
        domain_patterns = [
            r'domain["\s]+([^"]+)["\s]',
            r'domain:?\s*([^\n,.]+)'
        ]

        domain = ""
        for pattern in domain_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                domain = match.group(1).strip()
                break

        # Generate topic ID
        import uuid
        topic_id = f"topic_{uuid.uuid4().hex[:16]}"

        return TopicInfo(
            topic_id=topic_id,
            topic_name=topic_name,
            domain=domain,
            message_count=1,
            summary=f"Discussion about {topic_name}",
            start_message_index=message_index,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            llm_confidence=0.6,  # Default confidence
            llm_reasoning=f"Extracted from LLM analysis: {response_text[:200]}",
            llm_last_analyzed=datetime.now()
        )

    def _create_fallback_topic(self, user_message: str, message_index: int) -> TopicInfo:
        """Create a fallback topic when LLM analysis fails."""
        import uuid

        # Simple topic extraction
        words = user_message.split()[:5]
        topic_name = " ".join(words) if words else "General Conversation"

        topic_id = f"topic_{uuid.uuid4().hex[:16]}"

        return TopicInfo(
            topic_id=topic_id,
            topic_name=topic_name,
            domain="general",
            message_count=1,
            summary=f"Discussion about {topic_name}",
            start_message_index=message_index,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            llm_confidence=0.3,  # Low confidence for fallback
            llm_reasoning="Fallback topic creation due to analysis failure",
            llm_last_analyzed=datetime.now()
        )

    def get_topic_by_id(self, topic_id: str) -> Optional[TopicInfo]:
        """Get topic by ID."""
        return self.topic_storage.load_topic(topic_id)

    def get_all_topics_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all topics."""
        topics = self.topic_storage.get_all_topics()

        # Sort by recency
        topics.sort(key=lambda x: x.updated_at, reverse=True)

        return [
            {
                "id": topic.topic_id,
                "name": topic.topic_name,
                "domain": topic.domain,
                "message_count": topic.message_count,
                "last_updated": topic.updated_at.isoformat(),
                "summary": topic.summary[:100] + "..." if len(topic.summary) > 100 else topic.summary,
                "llm_confidence": topic.llm_confidence
            }
            for topic in topics
        ]

    def cleanup_old_topics(self, max_age_days: int = 365) -> int:
        """Clean up old topics."""
        return self.topic_storage.cleanup_old_topics(max_age_days)

    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        storage_stats = self.topic_storage.get_topic_stats()

        return {
            **storage_stats,
            "analyzer_type": "llm_tool_calling",
            "llm_provider": getattr(self.llm_adapter, 'provider', 'unknown'),
            "available_tools": len(self.topic_tools.get_all_tools())
        }
