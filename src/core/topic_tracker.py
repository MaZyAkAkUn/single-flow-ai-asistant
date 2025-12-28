"""
LLM-based topic tracker for conversation topic management.
Uses LLM tool calling for intelligent topic analysis and management.
"""
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from datetime import datetime

from ..data.schemas import TopicInfo
from .topic_storage import TopicStorage
# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    from .llm_topic_analyzer import LLMTopicAnalyzer
from .logging_config import get_logger

logger = get_logger(__name__)


class TopicTracker:
    """
    LLM-based topic tracking orchestrator.
    Uses tool calling for intelligent topic lifecycle management.
    """

    def __init__(self, topic_storage: TopicStorage, llm_topic_analyzer):
        """
        Initialize LLM-based topic tracker.

        Args:
            topic_storage: TopicStorage instance for persistence
            llm_topic_analyzer: LLMTopicAnalyzer instance for topic analysis
        """
        self.storage = topic_storage
        self.analyzer = llm_topic_analyzer
        self.current_topic: Optional[TopicInfo] = None
        self.conversation_topics: Dict[int, str] = {}  # message_index -> topic_id

        logger.info("LLM-based TopicTracker initialized")

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
        # Use LLM analyzer for topic analysis
        topic_info = self.analyzer.analyze_message_topic(
            user_message, message_index, conversation_context
        )

        # Update current topic and conversation mapping
        self.current_topic = topic_info
        self.conversation_topics[message_index] = topic_info.topic_id

        logger.info(f"Topic analysis completed: '{topic_info.topic_name}' (domain: {topic_info.domain}, confidence: {topic_info.llm_confidence})")
        return topic_info

    def add_message_to_current_topic(self, message: Dict[str, Any]) -> bool:
        """
        Add a message to the current topic's conversation history.

        Args:
            message: Message dictionary with role, content, timestamp, etc.

        Returns:
            True if successful
        """
        if not self.current_topic:
            logger.warning("No current topic set, cannot add message")
            return False

        try:
            # Add message to topic storage
            success = self.storage.add_message_to_topic(self.current_topic.topic_id, message)

            if success:
                logger.info(f"Added message to current topic '{self.current_topic.topic_name}'")
            else:
                logger.error(f"Failed to add message to current topic '{self.current_topic.topic_name}'")

            return success

        except Exception as e:
            logger.error(f"Error adding message to current topic: {e}")
            return False

    def get_topic_conversation_history(self, topic_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for a specific topic.

        Args:
            topic_id: Topic ID to get history for
            limit: Maximum number of messages to return

        Returns:
            List of message dictionaries
        """
        try:
            return self.storage.get_messages_for_topic(topic_id, limit)

        except Exception as e:
            logger.error(f"Failed to get topic conversation history: {e}")
            return []

    def get_current_topic_with_messages(self) -> Dict[str, Any]:
        """
        Get the current topic with its associated messages.

        Returns:
            Dictionary with topic info and messages, or None if no current topic
        """
        if not self.current_topic:
            return None

        try:
            return self.storage.get_topic_with_messages(self.current_topic.topic_id)

        except Exception as e:
            logger.error(f"Failed to get current topic with messages: {e}")
            return None

    def get_related_topics_for_current(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get topics related to the current topic.

        Args:
            limit: Maximum number of related topics to return

        Returns:
            List of related topic summaries
        """
        if not self.current_topic:
            return []

        try:
            return self.storage.find_related_topics(self.current_topic.topic_id, limit=limit)

        except Exception as e:
            logger.error(f"Failed to get related topics: {e}")
            return []

    def get_topic_context_summary(self, topic_id: str = None) -> Dict[str, Any]:
        """
        Get a context summary for a topic.

        Args:
            topic_id: Optional topic ID. If None, uses current topic.

        Returns:
            Dictionary with topic context summary
        """
        try:
            target_topic_id = topic_id or (self.current_topic.topic_id if self.current_topic else None)
            if not target_topic_id:
                return None

            return self.storage.get_topic_context_summary(target_topic_id)

        except Exception as e:
            logger.error(f"Failed to get topic context summary: {e}")
            return None

    def get_current_topic(self) -> Optional[TopicInfo]:
        """Get currently active topic."""
        return self.current_topic

    def get_topic_by_id(self, topic_id: str) -> Optional[TopicInfo]:
        """Get topic by ID."""
        return self.storage.load_topic(topic_id)

    def get_topic_history(self, topic_id: str) -> Dict[str, Any]:
        """
        Get complete history and context for a topic.

        Args:
            topic_id: Topic ID to get history for

        Returns:
            Dictionary with topic info, messages, and related data
        """
        topic = self.storage.load_topic(topic_id)
        if not topic:
            return {}

        # Get messages in this conversation related to the topic
        topic_messages = [
            {"index": msg_idx, "topic_id": tid}
            for msg_idx, tid in self.conversation_topics.items()
            if tid == topic_id
        ]

        # Find related topics (same domain)
        related_topics = []
        if topic.domain:
            domain_topics = self.storage.find_topics_by_domain(topic.domain)
            related_topics = [
                {"id": t.topic_id, "name": t.topic_name}
                for t in domain_topics
                if t.topic_id != topic_id
            ][:5]  # Limit to 5 related topics

        return {
            "topic_info": topic,
            "conversation_messages": topic_messages,
            "related_topics": related_topics,
            "last_activity": topic.updated_at.isoformat()
        }

    def get_all_topics_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all topics for UI display.

        Returns:
            List of topic summaries
        """
        topics = self.storage.get_all_topics()

        # Sort by recency
        topics.sort(key=lambda x: x.updated_at, reverse=True)

        return [
            {
                "id": topic.topic_id,
                "name": topic.topic_name,
                "domain": topic.domain,
                "message_count": topic.message_count,
                "last_updated": topic.updated_at.isoformat(),
                "summary": topic.summary[:100] + "..." if len(topic.summary) > 100 else topic.summary
            }
            for topic in topics
        ]

    def force_topic_switch(self, topic_name: str, domain: str = "") -> TopicInfo:
        """
        Force creation of a new topic (admin/debug function).

        Args:
            topic_name: Name for the new topic
            domain: Domain for the new topic

        Returns:
            Created TopicInfo
        """
        topic = self._create_new_topic(topic_name, domain, 0)
        self.current_topic = topic
        logger.info(f"Force-switched to new topic: '{topic_name}' (ID: {topic.topic_id})")
        return topic

    def update_topic_summary(self, topic_id: str, new_content: str):
        """
        Update topic summary with new content.

        Args:
            topic_id: Topic to update
            new_content: New content to incorporate
        """
        topic = self.storage.load_topic(topic_id)
        if not topic:
            return

        # Simple summary update (can be enhanced with LLM)
        current_summary = topic.summary
        if current_summary:
            # Append new content, keeping reasonable length
            combined = f"{current_summary} {new_content}"
            topic.summary = combined[:500] + "..." if len(combined) > 500 else combined
        else:
            topic.summary = new_content[:200]

        topic.updated_at = datetime.now()
        self.storage.save_topic(topic)

    def _create_new_topic(self, topic_name: str, domain: str, message_index: int) -> TopicInfo:
        """
        Create a new topic (fallback method).

        Args:
            topic_name: Name of the topic
            domain: Domain/category
            message_index: Starting message index

        Returns:
            Created TopicInfo
        """
        import uuid

        topic_id = f"topic_{uuid.uuid4().hex[:16]}"

        topic = TopicInfo(
            topic_id=topic_id,
            topic_name=topic_name,
            domain=domain,
            message_count=1,
            summary=f"Started discussion about {topic_name}",
            start_message_index=message_index,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            llm_confidence=0.5,
            llm_reasoning="Created via fallback method",
            llm_last_analyzed=datetime.now()
        )

        self.storage.save_topic(topic)
        return topic

    def cleanup_old_topics(self, max_age_days: int = 365) -> int:
        """
        Clean up old inactive topics.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of topics cleaned up
        """
        return self.storage.cleanup_old_topics(max_age_days)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get topic tracking statistics.

        Returns:
            Statistics dictionary
        """
        storage_stats = self.storage.get_topic_stats()
        analyzer_stats = self.analyzer.get_statistics()

        return {
            **storage_stats,
            **analyzer_stats,
            "current_topic": self.current_topic.topic_name if self.current_topic else None,
            "conversation_topics_tracked": len(self.conversation_topics)
        }
