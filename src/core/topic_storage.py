"""
Topic storage for persistent conversation topics.
Handles saving/loading topics to/from JSON files.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..data.schemas import TopicInfo
from .logging_config import get_logger

logger = get_logger(__name__)


class TopicStorage:
    """
    Persistent storage for conversation topics using JSON files.
    """

    def __init__(self, storage_path: str = "./data/topics/"):
        """
        Initialize topic storage.

        Args:
            storage_path: Directory to store topic files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.topic_db: Dict[str, TopicInfo] = {}
        self.topic_index: Dict[str, List[str]] = {}  # topic_name -> [topic_ids]
        self.topic_messages: Dict[str, List[Dict[str, Any]]] = {}  # topic_id -> [message_dicts]

        # Load existing topics
        self._load_all_topics()
        # Load topic messages
        self._load_all_topic_messages()
        logger.info(f"TopicStorage initialized with {len(self.topic_db)} topics at {storage_path}")

    def save_topic(self, topic: TopicInfo) -> bool:
        """
        Save a topic to persistent storage.

        Args:
            topic: TopicInfo to save

        Returns:
            True if successful
        """
        try:
            # Update in-memory cache
            self.topic_db[topic.topic_id] = topic

            # Update index
            self._update_index(topic)

            # Save to file
            file_path = self._get_topic_file_path(topic.topic_id)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(topic.to_dict(), f, ensure_ascii=False, indent=2)

            logger.info(f"Saved topic '{topic.topic_name}' (ID: {topic.topic_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to save topic {topic.topic_id}: {e}")
            return False

    def load_topic(self, topic_id: str) -> Optional[TopicInfo]:
        """
        Load a topic from storage.

        Args:
            topic_id: Topic ID to load

        Returns:
            TopicInfo if found, None otherwise
        """
        # Check cache first
        if topic_id in self.topic_db:
            return self.topic_db[topic_id]

        # Load from file
        try:
            file_path = self._get_topic_file_path(topic_id)
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                topic = TopicInfo.from_dict(data)

                # Add to cache
                self.topic_db[topic_id] = topic
                self._update_index(topic)

                return topic

        except Exception as e:
            logger.error(f"Failed to load topic {topic_id}: {e}")

        return None

    def get_all_topics(self) -> List[TopicInfo]:
        """
        Get all stored topics.

        Returns:
            List of all TopicInfo objects
        """
        return list(self.topic_db.values())

    def find_topics_by_name(self, topic_name: str, limit: int = 10) -> List[TopicInfo]:
        """
        Find topics by name (partial match).

        Args:
            topic_name: Topic name to search for
            limit: Maximum number of results

        Returns:
            List of matching TopicInfo objects
        """
        results = []
        search_name = topic_name.lower()

        for topic in self.topic_db.values():
            if search_name in topic.topic_name.lower():
                results.append(topic)

        # Sort by recency
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results[:limit]

    def find_topics_by_domain(self, domain: str) -> List[TopicInfo]:
        """
        Find topics by domain.

        Args:
            domain: Domain to search for

        Returns:
            List of matching TopicInfo objects
        """
        results = []
        search_domain = domain.lower()

        for topic in self.topic_db.values():
            if search_domain in topic.domain.lower():
                results.append(topic)

        # Sort by recency
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results

    def delete_topic(self, topic_id: str) -> bool:
        """
        Delete a topic from storage.

        Args:
            topic_id: Topic ID to delete

        Returns:
            True if successful
        """
        try:
            # Remove from cache
            if topic_id in self.topic_db:
                topic = self.topic_db[topic_id]
                del self.topic_db[topic_id]

                # Update index
                self._remove_from_index(topic)

            # Delete file
            file_path = self._get_topic_file_path(topic_id)
            if file_path.exists():
                file_path.unlink()

            logger.info(f"Deleted topic {topic_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete topic {topic_id}: {e}")
            return False

    def get_topic_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored topics.

        Returns:
            Dictionary with statistics
        """
        total_topics = len(self.topic_db)
        domains = {}
        total_messages = 0

        for topic in self.topic_db.values():
            # Count domains
            domain = topic.domain or "unknown"
            domains[domain] = domains.get(domain, 0) + 1

            # Count messages
            total_messages += topic.message_count

        return {
            "total_topics": total_topics,
            "total_messages": total_messages,
            "domains": domains,
            "avg_messages_per_topic": total_messages / max(total_topics, 1)
        }

    def _get_topic_file_path(self, topic_id: str) -> Path:
        """Get file path for a topic."""
        return self.storage_path / f"{topic_id}.json"

    def _load_all_topics(self):
        """Load all topics from storage directory."""
        if not self.storage_path.exists():
            return

        loaded_count = 0
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                topic = TopicInfo.from_dict(data)

                self.topic_db[topic.topic_id] = topic
                self._update_index(topic)
                loaded_count += 1

            except Exception as e:
                logger.warning(f"Failed to load topic from {file_path}: {e}")

        logger.info(f"Loaded {loaded_count} topics from storage")

    def _update_index(self, topic: TopicInfo):
        """Update the topic name index."""
        name_key = topic.topic_name.lower()
        if name_key not in self.topic_index:
            self.topic_index[name_key] = []
        if topic.topic_id not in self.topic_index[name_key]:
            self.topic_index[name_key].append(topic.topic_id)

    def _remove_from_index(self, topic: TopicInfo):
        """Remove topic from index."""
        name_key = topic.topic_name.lower()
        if name_key in self.topic_index:
            if topic.topic_id in self.topic_index[name_key]:
                self.topic_index[name_key].remove(topic.topic_id)
            if not self.topic_index[name_key]:
                del self.topic_index[name_key]

    def cleanup_old_topics(self, max_age_days: int = 365) -> int:
        """
        Clean up topics older than specified days.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of topics deleted
        """
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - max_age_days)

        topics_to_delete = []
        for topic in self.topic_db.values():
            if topic.updated_at < cutoff_date:
                topics_to_delete.append(topic.topic_id)

        deleted_count = 0
        for topic_id in topics_to_delete:
            if self.delete_topic(topic_id):
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old topics (older than {max_age_days} days)")

        return deleted_count

    def add_message_to_topic(self, topic_id: str, message: Dict[str, Any]) -> bool:
        """
        Add a message to a topic's conversation history.

        Args:
            topic_id: Topic ID to add message to
            message: Message dictionary with role, content, timestamp, etc.

        Returns:
            True if successful
        """
        try:
            # Initialize topic messages list if it doesn't exist
            if topic_id not in self.topic_messages:
                self.topic_messages[topic_id] = []

            # Add message to the topic
            self.topic_messages[topic_id].append(message)

            # Update topic's message count
            if topic_id in self.topic_db:
                topic = self.topic_db[topic_id]
                topic.message_count = len(self.topic_messages[topic_id])
                topic.updated_at = datetime.now()
                self.save_topic(topic)

            logger.info(f"Added message to topic {topic_id} (total: {len(self.topic_messages[topic_id])})")
            return True

        except Exception as e:
            logger.error(f"Failed to add message to topic {topic_id}: {e}")
            return False

    def get_messages_for_topic(self, topic_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get all messages associated with a topic.

        Args:
            topic_id: Topic ID to get messages for
            limit: Maximum number of messages to return

        Returns:
            List of message dictionaries
        """
        try:
            messages = self.topic_messages.get(topic_id, [])

            # Apply limit if specified
            if limit is not None and len(messages) > limit:
                return messages[-limit:]  # Return most recent messages

            return messages.copy()

        except Exception as e:
            logger.error(f"Failed to get messages for topic {topic_id}: {e}")
            return []

    def get_topic_with_messages(self, topic_id: str) -> Dict[str, Any]:
        """
        Get a topic with its associated messages.

        Args:
            topic_id: Topic ID to retrieve

        Returns:
            Dictionary with topic info and messages
        """
        try:
            topic = self.load_topic(topic_id)
            if not topic:
                return None

            messages = self.get_messages_for_topic(topic_id)

            return {
                "topic_info": topic,
                "messages": messages,
                "message_count": len(messages)
            }

        except Exception as e:
            logger.error(f"Failed to get topic with messages {topic_id}: {e}")
            return None

    def find_related_topics(self, current_topic_id: str, domain: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find topics related to the current topic.

        Args:
            current_topic_id: Current topic ID
            domain: Optional domain filter
            limit: Maximum number of related topics to return

        Returns:
            List of related topic summaries
        """
        try:
            related_topics = []

            # Get current topic
            current_topic = self.load_topic(current_topic_id)
            if not current_topic:
                return related_topics

            # Find topics in the same domain
            if domain or current_topic.domain:
                search_domain = domain or current_topic.domain
                domain_topics = self.find_topics_by_domain(search_domain)

                # Exclude current topic
                for topic in domain_topics:
                    if topic.topic_id != current_topic_id:
                        related_topics.append({
                            "topic_id": topic.topic_id,
                            "topic_name": topic.topic_name,
                            "domain": topic.domain,
                            "message_count": topic.message_count,
                            "summary": topic.summary[:100] + "..." if len(topic.summary) > 100 else topic.summary,
                            "relatedness": "same_domain"
                        })

            # Find topics with similar names
            if current_topic.topic_name:
                similar_topics = self.find_topics_by_name(current_topic.topic_name)
                for topic in similar_topics:
                    if topic.topic_id != current_topic_id and topic.topic_id not in [t["topic_id"] for t in related_topics]:
                        related_topics.append({
                            "topic_id": topic.topic_id,
                            "topic_name": topic.topic_name,
                            "domain": topic.domain,
                            "message_count": topic.message_count,
                            "summary": topic.summary[:100] + "..." if len(topic.summary) > 100 else topic.summary,
                            "relatedness": "similar_name"
                        })

            # Limit results
            return related_topics[:limit]

        except Exception as e:
            logger.error(f"Failed to find related topics: {e}")
            return []

    def get_topic_context_summary(self, topic_id: str) -> Dict[str, Any]:
        """
        Get a summary of context for a topic.

        Args:
            topic_id: Topic ID to summarize

        Returns:
            Dictionary with topic context summary
        """
        try:
            topic = self.load_topic(topic_id)
            if not topic:
                return None

            messages = self.get_messages_for_topic(topic_id, limit=5)  # Get recent messages

            # Extract key information from messages
            message_summaries = []
            for msg in messages:
                content_preview = msg.get('content', '')[:50] + "..." if len(msg.get('content', '')) > 50 else msg.get('content', '')
                message_summaries.append({
                    "role": msg.get('role', 'unknown'),
                    "content_preview": content_preview,
                    "timestamp": msg.get('timestamp', '')
                })

            return {
                "topic_id": topic.topic_id,
                "topic_name": topic.topic_name,
                "domain": topic.domain,
                "message_count": topic.message_count,
                "summary": topic.summary,
                "recent_messages": message_summaries,
                "last_updated": topic.updated_at.isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get topic context summary: {e}")
            return None

    def save_topic_messages(self, topic_id: str) -> bool:
        """
        Save topic messages to a separate file for persistence.

        Args:
            topic_id: Topic ID to save messages for

        Returns:
            True if successful
        """
        try:
            if topic_id not in self.topic_messages or not self.topic_messages[topic_id]:
                return True  # Nothing to save

            messages_file = self.storage_path / f"{topic_id}_messages.json"
            with open(messages_file, 'w', encoding='utf-8') as f:
                json.dump(self.topic_messages[topic_id], f, ensure_ascii=False, indent=2)

            logger.info(f"Saved {len(self.topic_messages[topic_id])} messages for topic {topic_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save topic messages: {e}")
            return False

    def load_topic_messages(self, topic_id: str) -> bool:
        """
        Load topic messages from file.

        Args:
            topic_id: Topic ID to load messages for

        Returns:
            True if successful
        """
        try:
            messages_file = self.storage_path / f"{topic_id}_messages.json"
            if messages_file.exists():
                with open(messages_file, 'r', encoding='utf-8') as f:
                    messages = json.load(f)
                self.topic_messages[topic_id] = messages
                logger.info(f"Loaded {len(messages)} messages for topic {topic_id}")
                return True

            return True  # File doesn't exist, which is fine

        except Exception as e:
            logger.error(f"Failed to load topic messages: {e}")
            return False

    def _load_all_topic_messages(self):
        """
        Load messages for all topics.
        """
        if not self.storage_path.exists():
            return

        for file_path in self.storage_path.glob("*_messages.json"):
            try:
                topic_id = file_path.stem.replace("_messages", "")
                if topic_id in self.topic_db:  # Only load messages for existing topics
                    with open(file_path, 'r', encoding='utf-8') as f:
                        messages = json.load(f)
                    self.topic_messages[topic_id] = messages
                    logger.info(f"Loaded {len(messages)} messages for topic {topic_id}")

            except Exception as e:
                logger.warning(f"Failed to load topic messages from {file_path}: {e}")

    def cleanup_topic_messages(self, topic_id: str, max_messages: int = 100) -> int:
        """
        Clean up old messages for a topic, keeping only the most recent.

        Args:
            topic_id: Topic ID to clean up
            max_messages: Maximum number of messages to keep

        Returns:
            Number of messages removed
        """
        try:
            if topic_id not in self.topic_messages:
                return 0

            messages = self.topic_messages[topic_id]
            if len(messages) <= max_messages:
                return 0

            removed_count = len(messages) - max_messages
            self.topic_messages[topic_id] = messages[-max_messages:]  # Keep most recent

            # Update topic message count
            if topic_id in self.topic_db:
                topic = self.topic_db[topic_id]
                topic.message_count = len(self.topic_messages[topic_id])
                self.save_topic(topic)

            logger.info(f"Cleaned up {removed_count} old messages for topic {topic_id}")
            return removed_count

        except Exception as e:
            logger.error(f"Failed to cleanup topic messages: {e}")
            return 0
