"""
Topic management tools for LLM-based topic tracking using tool calling.
Provides MCP-compatible tools that LLMs can use to manage conversation topics.
"""
import json
from typing import List, Dict, Optional, Any
from langchain.tools import tool
from ..core.logging_config import get_logger
from .topic_storage import TopicStorage

logger = get_logger(__name__)


class TopicTools:
    """
    Collection of tools for LLM-based topic management.
    These tools allow LLMs to perform topic operations through tool calling.
    """

    def __init__(self, topic_storage: TopicStorage):
        """
        Initialize topic tools.

        Args:
            topic_storage: TopicStorage instance for persistence
        """
        self.topic_storage = topic_storage
        logger.info("TopicTools initialized with storage backend")

    def get_all_tools(self) -> List[Any]:
        """
        Get all available topic management tools.

        Returns:
            List of LangChain tool instances
        """
        return [
            self.get_available_topics_tool(),
            self.create_new_topic_tool(),
            self.update_topic_summary_tool(),
            self.merge_topics_tool(),
            self.get_topic_details_tool(),
        ]

    def get_available_topics_tool(self):
        """Tool to retrieve existing topics with summaries."""
        topic_storage = self.topic_storage

        @tool("get_available_topics")
        def get_available_topics(limit: int = 10) -> str:
            """
            Retrieve existing conversation topics with their summaries.

            Args:
                limit: Maximum number of topics to return (default: 10)

            Returns:
                JSON string with available topics
            """
            try:
                topics = topic_storage.get_all_topics()

                # Sort by recency
                topics.sort(key=lambda x: x.updated_at, reverse=True)

                # Format topics for LLM consumption
                topic_list = []
                for topic in topics[:limit]:
                    topic_data = {
                        "topic_id": topic.topic_id,
                        "topic_name": topic.topic_name,
                        "domain": topic.domain,
                        "message_count": topic.message_count,
                        "summary": topic.summary[:200] + "..." if len(topic.summary) > 200 else topic.summary,
                        "llm_confidence": topic.llm_confidence,
                        "last_updated": topic.updated_at.isoformat()
                    }
                    topic_list.append(topic_data)

                result = {
                    "topics": topic_list,
                    "total_count": len(topics),
                    "returned_count": len(topic_list)
                }

                logger.info(f"Retrieved {len(topic_list)} topics for LLM analysis")
                return json.dumps(result, ensure_ascii=False)

            except Exception as e:
                logger.error(f"Failed to get available topics: {e}")
                return json.dumps({"error": f"Failed to retrieve topics: {str(e)}"})

        return get_available_topics

    def create_new_topic_tool(self):
        """Tool to create a new topic."""
        topic_storage = self.topic_storage

        @tool("create_new_topic")
        def create_new_topic(
            topic_name: str,
            domain: str = "",
            summary: str = "",
            message_index: int = 0,
            confidence: float = 0.5,
            reasoning: str = ""
        ) -> str:
            """
            Create a new conversation topic.

            Args:
                topic_name: Name of the topic (2-5 words describing the subject)
                domain: Topic domain/category (e.g., 'programming', 'economics', 'military')
                summary: Brief summary of the topic (1-2 sentences)
                message_index: Starting message index in conversation
                confidence: LLM confidence score (0.0-1.0)
                reasoning: LLM reasoning for topic creation

            Returns:
                JSON string with created topic information
            """
            try:
                from datetime import datetime
                import uuid

                # Generate unique topic ID
                topic_id = f"topic_{uuid.uuid4().hex[:16]}"

                # Import TopicInfo here to avoid circular imports
                from ..data.schemas import TopicInfo

                # Create new topic
                topic = TopicInfo(
                    topic_id=topic_id,
                    topic_name=topic_name,
                    domain=domain,
                    message_count=1,
                    summary=summary,
                    start_message_index=message_index,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    llm_confidence=confidence,
                    llm_reasoning=reasoning,
                    llm_last_analyzed=datetime.now()
                )

                # Save to storage
                success = topic_storage.save_topic(topic)

                if success:
                    result = {
                        "success": True,
                        "topic_id": topic_id,
                        "topic_name": topic_name,
                        "domain": domain,
                        "message": f"Created new topic '{topic_name}' in domain '{domain}'"
                    }
                    logger.info(f"LLM created new topic: {topic_name} (ID: {topic_id})")
                else:
                    result = {
                        "success": False,
                        "error": "Failed to save topic to storage"
                    }
                    logger.error(f"Failed to save new topic: {topic_name}")

                return json.dumps(result, ensure_ascii=False)

            except Exception as e:
                logger.error(f"Failed to create new topic: {e}")
                return json.dumps({"success": False, "error": f"Failed to create topic: {str(e)}"})

        return create_new_topic

    def update_topic_summary_tool(self):
        """Tool to update an existing topic's summary and metadata."""
        topic_storage = self.topic_storage

        @tool("update_topic_summary")
        def update_topic_summary(
            topic_id: str,
            new_content: str,
            confidence: float = 0.5,
            reasoning: str = ""
        ) -> str:
            """
            Update an existing topic's summary with new content.

            Args:
                topic_id: ID of the topic to update
                new_content: New content to incorporate into the topic
                confidence: LLM confidence score for the update
                reasoning: LLM reasoning for the update

            Returns:
                JSON string with update result
            """
            try:
                from datetime import datetime

                # Load existing topic
                topic = topic_storage.load_topic(topic_id)
                if not topic:
                    return json.dumps({"success": False, "error": f"Topic {topic_id} not found"})

                # Update topic with new content
                current_summary = topic.summary
                if current_summary:
                    # Combine summaries intelligently
                    combined = f"{current_summary} {new_content}"
                    topic.summary = combined[:500] + "..." if len(combined) > 500 else combined
                else:
                    topic.summary = new_content[:200]

                # Update metadata
                topic.message_count += 1
                topic.updated_at = datetime.now()
                topic.llm_confidence = confidence
                topic.llm_reasoning = reasoning
                topic.llm_last_analyzed = datetime.now()

                # Save updated topic
                success = topic_storage.save_topic(topic)

                if success:
                    result = {
                        "success": True,
                        "topic_id": topic_id,
                        "topic_name": topic.topic_name,
                        "updated_summary": topic.summary[:100] + "..." if len(topic.summary) > 100 else topic.summary,
                        "message_count": topic.message_count,
                        "message": f"Updated topic '{topic.topic_name}' with new content"
                    }
                    logger.info(f"LLM updated topic {topic_id}: {topic.topic_name}")
                else:
                    result = {"success": False, "error": "Failed to save topic updates"}

                return json.dumps(result, ensure_ascii=False)

            except Exception as e:
                logger.error(f"Failed to update topic summary: {e}")
                return json.dumps({"success": False, "error": f"Failed to update topic: {str(e)}"})

        return update_topic_summary

    def merge_topics_tool(self):
        """Tool to merge two topics into one."""
        topic_storage = self.topic_storage

        @tool("merge_topics")
        def merge_topics(
            primary_topic_id: str,
            secondary_topic_id: str,
            reasoning: str = ""
        ) -> str:
            """
            Merge two topics into one, keeping the primary topic and absorbing the secondary.

            Args:
                primary_topic_id: ID of the primary topic to keep
                secondary_topic_id: ID of the secondary topic to merge into primary
                reasoning: LLM reasoning for the merge decision

            Returns:
                JSON string with merge result
            """
            try:
                from datetime import datetime

                # Load both topics
                primary_topic = topic_storage.load_topic(primary_topic_id)
                secondary_topic = topic_storage.load_topic(secondary_topic_id)

                if not primary_topic:
                    return json.dumps({"success": False, "error": f"Primary topic {primary_topic_id} not found"})
                if not secondary_topic:
                    return json.dumps({"success": False, "error": f"Secondary topic {secondary_topic_id} not found"})

                # Merge topic data
                primary_topic.message_count += secondary_topic.message_count
                primary_topic.updated_at = datetime.now()

                # Combine summaries
                if secondary_topic.summary:
                    if primary_topic.summary:
                        primary_topic.summary = f"{primary_topic.summary} {secondary_topic.summary}"
                        # Truncate if too long
                        if len(primary_topic.summary) > 500:
                            primary_topic.summary = primary_topic.summary[:497] + "..."
                    else:
                        primary_topic.summary = secondary_topic.summary

                # Update LLM metadata
                primary_topic.llm_reasoning = f"Merged with topic '{secondary_topic.topic_name}': {reasoning}"
                primary_topic.llm_last_analyzed = datetime.now()

                # Save updated primary topic
                success = topic_storage.save_topic(primary_topic)

                if success:
                    # Delete the secondary topic
                    delete_success = topic_storage.delete_topic(secondary_topic_id)

                    result = {
                        "success": True,
                        "primary_topic_id": primary_topic_id,
                        "primary_topic_name": primary_topic.topic_name,
                        "merged_topic_id": secondary_topic_id,
                        "merged_topic_name": secondary_topic.topic_name,
                        "final_message_count": primary_topic.message_count,
                        "message": f"Merged topic '{secondary_topic.topic_name}' into '{primary_topic.topic_name}'"
                    }

                    logger.info(f"LLM merged topics: {secondary_topic.topic_name} -> {primary_topic.topic_name}")
                else:
                    result = {"success": False, "error": "Failed to save merged topic"}

                return json.dumps(result, ensure_ascii=False)

            except Exception as e:
                logger.error(f"Failed to merge topics: {e}")
                return json.dumps({"success": False, "error": f"Failed to merge topics: {str(e)}"})

        return merge_topics

    def get_topic_details_tool(self):
        """Tool to get detailed information about a specific topic."""
        topic_storage = self.topic_storage

        @tool("get_topic_details")
        def get_topic_details(topic_id: str) -> str:
            """
            Get detailed information about a specific topic.

            Args:
                topic_id: ID of the topic to retrieve

            Returns:
                JSON string with detailed topic information
            """
            try:
                topic = topic_storage.load_topic(topic_id)
                if not topic:
                    return json.dumps({"success": False, "error": f"Topic {topic_id} not found"})

                # Get related topics (same domain)
                related_topics = []
                if topic.domain:
                    domain_topics = topic_storage.find_topics_by_domain(topic.domain)
                    related_topics = [
                        {
                            "topic_id": t.topic_id,
                            "topic_name": t.topic_name,
                            "message_count": t.message_count
                        }
                        for t in domain_topics
                        if t.topic_id != topic_id
                    ][:3]  # Limit to 3 related topics

                topic_details = {
                    "success": True,
                    "topic": {
                        "topic_id": topic.topic_id,
                        "topic_name": topic.topic_name,
                        "domain": topic.domain,
                        "message_count": topic.message_count,
                        "summary": topic.summary,
                        "start_message_index": topic.start_message_index,
                        "tags": topic.tags,
                        "created_at": topic.created_at.isoformat(),
                        "updated_at": topic.updated_at.isoformat(),
                        "llm_confidence": topic.llm_confidence,
                        "llm_reasoning": topic.llm_reasoning,
                        "llm_analysis_metadata": topic.llm_analysis_metadata
                    },
                    "related_topics": related_topics,
                    "related_count": len(related_topics)
                }

                logger.info(f"Retrieved details for topic: {topic.topic_name} (ID: {topic_id})")
                return json.dumps(topic_details, ensure_ascii=False)

            except Exception as e:
                logger.error(f"Failed to get topic details: {e}")
                return json.dumps({"success": False, "error": f"Failed to get topic details: {str(e)}"})

        return get_topic_details
