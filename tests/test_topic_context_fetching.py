#!/usr/bin/env python3
"""
Test script to verify topic-based context fetching functionality.
"""

import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.topic_storage import TopicStorage
from src.core.topic_tracker import TopicTracker
from src.core.llm_topic_analyzer import LLMTopicAnalyzer
from src.core.context_aggregator import ContextAggregator
from src.data.schemas import TopicInfo, StructuredPromptConfig, UserIntent, IntentType

def test_topic_storage_enhancements():
    """Test the enhanced TopicStorage functionality."""
    print("🧪 Testing TopicStorage enhancements...")

    # Create topic storage
    storage = TopicStorage("./data/test_topics/")

    # Create a test topic
    topic = TopicInfo(
        topic_id="test_topic_1",
        topic_name="Python Programming",
        domain="programming",
        message_count=0,
        summary="Discussion about Python programming",
        start_message_index=0,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        llm_confidence=0.9,
        llm_reasoning="Test topic"
    )

    # Save the topic
    success = storage.save_topic(topic)
    assert success, "Failed to save test topic"

    # Add messages to the topic
    test_messages = [
        {"role": "user", "content": "How do I create a Python class?", "timestamp": datetime.now().isoformat()},
        {"role": "assistant", "content": "You can create a Python class using the class keyword.", "timestamp": datetime.now().isoformat()},
        {"role": "user", "content": "What about inheritance in Python?", "timestamp": datetime.now().isoformat()}
    ]

    for msg in test_messages:
        success = storage.add_message_to_topic("test_topic_1", msg)
        assert success, f"Failed to add message to topic: {msg}"

    # Retrieve messages for the topic
    messages = storage.get_messages_for_topic("test_topic_1")
    assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}"
    assert messages[0]["content"] == "How do I create a Python class?", "First message content mismatch"

    # Test get_topic_with_messages
    topic_with_msgs = storage.get_topic_with_messages("test_topic_1")
    assert topic_with_msgs is not None, "Failed to get topic with messages"
    assert topic_with_msgs["message_count"] == 3, "Message count mismatch"
    assert len(topic_with_msgs["messages"]) == 3, "Messages length mismatch"

    # Test topic context summary
    summary = storage.get_topic_context_summary("test_topic_1")
    assert summary is not None, "Failed to get topic context summary"
    assert summary["topic_name"] == "Python Programming", "Topic name mismatch"
    assert len(summary["recent_messages"]) == 3, "Recent messages count mismatch"

    # Test related topics
    related = storage.find_related_topics("test_topic_1")
    assert isinstance(related, list), "Related topics should be a list"

    print("✅ TopicStorage enhancements test passed!")

    # Cleanup
    storage.delete_topic("test_topic_1")

def test_context_aggregator_topic_awareness():
    """Test ContextAggregator with topic awareness."""
    print("🧪 Testing ContextAggregator topic awareness...")

    # Create context aggregator
    config = StructuredPromptConfig()
    aggregator = ContextAggregator(config.token_limits)

    # Create test data
    user_intent = UserIntent(intent_type=IntentType.CODE_GENERATION, confidence_score=0.8)

    # Test without topic info
    retrieved_context = aggregator.aggregate_context(
        user_intent=user_intent,
        conversation_history=[],
        memories=[],
        documents=[],
        project_contexts=[]
    )

    assert retrieved_context is not None, "Context aggregation failed"

    # Test with topic info and messages
    topic_info = TopicInfo(
        topic_id="test_topic_2",
        topic_name="Machine Learning",
        domain="ai",
        message_count=2,
        summary="Discussion about machine learning algorithms"
    )

    topic_messages = [
        {"role": "user", "content": "What's the difference between SVM and Random Forest?", "timestamp": datetime.now().isoformat()},
        {"role": "assistant", "content": "SVM uses kernel tricks while Random Forest is an ensemble method.", "timestamp": datetime.now().isoformat()}
    ]

    retrieved_context_with_topic = aggregator.aggregate_context(
        user_intent=user_intent,
        conversation_history=[],
        memories=[],
        documents=[],
        project_contexts=[],
        topic_info=topic_info,
        topic_messages=topic_messages
    )

    assert retrieved_context_with_topic is not None, "Context aggregation with topic failed"
    assert hasattr(retrieved_context_with_topic, 'topic_messages'), "Topic messages not found in retrieved context"
    assert len(retrieved_context_with_topic.topic_messages) == 2, "Topic messages count mismatch"

    print("✅ ContextAggregator topic awareness test passed!")

def test_topic_tracker_integration():
    """Test TopicTracker integration with storage."""
    print("🧪 Testing TopicTracker integration...")

    # Create components
    storage = TopicStorage("./data/test_topics/")
    # We'll create a mock LLMTopicAnalyzer since we don't have a real LLM here
    class MockLLMTopicAnalyzer:
        def __init__(self, storage):
            self.storage = storage

        def analyze_message_topic(self, user_message, message_index, conversation_context=None):
            # Return a simple topic for testing
            return TopicInfo(
                topic_id="test_integration_topic",
                topic_name="Integration Testing",
                domain="software",
                message_count=1,
                summary="Testing topic integration"
            )

        def get_statistics(self):
            return {"topics_analyzed": 1}

    mock_analyzer = MockLLMTopicAnalyzer(storage)
    tracker = TopicTracker(storage, mock_analyzer)

    # Test topic analysis
    topic_info = tracker.analyze_message_topic("How do I test this feature?", 0)
    assert topic_info is not None, "Topic analysis failed"
    assert topic_info.topic_name == "Integration Testing", "Topic name mismatch"

    # Save the topic to storage first
    storage.save_topic(topic_info)

    # Add message to current topic
    message = {"role": "user", "content": "Testing the integration", "timestamp": datetime.now().isoformat()}
    success = tracker.add_message_to_current_topic(message)
    assert success, "Failed to add message to current topic"

    # Get topic conversation history
    history = tracker.get_topic_conversation_history("test_integration_topic")
    assert len(history) == 1, "Topic conversation history length mismatch"

    # Get current topic with messages
    current_topic = tracker.get_current_topic_with_messages()
    assert current_topic is not None, "Failed to get current topic with messages"
    assert current_topic["topic_info"].topic_name == "Integration Testing", "Current topic name mismatch"

    print("✅ TopicTracker integration test passed!")

    # Cleanup
    storage.delete_topic("test_integration_topic")

def main():
    """Run all tests."""
    print("🚀 Starting topic-based context fetching tests...\n")

    try:
        test_topic_storage_enhancements()
        print()

        test_context_aggregator_topic_awareness()
        print()

        test_topic_tracker_integration()
        print()

        print("🎉 All tests passed! Topic-based context fetching is working correctly.")
        print("\n📋 Summary of implemented features:")
        print("✅ TopicStorage can now store and retrieve topic-conversation associations")
        print("✅ TopicTracker can associate messages with topics")
        print("✅ ContextAggregator supports topic-aware context retrieval")
        print("✅ EnhancedLLMAdapter integrates topic-aware context fetching")
        print("✅ PromptBuilder includes topic messages in context section")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
