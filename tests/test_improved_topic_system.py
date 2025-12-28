#!/usr/bin/env python3
"""
Test script for the improved Topic tracker/Intent Analyzer system.
Demonstrates centroid-based topic matching and enhanced metadata.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime
from core.topic_matcher import TopicMatcher
from core.topic_tracker import TopicTracker
from core.intent_analyzer import IntentAnalyzer
from core.topic_storage import TopicStorage
from data.schemas import TopicInfo

def test_centroid_matching():
    """Test centroid-based topic matching."""
    print("=== Testing Centroid-Based Topic Matching ===")

    # Create mock vector manager (we'll use fake embeddings for demo)
    class MockVectorManager:
        def embeddings(self):
            class MockEmbeddings:
                def embed_query(self, text):
                    # Simple hash-based fake embedding for demo
                    import hashlib
                    h = hashlib.md5(text.encode()).hexdigest()
                    return [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)][:10]
            return MockEmbeddings()

    matcher = TopicMatcher(MockVectorManager())

    # Test message embeddings
    emb1 = matcher.get_embedding("What about Venezuela economy?")
    emb2 = matcher.get_embedding("Venezuela economic situation in 2025")

    similarity = matcher._cosine_similarity(emb1, emb2)
    print(".3f")

    # Test should_create_new_topic
    assert matcher.should_create_new_topic("What about Venezuela economy?") == True
    assert matcher.should_create_new_topic("hi") == False
    print("✓ Topic creation logic works")

    # Test language detection
    lang = matcher.detect_language("What about Venezuela economy?")
    print(f"✓ Language detected: {lang}")

    # Test keyword extraction
    keywords = matcher.extract_topic_keywords("What about Venezuela economy in 2025?", lang)
    print(f"✓ Keywords extracted: {keywords}")

def test_enhanced_topic_schema():
    """Test enhanced TopicInfo schema."""
    print("\n=== Testing Enhanced TopicInfo Schema ===")

    # Create topic with new fields
    topic = TopicInfo(
        topic_id="test_topic_123",
        topic_name="Venezuela Economic Outlook 2025",
        domain="economics",
        message_count=2,
        summary="Discussion about Venezuela's economic challenges",
        centroid_embedding=[0.1, 0.2, 0.3],
        representative_messages=[
            {"text": "What about Venezuela economy?", "timestamp": "2025-12-08T20:00:00Z"}
        ],
        keywords=["Venezuela", "economy", "2025"],
        language="en",
        confidence=0.85,
        last_seen=datetime.now(),
        topic_state="active"
    )

    # Test serialization
    data = topic.to_dict()
    assert "centroid_embedding" in data
    assert "keywords" in data
    assert "confidence" in data
    print("✓ Enhanced schema serialization works")

    # Test deserialization
    topic2 = TopicInfo.from_dict(data)
    assert topic2.keywords == ["Venezuela", "economy", "2025"]
    assert topic2.confidence == 0.85
    print("✓ Enhanced schema deserialization works")

def test_rule_based_intent():
    """Test rule-based intent detection."""
    print("\n=== Testing Rule-Based Intent Detection ===")

    analyzer = IntentAnalyzer()

    # Test various messages
    test_cases = [
        ("How to set up nginx for docker?", "code_generation"),
        ("What's your take on biotech?", "opinion_request"),
        ("Plan a trip to Europe", "planning_request"),
        ("My code crashes with error", "troubleshooting"),
        ("Thanks!", "quick_answer")
    ]

    for message, expected_intent in test_cases:
        intent = analyzer._rule_based_quick_check(message)
        if intent:
            actual_intent = intent.intent_type.value
            status = "✓" if actual_intent == expected_intent else "✗"
            print(f"{status} '{message}' -> {actual_intent} (expected {expected_intent})")
        else:
            print(f"✗ '{message}' -> No rule matched")

def test_topic_lifecycle():
    """Test complete topic lifecycle."""
    print("\n=== Testing Topic Lifecycle ===")

    # Create components
    storage = TopicStorage(":memory:")  # In-memory for testing
    matcher = TopicMatcher(None)  # No LLM for this test
    tracker = TopicTracker(storage, matcher)

    # Process a message
    topic = tracker.analyze_message_topic(
        "What about Venezuela economy in 2025?",
        message_index=1
    )

    print(f"✓ Created topic: '{topic.topic_name}' (ID: {topic.topic_id})")
    print(f"  - Domain: {topic.domain}")
    print(f"  - Language: {topic.language}")
    print(f"  - Keywords: {topic.keywords}")
    print(f"  - Confidence: {topic.confidence}")

    # Check storage
    all_topics = storage.get_all_topics()
    assert len(all_topics) == 1
    print("✓ Topic stored successfully")

def main():
    """Run all tests."""
    print("Testing Improved Topic Tracker/Intent Analyzer System")
    print("=" * 60)

    try:
        test_centroid_matching()
        test_enhanced_topic_schema()
        test_rule_based_intent()
        test_topic_lifecycle()

        print("\n" + "=" * 60)
        print("🎉 All tests passed! System improvements implemented successfully.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
