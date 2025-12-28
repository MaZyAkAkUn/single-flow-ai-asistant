#!/usr/bin/env python3
"""
Test script to verify that intent and topic information is properly injected into prompts.
"""

import sys
import os
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'data'))

from core.prompt_builder import StructuredPromptBuilder
from data.schemas import UserIntent, IntentType, TopicInfo

def test_intent_injection():
    """Test that intent information is included in system section."""
    print("=== Testing Intent Information Injection ===")

    builder = StructuredPromptBuilder()

    # Create a test intent
    intent = UserIntent(
        intent_type=IntentType.CODE_GENERATION,
        confidence_score=0.9,
        expected_detail_level="comprehensive",
        context_requirements=["programming_language"]
    )

    # Build prompt
    prompt, flattened = builder.build_structured_prompt(
        user_message="Write a Python function to calculate fibonacci numbers",
        conversation_history=[],
        retrieved_context=None,  # Will use empty context
        agent_state=None,  # Will use defaults
        conversation_frame=None,  # Will use defaults
        memory_rules=None,  # Will use defaults
        user_personalization=None,  # Will use defaults
        user_intent=intent,
        settings=None
    )

    # Parse the JSON prompt
    prompt_data = json.loads(prompt)

    # Check that intent is in system section
    system = prompt_data.get('system', {})
    user_intent_data = system.get('user_intent', {})

    assert user_intent_data.get('type') == 'code_generation', f"Expected 'code_generation', got {user_intent_data.get('type')}"
    assert user_intent_data.get('confidence') == 0.9, f"Expected 0.9, got {user_intent_data.get('confidence')}"
    assert user_intent_data.get('detail_level') == 'comprehensive', f"Expected 'comprehensive', got {user_intent_data.get('detail_level')}"

    print("✓ Intent information correctly injected into system section")

    # Check that new rule is included
    rules = system.get('rules', [])
    has_intent_rule = any("user_intent" in rule for rule in rules)
    assert has_intent_rule, "Expected rule about adapting based on user_intent"
    print("✓ New rule about user_intent adaptation included")

def test_topic_injection():
    """Test that topic information is always included in context section."""
    print("\n=== Testing Topic Information Injection ===")

    builder = StructuredPromptBuilder()

    # Test 1: With topic info
    topic = TopicInfo(
        topic_id="test_topic_123",
        topic_name="Python Programming",
        domain="programming",
        message_count=5,
        summary="Discussion about Python programming concepts",
        keywords=["python", "programming", "code"]
    )

    prompt, _ = builder.build_structured_prompt(
        user_message="How do I use list comprehensions?",
        conversation_history=[],
        retrieved_context=None,
        agent_state=None,
        conversation_frame=None,
        memory_rules=None,
        user_personalization=None,
        user_intent=None,
        settings=None,
        topic_info=topic
    )

    prompt_data = json.loads(prompt)
    context = prompt_data.get('context', {})
    topic_data = context.get('topic', {})

    assert topic_data.get('status') == 'detected', f"Expected 'detected', got {topic_data.get('status')}"
    assert topic_data.get('name') == 'Python Programming', f"Expected 'Python Programming', got {topic_data.get('name')}"
    assert topic_data.get('domain') == 'programming', f"Expected 'programming', got {topic_data.get('domain')}"

    print("✓ Topic information correctly injected when available")

    # Test 2: Without topic info (should show "not_detected_yet")
    prompt2, _ = builder.build_structured_prompt(
        user_message="What is machine learning?",
        conversation_history=[],
        retrieved_context=None,
        agent_state=None,
        conversation_frame=None,
        memory_rules=None,
        user_personalization=None,
        user_intent=None,
        settings=None,
        topic_info=None  # No topic info
    )

    prompt_data2 = json.loads(prompt2)
    context2 = prompt_data2.get('context', {})
    topic_data2 = context2.get('topic', {})

    assert topic_data2.get('status') == 'not_detected_yet', f"Expected 'not_detected_yet', got {topic_data2.get('status')}"
    assert 'message' in topic_data2, "Expected fallback message to be present"

    print("✓ Fallback topic information correctly injected when not available")

def test_no_intent_fallback():
    """Test that system works without intent information."""
    print("\n=== Testing No Intent Fallback ===")

    builder = StructuredPromptBuilder()

    prompt, _ = builder.build_structured_prompt(
        user_message="Hello world",
        conversation_history=[],
        retrieved_context=None,
        agent_state=None,
        conversation_frame=None,
        memory_rules=None,
        user_personalization=None,
        user_intent=None,  # No intent
        settings=None
    )

    prompt_data = json.loads(prompt)
    system = prompt_data.get('system', {})
    user_intent_data = system.get('user_intent', {})

    assert user_intent_data.get('type') == 'not_analyzed', f"Expected 'not_analyzed', got {user_intent_data.get('type')}"
    assert user_intent_data.get('confidence') == 0.0, f"Expected 0.0, got {user_intent_data.get('confidence')}"

    print("✓ System correctly handles missing intent information")

def main():
    """Run all tests."""
    print("Testing Topic and Intent Injection into Prompts")
    print("=" * 60)

    try:
        test_intent_injection()
        test_topic_injection()
        test_no_intent_fallback()

        print("\n" + "=" * 60)
        print("🎉 All tests passed! Topic and intent injection working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
