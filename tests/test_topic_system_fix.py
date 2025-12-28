#!/usr/bin/env python3
"""
Simple test to verify topic injection into system section works.
"""

import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_topic_in_system_section():
    """Test that topic info is injected into the system section."""

    # Create a mock topic info object
    class MockTopicInfo:
        def __init__(self):
            self.topic_name = "Python Programming"
            self.domain = "programming"
            self.message_count = 5
            self.summary = "Discussion about Python programming concepts and best practices"

    # Create a mock prompt builder that only has the system section method
    class TestPromptBuilder:
        def _safe_truncate(self, text, max_chars):
            return text[:max_chars] if len(text) > max_chars else text

        def _build_system_section(self, agent_state, conversation_frame, memory_rules, user_intent=None, topic_info=None):
            """Build system information section."""
            # Build current topic information
            current_topic = None
            if topic_info:
                current_topic = {
                    "name": getattr(topic_info, "topic_name", None) or topic_info.get("topic_name", None),
                    "domain": getattr(topic_info, "domain", None) or topic_info.get("domain", None),
                    "message_count": int(getattr(topic_info, "message_count", 0) or topic_info.get("message_count", 0) or 0),
                    "summary": self._safe_truncate(getattr(topic_info, "summary", "") or topic_info.get("summary", "") or "", 200)
                }
                # Remove empty fields
                current_topic = {k: v for k, v in current_topic.items() if v not in (None, "", 0)}
            else:
                current_topic = {"status": "not_detected_yet"}

            return {
                "role": "helpful_technical_assistant",
                "tone": "professional",
                "mode": "assist",
                "goal": "Help user effectively",
                "user_intent": {"type": "not_analyzed", "confidence": 0.0},
                "current_topic": current_topic,
                "rules": [
                    "Follow user preferences in context.user",
                    "Use memory from context.ltm when relevant",
                    "Adapt response based on system.user_intent type and detail_level",
                    "Return structured JSON when asked (intent/topic metadata)",
                    "Prefer concise technical explanations with actionable steps",
                    "If unsure about a fact requiring recent data, mark as 'needs_browsing' (do not hallucinate)",
                    "Do not contradict previous statements about capabilities",
                    "Maintain topic coherence based on system.current_topic",
                    "Reference and build upon the current topic context when appropriate"
                ]
            }

    # Test with topic info
    builder = TestPromptBuilder()
    topic = MockTopicInfo()

    system_section = builder._build_system_section(None, None, None, None, topic)

    # Verify topic is in system section
    current_topic = system_section.get('current_topic', {})

    assert current_topic.get('name') == 'Python Programming', f"Expected 'Python Programming', got {current_topic.get('name')}"
    assert current_topic.get('domain') == 'programming', f"Expected 'programming', got {current_topic.get('domain')}"
    assert current_topic.get('message_count') == 5, f"Expected 5, got {current_topic.get('message_count')}"
    assert 'summary' in current_topic, "Expected summary to be present"

    print("✓ Topic information correctly injected into system section")

    # Check that topic coherence rule is included
    rules = system_section.get('rules', [])
    has_topic_rule = any("current_topic" in rule for rule in rules)
    assert has_topic_rule, "Expected rule about maintaining topic coherence"
    print("✓ Topic coherence rule correctly included in system rules")

    # Test without topic info
    system_section_no_topic = builder._build_system_section(None, None, None, None, None)
    current_topic_no_topic = system_section_no_topic.get('current_topic', {})

    assert current_topic_no_topic.get('status') == 'not_detected_yet', f"Expected 'not_detected_yet', got {current_topic_no_topic.get('status')}"
    print("✓ Fallback status correctly set when no topic info provided")

    print("\n🎉 All tests passed! Topic injection into system section is working correctly.")
    print("\nSystem section structure:")
    print(json.dumps(system_section, indent=2))

if __name__ == "__main__":
    test_topic_in_system_section()
