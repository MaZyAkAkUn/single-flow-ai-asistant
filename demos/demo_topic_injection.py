#!/usr/bin/env python3
"""
Demonstration of the topic and intent injection improvements.
Shows the JSON structure of prompts with and without topic/intent data.
"""

import json

def demo_prompt_structure():
    """Demonstrate the new prompt structure with intent and topic injection."""

    print("=== TOPIC AND INTENT INJECTION DEMONSTRATION ===\n")

    # Example 1: Prompt with detected intent and topic
    print("1. PROMPT WITH DETECTED INTENT AND TOPIC:")
    print("-" * 50)

    prompt_with_intent_topic = {
        "system": {
            "role": "helpful_technical_assistant",
            "tone": True,
            "mode": "casual_conversation",
            "goal": "",
            "user_intent": {  # <-- NEW: Intent information
                "type": "code_generation",
                "confidence": 0.9,
                "detail_level": "comprehensive"
            },
            "current_topic": {  # <-- NEW: Topic awareness in system section
                "name": "Python Programming",
                "domain": "programming",
                "message_count": 3,
                "summary": "Discussion about Python programming concepts and best practices"
            },
            "rules": [
                "Follow user preferences in context.user",
                "Use memory from context.ltm when relevant",
                "Adapt response based on system.user_intent type and detail_level",  # <-- NEW: Intent adaptation rule
                "Return structured JSON when asked (intent/topic metadata)",
                "Prefer concise technical explanations with actionable steps",
                "If unsure about a fact requiring recent data, mark as 'needs_browsing' (do not hallucinate)",
                "Do not contradict previous statements about capabilities",
                "Maintain topic coherence based on system.current_topic",  # <-- NEW: Topic coherence rule
                "Reference and build upon the current topic context when appropriate"  # <-- NEW: Topic reference rule
            ]
        },
        "system_info": {
            "current_date": "2025-12-08",
            "current_time": "22:07:15",
            "platform": "Windows",
            "timezone": "FLE Standard Time",
            "user_location": "",
            "main_spoken_language": ""
        },
        "context": {
            "user": {
                "style": "casual_technical",
                "detail": "comprehensive",
                "code_preference": "readable"
            },
            "history": [],
            "topic": {  # <-- IMPROVED: Always present topic section
                "status": "detected",  # <-- NEW: Status indicator
                "id": "topic_abc123",
                "name": "Python Programming",
                "domain": "programming",
                "message_count": 3,
                "summary": "Discussion about Python programming concepts and best practices",
                "keywords": ["python", "programming", "code"],
                "confidence": 0.85,
                "current_topic": True  # <-- NEW: Explicit current topic marker
            }
        },
        "task": "Write a Python function to calculate fibonacci numbers"
    }

    print(json.dumps(prompt_with_intent_topic, indent=2))

    print("\n\n2. PROMPT WITHOUT DETECTED INTENT AND TOPIC:")
    print("-" * 50)

    prompt_without_intent_topic = {
        "system": {
            "role": "helpful_technical_assistant",
            "tone": True,
            "mode": "casual_conversation",
            "goal": "",
            "user_intent": {  # <-- NEW: Still present, but shows "not_analyzed"
                "type": "not_analyzed",
                "confidence": 0.0
            },
            "current_topic": {  # <-- NEW: Fallback topic status in system section
                "status": "not_detected_yet"
            },
            "rules": [
                "Follow user preferences in context.user",
                "Use memory from context.ltm when relevant",
                "Adapt response based on system.user_intent type and detail_level",  # <-- NEW: Intent adaptation rule
                "Return structured JSON when asked (intent/topic metadata)",
                "Prefer concise technical explanations with actionable steps",
                "If unsure about a fact requiring recent data, mark as 'needs_browsing' (do not hallucinate)",
                "Do not contradict previous statements about capabilities",
                "Maintain topic coherence based on system.current_topic",  # <-- NEW: Topic coherence rule
                "Reference and build upon the current topic context when appropriate"  # <-- NEW: Topic reference rule
            ]
        },
        "system_info": {
            "current_date": "2025-12-08",
            "current_time": "22:07:15",
            "platform": "Windows",
            "timezone": "FLE Standard Time",
            "user_location": "",
            "main_spoken_language": ""
        },
        "context": {
            "user": {
                "style": "casual_technical",
                "detail": "comprehensive",
                "code_preference": "readable"
            },
            "history": [],
            "topic": {  # <-- IMPROVED: Always present, even when not detected
                "status": "not_detected_yet",  # <-- NEW: Fallback status
                "message": "Topic tracking is analyzing conversation. Continue naturally."  # <-- NEW: Helpful message
            }
        },
        "task": "What is machine learning?"
    }

    print(json.dumps(prompt_without_intent_topic, indent=2))

    print("\n\n3. STATUS EVENTS DURING PROCESSING:")
    print("-" * 50)
    print("The system now yields real-time status events:")
    print("- 'Analyzing user intent...' -> 'Intent detected: code_generation (confidence: 0.90)'")
    print("- 'Analyzing conversation topic...' -> 'Topic: Python Programming (domain: programming)'")
    print("- Or when unavailable: 'Topic tracker not initialized'")
    print("- Or when failed: 'Topic analysis failed, will continue without topic context'")

    print("\n\n4. MESSAGE STORAGE ENHANCEMENTS:")
    print("-" * 50)
    print("Conversation messages now include metadata fields:")
    print("- User messages get 'metadata': {} (populated by adapter)")
    print("- Assistant messages get intent/topic metadata")
    print("- Enables conversation reload with full context")

    print("\n\n5. IMPROVED ERROR HANDLING:")
    print("-" * 50)
    print("✅ Topic tracker initialization failures logged with clear messages")
    print("✅ Fallback topic status always provided")
    print("✅ Intent analysis failures gracefully handled")
    print("✅ System continues working even with partial failures")

    print("\n" + "=" * 60)
    print("🎉 IMPLEMENTATION COMPLETE!")
    print("The assistant now always knows the current intent and topic state!")

if __name__ == "__main__":
    demo_prompt_structure()
