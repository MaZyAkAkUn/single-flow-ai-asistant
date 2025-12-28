"""
Core structured prompt builder implementation.
Migrated to JSON-based structure for better LLM compatibility and token efficiency.
"""
import hashlib
import logging
import json
import platform
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from ..data.schemas import (
    AgentState, AgentMode, ConversationFrame, MemoryRules,
    UserPersonalization, UserIntent, RetrievedContext,
    StructuredPromptConfig, ProjectContext, TopicInfo
)
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class StructuredPromptBuilder:
    """
    Core class for building structured prompts with JSON formatting.
    Implements optimized context handling and clean separation of concerns.
    """
    
    def __init__(self, config: Optional[StructuredPromptConfig] = None):
        """Initialize the prompt builder with configuration."""
        self.config = config or StructuredPromptConfig()

    def get_system_info(self, settings: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Get system information including auto-detected and user-configured data.

        Args:
            settings: Application settings containing user-configured system info

        Returns:
            Dictionary with system information
        """
        system_info = {
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "current_time": datetime.now().strftime("%H:%M:%S"),
            "platform": platform.system(),
            "timezone": datetime.now().astimezone().tzname()
        }

        # Add user-configured information if available
        if settings:
            system_info_settings = settings.get('system_info', {})
            system_info["user_location"] = system_info_settings.get('user_location', '')
            system_info["main_spoken_language"] = system_info_settings.get('main_spoken_language', '')

        return system_info
        
    def build_structured_prompt(
        self,
        user_message: str,
        conversation_history: List[Dict[str, Any]],
        retrieved_context: RetrievedContext,
        agent_state: AgentState,
        conversation_frame: ConversationFrame,
        memory_rules: MemoryRules,
        user_personalization: UserPersonalization,
        user_intent: UserIntent,
        settings: Optional[Dict[str, Any]] = None,
        current_project: Optional[ProjectContext] = None,
        other_projects: List[ProjectContext] = None,
        topic_info: Optional[TopicInfo] = None
    ) -> Tuple[str, str]:
        """
        Build complete structured prompt with all sections in JSON format.
        
        Args:
            user_message: Current user input
            conversation_history: Recent message history (STM)
            retrieved_context: Retrieved memories, docs, etc.
            agent_state: Current agent operational state
            conversation_frame: Conversation continuity settings
            memory_rules: Memory management instructions
            user_personalization: User preferences
            user_intent: Analyzed user intent
            current_project: Currently active project
            other_projects: Other user projects (minimal info)
            
        Returns:
            Tuple of (structured_json_prompt, flattened_prompt)
        """
        try:
            # 1. System Section (Static Rules & Philosophy)
            system_section = self._build_system_section(
                agent_state, conversation_frame, memory_rules, user_intent, topic_info
            )

            # 2. System Info Section (Auto-detected and user-configured system data)
            system_info_section = self.get_system_info(settings)

            # 3. Context Section (Dynamic Memory & Data)
            context_section = self._build_context_section(
                retrieved_context,
                conversation_history,
                user_personalization,
                current_project,
                other_projects or [],
                topic_info
            )

            # 4. Task Section (The specific request)
            task_section = user_message

            # Assemble complete JSON structure
            prompt_structure = {
                "system": system_section,
                "system_info": system_info_section,
                "context": context_section,
                "task": task_section
            }
            
            # Serialize to string (Compact JSON)
            structured_prompt = json.dumps(prompt_structure, ensure_ascii=False, indent=2)
            
            # Flattened version is just the JSON string (LLMs handle JSON natively)
            # or we can provide a readable text version if really needed for debugging
            flattened_prompt = structured_prompt 
            
            logger.info(f"Built JSON prompt: {len(structured_prompt)} chars")
            
            return structured_prompt, flattened_prompt
            
        except Exception as e:
            logger.error(f"Error building JSON prompt: {e}")
            # Fallback to simple string
            return self._build_fallback_prompt(user_message), user_message

    def _build_system_section(
        self,
        agent_state: AgentState,
        conversation_frame: ConversationFrame,
        memory_rules: MemoryRules,
        user_intent: Optional[UserIntent] = None,
        topic_info: Optional[TopicInfo] = None
    ) -> Dict[str, Any]:
        """Build system information section."""
        # Build current topic information
        current_topic = None
        if topic_info:
            current_topic = {
                "name": getattr(topic_info, "topic_name", None),
                "domain": getattr(topic_info, "domain", None),
                "message_count": int(getattr(topic_info, "message_count", 0) or 0),
                "summary": self._safe_truncate(getattr(topic_info, "summary", "") or "", 200)
            }
            # Remove empty fields
            current_topic = {k: v for k, v in current_topic.items() if v not in (None, "", 0)}
        else:
            current_topic = {"status": "not_detected_yet"}

        return {
            "role": "helpful_technical_assistant",
            "tone": conversation_frame.maintain_tone,
            "mode": agent_state.current_mode.value,
            "goal": agent_state.current_goal,
            "user_intent": {
                "type": user_intent.intent_type.value if user_intent else "not_analyzed",
                "confidence": user_intent.confidence_score if user_intent else 0.0,
                "detail_level": user_intent.expected_detail_level if user_intent else "adaptive"
            } if user_intent else {"type": "not_analyzed", "confidence": 0.0},
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
                "Reference and build upon the current topic context when appropriate",
                "When executing web search, and got in result an Image urls/image tags(in markdown) embedd it in the beggining of ur response, so user can seee founded images in ur response"
            ]
        }

    def _build_context_section(
        self,
        retrieved_context: RetrievedContext,
        conversation_history: List[Dict[str, Any]],
        user_personalization: UserPersonalization,
        current_project: Optional[ProjectContext],
        other_projects: List[ProjectContext],
        topic_info: Optional[TopicInfo] = None
    ) -> Dict[str, Any]:
        """
        Build a structured context payload for LLM prompts.
        Adds deduplicated LTM and RAG, safe truncation and enriched topic metadata.
        """
        # base user + history
        context: Dict[str, Any] = {
            "user": self._build_user_section(user_personalization),
            "history": self._build_history_section(conversation_history),
        }

        # Long Term Memory (dedupe + limit)
        if retrieved_context.memories:
            raw_mem = []
            for m in retrieved_context.memories:
                # memory object may be simple string or object with attributes
                content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else str(m))
                mtype = getattr(m, "memory_type", None) or (m.get("memory_type") if isinstance(m, dict) else "general")
                ts = getattr(m, "timestamp", None) or (m.get("timestamp") if isinstance(m, dict) else None)
                raw_mem.append({"content": content, "type": mtype, "timestamp": ts})
            deduped = self._unique_by_content(raw_mem, key="content", limit=5)
            context["ltm"] = deduped

        # RAG / Documents (dedupe + excerpt)
        if retrieved_context.documents:
            raw_docs = []
            for doc in retrieved_context.documents:
                content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
                source = doc.get("source", "unknown") if isinstance(doc, dict) else "unknown"
                raw_docs.append({"content": self._safe_truncate(content, 300), "source": source})
            context["rag"] = self._unique_by_content(raw_docs, key="content", limit=3)

        # Project context
        if current_project:
            proj = {
                "name": getattr(current_project, "project_name", None) or current_project.get("project_name", None),
                "description": getattr(current_project, "project_description", None) or current_project.get("project_description", None),
                "status": getattr(current_project, "project_status", None) or current_project.get("project_status", None),
            }
            # remove None entries
            context["project"] = {k: v for k, v in proj.items() if v is not None}
            if other_projects:
                other_names = []
                for p in other_projects[:3]:
                    pname = getattr(p, "project_name", None) or (p.get("project_name") if isinstance(p, dict) else None)
                    if pname:
                        other_names.append(pname)
                if other_names:
                    context["other_projects"] = other_names

        # Topic context — ALWAYS include, even if not detected
        if topic_info:
            # gather representative messages safely
            reps = []
            if getattr(topic_info, "representative_messages", None):
                for m in topic_info.representative_messages[:5]:
                    text = m.get("text") if isinstance(m, dict) else getattr(m, "text", None)
                    ts = m.get("timestamp") if isinstance(m, dict) else getattr(m, "timestamp", None)
                    if text:
                        reps.append({"text": self._safe_truncate(text, 500), "timestamp": ts})
            else:
                # fallback: try first message or summary
                first_msg = getattr(topic_info, "first_message", None) or (topic_info.summary if getattr(topic_info, "summary", None) else None)
                if first_msg:
                    reps.append({"text": self._safe_truncate(first_msg, 500), "timestamp": None})

            topic_obj = {
                "status": "detected",  # NEW FIELD
                "id": getattr(topic_info, "topic_id", None),
                "name": getattr(topic_info, "topic_name", None),
                "domain": getattr(topic_info, "domain", None),
                "message_count": int(getattr(topic_info, "message_count", 0) or 0),
                "summary": self._safe_truncate(getattr(topic_info, "summary", "") or "", 300),
                "representative_messages": reps,
                "language": getattr(topic_info, "language", None),
                "confidence": float(getattr(topic_info, "llm_confidence", 0.0) or 0.0),
                "last_seen": getattr(topic_info, "last_seen", None),
                # optional: id of stored centroid/embedding vector in vector DB
                "centroid_id": getattr(topic_info, "centroid_id", None),
                "keywords": getattr(topic_info, "keywords", None) or []
            }
            # remove empty keys to keep payload compact
            context["topic"] = {k: v for k, v in topic_obj.items() if v not in (None, "", [], 0)}
        else:
            # NEW: Always include topic section, even if not detected
            context["topic"] = {
                "status": "not_detected_yet",
                "message": "Topic tracking is analyzing conversation. Continue naturally."
            }

        # Topic messages - add if available in retrieved context
        if hasattr(retrieved_context, 'topic_messages') and retrieved_context.topic_messages:
            topic_msgs = []
            for msg in retrieved_context.topic_messages[:8]:  # Limit to 8 most relevant messages
                msg_content = msg.get('content', '')[:200] + "..." if len(msg.get('content', '')) > 200 else msg.get('content', '')
                msg_role = msg.get('role', 'unknown')
                msg_timestamp = msg.get('timestamp', '')
                if msg_content:
                    topic_msgs.append({
                        "role": msg_role,
                        "content": msg_content,
                        "timestamp": msg_timestamp
                    })
            if topic_msgs:
                context["topic_messages"] = topic_msgs

        return context
        
    def _build_user_section(self, user_personalization: UserPersonalization) -> Dict[str, Any]:
        """Build compact user personalization section."""
        return {
            "style": user_personalization.communication_style,
            "detail": user_personalization.detail_preference,
            "code_preference": user_personalization.code_style_preference
        }
        
    def _build_history_section(self, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Build compact message history."""
        # Get last N messages
        max_messages = self.config.memory_management.get("stm_buffer_size", 5)
        recent = conversation_history[-max_messages:]
        
        return [
            {
                "role": msg.get('role', 'unknown'),
                "content": msg.get('content', '')[:500] # Limit individual message size
            }
            for msg in recent
        ]

    def _build_fallback_prompt(self, user_message: str) -> str:
        """Build minimal fallback prompt."""
        return json.dumps({
            "role": "assistant",
            "task": user_message
        })

    def get_message_hash(self, message: str) -> str:
        """Get hash for message-prompt pairing."""
        return hashlib.md5(message.encode()).hexdigest()

    def _safe_truncate(self, text: str, max_chars: int) -> str:
        """
        Safely truncate text without breaking words.

        Args:
            text: Text to truncate
            max_chars: Maximum character length

        Returns:
            Truncated text
        """
        if not text or len(text) <= max_chars:
            return text or ""

        import textwrap
        # Use textwrap.shorten for word-boundary aware truncation
        return textwrap.shorten(text, width=max_chars, placeholder="...")

    def _unique_by_content(self, items: List[Dict[str, Any]], key: str = "content", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Deduplicate items by content hash.

        Args:
            items: List of dictionaries to deduplicate
            key: Key to use for content hashing
            limit: Maximum number of unique items to return

        Returns:
            Deduplicated list
        """
        seen = set()
        out = []
        for it in items:
            content = (it.get(key) or "").strip()
            if not content:
                continue
            # Create hash of content
            content_hash = hashlib.sha1(content.encode("utf-8")).hexdigest()
            if content_hash in seen:
                continue
            seen.add(content_hash)
            out.append(it)
            if len(out) >= limit:
                break
        return out
