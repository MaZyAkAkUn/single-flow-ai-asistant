"""
Context aggregator for intelligent context assembly from multiple sources.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

from ..data.schemas import (
    RetrievedContext, MemoryItem, ProjectContext, AgentState,
    UserPersonalization, UserIntent, ConversationFrame, MemoryRules, TopicInfo
)
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class ContextAggregator:
    """
    Intelligent context assembly from multiple sources (STM, LTM, RAG, Projects).
    Implements token optimization and context relevance scoring.
    """
    
    def __init__(self, token_limits: Dict[str, int]):
        """Initialize context aggregator."""
        self.token_limits = token_limits
        self._context_cache: Dict[str, Any] = {}
        self._relevance_cache: Dict[str, float] = {}
        
    def aggregate_context(
        self,
        user_intent: UserIntent,
        conversation_history: List[Dict[str, Any]],
        memories: List[MemoryItem],
        documents: List[Dict[str, Any]],
        project_contexts: List[ProjectContext],
        web_results: List[Dict[str, Any]] = None,
        agent_state: Optional[AgentState] = None,
        user_profile: Optional[UserPersonalization] = None,
        topic_info: Optional[TopicInfo] = None,
        topic_messages: List[Dict[str, Any]] = None
    ) -> RetrievedContext:
        """
        Aggregate all context sources with intelligent optimization.

        Args:
            user_intent: Analyzed user intent
            conversation_history: Recent conversation (STM)
            memories: Retrieved memories (LTM)
            documents: Retrieved documents (RAG)
            project_contexts: Project contexts
            web_results: Web search results
            agent_state: Current agent state
            user_profile: User personalization
            topic_info: Current topic information
            topic_messages: Messages related to current topic

        Returns:
            Aggregated context with optimized content
        """
        try:
            logger.info("Starting context aggregation")

            # Calculate relevance scores for all sources
            relevance_scores = self._calculate_relevance_scores(
                user_intent, conversation_history, memories, documents, project_contexts, topic_info
            )

            # Select and optimize content within token limits
            optimized_memories = self._optimize_memories(memories, relevance_scores.get('memories', {}))
            optimized_documents = self._optimize_documents(documents, relevance_scores.get('documents', {}))
            optimized_projects = self._optimize_projects(project_contexts, relevance_scores.get('projects', {}))
            optimized_topic_messages = self._optimize_topic_messages(topic_messages or [], relevance_scores.get('topic_messages', {}))

            # Build retrieved context
            retrieved_context = RetrievedContext(
                memories=optimized_memories,
                documents=optimized_documents,
                project_contexts=optimized_projects,
                web_results=web_results or []
            )

            # Add topic messages to context if available
            if optimized_topic_messages:
                # Store topic messages in a separate attribute for now
                retrieved_context.topic_messages = optimized_topic_messages

            # Cache context for efficiency
            context_key = self._generate_context_key(user_intent, conversation_history[-5:])
            self._context_cache[context_key] = retrieved_context

            logger.info(f"Aggregated context: {len(optimized_memories)} memories, "
                       f"{len(optimized_documents)} docs, {len(optimized_projects)} projects, "
                       f"{len(optimized_topic_messages)} topic messages")

            return retrieved_context

        except Exception as e:
            logger.error(f"Failed to aggregate context: {e}")
            return RetrievedContext()
            
    def _calculate_relevance_scores(
        self,
        user_intent: UserIntent,
        conversation_history: List[Dict[str, Any]],
        memories: List[MemoryItem],
        documents: List[Dict[str, Any]],
        project_contexts: List[ProjectContext],
        topic_info: Optional[TopicInfo] = None
    ) -> Dict[str, Dict[str, float]]:
        """Calculate relevance scores for all context sources."""
        scores = {
            'memories': {},
            'documents': {},
            'projects': {},
            'topic_messages': {}
        }

        try:
            # Intent-based scoring weights
            intent_weights = self._get_intent_scoring_weights(user_intent.intent_type)

            # Score memories
            for memory in memories:
                score = self._score_memory_relevance(memory, user_intent, conversation_history)
                scores['memories'][memory.memory_id] = score * intent_weights.get('memory', 1.0)

            # Score documents
            for doc in documents:
                score = self._score_document_relevance(doc, user_intent)
                scores['documents'][doc.get('id', str(hash(str(doc))))] = score * intent_weights.get('document', 1.0)

            # Score projects
            for project in project_contexts:
                score = self._score_project_relevance(project, user_intent, conversation_history)
                scores['projects'][project.project_id] = score * intent_weights.get('project', 1.0)

            # Score topic messages if available and topic info is provided
            if topic_info:
                # Topic messages get higher relevance when topic is well-established
                topic_confidence = topic_info.llm_confidence or 0.5
                topic_message_weight = 0.8 + (topic_confidence * 0.4)  # 0.8 to 1.2 based on confidence

                # For now, we'll score topic messages uniformly high since they're directly relevant
                # In a more sophisticated implementation, we could analyze individual message relevance
                for i, msg in enumerate(conversation_history or []):
                    # Give higher score to more recent messages in the topic
                    recency_factor = min(1.0, (i + 1) / len(conversation_history)) if conversation_history else 1.0
                    scores['topic_messages'][i] = 0.9 * topic_message_weight * recency_factor

        except Exception as e:
            logger.error(f"Failed to calculate relevance scores: {e}")

        return scores
        
    def _get_intent_scoring_weights(self, intent_type) -> Dict[str, float]:
        """Get scoring weights based on intent type."""
        weights = {
            "opinion_request": {"memory": 1.5, "document": 0.8, "project": 1.0},
            "code_generation": {"memory": 0.7, "document": 1.2, "project": 1.3},
            "analysis_and_opinion": {"memory": 1.2, "document": 1.1, "project": 0.9},
            "quick_answer": {"memory": 0.5, "document": 0.6, "project": 0.4},
            "planning_request": {"memory": 0.9, "document": 0.8, "project": 1.4},
            "explanation": {"memory": 0.8, "document": 1.0, "project": 0.7},
            "troubleshooting": {"memory": 1.1, "document": 1.3, "project": 1.2},
            "creative_writing": {"memory": 1.0, "document": 0.7, "project": 0.6}
        }
        return weights.get(intent_type.value, {"memory": 1.0, "document": 1.0, "project": 1.0})
        
    def _score_memory_relevance(
        self, 
        memory: MemoryItem, 
        user_intent: UserIntent,
        conversation_history: List[Dict[str, Any]]
    ) -> float:
        """Score memory relevance based on multiple factors."""
        score = 0.0
        
        try:
            # Base importance score
            importance_scores = {"low": 0.3, "normal": 0.6, "high": 0.8, "critical": 1.0}
            score += importance_scores.get(memory.importance, 0.5)
            
            # Access frequency bonus
            if memory.access_count > 5:
                score += 0.2
            elif memory.access_count > 10:
                score += 0.3
                
            # Memory type relevance to intent
            type_relevance = {
                "preference": 1.2 if user_intent.intent_type.value in ["opinion_request", "explanation"] else 0.8,
                "project_metadata": 1.3 if user_intent.intent_type.value in ["planning_request", "troubleshooting"] else 0.9,
                "decision": 1.1 if user_intent.intent_type.value in ["analysis_and_opinion"] else 0.9,
                "fact": 0.9,
                "skill": 1.0
            }
            score += type_relevance.get(memory.memory_type, 0.8) * 0.3
            
            # Tag relevance (bonus for matching tags)
            # This would need more sophisticated matching in real implementation
            
        except Exception as e:
            logger.error(f"Failed to score memory relevance: {e}")
            
        return min(score, 1.0)
        
    def _score_document_relevance(self, doc: Dict[str, Any], user_intent: UserIntent) -> float:
        """Score document relevance."""
        score = 0.0
        
        try:
            # Use similarity score if available
            if 'similarity_score' in doc:
                score = doc['similarity_score']
            elif 'score' in doc:
                score = doc['score']
            else:
                # Default scoring based on content length and structure
                content = doc.get('content', '')
                if len(content) > 100:  # Substantial content
                    score = 0.7
                else:
                    score = 0.4
                    
            # Intent-based adjustments
            if user_intent.intent_type.value == "code_generation":
                if "code" in doc.get('content', '').lower():
                    score += 0.2
            elif user_intent.intent_type.value == "explanation":
                if any(word in doc.get('content', '').lower() for word in ["what", "how", "why", "explain"]):
                    score += 0.1
                    
        except Exception as e:
            logger.error(f"Failed to score document relevance: {e}")
            
        return min(score, 1.0)
        
    def _score_project_relevance(
        self, 
        project: ProjectContext, 
        user_intent: UserIntent,
        conversation_history: List[Dict[str, Any]]
    ) -> float:
        """Score project relevance."""
        score = 0.0
        
        try:
            # Base project status score
            status_scores = {"active": 1.0, "paused": 0.7, "completed": 0.5, "archived": 0.2}
            score = status_scores.get(project.project_status, 0.6)
            
            # Recent activity bonus (if last_updated is recent)
            days_since_update = (datetime.now() - project.last_updated).days
            if days_since_update < 7:
                score += 0.2
            elif days_since_update < 30:
                score += 0.1
                
            # Intent-based relevance
            if user_intent.intent_type.value == "planning_request":
                if "project" in user_intent.context_requirements:
                    score += 0.3
            elif user_intent.intent_type.value == "troubleshooting":
                # Projects are relevant for debugging
                score += 0.2
                
        except Exception as e:
            logger.error(f"Failed to score project relevance: {e}")
            
        return min(score, 1.0)
        
    def _optimize_memories(self, memories: List[MemoryItem], scores: Dict[str, float]) -> List[MemoryItem]:
        """Optimize memory selection within token limits."""
        token_limit = self.token_limits.get('retrieved_context', 800)
        
        # Sort by relevance score
        sorted_memories = sorted(
            [(memory, scores.get(memory.memory_id, 0.0)) for memory in memories],
            key=lambda x: x[1],
            reverse=True
        )
        
        selected_memories = []
        total_tokens = 0
        
        for memory, score in sorted_memories:
            estimated_tokens = self._estimate_memory_tokens(memory)
            if total_tokens + estimated_tokens <= token_limit:
                selected_memories.append(memory)
                total_tokens += estimated_tokens
            else:
                break
                
        return selected_memories
        
    def _optimize_documents(self, documents: List[Dict[str, Any]], scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Optimize document selection within token limits."""
        token_limit = self.token_limits.get('retrieved_context', 800)
        
        # Sort by relevance score
        sorted_docs = sorted(
            [(doc, scores.get(doc.get('id', str(hash(str(doc)))), 0.0)) for doc in documents],
            key=lambda x: x[1],
            reverse=True
        )
        
        selected_docs = []
        total_tokens = 0
        
        for doc, score in sorted_docs:
            estimated_tokens = self._estimate_document_tokens(doc)
            if total_tokens + estimated_tokens <= token_limit:
                selected_docs.append(doc)
                total_tokens += estimated_tokens
            else:
                break
                
        return selected_docs
        
    def _optimize_projects(self, projects: List[ProjectContext], scores: Dict[str, float]) -> List[ProjectContext]:
        """Optimize project selection within token limits."""
        token_limit = self.token_limits.get('project_context_reference', 200)
        
        # Sort by relevance score
        sorted_projects = sorted(
            [(project, scores.get(project.project_id, 0.0)) for project in projects],
            key=lambda x: x[1],
            reverse=True
        )
        
        selected_projects = []
        total_tokens = 0
        
        for project, score in sorted_projects:
            estimated_tokens = self._estimate_project_tokens(project)
            if total_tokens + estimated_tokens <= token_limit:
                selected_projects.append(project)
                total_tokens += estimated_tokens
            else:
                break
                
        return selected_projects
        
    def _estimate_memory_tokens(self, memory: MemoryItem) -> int:
        """Estimate token count for a memory item."""
        content_tokens = len(memory.content.split()) * 1.3
        metadata_tokens = len(memory.memory_type) + len(memory.importance) + len(memory.tags) * 2
        return int(content_tokens + metadata_tokens)
        
    def _estimate_document_tokens(self, doc: Dict[str, Any]) -> int:
        """Estimate token count for a document."""
        content = doc.get('content', '')
        content_tokens = len(content.split()) * 1.3
        
        # Include metadata
        metadata_tokens = 0
        for key, value in doc.items():
            if key != 'content' and isinstance(value, str):
                metadata_tokens += len(value.split()) * 1.3
                
        return int(content_tokens + metadata_tokens)
        
    def _estimate_project_tokens(self, project: ProjectContext) -> int:
        """Estimate token count for a project context."""
        desc_tokens = len(project.project_description.split()) * 1.3
        summary_tokens = len(project.context_summary.split()) * 1.3 if project.context_summary else 0
        tags_tokens = len(project.tags) * 2
        
        return int(desc_tokens + summary_tokens + tags_tokens)
        
    def _generate_context_key(self, user_intent: UserIntent, recent_history: List[Dict[str, Any]]) -> str:
        """Generate cache key for context."""
        try:
            import hashlib
            key_data = f"{user_intent.intent_type.value}_{len(recent_history)}"
            for msg in recent_history:
                key_data += f"{msg.get('role', '')}{len(msg.get('content', ''))}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except Exception:
            return "default_context_key"
            
    def get_context_summary(self, retrieved_context: RetrievedContext) -> Dict[str, Any]:
        """Get summary of retrieved context."""
        return {
            "total_memories": len(retrieved_context.memories),
            "total_documents": len(retrieved_context.documents),
            "total_projects": len(retrieved_context.project_contexts),
            "total_web_results": len(retrieved_context.web_results),
            "estimated_tokens": self._estimate_total_tokens(retrieved_context)
        }
        
    def _estimate_total_tokens(self, retrieved_context: RetrievedContext) -> int:
        """Estimate total tokens in retrieved context."""
        total = 0
        for memory in retrieved_context.memories:
            total += self._estimate_memory_tokens(memory)
        for doc in retrieved_context.documents:
            total += self._estimate_document_tokens(doc)
        for project in retrieved_context.project_contexts:
            total += self._estimate_project_tokens(project)
        # Add topic messages if present
        if hasattr(retrieved_context, 'topic_messages') and retrieved_context.topic_messages:
            for msg in retrieved_context.topic_messages:
                total += self._estimate_topic_message_tokens(msg)
        return total

    def _optimize_topic_messages(self, topic_messages: List[Dict[str, Any]], scores: Dict[int, float]) -> List[Dict[str, Any]]:
        """
        Optimize topic message selection within token limits.
        Topic messages get higher priority since they're directly relevant to current conversation.
        """
        # Allocate more tokens for topic messages since they're highly relevant
        token_limit = self.token_limits.get('topic_context', 1200)  # Higher limit for topic context

        # Sort by relevance score (index-based)
        sorted_messages = sorted(
            [(i, msg, scores.get(i, 0.9)) for i, msg in enumerate(topic_messages)],
            key=lambda x: x[2],
            reverse=True
        )

        selected_messages = []
        total_tokens = 0

        for index, message, score in sorted_messages:
            estimated_tokens = self._estimate_topic_message_tokens(message)
            if total_tokens + estimated_tokens <= token_limit:
                selected_messages.append(message)
                total_tokens += estimated_tokens
            else:
                break

        return selected_messages

    def _estimate_topic_message_tokens(self, message: Dict[str, Any]) -> int:
        """Estimate token count for a topic message."""
        content = message.get('content', '')
        content_tokens = len(content.split()) * 1.3

        # Include role and minimal metadata
        role_tokens = len(message.get('role', '')) * 0.5
        timestamp_tokens = len(message.get('timestamp', '')) * 0.3 if message.get('timestamp') else 0

        return int(content_tokens + role_tokens + timestamp_tokens)
        
    def clear_cache(self):
        """Clear context cache."""
        self._context_cache.clear()
        self._relevance_cache.clear()
        logger.info("Context aggregator cache cleared")
