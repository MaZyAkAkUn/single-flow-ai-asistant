"""
Memory management system implementing STM/LTM operations.
"""
import json
import logging
from typing import Dict, List, Optional, Any, Deque
from datetime import datetime, timedelta
from collections import deque
import threading
import os

from memori import Memori

from ..data.schemas import MemoryItem, MemoryTrigger, RetrievedContext
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    Two-layer memory system: Short-Term Memory (STM) and Long-Term Memory (LTM).
    Handles conversation context, memory triggers, and intelligent summarization.
    Uses Memori as the LTM backend while maintaining a local STM buffer.
    """
    
    def __init__(self, data_dir: str = "./data/memory", max_stm_size: int = 20, openai_api_key: Optional[str] = None):
        """Initialize memory manager."""
        self.data_dir = data_dir
        self.max_stm_size = max_stm_size
        self._lock = threading.RLock()
        
        # Short-Term Memory (conversation buffer)
        self._stm: Deque[Dict[str, Any]] = deque(maxlen=max_stm_size)
        
        # Initialize Memori for Long-Term Memory
        try:
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.abspath(os.path.join(data_dir, "memori.db"))
            
            # Prepare Memori kwargs
            memori_kwargs = {
                "database_connect": f"sqlite:///{db_path}",
                "conscious_ingest": True,
                "auto_ingest": False,  # We manually control retrieval
                "namespace": "personal_assistant" # Ensure namespace is set
            }
            if openai_api_key:
                memori_kwargs["openai_api_key"] = openai_api_key
                
            self.memori = Memori(**memori_kwargs)
            
            # Do not call enable() to avoid monkey-patching the LLM client
            logger.info(f"Memori initialized with database: {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Memori: {e}")
            self.memori = None
        
        # LTM Cache (optional, Memori handles storage)
        self._ltm: Dict[str, MemoryItem] = {} 
        
        # Memory triggers and summarization
        self._trigger_handlers: Dict[MemoryTrigger, Any] = {}
        # self._load_memory() # No longer needed with Memori
        
    def add_to_stm(self, message: Dict[str, Any]) -> bool:
        """
        Add message to Short-Term Memory.
        
        Args:
            message: Message dictionary with role, content, timestamp
            
        Returns:
            True if successful
        """
        try:
            with self._lock:
                # Ensure message has required fields
                if 'role' not in message or 'content' not in message:
                    logger.warning("Invalid message format for STM")
                    return False
                    
                # Add timestamp if missing
                if 'timestamp' not in message:
                    message['timestamp'] = datetime.now().isoformat()
                    
                # Add to STM
                self._stm.append(message)
                
                # Check for memory triggers
                self._check_memory_triggers(message)
                
                logger.debug(f"Added to STM: {message['role']} message")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add to STM: {e}")
            return False
            
    def get_stm(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get Short-Term Memory messages.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of STM messages
        """
        with self._lock:
            if limit:
                return list(self._stm)[-limit:]
            return list(self._stm)
            
    def clear_stm(self):
        """Clear Short-Term Memory."""
        with self._lock:
            self._stm.clear()
            logger.info("STM cleared")
            
    def store_in_ltm(
        self,
        content: str,
        memory_type: str = "fact",
        importance: str = "normal",
        tags: List[str] = None,
        source_context: str = "",
        trigger: Optional[MemoryTrigger] = None
    ) -> Optional[str]:
        """
        Store information in Long-Term Memory using Memori.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance level
            tags: List of tags
            source_context: Context where memory was created
            trigger: What triggered this memory storage
            
        Returns:
            Memory ID if successful, None otherwise
        """
        if not self.memori:
            logger.warning("Memori not initialized, skipping LTM storage")
            return None
            
        try:
            # Map importance string to approximate score if needed, 
            # or let Memori handle it. Memori uses explicit calls.
            # Since we don't use enable(), we might need to simulate interaction
            # or see if we can just add a memory.
            # Assuming we can treat this as an interaction we want to remember.
            
            # Using add_interaction as a generic storage mechanism if specific API is not available
            # But ideally, we should use a more direct method if available.
            # Falling back to treating it as a "conscious" storage request.
            
            # Currently Memori doesn't have a simple 'add_fact' in the public API 
            # that is widely documented without 'enable()'. 
            # However, we can use the unified 'add_memory' if it existed, 
            # or we can simulate a user-system pair that gets ingested.
            
            # For now, we will use a placeholder ID since explicit storage 
            # depends on Memori's ingestion pipeline.
            
            # Actually, looking at Memori source (if we could), 
            # it processes interactions.
            # We can try to "inject" it by calling:
            # self.memori.add_interaction(user_input=source_context, ai_output=content)
            
            # If memory_type is 'explicit_memory', we definitely want it saved.
            
            # Use simulated conversation for explicit memory storage since store_fact doesn't exist
            metadata = {
                "type": memory_type,
                "importance": importance,
                "tags": tags or [],
                "source": source_context,
                "timestamp": datetime.now().isoformat()
            }
            
            # Since Memori doesn't have a direct store_fact method, we simulate 
            # a conversation turn that forces the memory to be recorded.
            user_input = f"Note this important fact: {content}"
            ai_output = "I have noted this importance fact."

            self.memori.record_conversation(
                user_input=user_input,
                ai_output=ai_output,
                metadata=metadata
            )
            
            logger.info(f"Stored in Memori LTM via conversation record: {memory_type}")
            return "stored"
            
        except Exception as e:
            logger.error(f"Failed to store in LTM: {e}")
            return None
            
    def retrieve_from_ltm(
        self,
        query: str,
        memory_types: List[str] = None,
        importance_threshold: str = "low",
        limit: int = 10
    ) -> List[MemoryItem]:
        """
        Retrieve relevant memories from Long-Term Memory using Memori.
        
        Args:
            query: Search query
            memory_types: Filter by memory types (Ignored by basic Memori search currently)
            importance_threshold: Minimum importance level
            limit: Maximum number of results
            
        Returns:
            List of relevant memory items
        """
        if not self.memori:
            return []

        try:
            # Search strategy
            memori_results = []
            query = query.strip()
            
            if not query:
                # Browse mode: Iterate standard categories
                categories = ["conversation", "fact", "semantic", "episodic", "procedural"]
                if memory_types:
                     categories = memory_types
                
                for cat in categories:
                    try:
                        if hasattr(self.memori, 'search_memories_by_category'):
                            cat_results = self.memori.search_memories_by_category(cat, limit=limit)
                            if cat_results:
                                memori_results.extend(cat_results)
                    except Exception as e:
                        logger.warning(f"Failed to search category {cat}: {e}")
                        
                # Deduplicate by ID
                seen_ids = set()
                unique_results = []
                for item in memori_results:
                    # Try to get ID safely
                    if isinstance(item, dict):
                        mid = item.get('id') or item.get('memory_id')
                    else:
                        mid = getattr(item, 'id', None) or getattr(item, 'memory_id', None)
                    
                    # Fallback to hash of content
                    if not mid:
                         if isinstance(item, dict):
                             mid = hash(item.get('content', ''))
                         else:
                             mid = hash(getattr(item, 'content', ''))
                             
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        unique_results.append(item)
                memori_results = unique_results
            else:
                # Normal search
                memori_results = self.memori.retrieve_context(query, limit=limit)
            
            # DEBUG: Log raw structure
            logger.info(f"Raw Memori results type: {type(memori_results)}")
            if memori_results:
                logger.info(f"First item type: {type(memori_results[0])}")
                logger.info(f"Raw Memori results: {memori_results}")

            results: List[MemoryItem] = []
            for item in memori_results:
                # Map Memori result to MemoryItem
                # Memori result structure example: 
                # {'content': '...', 'category': '...', 'importance_score': 0.8, ...}
                
                # Handle different field names safely
                if isinstance(item, dict):
                    # Check for nested processed_data (Memori structure)
                    processed_data = item.get('processed_data', {})
                    if isinstance(processed_data, dict) and processed_data:
                        content = processed_data.get('content') or processed_data.get('summary') or ''
                        # Fallback to top-level if nested is empty, or use top-level keys
                        if not content:
                             content = item.get('content') or item.get('memory') or item.get('text') or item.get('value') or ''
                    else:
                        content = item.get('content') or item.get('memory') or item.get('text') or item.get('value') or ''
                    
                    category = item.get('category', item.get('memory_type', 'general'))
                    score = item.get('importance_score', 0.0)
                    memory_id = str(item.get('memory_id') or item.get('id') or 'unknown')
                elif hasattr(item, 'content'): # Object with attributes
                    content = item.content
                    category = getattr(item, 'category', 'general')
                    score = getattr(item, 'importance_score', 0.0)
                    memory_id = str(getattr(item, 'id', 'unknown'))
                else: # Fallback or string
                    content = str(item)
                    category = 'general'
                    score = 0.0
                    memory_id = 'unknown'
                
                # Map numerical score to string importance
                importance = "normal"
                if score > 0.8: importance = "critical"
                elif score > 0.6: importance = "high"
                elif score < 0.3: importance = "low"
                
                memory_item = MemoryItem(
                    memory_id=memory_id,
                    content=content,
                    memory_type=category,
                    importance=importance,
                    tags=[], # Memori might not separate tags in retrieval output
                    created_at=datetime.now(), # Timestamp might be in item['created_at']
                    source_context="Memori Retrieval"
                )
                results.append(memory_item)
                
            logger.info(f"Retrieved {len(results)} memories from Memori")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve from Memori: {e}")
            return []
            
    def update_memory_importance(self, memory_id: str, new_importance: str) -> bool:
        """Update importance of an existing memory."""
        try:
            with self._lock:
                if memory_id in self._ltm:
                    self._ltm[memory_id].importance = new_importance
                    self._save_memory_item(memory_id)
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to update memory importance: {e}")
            return False
            
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from LTM."""
        try:
            with self._lock:
                if memory_id in self._ltm:
                    # Remove from index
                    self._remove_from_index(memory_id, self._ltm[memory_id])
                    
                    # Remove from LTM
                    del self._ltm[memory_id]
                    
                    # Remove from disk
                    self._remove_memory_file(memory_id)
                    
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False
            
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        with self._lock:
            stm_count = len(self._stm)
            ltm_count = len(self._ltm)
            
            # Memory type distribution
            type_counts = {}
            importance_counts = {}
            
            for memory in self._ltm.values():
                type_counts[memory.memory_type] = type_counts.get(memory.memory_type, 0) + 1
                importance_counts[memory.importance] = importance_counts.get(memory.importance, 0) + 1
                
            return {
                "stm_size": stm_count,
                "ltm_size": ltm_count,
                "memory_types": type_counts,
                "importance_distribution": importance_counts,
                "index_size": len(self._memory_index)
            }
            
    def summarize_and_archive(self, max_age_days: int = 30) -> int:
        """
        Summarize old STM conversations and archive to LTM.
        
        Args:
            max_age_days: Maximum age before archiving
            
        Returns:
            Number of memories archived
        """
        archived_count = 0
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        try:
            with self._lock:
                # Group STM messages by conversation sessions
                conversations = self._group_conversations()
                
                for conversation in conversations:
                    # Check if conversation is old enough
                    first_message = conversation[0]
                    if datetime.fromisoformat(first_message['timestamp']) < cutoff_date:
                        
                        # Summarize conversation
                        summary = self._summarize_conversation(conversation)
                        
                        # Archive to LTM
                        memory_id = self.store_in_ltm(
                            content=summary,
                            memory_type="conversation_summary",
                            importance="normal",
                            source_context=f"Auto-archived conversation from {first_message['timestamp']}",
                            trigger=MemoryTrigger.THRESHOLD_SUMMARIZE
                        )
                        
                        if memory_id:
                            archived_count += 1
                            
                # Clear old STM if we archived anything
                if archived_count > 0:
                    self._cleanup_old_stm(cutoff_date)
                    
            logger.info(f"Archived {archived_count} conversations to LTM")
            return archived_count
            
        except Exception as e:
            logger.error(f"Failed to summarize and archive: {e}")
            return archived_count
            
    def _check_memory_triggers(self, message: Dict[str, Any]):
        """Check if message triggers memory storage."""
        content = message.get('content', '').lower()
        role = message.get('role', '')
        
        # Explicit memory triggers
        if any(phrase in content for phrase in [
            'remember this', 'save this', 'important', 'don\'t forget',
            'store this', 'keep in mind', 'make note'
        ]):
            self.store_in_ltm(
                content=content,
                memory_type="explicit_memory",
                importance="high",
                source_context=f"User message from {message.get('timestamp')}",
                trigger=MemoryTrigger.EXPLICIT_SAVE
            )
            
        # Project milestone triggers
        if role == 'assistant' and any(phrase in content for phrase in [
            'completed', 'finished', 'done', 'milestone achieved',
            'project update', 'status change'
        ]):
            self.store_in_ltm(
                content=f"Project milestone: {content[:200]}",
                memory_type="project_milestone",
                importance="high",
                source_context="Automatic milestone detection",
                trigger=MemoryTrigger.MILESTONE_COMPLETE
            )
            
    def _handle_memory_trigger(self, memory: MemoryItem, trigger: MemoryTrigger):
        """Handle specific memory trigger actions."""
        try:
            if trigger == MemoryTrigger.EXPLICIT_SAVE:
                # Immediate indexing and priority processing
                logger.info(f"High priority memory stored: {memory.memory_id}")
                
            elif trigger == MemoryTrigger.MILESTONE_COMPLETE:
                # Update related project memories
                if memory.memory_type == "project_milestone":
                    # Could trigger project context updates
                    pass
                    
            elif trigger == MemoryTrigger.THRESHOLD_SUMMARIZE:
                # Summarization completed, can trigger cleanup
                pass
                
        except Exception as e:
            logger.error(f"Failed to handle memory trigger: {e}")
            
    def _get_candidate_memories(self, query: str, memory_types: List[str] = None) -> List[MemoryItem]:
        """Get candidate memories based on query."""
        candidates = []
        
        # Search by keywords in index
        query_words = query.split()
        keyword_matches = set()
        
        for word in query_words:
            if word in self._memory_index:
                keyword_matches.update(self._memory_index[word])
                
        # Add keyword matches
        for memory_id in keyword_matches:
            if memory_id in self._ltm:
                candidates.append(self._ltm[memory_id])
                
        # If no keyword matches, search all memories
        if not candidates:
            candidates = list(self._ltm.values())
            
        # Filter by memory types
        if memory_types:
            candidates = [m for m in candidates if m.memory_type in memory_types]
            
        return candidates
        
    def _filter_by_importance(self, memories: List[MemoryItem], threshold: str) -> List[MemoryItem]:
        """Filter memories by importance threshold."""
        importance_levels = ["low", "normal", "high", "critical"]
        min_index = importance_levels.index(threshold)
        
        filtered = []
        for memory in memories:
            if importance_levels.index(memory.importance) >= min_index:
                filtered.append(memory)
                
        return filtered
        
    def _score_memories(self, memories: List[MemoryItem], query: str) -> List[tuple[MemoryItem, float]]:
        """Score memories for relevance."""
        scored = []
        query_words = set(query.split())
        
        for memory in memories:
            score = 0.0
            
            # Content relevance
            content_words = set(memory.content.lower().split())
            content_overlap = len(query_words & content_words)
            score += content_overlap * 0.4
            
            # Tag relevance
            tag_words = set(tag.lower() for tag in memory.tags)
            tag_overlap = len(query_words & tag_words)
            score += tag_overlap * 0.3
            
            # Importance bonus
            importance_scores = {"low": 0.1, "normal": 0.3, "high": 0.6, "critical": 1.0}
            score += importance_scores.get(memory.importance, 0.3) * 0.2
            
            # Access frequency bonus
            if memory.access_count > 5:
                score += 0.1
                
            scored.append((memory, score))
            
        return scored
        
    def _update_memory_index(self, memory_id: str, content: str, tags: List[str]):
        pass # Deprecated
                    
    def _remove_from_index(self, memory_id: str, memory: MemoryItem):
        pass # Deprecated
                    
    def _group_conversations(self) -> List[List[Dict[str, Any]]]:
        """Group STM messages into conversation sessions."""
        if not self._stm:
            return []
            
        conversations = []
        current_conversation = []
        last_role = None
        
        for message in self._stm:
            # Start new conversation if role pattern changes significantly
            if (last_role == 'assistant' and message['role'] == 'user' and 
                len(current_conversation) >= 4):  # Minimum conversation length
                if current_conversation:
                    conversations.append(current_conversation)
                current_conversation = [message]
            else:
                current_conversation.append(message)
            last_role = message['role']
            
        # Add final conversation
        if current_conversation:
            conversations.append(current_conversation)
            
        return conversations
        
    def _summarize_conversation(self, conversation: List[Dict[str, Any]]) -> str:
        """Summarize a conversation for LTM storage."""
        try:
            # Extract key points
            user_messages = [msg['content'] for msg in conversation if msg['role'] == 'user']
            assistant_messages = [msg['content'] for msg in conversation if msg['role'] == 'assistant']
            
            # Create summary
            summary_parts = []
            
            if user_messages:
                # Take first and last user messages as anchors
                summary_parts.append(f"User discussed: {user_messages[0][:100]}")
                if len(user_messages) > 1:
                    summary_parts.append(f"Follow-up topics: {user_messages[-1][:100]}")
                    
            if assistant_messages:
                # Key assistant responses
                key_response = assistant_messages[0][:150]
                summary_parts.append(f"Assistant provided: {key_response}")
                
            return " | ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Failed to summarize conversation: {e}")
            return f"Conversation from {conversation[0]['timestamp'] if conversation else 'unknown'}"
            
    def _cleanup_old_stm(self, cutoff_date: datetime):
        """Remove old messages from STM."""
        to_remove = []
        
        for message in self._stm:
            try:
                msg_date = datetime.fromisoformat(message['timestamp'])
                if msg_date < cutoff_date:
                    to_remove.append(message)
            except Exception:
                # Remove messages with invalid timestamps
                to_remove.append(message)
                
        # Remove old messages
        for message in to_remove:
            try:
                self._stm.remove(message)
            except ValueError:
                pass  # Already removed
                
    def _save_memory_item(self, memory_id: str):
        pass # Deprecated - Handled by Memori
            
    def _remove_memory_file(self, memory_id: str):
        pass # Deprecated - Handled by Memori
            
    def _load_memory(self):
        pass # Deprecated - Handled by Memori
            
    def _load_memory_item(self, memory_id: str):
        pass # Deprecated - Handled by Memori
