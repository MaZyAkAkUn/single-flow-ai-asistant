"""
Memory Adapter - Memori SDK integration for long-term memory management.
Provides LangChain-compatible memory interface with advanced retrieval capabilities.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio

from .logging_config import get_logger

logger = get_logger(__name__)


class MemoryAdapter:
    """
    Adapter for Memori SDK providing LangChain-compatible memory interface.
    Handles episodic, semantic, and procedural memory types with advanced retrieval.
    """

    def __init__(self, database_path: str = "data/memory/memori.db", namespace: str = "personal_assistant", openai_api_key: Optional[str] = None):
        """
        Initialize the memory adapter.

        Args:
            database_path: Path to SQLite database for memory storage
            namespace: Namespace for memory isolation
            openai_api_key: OpenAI API key for Memori operations
        """
        self.database_path = database_path
        self.namespace = namespace
        self.openai_api_key = openai_api_key
        self.memory_system = None
        self.memory_tool = None
        self.is_enabled = False

        # Ensure data directory exists
        os.makedirs(os.path.dirname(database_path), exist_ok=True)

        # Memory configuration
        self.max_memories_per_query = 5
        self.relevance_threshold = 0.1
        self.recency_weight = 0.3
        self.relevance_weight = 0.7

        # Initialize memory system only if API key is available
        if openai_api_key:
            self._initialize_memory_system()
        else:
            logger.info("Memory adapter initialized without API key - will initialize when key becomes available")

    def _initialize_memory_system(self):
        """Initialize the Memori memory system."""
        try:
            # Check if Memori SDK is available
            try:
                import memori
            except ImportError:
                logger.warning("Memori SDK not installed. Install with: pip install memori. Memory features will be disabled.")
                self.is_enabled = False
                return

            # Import Memori components
            from memori import Memori, create_memory_tool

            # Validate API key if provided
            if not self.openai_api_key:
                logger.warning("OpenAI API key required for memory system but not provided. Memory features will be disabled.")
                self.is_enabled = False
                return

            # Validate API key format
            if not isinstance(self.openai_api_key, str) or not self.openai_api_key.strip().startswith('sk-'):
                logger.warning("Invalid OpenAI API key format for memory system. Memory features will be disabled.")
                self.is_enabled = False
                return

            # Prepare Memori initialization parameters
            memori_kwargs = {
                "database_connect": f"sqlite:///{self.database_path}",
                "conscious_ingest": True,
                "verbose": False,
                "namespace": self.namespace,
                "openai_api_key": self.openai_api_key
            }

            # Initialize Memori
            self.memory_system = Memori(**memori_kwargs)

            # Enable the memory system
            self.memory_system.enable()
            self.is_enabled = True

            # Create memory tool for advanced operations
            self.memory_tool = create_memory_tool(self.memory_system)

            logger.info(f"Memori memory system initialized with database: {self.database_path}")

        except ImportError as e:
            logger.warning(f"Memori SDK import failed: {e}. Memory features will be disabled.")
            self.is_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            self.is_enabled = False

    def is_available(self) -> bool:
        """Check if memory system is available and enabled."""
        return self.is_enabled and self.memory_system is not None

    def record_conversation(self, user_input: str, ai_output: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Record a conversation turn in memory.

        Args:
            user_input: User's message
            ai_output: AI's response
            metadata: Additional metadata (tags, importance, etc.)
        """
        if not self.is_available():
            return

        try:
            # Prepare metadata
            conversation_metadata = {
                "timestamp": datetime.now().isoformat(),
                "type": "conversation",
                "importance": metadata.get("importance", "normal") if metadata else "normal",
                "tags": metadata.get("tags", []) if metadata else [],
            }

            # Record the conversation
            self.memory_system.record_conversation(
                user_input=user_input,
                ai_output=ai_output,
                metadata=conversation_metadata
            )

            logger.debug("Conversation recorded in memory")

        except Exception as e:
            logger.error(f"Failed to record conversation: {e}")

    def store_fact(self, fact: str, fact_type: str = "semantic", importance: str = "normal",
                   tags: Optional[List[str]] = None, expires_at: Optional[datetime] = None):
        """
        Store a specific fact in memory.

        Args:
            fact: The fact to store
            fact_type: Type of fact (semantic, episodic, procedural)
            importance: Importance level (low, normal, high, critical)
            tags: List of tags for categorization
            expires_at: Optional expiration date
        """
        if not self.is_available():
            return

        try:
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "type": fact_type,
                "importance": importance,
                "tags": tags or [],
                "expires_at": expires_at.isoformat() if expires_at else None,
            }

            # Since Memori doesn't have a direct store_fact method, we simulate 
            # a conversation turn that forces the memory to be recorded.
            # This allows Memori's extraction system to process it naturally.
            user_input = f"Note this fact: {fact}"
            ai_output = "I have noted that fact."

            # Record the conversation simulation
            self.memory_system.record_conversation(
                user_input=user_input,
                ai_output=ai_output,
                metadata=metadata
            )

            logger.debug(f"Fact stored in memory via conversation record: {fact[:50]}...")

        except Exception as e:
            logger.error(f"Failed to store fact: {e}")

    def search_memory(self, query: str, limit: Optional[int] = None,
                     memory_types: Optional[List[str]] = None,
                     tags: Optional[List[str]] = None,
                     min_importance: str = "low") -> List[Dict[str, Any]]:
        """
        Search memory for relevant information.

        Args:
            query: Search query
            limit: Maximum number of results
            memory_types: Filter by memory types
            tags: Filter by tags
            min_importance: Minimum importance level

        Returns:
            List of relevant memories with scores
        """
        if not self.is_available():
            return []

        try:
            limit = limit or self.max_memories_per_query
            query = query.strip() if query else ""
            raw_results = []

            if not query:
                # Browse mode: Iterate standard categories if query is empty
                categories = ["conversation", "fact", "semantic", "episodic", "procedural"]
                if memory_types:
                     # If types specified, use those instead
                     categories = memory_types
                
                all_results = []
                for cat in categories:
                    try:
                        # Check if search_memories_by_category exists on the instance
                        if hasattr(self.memory_system, 'search_memories_by_category'):
                            cat_results = self.memory_system.search_memories_by_category(cat, limit=limit)
                            if cat_results:
                                all_results.extend(cat_results)
                        else:
                             # Fallback if method missing
                             pass
                    except Exception as e:
                        logger.warning(f"Failed to search category {cat}: {e}")
                
                raw_results = all_results
            else:
                # Use native Memori context retrieval
                # This is more efficient and accurate than using the tool wrapper
                raw_results = self.memory_system.retrieve_context(query, limit=limit)
            
            # Parse and normalize results
            memories = self._parse_memori_results(raw_results)
            
            # Deduplicate by ID if browsing (assuming retrieve_context does it for search)
            if not query:
                seen_ids = set()
                unique_memories = []
                for mem in memories:
                    mem_id = mem['metadata'].get('id')
                    # Use content hash if ID missing? content is good proxy
                    if not mem_id:
                        mem_id = hash(mem.get('content', ''))
                    
                    if mem_id not in seen_ids:
                        seen_ids.add(mem_id)
                        unique_memories.append(mem)
                memories = unique_memories

            # Filter memories based on criteria
            filtered_memories = self._filter_memories(
                memories, memory_types, tags, min_importance
            )

            # Sort by relevance and recency
            sorted_memories = self._sort_memories(filtered_memories)

            return sorted_memories[:limit]

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []

    def _parse_memori_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse and normalize results from Memori SDK.
        
        Args:
            results: List of raw memory dictionaries from Memori
            
        Returns:
            List of normalized memory dictionaries
        """
        normalized = []
        
        try:
            for item in results:
                # Handle timestamp conversion
                created_at = item.get("created_at")
                if isinstance(created_at, datetime):
                    timestamp_str = created_at.isoformat()
                elif isinstance(created_at, str):
                    timestamp_str = created_at
                else:
                    timestamp_str = datetime.now().isoformat()

                # Memori returns dicts with 'content', 'category', 'importance_score', 'created_at' etc.
                # It may also include the original metadata we stored
                metadata = item.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        import json
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                # Determine type: try 'category' first, then look in metadata, default to 'unknown'
                mem_type = item.get("category")
                if not mem_type or mem_type == "unknown":
                    mem_type = metadata.get("type", "unknown")
                
                normalized_item = {
                    "content": item.get("content", ""),
                    "score": float(item.get("importance_score", 0.0)),
                    "metadata": {
                        "type": mem_type,
                        "timestamp": timestamp_str,
                        # Map score to importance levels if needed, or keep explicit importance if available
                        "importance": self._score_to_importance(item.get("importance_score", 0.0)),
                        "tags": item.get("tags", []),
                        "id": item.get("id")
                    }
                }
                
                # Merge any other metadata fields we might want
                if isinstance(metadata, dict):
                    if "tags" in metadata and not normalized_item["metadata"]["tags"]:
                         normalized_item["metadata"]["tags"] = metadata["tags"]
                    if "importance" in metadata:
                         # Use explicit importance if available instead of derived score
                         normalized_item["metadata"]["importance"] = metadata["importance"]
                normalized.append(normalized_item)
                
        except Exception as e:
            logger.error(f"Error parsing Memori results: {e}")
            
        return normalized

    def _score_to_importance(self, score: float) -> str:
        """Convert float score to importance string."""
        if score >= 0.8: return "critical"
        if score >= 0.6: return "high"
        if score >= 0.3: return "normal"
        return "low"

    def _parse_search_results(self, result: Any) -> List[Dict[str, Any]]:
        """Parse search results from memory tool (Legacy)."""
        try:
            if isinstance(result, str):
                # Try to parse as JSON
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    return [parsed]
            elif isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return [result]

            return []

        except (json.JSONDecodeError, TypeError):
            # If parsing fails, return empty list
            return []

    def _filter_memories(self, memories: List[Dict[str, Any]],
                        memory_types: Optional[List[str]] = None,
                        tags: Optional[List[str]] = None,
                        min_importance: str = "low") -> List[Dict[str, Any]]:
        """Filter memories based on criteria."""
        filtered = []

        importance_levels = {"low": 0, "normal": 1, "high": 2, "critical": 3}
        min_level = importance_levels.get(min_importance, 0)

        for memory in memories:
            # Filter by memory type
            if memory_types:
                mem_type = memory.get("metadata", {}).get("type", "unknown")
                if mem_type not in memory_types:
                    continue

            # Filter by tags
            if tags:
                mem_tags = memory.get("metadata", {}).get("tags", [])
                if not any(tag in mem_tags for tag in tags):
                    continue

            # Filter by importance
            importance = memory.get("metadata", {}).get("importance", "normal")
            if importance_levels.get(importance, 0) < min_level:
                continue

            # Check expiration
            expires_at = memory.get("metadata", {}).get("expires_at")
            if expires_at:
                try:
                    if datetime.fromisoformat(expires_at) < datetime.now():
                        continue  # Expired
                except ValueError:
                    pass  # Invalid date format, keep the memory

            filtered.append(memory)

        return filtered

    def _sort_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort memories by relevance and recency."""
        def sort_key(memory):
            score = memory.get("score", 0.0)

            # Apply recency weighting
            timestamp = memory.get("metadata", {}).get("timestamp")
            if timestamp:
                try:
                    mem_time = datetime.fromisoformat(timestamp)
                    hours_old = (datetime.now() - mem_time).total_seconds() / 3600

                    # Recency score (newer = higher score)
                    recency_score = max(0, 1 - (hours_old / 168))  # 168 hours = 1 week
                    score = (self.relevance_weight * score) + (self.recency_weight * recency_score)
                except ValueError:
                    pass

            return score

        return sorted(memories, key=sort_key, reverse=True)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        if not self.is_available():
            return {"enabled": False}

        try:
            # This would need to be implemented based on Memori SDK capabilities
            # For now, return basic info
            return {
                "enabled": True,
                "database_path": self.database_path,
                "namespace": self.namespace,
                "max_memories_per_query": self.max_memories_per_query,
            }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"enabled": False, "error": str(e)}

    def clear_memory(self, memory_types: Optional[List[str]] = None,
                    tags: Optional[List[str]] = None,
                    older_than: Optional[datetime] = None):
        """
        Clear memories based on filters.

        Args:
            memory_types: Types of memories to clear
            tags: Tags to filter by
            older_than: Clear memories older than this date
        """
        if not self.is_available():
            return

        try:
            # This would need to be implemented based on Memori SDK capabilities
            # For now, log the request
            logger.info(f"Memory clearing requested - types: {memory_types}, tags: {tags}, older_than: {older_than}")

        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")

    def export_memory(self, file_path: str):
        """
        Export memory data to a file.

        Args:
            file_path: Path to export file
        """
        if not self.is_available():
            return

        try:
            # This would need to be implemented based on Memori SDK capabilities
            logger.info(f"Memory export requested to: {file_path}")

        except Exception as e:
            logger.error(f"Failed to export memory: {e}")

    def import_memory(self, file_path: str):
        """
        Import memory data from a file.

        Args:
            file_path: Path to import file
        """
        if not self.is_available():
            return

        try:
            # This would need to be implemented based on Memori SDK capabilities
            logger.info(f"Memory import requested from: {file_path}")

        except Exception as e:
            logger.error(f"Failed to import memory: {e}")

    # LangChain Memory Interface Methods

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        """
        Save context from LangChain conversation.
        LangChain Memory interface method.
        """
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")

        self.record_conversation(user_input, ai_output)

    def clear(self):
        """Clear all memories. LangChain Memory interface method."""
        self.clear_memory()

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables. LangChain Memory interface method."""
        return ["memory_context"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables for LangChain.
        LangChain Memory interface method.
        """
        query = inputs.get("input", "")
        memories = self.search_memory(query)

        # Format memories for prompt injection
        memory_context = self._format_memories_for_prompt(memories)

        return {"memory_context": memory_context}

    def _format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories for injection into prompts."""
        if not memories:
            return ""

        formatted = ["Relevant memories:"]
        for i, memory in enumerate(memories, 1):
            content = memory.get("content", "")
            score = memory.get("score", 0.0)
            mem_type = memory.get("metadata", {}).get("type", "unknown")

            formatted.append(f"[{i}] ({mem_type}, relevance: {score:.2f}): {content}")

        return "\n".join(formatted)

    # Async methods for future use

    async def arecord_conversation(self, user_input: str, ai_output: str,
                                  metadata: Optional[Dict[str, Any]] = None):
        """Async version of record_conversation."""
        # For now, just call sync version in thread pool
        await asyncio.get_event_loop().run_in_executor(
            None, self.record_conversation, user_input, ai_output, metadata
        )

    async def asearch_memory(self, query: str, limit: Optional[int] = None,
                           memory_types: Optional[List[str]] = None,
                           tags: Optional[List[str]] = None,
                           min_importance: str = "low") -> List[Dict[str, Any]]:
        """Async version of search_memory."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.search_memory, query, limit, memory_types, tags, min_importance
        )
