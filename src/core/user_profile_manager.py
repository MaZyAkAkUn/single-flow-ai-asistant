"""
User profile manager for handling user personalization data.
"""
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

from ..data.schemas import UserPersonalization, MemoryItem
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class UserProfileManager:
    """
    Manages user personalization preferences and long-term memory.
    Handles user preferences, interaction patterns, and persistent data.
    """
    
    def __init__(self, data_dir: str = "./data/user_profile"):
        """Initialize user profile manager."""
        self.data_dir = data_dir
        self._ensure_data_dir()
        self._user_profile: UserPersonalization = UserPersonalization()
        self._user_memories: Dict[str, MemoryItem] = {}
        self._interaction_patterns: Dict[str, Any] = {}
        self._load_profile()
        
    def _ensure_data_dir(self):
        """Ensure user profile data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_user_profile(self) -> UserPersonalization:
        """Get current user profile."""
        return self._user_profile
        
    def update_user_profile(self, 
                           communication_style: Optional[str] = None,
                           detail_preference: Optional[str] = None,
                           opinion_requested: Optional[bool] = None,
                           code_style_preference: Optional[str] = None,
                           explanation_depth: Optional[str] = None,
                           response_format: Optional[str] = None,
                           cultural_context: Optional[str] = None) -> bool:
        """Update user profile preferences."""
        try:
            if communication_style:
                self._user_profile.communication_style = communication_style
            if detail_preference:
                self._user_profile.detail_preference = detail_preference
            if opinion_requested is not None:
                self._user_profile.opinion_requested = opinion_requested
            if code_style_preference:
                self._user_profile.code_style_preference = code_style_preference
            if explanation_depth:
                self._user_profile.explanation_depth = explanation_depth
            if response_format:
                self._user_profile.response_format = response_format
            if cultural_context:
                self._user_profile.cultural_context = cultural_context
                
            self._save_profile()
            logger.info(f"Updated user profile: {self._user_profile}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user profile: {e}")
            return False
            
    def store_memory(self, 
                    content: str,
                    memory_type: str = "fact",
                    importance: str = "normal",
                    tags: List[str] = None,
                    source_context: str = "") -> str:
        """
        Store a memory item for the user.
        
        Args:
            content: Memory content
            memory_type: Type of memory (preference, project_metadata, decision, fact, skill)
            importance: Importance level (low, normal, high, critical)
            tags: List of tags for categorization
            source_context: Context where memory was created
            
        Returns:
            Memory ID if successful, None otherwise
        """
        try:
            import uuid
            memory_id = str(uuid.uuid4())
            
            memory_item = MemoryItem(
                memory_id=memory_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags or [],
                source_context=source_context,
                created_at=datetime.now()
            )
            
            self._user_memories[memory_id] = memory_item
            self._save_memories()
            
            logger.info(f"Stored memory: {memory_type} ({memory_id})")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return None
            
    def retrieve_memories(self, 
                         query: str,
                         memory_types: List[str] = None,
                         importance_threshold: str = "low",
                         limit: int = 10) -> List[MemoryItem]:
        """
        Retrieve relevant memories based on query and filters.
        
        Args:
            query: Search query for memory content
            memory_types: Filter by memory types
            importance_threshold: Minimum importance level
            limit: Maximum number of results
            
        Returns:
            List of relevant memory items
        """
        try:
            query_lower = query.lower()
            relevant_memories = []
            
            importance_levels = ["low", "normal", "high", "critical"]
            min_importance_index = importance_levels.index(importance_threshold)
            
            for memory in self._user_memories.values():
                # Check importance threshold
                if importance_levels.index(memory.importance) < min_importance_index:
                    continue
                    
                # Check memory type filter
                if memory_types and memory.memory_type not in memory_types:
                    continue
                    
                # Check content relevance
                if (query_lower in memory.content.lower() or 
                    any(query_lower in tag.lower() for tag in memory.tags)):
                    relevant_memories.append(memory)
                    
            # Sort by importance and access count
            relevant_memories.sort(key=lambda m: (
                importance_levels.index(m.importance),
                m.access_count
            ), reverse=True)
            
            # Update access tracking
            for memory in relevant_memories[:limit]:
                memory.last_accessed = datetime.now()
                memory.access_count += 1
                
            return relevant_memories[:limit]
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
            
    def learn_from_interaction(self, 
                              user_message: str,
                              assistant_response: str,
                              user_satisfaction: Optional[float] = None):
        """
        Learn from user interaction to improve personalization.
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
            user_satisfaction: Optional satisfaction rating (0.0-1.0)
        """
        try:
            # Track interaction patterns
            interaction_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._interaction_patterns[interaction_id] = {
                "user_message_length": len(user_message),
                "response_length": len(assistant_response),
                "user_satisfaction": user_satisfaction,
                "timestamp": datetime.now().isoformat()
            }
            
            # Analyze preferences from content
            self._analyze_preferences(user_message, assistant_response)
            
            # Keep only recent interactions
            if len(self._interaction_patterns) > 1000:
                # Keep last 500 interactions
                recent_ids = sorted(self._interaction_patterns.keys())[-500:]
                self._interaction_patterns = {
                    k: v for k, v in self._interaction_patterns.items() 
                    if k in recent_ids
                }
                
            self._save_interaction_patterns()
            
        except Exception as e:
            logger.error(f"Failed to learn from interaction: {e}")
            
    def _analyze_preferences(self, user_message: str, assistant_response: str):
        """Analyze user message and response to infer preferences."""
        try:
            message_lower = user_message.lower()
            response_lower = assistant_response.lower()
            
            # Detect communication style preferences
            if any(word in message_lower for word in ["brief", "short", "concise"]):
                if self._user_profile.detail_preference not in ["brief", "balanced"]:
                    self._user_profile.detail_preference = "brief"
                    
            if any(word in message_lower for word in ["detailed", "explain more", "thorough"]):
                if self._user_profile.detail_preference != "comprehensive":
                    self._user_profile.detail_preference = "comprehensive"
                    
            # Detect code style preferences
            if "```python" in message_lower or "python" in message_lower:
                self._user_profile.code_style_preference = "pythonic"
            elif "```javascript" in message_lower or "javascript" in message_lower:
                self._user_profile.code_style_preference = "modern_js"
                
        except Exception as e:
            logger.error(f"Failed to analyze preferences: {e}")
            
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about user memories."""
        total_memories = len(self._user_memories)
        memory_types = {}
        importance_counts = {}
        
        for memory in self._user_memories.values():
            memory_types[memory.memory_type] = memory_types.get(memory.memory_type, 0) + 1
            importance_counts[memory.importance] = importance_counts.get(memory.importance, 0) + 1
            
        return {
            "total_memories": total_memories,
            "memory_types": memory_types,
            "importance_counts": importance_counts,
            "total_interactions": len(self._interaction_patterns)
        }
        
    def _save_profile(self):
        """Save user profile to disk."""
        try:
            profile_file = os.path.join(self.data_dir, "user_profile.json")
            profile_data = {
                "communication_style": self._user_profile.communication_style,
                "detail_preference": self._user_profile.detail_preference,
                "opinion_requested": self._user_profile.opinion_requested,
                "code_style_preference": self._user_profile.code_style_preference,
                "explanation_depth": self._user_profile.explanation_depth,
                "response_format": self._user_profile.response_format,
                "cultural_context": self._user_profile.cultural_context,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save user profile: {e}")
            
    def _save_memories(self):
        """Save memories to disk."""
        try:
            memories_file = os.path.join(self.data_dir, "memories.json")
            memories_data = {}
            
            for memory_id, memory in self._user_memories.items():
                memories_data[memory_id] = {
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "memory_type": memory.memory_type,
                    "importance": memory.importance,
                    "tags": memory.tags,
                    "created_at": memory.created_at.isoformat(),
                    "last_accessed": memory.last_accessed.isoformat(),
                    "access_count": memory.access_count,
                    "source_context": memory.source_context
                }
                
            with open(memories_file, 'w', encoding='utf-8') as f:
                json.dump(memories_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
            
    def _save_interaction_patterns(self):
        """Save interaction patterns to disk."""
        try:
            patterns_file = os.path.join(self.data_dir, "interaction_patterns.json")
            
            with open(patterns_file, 'w', encoding='utf-8') as f:
                json.dump(self._interaction_patterns, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save interaction patterns: {e}")
            
    def _load_profile_data(self):
        """Load user profile from disk."""
        try:
            profile_file = os.path.join(self.data_dir, "user_profile.json")
            if not os.path.exists(profile_file):
                return
                
            with open(profile_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self._user_profile.communication_style = data.get("communication_style", "casual_technical")
            self._user_profile.detail_preference = data.get("detail_preference", "comprehensive")
            self._user_profile.opinion_requested = data.get("opinion_requested", True)
            self._user_profile.code_style_preference = data.get("code_style_preference", "readable")
            self._user_profile.explanation_depth = data.get("explanation_depth", "adaptive")
            self._user_profile.response_format = data.get("response_format", "conversational")
            self._user_profile.cultural_context = data.get("cultural_context", "western")
            
        except Exception as e:
            logger.error(f"Failed to load user profile: {e}")
            
    def _load_memories(self):
        """Load memories from disk."""
        try:
            memories_file = os.path.join(self.data_dir, "memories.json")
            if not os.path.exists(memories_file):
                return
                
            with open(memories_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for memory_id, memory_data in data.items():
                memory_item = MemoryItem(
                    memory_id=memory_data["memory_id"],
                    content=memory_data["content"],
                    memory_type=memory_data["memory_type"],
                    importance=memory_data["importance"],
                    tags=memory_data["tags"],
                    created_at=datetime.fromisoformat(memory_data["created_at"]),
                    last_accessed=datetime.fromisoformat(memory_data["last_accessed"]),
                    access_count=memory_data["access_count"],
                    source_context=memory_data["source_context"]
                )
                self._user_memories[memory_id] = memory_item
                
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
            
    def _load_interaction_patterns(self):
        """Load interaction patterns from disk."""
        try:
            patterns_file = os.path.join(self.data_dir, "interaction_patterns.json")
            if not os.path.exists(patterns_file):
                return
                
            with open(patterns_file, 'r', encoding='utf-8') as f:
                self._interaction_patterns = json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load interaction patterns: {e}")
            
    def _load_profile(self):
        """Load complete profile data."""
        self._load_profile_data()
        self._load_memories()
        self._load_interaction_patterns()
