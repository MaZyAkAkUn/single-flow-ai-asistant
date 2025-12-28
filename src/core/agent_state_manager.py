"""
Agent state manager for tracking global and local agent status.
"""
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from ..data.schemas import AgentState, AgentMode, IntentType
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class AgentStateManager:
    """
    Manages agent operational state including global and local status tracking.
    Handles agent mode transitions, goal tracking, and session management.
    """
    
    def __init__(self, state_file: str = "./data/agent_state.json"):
        """Initialize agent state manager."""
        self.state_file = state_file
        self._current_state: AgentState = AgentState()
        self._state_history: List[Dict[str, Any]] = []
        self._session_goals: Dict[str, Dict[str, Any]] = {}
        self._load_state()
        
    def get_current_state(self) -> AgentState:
        """Get current agent state."""
        return self._current_state
        
    def update_agent_mode(self, mode: AgentMode, goal: str = "") -> bool:
        """
        Update agent mode and current goal.
        
        Args:
            mode: New agent mode
            goal: Current goal or task description
            
        Returns:
            True if state changed, False otherwise
        """
        try:
            old_mode = self._current_state.current_mode
            
            # Update state
            self._current_state.current_mode = mode
            self._current_state.current_goal = goal
            self._current_state.updated_at = datetime.now()
            
            # Log state transition
            if old_mode != mode:
                logger.info(f"Agent mode changed: {old_mode.value} -> {mode.value}")
                self._log_state_transition("mode", old_mode.value, mode.value)
                
            # Update session goals if needed
            if goal and mode in [AgentMode.PROJECT_WORK, AgentMode.PLANNING, AgentMode.CODING]:
                self._session_goals[str(uuid.uuid4())] = {
                    "goal": goal,
                    "mode": mode.value,
                    "created_at": datetime.now().isoformat(),
                    "status": "active"
                }
                
            self._save_state()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent mode: {e}")
            return False
            
    def update_global_status(self, status: Dict[str, Any]) -> bool:
        """Update global agent status (persistent across sessions)."""
        try:
            self._current_state.global_status = status
            self._current_state.updated_at = datetime.now()
            self._save_state()
            logger.info(f"Global status updated: {status}")
            return True
        except Exception as e:
            logger.error(f"Failed to update global status: {e}")
            return False
            
    def update_local_status(self, status: Dict[str, Any]) -> bool:
        """Update local agent status (session-specific)."""
        try:
            self._current_state.local_status = status
            self._current_state.updated_at = datetime.now()
            self._save_state()
            logger.info(f"Local status updated: {status}")
            return True
        except Exception as e:
            logger.error(f"Failed to update local status: {e}")
            return False
            
    def set_session_id(self, session_id: str):
        """Set current session identifier."""
        self._current_state.session_id = session_id
        self._current_state.updated_at = datetime.now()
        
    def detect_agent_mode_from_intent(self, user_intent: IntentType, 
                                     user_message: str = "") -> AgentMode:
        """
        Auto-detect appropriate agent mode based on user intent and message content.
        
        Args:
            user_intent: Classified user intent
            user_message: User message for additional context
            
        Returns:
            Recommended agent mode
        """
        # Mode mapping based on intent
        intent_to_mode = {
            IntentType.CODE_GENERATION: AgentMode.CODING,
            IntentType.PLANNING_REQUEST: AgentMode.PLANNING,
            IntentType.TROUBLESHOOTING: AgentMode.PROBLEM_SOLVING,
            IntentType.CREATIVE_WRITING: AgentMode.DRAFTING,
            IntentType.EXPLAINATION: AgentMode.TUTORING,
            IntentType.ANALYSIS: AgentMode.DEEP_THINK,
        }
        
        mode = intent_to_mode.get(user_intent, AgentMode.CASUAL_CONVERSATION)
        
        # Additional heuristics based on message content
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ["project", "build", "develop", "implement"]):
            mode = AgentMode.PROJECT_WORK
        elif any(word in message_lower for word in ["search", "research", "find information"]):
            mode = AgentMode.RESEARCH
        elif any(word in message_lower for word in ["explain", "how does", "what is"]):
            mode = AgentMode.TUTORING
            
        return mode
        
    def get_active_goals(self) -> List[Dict[str, Any]]:
        """Get list of active goals for current session."""
        return [
            goal for goal in self._session_goals.values()
            if goal.get("status") == "active"
        ]
        
    def mark_goal_complete(self, goal_description: str) -> bool:
        """Mark a goal as completed."""
        try:
            for goal_id, goal in self._session_goals.items():
                if (goal.get("status") == "active" and 
                    goal_description.lower() in goal.get("goal", "").lower()):
                    goal["status"] = "completed"
                    goal["completed_at"] = datetime.now().isoformat()
                    logger.info(f"Goal completed: {goal.get('goal')}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to mark goal complete: {e}")
            return False
            
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current agent state."""
        return {
            "current_mode": self._current_state.current_mode.value,
            "current_goal": self._current_state.current_goal,
            "session_id": self._current_state.session_id,
            "global_status": self._current_state.global_status,
            "local_status": self._current_state.local_status,
            "active_goals": len(self.get_active_goals()),
            "total_goals": len(self._session_goals),
            "last_updated": self._current_state.updated_at.isoformat()
        }
        
    def auto_transition_mode(self, message_patterns: List[str]) -> bool:
        """
        Attempt automatic mode transitions based on message patterns.
        
        Args:
            message_patterns: Recent user message patterns
            
        Returns:
            True if transition occurred
        """
        # Analyze patterns for mode suggestions
        coding_indicators = ["code", "function", "class", "bug", "error", "debug"]
        planning_indicators = ["plan", "roadmap", "strategy", "timeline", "steps"]
        research_indicators = ["research", "search", "find", "investigate", "analyze"]
        
        scores = {
            AgentMode.CODING: sum(1 for pattern in message_patterns 
                                 if any(indicator in pattern.lower() for indicator in coding_indicators)),
            AgentMode.PLANNING: sum(1 for pattern in message_patterns 
                                   if any(indicator in pattern.lower() for indicator in planning_indicators)),
            AgentMode.RESEARCH: sum(1 for pattern in message_patterns 
                                  if any(indicator in pattern.lower() for indicator in research_indicators))
        }
        
        # Find highest scoring mode
        best_mode = max(scores, key=scores.get)
        best_score = scores[best_mode]
        
        # Only transition if significant pattern match
        if best_score >= 2 and best_mode != self._current_state.current_mode:
            logger.info(f"Auto-transitioning to mode: {best_mode.value} (score: {best_score})")
            return self.update_agent_mode(best_mode, f"Auto-detected from message patterns")
            
        return False
        
    def get_persistence_data(self) -> Dict[str, Any]:
        """Get data that should persist across sessions."""
        return {
            "global_status": self._current_state.global_status,
            "persona_profile": self._current_state.persona_profile,
            "state_history": self._state_history[-10:]  # Last 10 transitions
        }
        
    def restore_persistence_data(self, data: Dict[str, Any]):
        """Restore persisted data."""
        try:
            if "global_status" in data:
                self._current_state.global_status = data["global_status"]
            if "persona_profile" in data:
                self._current_state.persona_profile = data["persona_profile"]
            if "state_history" in data:
                self._state_history.extend(data["state_history"])
                
            logger.info("Restored agent state persistence data")
            
        except Exception as e:
            logger.error(f"Failed to restore persistence data: {e}")
            
    def _log_state_transition(self, field: str, old_value: str, new_value: str):
        """Log state transition for history."""
        transition = {
            "timestamp": datetime.now().isoformat(),
            "field": field,
            "old_value": old_value,
            "new_value": new_value,
            "session_id": self._current_state.session_id
        }
        self._state_history.append(transition)
        
        # Keep only recent history
        if len(self._state_history) > 100:
            self._state_history = self._state_history[-50:]
            
    def _save_state(self):
        """Save current state to disk."""
        try:
            state_data = {
                "current_state": {
                    "current_mode": self._current_state.current_mode.value,
                    "current_goal": self._current_state.current_goal,
                    "persona_profile": self._current_state.persona_profile,
                    "global_status": self._current_state.global_status,
                    "local_status": self._current_state.local_status,
                    "session_id": self._current_state.session_id,
                    "created_at": self._current_state.created_at.isoformat(),
                    "updated_at": self._current_state.updated_at.isoformat()
                },
                "session_goals": self._session_goals,
                "state_history": self._state_history
            }
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
            
    def _load_state(self):
        """Load state from disk."""
        try:
            if not os.path.exists(self.state_file):
                return
                
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            state_data = data.get("current_state", {})
            
            # Restore state
            self._current_state.current_mode = AgentMode(state_data.get("current_mode", "casual_conversation"))
            self._current_state.current_goal = state_data.get("current_goal", "")
            self._current_state.persona_profile = state_data.get("persona_profile", "helpful_assistant")
            self._current_state.global_status = state_data.get("global_status")
            self._current_state.local_status = state_data.get("local_status")
            self._current_state.session_id = state_data.get("session_id", "")
            
            created_at = state_data.get("created_at")
            if created_at:
                self._current_state.created_at = datetime.fromisoformat(created_at)
                
            updated_at = state_data.get("updated_at")
            if updated_at:
                self._current_state.updated_at = datetime.fromisoformat(updated_at)
                
            # Restore goals and history
            self._session_goals = data.get("session_goals", {})
            self._state_history = data.get("state_history", [])
            
            logger.info("Loaded agent state from disk")
            
        except Exception as e:
            logger.error(f"Failed to load agent state: {e}")
            # Initialize with default state
            self._current_state = AgentState()
            
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old state history and completed goals."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # Clean old history
        self._state_history = [
            entry for entry in self._state_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
        ]
        
        # Clean completed goals older than cutoff
        completed_goals = [
            goal_id for goal_id, goal in self._session_goals.items()
            if goal.get("status") == "completed"
        ]
        
        for goal_id in completed_goals:
            goal = self._session_goals[goal_id]
            completed_at = goal.get("completed_at")
            if completed_at:
                if datetime.fromisoformat(completed_at) < cutoff_date:
                    del self._session_goals[goal_id]
                    
        logger.info(f"Cleaned up old agent state data")
