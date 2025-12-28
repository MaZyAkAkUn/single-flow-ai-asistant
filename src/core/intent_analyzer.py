"""
Intent analyzer for classifying user intent and determining agent mode.
"""
import re
import logging
import json
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

from ..data.schemas import UserIntent, IntentType, AgentMode
from ..core.logging_config import get_logger

if TYPE_CHECKING:
    from ..langchain_adapters.llm_adapter import LLMAdapter

logger = get_logger(__name__)


@dataclass
class IntentPattern:
    """Pattern for intent classification."""
    keywords: List[str]
    patterns: List[str]
    intent_type: IntentType
    confidence_boost: float
    required_context: List[str]


class IntentAnalyzer:
    """
    Analyzes user messages to classify intent and determine appropriate agent mode.
    Uses LLM-based analysis if available, falling back to pattern matching.
    """
    
    def __init__(self, llm_adapter: Optional['LLMAdapter'] = None):
        """
        Initialize intent analyzer.
        
        Args:
            llm_adapter: LLM adapter for advanced intent analysis
        """
        self.llm_adapter = llm_adapter
        self._intent_patterns: List[IntentPattern] = []
        self._initialize_patterns()
        
    def _initialize_patterns(self):
        """Initialize intent classification patterns (fallback mechanism)."""
        self._intent_patterns = [
            IntentPattern(
                keywords=["opinion", "think", "feel", "what do you think", "your opinion", "view", "perspective"],
                patterns=[r"what.*think", r"your opinion", r"how do you feel", r"what's your view"],
                intent_type=IntentType.OPINION_REQUEST,
                confidence_boost=0.3,
                required_context=["personal_preference", "subjective_analysis"]
            ),
            IntentPattern(
                keywords=["code", "function", "class", "implement", "write code", "program", "script", "algorithm"],
                patterns=[r"write.*code", r"create.*function", r"implement.*algorithm", r"debug.*code"],
                intent_type=IntentType.CODE_GENERATION,
                confidence_boost=0.4,
                required_context=["programming_language", "technical_requirements"]
            ),
            IntentPattern(
                keywords=["analyze", "analysis", "examine", "study", "investigate", "compare", "evaluate", "review"],
                patterns=[r"analyze.*", r"what is.*", r"explain.*", r"how does.*"],
                intent_type=IntentType.ANALYSIS,
                confidence_boost=0.2,
                required_context=["analytical_framework", "data_points"]
            ),
            IntentPattern(
                keywords=["quick", "brief", "simple", "fast", "just", "short answer", "summary"],
                patterns=[r"quick.*", r"brief.*", r"just.*", r"simple.*"],
                intent_type=IntentType.QUICK_ANSWER,
                confidence_boost=0.2,
                required_context=["concise_response"]
            ),
            IntentPattern(
                keywords=["plan", "roadmap", "strategy", "timeline", "steps", "roadmap", "approach", "structure"],
                patterns=[r"plan.*", r"how to.*", r"steps to.*", r"roadmap.*"],
                intent_type=IntentType.PLANNING_REQUEST,
                confidence_boost=0.3,
                required_context=["project_scope", "timeline_constraints"]
            ),
            IntentPattern(
                keywords=["explain", "what is", "how does", "why", "describe", "tell me about", "definition"],
                patterns=[r"what is.*", r"how does.*", r"explain.*", r"tell me about.*"],
                intent_type=IntentType.EXPLAINATION,
                confidence_boost=0.2,
                required_context=["explanation_depth", "audience_level"]
            ),
            IntentPattern(
                keywords=["error", "problem", "issue", "bug", "fix", "troubleshoot", "debug", "wrong", "broken"],
                patterns=[r"error.*", r"problem.*", r"not working", r"fix.*", r"debug.*"],
                intent_type=IntentType.TROUBLESHOOTING,
                confidence_boost=0.4,
                required_context=["error_details", "system_state"]
            ),
            IntentPattern(
                keywords=["write", "story", "creative", "poem", "essay", "content", "draft", "article"],
                patterns=[r"write.*", r"create.*story", r"poem.*", r"creative.*"],
                intent_type=IntentType.CREATIVE_WRITING,
                confidence_boost=0.3,
                required_context=["creative_style", "content_guidelines"]
            )
        ]
        
    def analyze_intent(self, user_message: str, conversation_context: List[Dict[str, Any]] = None) -> UserIntent:
        """
        Analyze user message to determine intent using rule-based quick path + LLM few-shot.

        Args:
            user_message: User's message text
            conversation_context: Recent conversation for context

        Returns:
            UserIntent object with classified intent and metadata
        """
        conversation_context = conversation_context or []

        # Step 1: Rule-based quick checks (fast path for obvious intents)
        rule_based_intent = self._rule_based_quick_check(user_message)
        if rule_based_intent:
            logger.debug(f"Rule-based intent detected: {rule_based_intent.intent_type.value}")
            return rule_based_intent

        # Step 2: LLM few-shot analysis if adapter available
        if self.llm_adapter:
            try:
                return self._analyze_with_llm_fewshot(user_message, conversation_context)
            except Exception as e:
                logger.warning(f"LLM intent analysis failed, falling back to patterns: {e}")

        # Step 3: Fallback to enhanced pattern-based analysis
        return self._analyze_with_patterns(user_message, conversation_context)

    def _rule_based_quick_check(self, user_message: str) -> Optional[UserIntent]:
        """
        Fast rule-based intent detection for obvious cases.

        Args:
            user_message: User's message text

        Returns:
            UserIntent if rule matches, None otherwise
        """
        message_lower = user_message.lower().strip()

        # Rule 1: Question starters
        if re.match(r'^(how|what|why|when|where|who|can you|tell me)', message_lower):
            if any(word in message_lower for word in ['think', 'opinion', 'view', 'feel']):
                return UserIntent(
                    intent_type=IntentType.OPINION_REQUEST,
                    confidence_score=0.8,
                    expected_detail_level="comprehensive",
                    context_requirements=["personal_preference"]
                )
            elif any(word in message_lower for word in ['code', 'function', 'program', 'implement', 'write']):
                return UserIntent(
                    intent_type=IntentType.CODE_GENERATION,
                    confidence_score=0.9,
                    expected_detail_level="comprehensive",
                    context_requirements=["programming_language", "technical_requirements"]
                )
            else:
                return UserIntent(
                    intent_type=IntentType.EXPLAINATION,
                    confidence_score=0.7,
                    expected_detail_level="adaptive",
                    context_requirements=["explanation_depth"]
                )

        # Rule 2: Code-related patterns
        if re.search(r'```|code|function|class|def |import |print\(', message_lower):
            return UserIntent(
                intent_type=IntentType.CODE_GENERATION,
                confidence_score=0.85,
                expected_detail_level="comprehensive",
                context_requirements=["programming_language"]
            )

        # Rule 3: Planning/roadmap keywords
        if any(word in message_lower for word in ['plan', 'roadmap', 'strategy', 'timeline', 'steps', 'how to']):
            return UserIntent(
                intent_type=IntentType.PLANNING_REQUEST,
                confidence_score=0.8,
                expected_detail_level="structured",
                context_requirements=["project_scope"]
            )

        # Rule 4: Analysis keywords
        if any(word in message_lower for word in ['analyze', 'analysis', 'compare', 'evaluate', 'review', 'study']):
            return UserIntent(
                intent_type=IntentType.ANALYSIS,
                confidence_score=0.8,
                expected_detail_level="comprehensive",
                context_requirements=["analytical_framework"]
            )

        # Rule 5: Troubleshooting keywords
        if any(word in message_lower for word in ['error', 'problem', 'issue', 'bug', 'fix', 'broken', 'not working']):
            return UserIntent(
                intent_type=IntentType.TROUBLESHOOTING,
                confidence_score=0.85,
                expected_detail_level="comprehensive",
                context_requirements=["error_details"]
            )

        # Rule 6: Very short messages (likely quick answers or conversational)
        if len(message_lower.split()) <= 3:
            if message_lower in ['yes', 'no', 'okay', 'thanks', 'thank you', 'please', 'sorry']:
                return UserIntent(
                    intent_type=IntentType.QUICK_ANSWER,
                    confidence_score=0.9,
                    expected_detail_level="brief",
                    context_requirements=[]
                )

        return None  # No rule matched

    def _analyze_with_llm_fewshot(self, user_message: str, conversation_context: List[Dict[str, Any]] = None) -> UserIntent:
        """
        Analyze intent using LLM with few-shot prompting.

        Args:
            user_message: User's message text
            conversation_context: Recent conversation history

        Returns:
            UserIntent with LLM-classified intent
        """
        conversation_context = conversation_context or []

        # Switch to intent analysis tool set (no tools needed for intent analysis)
        original_tool_set = None
        if hasattr(self.llm_adapter, 'get_current_tool_set'):
            original_tool_set = self.llm_adapter.get_current_tool_set()
            self.llm_adapter.set_tool_set("intent_analysis")

        try:
            # Format recent history for context
            recent_history = []
            for msg in conversation_context[-3:]:  # Last 3 messages for context
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:150]  # Truncate
                recent_history.append(f"{role}: {content}")

            history_str = "\n".join(recent_history) if recent_history else "No recent context"

            # Few-shot prompt with examples
            prompt = f"""You are an expert intent classifier. Map the user's message to a single intent from this list:
- opinion_request: User asks for your opinion, view, or perspective
- code_generation: User asks to write, fix, or modify code
- analysis_and_opinion: User asks for analysis, comparison, or evaluation of a topic
- quick_answer: User wants a short, factual answer without elaboration
- planning_request: User asks for a plan, roadmap, steps, or strategy
- explanation: User asks for an explanation, definition, or "how does it work"
- troubleshooting: User reports an error, bug, or problem
- creative_writing: User asks for stories, poems, or creative content

EXAMPLES:
- "How to set up nginx for docker?" -> code_generation
- "What's your take on biotech regulation?" -> opinion_request
- "Summarize economic situation in Venezuela 2025" -> analysis_and_opinion
- "What time is it?" -> quick_answer
- "Plan a trip to Europe" -> planning_request
- "Explain quantum computing" -> explanation
- "My code crashes with null pointer" -> troubleshooting
- "Write a short story about AI" -> creative_writing

RECENT CONVERSATION:
{history_str}

CURRENT MESSAGE: "{user_message}"

Return JSON only: {{"intent": "intent_name", "confidence": 0.95, "reasoning": "brief explanation"}}"""

            # Log the used model and provider
            logger.info(f"Using LLM provider: {self.llm_adapter.provider}, model: {self.llm_adapter.config.get('model', 'unknown')}")

            # Call LLM
            response_text = self.llm_adapter.generate_response(prompt)

            # Parse response
            try:
                # Clean response
                response_text = response_text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

                data = json.loads(response_text)

                # Map intent string to enum
                intent_str = data.get("intent", "quick_answer").lower()

                # Fix common variations
                intent_mapping = {
                    "opinion_request": IntentType.OPINION_REQUEST,
                    "code_generation": IntentType.CODE_GENERATION,
                    "analysis_and_opinion": IntentType.ANALYSIS,
                    "quick_answer": IntentType.QUICK_ANSWER,
                    "planning_request": IntentType.PLANNING_REQUEST,
                    "explanation": IntentType.EXPLAINATION,
                    "troubleshooting": IntentType.TROUBLESHOOTING,
                    "creative_writing": IntentType.CREATIVE_WRITING
                }

                intent_type = intent_mapping.get(intent_str, IntentType.QUICK_ANSWER)
                confidence = float(data.get("confidence", 0.7))

                # Determine detail level and structure based on intent
                detail_level, structure = self._get_detail_and_structure_for_intent(intent_type, user_message)

                return UserIntent(
                    intent_type=intent_type,
                    expected_detail_level=detail_level,
                    expected_structure=structure,
                    confidence_score=confidence,
                    context_requirements=self._get_context_requirements(intent_type),
                    task_complexity="medium"
                )

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse LLM few-shot response: {response_text}, error: {e}")
                # Fallback to pattern matching
                return self._analyze_with_patterns(user_message, conversation_context)
        finally:
            # Restore original tool set
            if original_tool_set and hasattr(self.llm_adapter, 'set_tool_set'):
                self.llm_adapter.set_tool_set(original_tool_set.name)

    def _get_detail_and_structure_for_intent(self, intent_type: IntentType, message: str) -> Tuple[str, str]:
        """
        Get expected detail level and structure for an intent.

        Args:
            intent_type: The classified intent
            message: Original message text

        Returns:
            Tuple of (detail_level, structure)
        """
        message_lower = message.lower()

        # Check for explicit detail indicators
        if any(word in message_lower for word in ["brief", "short", "quick"]):
            detail_level = "brief"
        elif any(word in message_lower for word in ["detailed", "comprehensive", "deep"]):
            detail_level = "comprehensive"
        else:
            # Default detail levels by intent
            detail_mapping = {
                IntentType.CODE_GENERATION: "comprehensive",
                IntentType.ANALYSIS: "comprehensive",
                IntentType.PLANNING_REQUEST: "structured",
                IntentType.TROUBLESHOOTING: "comprehensive",
                IntentType.EXPLAINATION: "adaptive",
                IntentType.OPINION_REQUEST: "comprehensive",
                IntentType.CREATIVE_WRITING: "comprehensive",
                IntentType.QUICK_ANSWER: "brief"
            }
            detail_level = detail_mapping.get(intent_type, "adaptive")

        # Structure based on intent
        structure_mapping = {
            IntentType.CODE_GENERATION: "technical",
            IntentType.PLANNING_REQUEST: "structured",
            IntentType.ANALYSIS: "analytical",
            IntentType.CREATIVE_WRITING: "narrative",
            IntentType.QUICK_ANSWER: "conversational"
        }
        structure = structure_mapping.get(intent_type, "conversational")

        return detail_level, structure


            
    def _analyze_with_patterns(self, user_message: str, conversation_context: List[Dict[str, Any]] = None) -> UserIntent:
        """Legacy pattern-based analysis."""
        try:
            message_lower = user_message.lower()
            conversation_context = conversation_context or []
            
            # Get pattern scores
            intent_scores = self._score_intents(message_lower, conversation_context)
            
            # Determine primary intent
            primary_intent = max(intent_scores, key=intent_scores.get)
            primary_score = intent_scores[primary_intent]
            
            # Determine expected detail level and structure
            detail_level = self._determine_detail_level(user_message, conversation_context)
            expected_structure = self._determine_structure(user_message, primary_intent)
            
            # Calculate confidence score
            confidence_score = min(primary_score, 1.0)
            
            # Determine task complexity
            task_complexity = self._assess_complexity(user_message, conversation_context)
            
            # Extract context requirements
            context_requirements = self._get_context_requirements(primary_intent)
            
            return UserIntent(
                intent_type=primary_intent,
                expected_detail_level=detail_level,
                expected_structure=expected_structure,
                confidence_score=confidence_score,
                context_requirements=context_requirements,
                task_complexity=task_complexity
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze intent with patterns: {e}")
            # Return default intent
            return UserIntent(
                intent_type=IntentType.QUICK_ANSWER,
                expected_detail_level="adaptive",
                expected_structure="conversational",
                confidence_score=0.5,
                context_requirements=[],
                task_complexity="simple"
            )
            
    def _score_intents(self, message_lower: str, conversation_context: List[Dict[str, Any]]) -> Dict[IntentType, float]:
        """Score each intent type based on message content."""
        scores = {}
        
        for pattern in self._intent_patterns:
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in pattern.keywords if keyword in message_lower)
            keyword_score = (keyword_matches / len(pattern.keywords)) * 0.6
            score += keyword_score
            
            # Pattern matching
            pattern_matches = sum(1 for regex_pattern in pattern.patterns 
                                if re.search(regex_pattern, message_lower))
            if pattern_matches > 0:
                pattern_score = (pattern_matches / len(pattern.patterns)) * 0.4
                score += pattern_score
                
            # Context bonus (from recent conversation)
            context_bonus = self._calculate_context_bonus(pattern, conversation_context)
            score += context_bonus
            
            # Apply confidence boost
            score += pattern.confidence_boost
            
            scores[pattern.intent_type] = min(score, 1.0)
            
        return scores
        
    def _calculate_context_bonus(self, pattern: IntentPattern, conversation_context: List[Dict[str, Any]]) -> float:
        """Calculate bonus score based on conversation context."""
        if not conversation_context:
            return 0.0
            
        bonus = 0.0
        recent_messages = conversation_context[-3:]  # Look at last 3 messages
        
        for msg in recent_messages:
            content_lower = msg.get('content', '').lower()
            
            # Check for context that supports this intent
            for keyword in pattern.keywords:
                if keyword in content_lower:
                    bonus += 0.1
                    
        return min(bonus, 0.3)  # Cap context bonus
        
    def _determine_detail_level(self, user_message: str, conversation_context: List[Dict[str, Any]]) -> str:
        """Determine expected detail level from message."""
        message_lower = user_message.lower()
        
        # Explicit detail indicators
        if any(word in message_lower for word in ["brief", "short", "concise", "quick"]):
            return "brief"
        elif any(word in message_lower for word in ["detailed", "thorough", "comprehensive", "deep"]):
            return "comprehensive"
        elif any(word in message_lower for word in ["simple", "basic", "beginner"]):
            return "beginner"
        elif any(word in message_lower for word in ["advanced", "complex", "expert"]):
            return "advanced"
        else:
            return "adaptive"
            
    def _determine_structure(self, user_message: str, intent_type: IntentType) -> str:
        """Determine expected response structure."""
        message_lower = user_message.lower()
        
        # Structure preferences from message
        if any(word in message_lower for word in ["bullet", "list", "points", "steps"]):
            return "bullet_points"
        elif any(word in message_lower for word in ["formal", "report", "document"]):
            return "formal"
        elif any(word in message_lower for word in ["conversational", "chat", "talk"]):
            return "conversational"
        else:
            # Default structure based on intent
            structure_mapping = {
                IntentType.CODE_GENERATION: "technical",
                IntentType.PLANNING_REQUEST: "structured",
                IntentType.CREATIVE_WRITING: "narrative",
                IntentType.ANALYSIS: "analytical"
            }
            return structure_mapping.get(intent_type, "conversational")
            
    def _assess_complexity(self, user_message: str, conversation_context: List[Dict[str, Any]]) -> str:
        """Assess task complexity."""
        message_lower = user_message.lower()
        
        # Complexity indicators
        complexity_indicators = {
            "simple": ["quick", "brief", "what", "how", "when", "where"],
            "medium": ["explain", "analyze", "compare", "describe", "implement"],
            "complex": ["design", "architecture", "strategy", "comprehensive", "multi-step", "system"]
        }
        
        scores = {"simple": 0, "medium": 0, "complex": 0}
        
        for complexity, indicators in complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in message_lower)
            scores[complexity] = score
            
        # Get highest scoring complexity
        max_score = max(scores.values())
        if max_score == 0:
            return "medium"  # Default to medium
        else:
            return max(scores, key=scores.get)
            
    def _get_context_requirements(self, intent_type: IntentType) -> List[str]:
        """Get context requirements for intent type."""
        requirements_mapping = {
            IntentType.OPINION_REQUEST: ["personal_preference", "user_context"],
            IntentType.CODE_GENERATION: ["programming_context", "technical_requirements"],
            IntentType.ANALYSIS: ["data_context", "analytical_framework"],
            IntentType.PLANNING_REQUEST: ["project_context", "constraints"],
            IntentType.TROUBLESHOOTING: ["error_context", "system_state"],
            IntentType.EXPLAINATION: ["explanation_level", "background_knowledge"],
            IntentType.CREATIVE_WRITING: ["creative_style", "content_guidelines"],
            IntentType.QUICK_ANSWER: ["minimal_context"]
        }
        return requirements_mapping.get(intent_type, [])
        
    def detect_multi_intent(self, user_message: str) -> List[UserIntent]:
        """
        Detect multiple intents in a single message.
        
        Args:
            user_message: User's message text
            
        Returns:
            List of UserIntent objects (sorted by confidence)
        """
        try:
            # Split message into sentences or clauses
            sentences = re.split(r'[.!?;]', user_message)
            intents = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Ignore very short fragments
                    intent = self.analyze_intent(sentence)
                    if intent.confidence_score > 0.3:  # Minimum confidence threshold
                        intents.append(intent)
                        
            # Sort by confidence score
            intents.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return intents
            
        except Exception as e:
            logger.error(f"Failed to detect multi-intent: {e}")
            return [self.analyze_intent(user_message)]
            
    def get_recommended_agent_mode(self, user_intent: UserIntent, conversation_context: List[Dict[str, Any]] = None) -> AgentMode:
        """
        Get recommended agent mode based on intent and context.
        
        Args:
            user_intent: Classified user intent
            conversation_context: Recent conversation context
            
        Returns:
            Recommended AgentMode
        """
        # Base mapping from intent to mode
        intent_to_mode = {
            IntentType.CODE_GENERATION: AgentMode.CODING,
            IntentType.PLANNING_REQUEST: AgentMode.PLANNING,
            IntentType.ANALYSIS: AgentMode.DEEP_THINK,
            IntentType.TROUBLESHOOTING: AgentMode.PROBLEM_SOLVING,
            IntentType.CREATIVE_WRITING: AgentMode.DRAFTING,
            IntentType.EXPLAINATION: AgentMode.TUTORING,
        }
        
        # Get base mode from intent
        base_mode = intent_to_mode.get(user_intent.intent_type, AgentMode.CASUAL_CONVERSATION)
        
        # Context-based mode adjustments
        conversation_context = conversation_context or []
        
        # Check for project-related keywords
        project_keywords = ["project", "task", "work", "develop", "build", "implement"]
        recent_content = " ".join([msg.get('content', '') for msg in conversation_context[-2:]]).lower()
        
        if any(keyword in recent_content for keyword in project_keywords):
            return AgentMode.PROJECT_WORK
            
        # Check for research indicators
        research_keywords = ["research", "search", "find information", "investigate"]
        if any(keyword in user_message.lower() for keyword in research_keywords):
            return AgentMode.RESEARCH
            
        # Check for code-related indicators (even in non-code intents)
        code_keywords = ["code", "function", "class", "debug", "implement"]
        if any(keyword in user_intent.context_requirements for keyword in ["technical_requirements", "programming_context"]):
            return AgentMode.CODING
            
        return base_mode
        
    def get_intent_suggestions(self, user_message: str) -> List[Tuple[IntentType, float]]:
        """
        Get intent suggestions with confidence scores.
        
        Args:
            user_message: User's message text
            
        Returns:
            List of (IntentType, confidence_score) tuples
        """
        try:
            # Fallback to pattern matching for suggestions to avoid LLM overhead
            message_lower = user_message.lower()
            intent_scores = self._score_intents(message_lower, [])
            
            # Convert to list and sort by score
            suggestions = [(intent_type, score) for intent_type, score in intent_scores.items()]
            suggestions.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 3 suggestions above threshold
            return [(intent, score) for intent, score in suggestions if score > 0.2][:3]
            
        except Exception as e:
            logger.error(f"Failed to get intent suggestions: {e}")
            return [(IntentType.QUICK_ANSWER, 0.5)]
            
    def learn_from_feedback(self, user_message: str, user_intent: UserIntent, actual_intent: IntentType):
        """
        Learn from user feedback to improve intent classification.
        
        Args:
            user_message: Original user message
            user_intent: Classified intent
            actual_intent: Actual intent (from user feedback)
        """
        try:
            if user_intent.intent_type != actual_intent:
                logger.info(f"Intent classification correction: {user_intent.intent_type.value} -> {actual_intent.value}")
                
                # Here we could implement learning mechanisms:
                # - Add new patterns
                # - Adjust confidence weights
                # - Update keyword associations
                
                # For now, just log the correction for manual analysis
                # In a production system, this could feed into a machine learning model
                pass
                
        except Exception as e:
            logger.error(f"Failed to learn from feedback: {e}")
            


    def get_intent_statistics(self) -> Dict[str, Any]:
        """Get statistics about intent analyzer usage."""
        # This would track usage patterns in a real implementation
        return {
            "total_patterns": len(self._intent_patterns),
            "supported_intents": len(set(pattern.intent_type for pattern in self._intent_patterns)),
            "analysis_methods": ["llm_analysis" if self.llm_adapter else "pattern_matching", "context_analysis", "topic_analysis"]
        }
