"""
DeepAgent integration using LangGraph for complex task orchestration.
"""
import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from .enhanced_llm_adapter import EnhancedLLMAdapter
from ..data.schemas import AgentState, AgentMode, ProjectContext, UserIntent
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class DeepAgentState(TypedDict):
    """State definition for DeepAgent graph."""
    messages: Annotated[List[BaseMessage], operator.add]
    user_intent: Optional[UserIntent]
    agent_state: Optional[AgentState]
    project_context: Optional[ProjectContext]
    plan: List[str]
    current_step: int
    final_response: str
    metadata: Dict[str, Any]


class DeepAgentIntegration:
    """
    Integrates DeepAgent architecture with structured prompt system.
    Uses LangGraph for orchestration and EnhancedLLMAdapter for generation.
    """
    
    def __init__(self, llm_adapter: EnhancedLLMAdapter):
        """
        Initialize DeepAgent integration.
        
        Args:
            llm_adapter: Configured EnhancedLLMAdapter
        """
        self.llm_adapter = llm_adapter
        self.graph = self._build_graph()
        logger.info("DeepAgent integration initialized")
        
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph orchestration graph.
        
        Returns:
            Compiled StateGraph
        """
        graph = StateGraph(DeepAgentState)
        
        # Define nodes
        graph.add_node("analyze", self._analyze_intent_node)
        graph.add_node("plan", self._planner_node)
        graph.add_node("execute", self._executor_node)
        graph.add_node("review", self._reviewer_node)
        
        # Define edges
        graph.set_entry_point("analyze")
        
        graph.add_edge("analyze", "plan")
        graph.add_edge("plan", "execute")
        # Conditional edge: if execution needs more steps or review
        graph.add_conditional_edges(
            "execute",
            self._check_execution_status,
            {
                "continue": "execute",
                "review": "review",
                "end": END
            }
        )
        graph.add_edge("review", END)
        
        return graph.compile()
        
    async def process_request(
        self,
        user_message: str,
        project_context: Optional[ProjectContext] = None,
        agent_state: Optional[AgentState] = None
    ) -> Dict[str, Any]:
        """
        Process complex request using DeepAgent workflow.
        
        Args:
            user_message: User input
            project_context: Active project context
            agent_state: Current agent state
            
        Returns:
            Processing result
        """
        try:
            initial_state = DeepAgentState(
                messages=[HumanMessage(content=user_message)],
                user_intent=None,
                agent_state=agent_state,
                project_context=project_context,
                plan=[],
                current_step=0,
                final_response="",
                metadata={}
            )
            
            # Execute graph
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                "response": final_state.get("final_response", ""),
                "plan": final_state.get("plan", []),
                "metadata": final_state.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"DeepAgent processing failed: {e}")
            return {
                "response": f"I encountered an error processing your request: {e}",
                "error": str(e)
            }
            
    async def _analyze_intent_node(self, state: DeepAgentState) -> Dict[str, Any]:
        """Node: Analyze user intent."""
        messages = state.get("messages", [])
        last_message = messages[-1].content if messages else ""
        
        # Prepare conversation history
        conversation_history = []
        for msg in messages[:-1]:  # Exclude last message
            role = "user" if isinstance(msg, HumanMessage) else "assistant" if isinstance(msg, AIMessage) else "system"
            conversation_history.append({"role": role, "content": msg.content})
        
        # Use existing intent analyzer
        user_intent = self.llm_adapter.intent_analyzer.analyze_intent(
            last_message, 
            conversation_context=conversation_history
        )
        
        return {
            "user_intent": user_intent,
            "metadata": {"intent_confidence": user_intent.confidence_score}
        }
        
    async def _planner_node(self, state: DeepAgentState) -> Dict[str, Any]:
        """Node: Create execution plan."""
        user_intent = state.get("user_intent")
        messages = state.get("messages", [])
        last_message = messages[-1].content
        
        # Only plan if complex task
        if user_intent and user_intent.task_complexity in ["medium", "complex"]:
            prompt = f"""
            Task: {last_message}
            Intent: {user_intent.intent_type.value}
            
            Create a step-by-step plan to accomplish this task. 
            Return ONLY the plan steps as a list, one per line.
            """
            
            response = await self.llm_adapter.agenerate_response(prompt)
            plan_steps = [idx.strip() for idx in response.split('\n') if idx.strip()]
            
            return {"plan": plan_steps}
        
        return {"plan": ["Execute task directly"]}
        
    async def _executor_node(self, state: DeepAgentState) -> Dict[str, Any]:
        """Node: Execute plan steps."""
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        messages = state.get("messages", [])
        
        if current_step < len(plan):
            step_description = plan[current_step]
            logger.info(f"Executing step {current_step + 1}: {step_description}")
            
            # Use EnhancedLLMAdapter to generate structured response/action for this step
            response, _, _ = await self.llm_adapter.aprocess_message_structured(
                user_message=f"Execute step: {step_description}",
                conversation_history=[{"role": "user", "content": msg.content} for msg in messages],
                project_contexts=[state["project_context"]] if state.get("project_context") else [],
                agent_state=state.get("agent_state")
            )
            
            # Extract actual content from XML prompt if needed (mocking execution here)
            # In real scenario, this would call tools via llm_adapter
            
            # Store result
            
            return {
                "current_step": current_step + 1,
                "final_response": response  # Accumulate or replace based on logic
            }
            
        return {}
        
    async def _reviewer_node(self, state: DeepAgentState) -> Dict[str, Any]:
        """Node: Review final output."""
        # Optional review step
        return {}
        
    def _check_execution_status(self, state: DeepAgentState) -> str:
        """Determine next graph step."""
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        
        if current_step < len(plan):
            return "continue"
        return "review"  # or "end"
