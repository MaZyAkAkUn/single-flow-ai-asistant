"""
Test script for DeepAgent integration and structured prompt flows.
"""
import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.langchain_adapters.enhanced_llm_adapter import EnhancedLLMAdapter
from src.langchain_adapters.deep_agent_integration import DeepAgentIntegration
from src.data.schemas import AgentState, AgentMode, ProjectContext, UserIntent, IntentType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_deep_agent_flow():
    logger.info("Starting DeepAgent integration test...")
    
    # 1. Mock LLM Adapter
    llm_adapter = MagicMock(spec=EnhancedLLMAdapter)
    llm_adapter.intent_analyzer = MagicMock()
    
    # Mock intent analysis
    mock_intent = UserIntent(
        intent_type=IntentType.PLANNING_REQUEST,
        confidence_score=0.9,
        task_complexity="medium",
        expected_detail_level="comprehensive",
        expected_structure="structured"
    )
    llm_adapter.intent_analyzer.analyze_intent.return_value = mock_intent
    
    # Mock response generation
    llm_adapter.agenerate_response = AsyncMock(return_value="1. Step 1: Analyze\n2. Step 2: Execute\n3. Step 3: Review")
    llm_adapter.aprocess_message_structured = AsyncMock(return_value=("structured_prompt", "Step execution output", {}))
    
    # 2. Initialize DeepAgent
    deep_agent = DeepAgentIntegration(llm_adapter)
    
    # 3. Test processing a complex request
    user_message = "Create a detailed project plan for a new Python web app."
    project_context = ProjectContext(
        project_id="test_p1", 
        project_name="Test Project",
        project_description="A test project"
    )
    agent_state = AgentState(current_mode=AgentMode.PROJECT_WORK)
    
    logger.info(f"Processing request: {user_message}")
    
    result = await deep_agent.process_request(
        user_message=user_message,
        project_context=project_context,
        agent_state=agent_state
    )
    
    # 4. Verify results
    logger.info("Result received")
    logger.info(f"Plan: {result['plan']}")
    logger.info(f"Final Response: {result['response']}")
    
    if result['plan'] and len(result['plan']) == 3:
        logger.info("✅ Plan generation successful")
    else:
        logger.error("❌ Plan generation failed")
        
    if result['response'] == "Step execution output":
        logger.info("✅ Execution successful")
    else:
        # Note: The mock returns the last step output as final response in this simple test
        logger.info(f"Execution output: {result['response']}")

if __name__ == "__main__":
    asyncio.run(test_deep_agent_flow())
