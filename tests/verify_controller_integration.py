"""
Verify AIController integration with EnhancedLLMAdapter.
"""
import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.controller import AIController
from src.langchain_adapters.enhanced_llm_adapter import EnhancedLLMAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_controller_flow():
    logger.info("Starting Controller integration test...")
    
    # 1. Initialize Controller
    # We'll let it create its default adapter, then replace it with a mock for verification
    controller = AIController()
    
    # Verify it created an EnhancedLLMAdapter
    if isinstance(controller.llm_adapter, EnhancedLLMAdapter):
        logger.info("✅ AIController initialized with EnhancedLLMAdapter")
    else:
        logger.error(f"❌ AIController initialized with {type(controller.llm_adapter)}")
        return

    # 2. Mock the adapter to verify calls
    mock_adapter = MagicMock(spec=EnhancedLLMAdapter)
    
    # Mock async methods
    mock_adapter.aprocess_message_structured = AsyncMock(
        return_value=(
            "<structured_prompt>...</structured_prompt>", 
            "Human readable prompt...", 
            {"metadata": "test"}
        )
    )
    mock_adapter.agenerate_response = AsyncMock(
        return_value="Response from structured flow"
    )
    # Mock get_tools to return empty list (no tool use for this test)
    mock_adapter.get_tools.return_value = []
    
    # Replace the real adapter with mock
    controller.llm_adapter = mock_adapter
    
    # 3. Process a message
    user_message = "Hello, I need help with a project."
    logger.info(f"Processing message: {user_message}")
    
    response = await controller.process_message(user_message)
    
    # 4. Verify interaction
    logger.info(f"Response received: {response}")
    
    if response == "Response from structured flow":
        logger.info("✅ Controller used the structured flow response")
    else:
        logger.error(f"❌ Unexpected response: {response}")
        
    # Verify aprocess_message_structured was called
    if mock_adapter.aprocess_message_structured.called:
        logger.info("✅ aprocess_message_structured was called")
        # Check arguments
        call_args = mock_adapter.aprocess_message_structured.call_args
        if call_args is not None:
            kwargs = call_args.kwargs
            if kwargs.get('user_message') == user_message:
                logger.info("✅ user_message passed correctly")
            else:
                logger.error("❌ user_message mismatch")
                
            if 'project_contexts' in kwargs:
                logger.info("✅ project_contexts passed")
            else:
                logger.error("❌ project_contexts missing")
    else:
        logger.error("❌ aprocess_message_structured was NOT called")

if __name__ == "__main__":
    asyncio.run(test_controller_flow())
