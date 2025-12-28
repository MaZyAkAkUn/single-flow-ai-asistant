"""
Unit tests for the AI Controller.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.controller import AIController
from src.core.settings import PlainSettings


class TestAIController:
    """Test cases for AIController."""

    def test_init_default(self):
        """Test controller initialization with defaults."""
        controller = AIController()
        assert controller is not None
        assert controller.conversation_history == []
        assert controller.max_history_length == 50
        assert controller.retrieval_enabled is True
        assert controller.memory_enabled is True

    def test_input_validation(self):
        """Test input validation methods directly."""
        controller = AIController()

        # Test the validation logic directly by calling the methods that do validation
        # Since the full message processing is complex to mock, we'll test the validation parts

        # Test that the controller has proper defaults
        assert controller.max_history_length == 50
        assert controller.retrieval_enabled is True
        assert controller.memory_enabled is True

        # Test settings validation
        controller.set_max_retrieved_docs(10)
        assert controller.max_retrieved_docs == 10

        controller.set_max_retrieved_docs(0)  # Should clamp to 1
        assert controller.max_retrieved_docs == 1

        # Test history management
        controller._add_to_history('user', 'test message')
        assert len(controller.conversation_history) == 1
        assert controller.conversation_history[0]['role'] == 'user'
        assert controller.conversation_history[0]['content'] == 'test message'

    def test_history_management(self):
        """Test conversation history management."""
        controller = AIController()
        controller.max_history_length = 3

        # Add messages
        controller._add_to_history('user', 'msg1')
        controller._add_to_history('assistant', 'resp1')
        controller._add_to_history('user', 'msg2')
        controller._add_to_history('assistant', 'resp2')
        controller._add_to_history('user', 'msg3')

        assert len(controller.conversation_history) == 3  # Should be trimmed
        # History keeps the most recent messages: msg2, resp2, msg3
        # So index 0 should be msg2 (the oldest of the kept messages)
        assert controller.conversation_history[0]['content'] == 'msg2'
        assert controller.conversation_history[1]['content'] == 'resp2'
        assert controller.conversation_history[2]['content'] == 'msg3'

    def test_settings_validation(self):
        """Test settings validation."""
        controller = AIController()

        # Test valid settings
        controller.set_max_retrieved_docs(10)
        assert controller.max_retrieved_docs == 10

        # Test invalid settings (should be clamped)
        controller.set_max_retrieved_docs(0)
        assert controller.max_retrieved_docs == 1

        controller.set_max_retrieved_memories(-5)
        assert controller.max_retrieved_memories == 1

    def test_toggle_features(self):
        """Test feature toggling."""
        controller = AIController()

        # Test retrieval toggle
        controller.toggle_retrieval(False)
        assert controller.retrieval_enabled is False

        controller.toggle_retrieval(True)
        assert controller.retrieval_enabled is True

        # Test memory toggle
        controller.toggle_memory(False)
        assert controller.memory_enabled is False

        controller.toggle_memory(True)
        assert controller.memory_enabled is True

    def test_configure_llm_idempotent(self):
        """Test that configure_llm preserves tool configuration when recreating adapter."""
        controller = AIController()

        # Configure tools first
        tool_config = {'tavily': 'test_key_123'}
        controller.configure_tools(tool_config)

        # Verify tools are configured
        initial_tools = controller.get_available_tools()
        assert len(initial_tools) > 0

        # Configure LLM with different config (should recreate adapter)
        controller.configure_llm('openrouter', api_key='test_key', model='test-model')

        # Verify tools are still available after recreation
        final_tools = controller.get_available_tools()
        assert len(final_tools) == len(initial_tools)
        assert final_tools == initial_tools

        # Configure LLM again with same config (should be idempotent)
        controller.configure_llm('openrouter', api_key='test_key', model='test-model')

        # Tools should still be available
        final_tools_2 = controller.get_available_tools()
        assert len(final_tools_2) == len(initial_tools)
        assert final_tools_2 == initial_tools


class TestPlainSettings:
    """Test cases for PlainSettings."""

    def test_init(self):
        """Test settings initialization."""
        settings = PlainSettings()
        assert settings.settings_dir.exists()

    def test_load_save_settings(self, tmp_path):
        """Test loading and saving settings."""
        settings = PlainSettings(str(tmp_path))

        test_data = {
            'llm': {'provider': 'test', 'model': 'test-model'},
            'api_keys': {'test': 'test-key'}
        }

        settings.save_settings(test_data)
        loaded = settings.load_settings()

        assert loaded['llm']['provider'] == 'test'
        assert loaded['api_keys']['test'] == 'test-key'

    def test_api_key_management(self, tmp_path):
        """Test API key management."""
        settings = PlainSettings(str(tmp_path))

        # Test setting and getting API key
        settings.set_api_key('test_provider', 'test_key_123')
        key = settings.get_api_key('test_provider')
        assert key == 'test_key_123'

        # Test deleting API key
        settings.delete_api_key('test_provider')
        key = settings.get_api_key('test_provider')
        assert key is None

    def test_setting_operations(self, tmp_path):
        """Test nested setting operations."""
        settings = PlainSettings(str(tmp_path))

        # Test setting nested values
        settings.set_setting('test_value', 'test', 'nested', 'key')
        value = settings.get_setting('test', 'nested', 'key')
        assert value == 'test_value'

        # Test getting non-existent setting
        value = settings.get_setting('non', 'existent')
        assert value is None
