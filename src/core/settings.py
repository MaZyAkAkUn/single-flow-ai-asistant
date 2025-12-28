"""
Plain settings management for the Personal Assistant.
Handles storage of API keys and configuration in plain JSON format.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

from .logging_config import get_logger

logger = get_logger(__name__)


class PlainSettings:
    """
    Manages plain storage of application settings.
    Stores API keys and configuration in plain JSON format.
    """

    def __init__(self, settings_dir: str = "./data"):
        """
        Initialize plain settings manager.

        Args:
            settings_dir: Directory to store settings files
        """
        self.settings_dir = Path(settings_dir)
        self.settings_dir.mkdir(parents=True, exist_ok=True)

        self.settings_file = self.settings_dir / "settings.json"

        # Default settings
        self._defaults = {
            'api_keys': {
                'openrouter': '',
                'openai': '',
                'tools': {
                    'tavily': '',
                    'exa': '',
                    'jina': ''
                }
            },
            'combos': {
                'default_llm': {
                    'name': 'Default LLM',
                    'provider': 'openrouter',
                    'model': 'openai/gpt-4o-mini'
                },
                'default_intent': {
                    'name': 'Default Intent Analysis',
                    'provider': 'openrouter',
                    'model': 'openai/gpt-4o-mini'
                },
                'default_embeddings': {
                    'name': 'Default Embeddings',
                    'provider': 'openrouter',
                    'model': 'text-embedding-3-small'
                }
            },
            'llm': {
                'combo': 'default_llm',
                'temperature': 0.4
            },
            'intent_analysis': {
                'enabled': False,
                'combo': 'default_intent'
            },
            'embeddings': {
                'combo': 'default_embeddings'
            },
            'retrieval': {
                'enabled': True,
                'max_docs': 3,
                'method': 'semantic',
                'semantic_weight': 0.7,
                'keyword_weight': 0.3
            },
            'vector_store': {
                'provider': 'faiss',
                'path': './data/vector_store'
            },
            'memory': {
                'enabled': False,
                'max_memories': 3
            },
            'audio': {
                'tts_enabled': True,
                'asr_enabled': True,
                'tts_voice': 'alloy',
                'tts_speed': 1.0,
                'asr_language': 'en-US'
            },
            'app': {
                'max_history': 50,
                'theme': 'light'
            },
            'window': {
                'width': 1400,
                'height': 800,
                'maximized': True,
                'fullscreen': False
            },
            'system_info': {
                'user_location': '',
                'main_spoken_language': ''
            },
            'tracing': {
                'enabled': False,
                'provider': 'langsmith',
                'api_key': '',
                'project': 'default',
                'endpoint': 'https://api.smith.langchain.com'
            }
        }

    def load_settings(self) -> Dict[str, Any]:
        """
        Load settings from plain JSON file.

        Returns:
            Settings dictionary
        """
        if not self.settings_file.exists():
            logger.info("No settings file found, returning defaults")
            return self._defaults.copy()

        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)

            # Merge with defaults for any missing keys
            merged_settings = self._defaults.copy()
            self._deep_update(merged_settings, settings)

            logger.info("Settings loaded successfully")
            return merged_settings

        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return self._defaults.copy()

    def save_settings(self, settings: Dict[str, Any]):
        """
        Save settings to plain JSON file.

        Args:
            settings: Settings dictionary to save
        """
        try:
            # Validate settings structure
            validated_settings = self._validate_settings(settings)

            # Save as plain JSON
            with open(self.settings_file, 'w') as f:
                json.dump(validated_settings, f, indent=2)

            logger.info("Settings saved successfully")

        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            raise RuntimeError(f"Failed to save settings: {e}")

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a provider.

        Args:
            provider: Provider name (e.g., 'openrouter', 'openai')

        Returns:
            API key or None if not found
        """
        settings = self.load_settings()
        api_keys = settings.get('api_keys', {})
        return api_keys.get(provider)

    def set_api_key(self, provider: str, api_key: str):
        """
        Set API key for a provider.

        Args:
            provider: Provider name
            api_key: API key to store
        """
        settings = self.load_settings()
        if 'api_keys' not in settings:
            settings['api_keys'] = {}
        settings['api_keys'][provider] = api_key
        self.save_settings(settings)

    def delete_api_key(self, provider: str):
        """
        Delete API key for a provider.

        Args:
            provider: Provider name
        """
        settings = self.load_settings()
        api_keys = settings.get('api_keys', {})
        if provider in api_keys:
            del api_keys[provider]
            self.save_settings(settings)

    def get_tool_api_key(self, tool: str) -> Optional[str]:
        """
        Get API key for a tool.

        Args:
            tool: Tool name (e.g., 'tavily', 'exa', 'jina')

        Returns:
            API key or None if not found
        """
        settings = self.load_settings()
        api_keys = settings.get('api_keys', {})
        tools = api_keys.get('tools', {})
        return tools.get(tool)

    def set_tool_api_key(self, tool: str, api_key: str):
        """
        Set API key for a tool.

        Args:
            tool: Tool name
            api_key: API key to store
        """
        settings = self.load_settings()
        if 'api_keys' not in settings:
            settings['api_keys'] = {}
        if 'tools' not in settings['api_keys']:
            settings['api_keys']['tools'] = {}
        settings['api_keys']['tools'][tool] = api_key
        self.save_settings(settings)

    def get_all_tool_api_keys(self) -> Dict[str, str]:
        """
        Get all tool API keys.

        Returns:
            Dictionary of tool API keys
        """
        settings = self.load_settings()
        api_keys = settings.get('api_keys', {})
        tools = api_keys.get('tools', {})
        return tools.copy()

    def get_setting(self, *keys: str) -> Any:
        """
        Get a nested setting value.

        Args:
            *keys: Key path (e.g., 'llm', 'model')

        Returns:
            Setting value or None if not found
        """
        settings = self.load_settings()
        current = settings

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def set_setting(self, value: Any, *keys: str):
        """
        Set a nested setting value.

        Args:
            value: Value to set
            *keys: Key path
        """
        settings = self.load_settings()
        current = settings

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value
        current[keys[-1]] = value
        self.save_settings(settings)

    def _deep_update(self, base: Dict[str, Any], update: Dict[str, Any]):
        """
        Deep update a dictionary.

        Args:
            base: Base dictionary to update
            update: Dictionary with updates
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def _validate_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate settings structure.

        Args:
            settings: Settings to validate

        Returns:
            Validated settings
        """
        # For now, just ensure it's a dictionary
        if not isinstance(settings, dict):
            raise ValueError("Settings must be a dictionary")

        return settings

    def export_settings(self, file_path: str):
        """
        Export settings to a plain JSON file (for backup/debugging).

        Args:
            file_path: Path to export file
        """
        settings = self.load_settings()
        with open(file_path, 'w') as f:
            json.dump(settings, f, indent=2)
        logger.info(f"Settings exported to {file_path}")

    def import_settings(self, file_path: str):
        """
        Import settings from a plain JSON file.

        Args:
            file_path: Path to import file
        """
        with open(file_path, 'r') as f:
            settings = json.load(f)
        self.save_settings(settings)
        logger.info(f"Settings imported from {file_path}")

    def reset_to_defaults(self):
        """Reset settings to defaults."""
        self.save_settings(self._defaults.copy())
        logger.info("Settings reset to defaults")

    # Combo management methods

    def get_combos(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all combos.

        Returns:
            Dictionary of combo_id -> combo data
        """
        settings = self.load_settings()
        return settings.get('combos', {}).copy()

    def get_combo(self, combo_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific combo.

        Args:
            combo_id: Combo ID

        Returns:
            Combo data or None if not found
        """
        combos = self.get_combos()
        return combos.get(combo_id)

    def create_combo(self, combo_id: str, name: str, provider: str, model: str):
        """
        Create a new combo.

        Args:
            combo_id: Unique combo identifier
            name: Display name for the combo
            provider: LLM provider
            model: Model name
        """
        settings = self.load_settings()
        if 'combos' not in settings:
            settings['combos'] = {}

        if combo_id in settings['combos']:
            raise ValueError(f"Combo with ID '{combo_id}' already exists")

        settings['combos'][combo_id] = {
            'name': name,
            'provider': provider,
            'model': model
        }

        self.save_settings(settings)
        logger.info(f"Created combo: {combo_id} ({name})")

    def update_combo(self, combo_id: str, name: Optional[str] = None,
                    provider: Optional[str] = None, model: Optional[str] = None):
        """
        Update an existing combo.

        Args:
            combo_id: Combo ID to update
            name: New display name (optional)
            provider: New provider (optional)
            model: New model (optional)
        """
        settings = self.load_settings()
        combos = settings.get('combos', {})

        if combo_id not in combos:
            raise ValueError(f"Combo with ID '{combo_id}' not found")

        combo = combos[combo_id]

        if name is not None:
            combo['name'] = name
        if provider is not None:
            combo['provider'] = provider
        if model is not None:
            combo['model'] = model

        self.save_settings(settings)
        logger.info(f"Updated combo: {combo_id}")

    def delete_combo(self, combo_id: str):
        """
        Delete a combo.

        Args:
            combo_id: Combo ID to delete
        """
        settings = self.load_settings()
        combos = settings.get('combos', {})

        if combo_id not in combos:
            raise ValueError(f"Combo with ID '{combo_id}' not found")

        # Check if combo is being used by any use case
        use_cases = ['llm', 'intent_analysis', 'embeddings']
        for use_case in use_cases:
            use_case_settings = settings.get(use_case, {})
            if use_case_settings.get('combo') == combo_id:
                raise ValueError(f"Cannot delete combo '{combo_id}' as it is being used by {use_case}")

        del combos[combo_id]
        self.save_settings(settings)
        logger.info(f"Deleted combo: {combo_id}")

    def resolve_combo(self, combo_id: str) -> Optional[Dict[str, Any]]:
        """
        Resolve a combo ID to its provider and model.

        Args:
            combo_id: Combo ID

        Returns:
            Dict with 'provider' and 'model' keys, or None if not found
        """
        combo = self.get_combo(combo_id)
        if combo:
            return {
                'provider': combo['provider'],
                'model': combo['model']
            }
        return None

    def migrate_legacy_settings(self):
        """
        Migrate legacy provider/model settings to use combos.
        This should be called once when upgrading from old settings format.
        """
        settings = self.load_settings()

        # Check if migration is needed
        needs_migration = False
        if 'llm' in settings and 'provider' in settings['llm']:
            needs_migration = True
        elif 'intent_analysis' in settings and 'provider' in settings['intent_analysis']:
            needs_migration = True
        elif 'embeddings' in settings and 'provider' in settings['embeddings']:
            needs_migration = True

        if not needs_migration:
            logger.info("No migration needed - settings already use combos")
            return

        logger.info("Migrating legacy settings to combo format...")

        # Create combos from existing settings
        combos = settings.get('combos', {})

        # LLM combo
        if 'llm' in settings and 'provider' in settings['llm']:
            llm_combo_id = 'migrated_llm'
            combos[llm_combo_id] = {
                'name': 'Migrated LLM',
                'provider': settings['llm']['provider'],
                'model': settings['llm']['model']
            }
            # Update llm settings to use combo
            settings['llm'] = {
                'combo': llm_combo_id,
                'temperature': settings['llm'].get('temperature', 0.4)
            }

        # Intent analysis combo
        if 'intent_analysis' in settings and 'provider' in settings['intent_analysis']:
            intent_combo_id = 'migrated_intent'
            combos[intent_combo_id] = {
                'name': 'Migrated Intent Analysis',
                'provider': settings['intent_analysis']['provider'],
                'model': settings['intent_analysis']['model']
            }
            # Update intent_analysis settings to use combo
            settings['intent_analysis'] = {
                'enabled': settings['intent_analysis'].get('enabled', False),
                'combo': intent_combo_id
            }

        # Embeddings combo
        if 'embeddings' in settings and 'provider' in settings['embeddings']:
            emb_combo_id = 'migrated_embeddings'
            combos[emb_combo_id] = {
                'name': 'Migrated Embeddings',
                'provider': settings['embeddings']['provider'],
                'model': settings['embeddings']['model']
            }
            # Update embeddings settings to use combo
            settings['embeddings'] = {
                'combo': emb_combo_id
            }

        # Save migrated settings
        settings['combos'] = combos
        self.save_settings(settings)
        logger.info("Migration completed successfully")
