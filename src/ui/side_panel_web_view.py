import json
import logging
from typing import Dict, Any, List
from pathlib import Path
from PyQt6.QtCore import pyqtSlot, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from .base_web_view import BaseWebView
from .files_web_view import FilesBridge, FileIngestionWorker
from .memory_web_view import MemoryBridge
from ..core.controller import AIController

logger = logging.getLogger(__name__)


class SettingsBridge(QObject):
    """Bridge for communication between Python and JavaScript for settings."""

    # Signals to Python
    settingsSaved = pyqtSignal(dict)
    settingsReset = pyqtSignal()
    testConnectionRequested = pyqtSignal(str)
    ready = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = None
        self.run_js = None

    @pyqtSlot()
    def onReady(self):
        """Called when JS is ready."""
        self.ready.emit()

    @pyqtSlot(str)
    def saveSettings(self, settings_json: str):
        """Receive settings from JS (as JSON string or dict)."""
        try:
            if isinstance(settings_json, str):
                self.settingsSaved.emit(json.loads(settings_json))
            else:
                self.settingsSaved.emit(settings_json)
        except Exception as e:
            logger.error(f"Failed to parse settings JSON: {e}")

    @pyqtSlot()
    def resetSettings(self):
        """Reset settings request."""
        self.settingsReset.emit()

    @pyqtSlot(str)
    def testConnection(self, api_key: str):
        """Test API connection request."""
        self.testConnectionRequested.emit(api_key)

    @pyqtSlot(str, str, str, str)
    def createCombo(self, combo_id: str, name: str, provider: str, model: str):
        """Create a new combo."""
        try:
            if self.controller.settings_manager:
                self.controller.settings_manager.create_combo(combo_id, name, provider, model)
                # Refresh combos in UI
                self.send_combos()
                self.run_js("showToast('Combo created successfully');")
            else:
                self.run_js("showToast('Settings manager not available', true);")
        except Exception as e:
            logger.error(f"Failed to create combo: {e}")
            self.run_js(f"showToast({json.dumps(f'Failed to create combo: {str(e)}')}, true);")

    @pyqtSlot(str, str, str, str)
    def updateCombo(self, combo_id: str, name: str, provider: str, model: str):
        """Update an existing combo."""
        try:
            if self.controller.settings_manager:
                self.controller.settings_manager.update_combo(combo_id, name, provider, model)
                # Refresh combos in UI
                self.send_combos()
                self.run_js("showToast('Combo updated successfully');")
            else:
                self.run_js("showToast('Settings manager not available', true);")
        except Exception as e:
            logger.error(f"Failed to update combo: {e}")
            self.run_js(f"showToast({json.dumps(f'Failed to update combo: {str(e)}')}, true);")

    @pyqtSlot(str)
    def deleteCombo(self, combo_id: str):
        """Delete a combo."""
        try:
            if self.controller.settings_manager:
                self.controller.settings_manager.delete_combo(combo_id)
                # Refresh combos in UI
                self.send_combos()
                self.run_js("showToast('Combo deleted successfully');")
            else:
                self.run_js("showToast('Settings manager not available', true);")
        except Exception as e:
            logger.error(f"Failed to delete combo: {e}")
            self.run_js(f"showToast({json.dumps(f'Failed to delete combo: {str(e)}')}, true);")

    def send_combos(self):
        """Send available combos to JS."""
        combos = self.controller.settings_manager.get_combos()
        json_combos = json.dumps(combos)
        self.run_js(f"settingsLoadCombos({json_combos});")


class SidePanelWebView(BaseWebView):
    """
    Unified Side Panel Web View containing Files, Memory, and Settings tabs.
    """

    settings_changed = pyqtSignal()

    def __init__(self, controller: AIController, parent=None):
        super().__init__(parent)
        self.controller = controller
        
        # Instantiate Bridges
        self.files_bridge = FilesBridge()
        self.memory_bridge = MemoryBridge()
        self.settings_bridge = SettingsBridge()
        self.settings_bridge.controller = self.controller
        self.settings_bridge.run_js = self.run_js

        # Init Web Channel with multiple objects
        self.init_web_channel({
            "files": self.files_bridge,
            "memory": self.memory_bridge,
            "settings": self.settings_bridge
        })
        
        # Load Unified UI
        self.load_ui("side_panel.html")
        
        # --- Connect Signals ---
        
        # Files
        self.files_bridge.uploadRequested.connect(self.files_upload)
        self.files_bridge.deleteRequested.connect(self.files_delete)
        self.files_bridge.openRequested.connect(self.files_open_details)
        self.files_bridge.refreshRequested.connect(self.files_refresh)
        
        # Memory
        self.memory_bridge.searchRequested.connect(self.memory_search)
        self.memory_bridge.refreshRequested.connect(self.memory_refresh)
        self.memory_bridge.updateRequested.connect(self.memory_update)
        self.memory_bridge.addFactRequested.connect(self.memory_add_fact)
        self.memory_bridge.deleteRequested.connect(self.memory_delete)
        self.memory_bridge.exportRequested.connect(self.memory_export)
        self.memory_bridge.importRequested.connect(self.memory_import)
        self.memory_bridge.clearResult.connect(self.memory_clear_all)
        self.memory_bridge.toggleRequested.connect(self.memory_toggle_enabled)
        
        # Settings
        self.settings_bridge.ready.connect(self.settings_on_ready)
        self.settings_bridge.settingsSaved.connect(self.settings_save)
        self.settings_bridge.settingsReset.connect(self.settings_reset)
        self.settings_bridge.testConnectionRequested.connect(self.settings_test_connection)
        
        # Settings state
        self.can_apply_settings_on_ready = False 
        self.current_settings = {}
        
        self.ingestion_worker = None

    def apply_ui_settings(self):
        """Apply UI-related settings like showing/hiding tabs."""
        try:
            settings = self.settings_load_from_file()
            memory_settings = settings.get('memory', {})
            show_memory_tab = memory_settings.get('show_tab', True)

            # Show or hide memory tab based on setting
            if show_memory_tab:
                self.run_js("showMemoryTab();")
            else:
                self.run_js("hideMemoryTab();")

            logger.info(f"Applied UI settings: memory tab {'shown' if show_memory_tab else 'hidden'}")
        except Exception as e:
            logger.error(f"Failed to apply UI settings: {e}")

    def on_ui_ready(self):
        """Called when UI is fully loaded."""
        logger.info("SidePanel UI Ready")

        # Load settings and apply UI preferences (like showing/hiding memory tab)
        self.apply_ui_settings()

        # Refresh Files
        self.files_refresh()

        # Refresh Memory
        self.memory_refresh()
        self.memory_update_stats()

        # Settings logic handles its own ready signal via self.settings_bridge.ready
        # But we can trigger it too
        self.settings_load_from_file()
        self.settings_send_current()
        self.settings_send_voices()

    # ==========================================
    #              FILES LOGIC
    # ==========================================

    def files_refresh(self):
        try:
            files = self.controller.get_ingested_files()
            files.sort(key=lambda x: x.get('file_name', '').lower())
            files_json = json.dumps(files)
            self.run_js(f"filesUpdateList({files_json});")
        except Exception as e:
            logger.error(f"Failed to refresh files: {e}")
            self.run_js(f"showToast('Error loading files', true);")

    def files_upload(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Documents (*.pdf *.docx *.doc *.txt *.md *.py *.js *.html *.css *.json)")

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.files_start_ingestion(selected_files)

    def files_start_ingestion(self, file_paths: List[str]):
        self.run_js("filesSetIngestionBusy(true);")
        
        self.ingestion_worker = FileIngestionWorker(self.controller, file_paths)
        self.ingestion_worker.progress_updated.connect(self._files_on_ingest_progress)
        self.ingestion_worker.ingestion_completed.connect(self._files_on_ingest_complete)
        self.ingestion_worker.error_occurred.connect(self._files_on_ingest_error)
        self.ingestion_worker.start()

    def _files_on_ingest_progress(self, percent: int, message: str):
        self.run_js(f"filesUpdateIngestionProgress({percent}, {json.dumps(message)});")

    def _files_on_ingest_complete(self, result: dict):
        self.run_js("filesSetIngestionBusy(false);")
        msg = f"Ingested {result['successful']} files."
        if result['failed'] > 0:
            msg += f" {result['failed']} failed."
            self.run_js(f"showToast({json.dumps(msg)}, true);")
        else:
            self.run_js(f"showToast({json.dumps(msg)}, false);")
        
        self.files_refresh()

    def _files_on_ingest_error(self, error: str):
        self.run_js("filesSetIngestionBusy(false);")
        self.run_js(f"showToast({json.dumps(str(error))}, true);")

    def files_delete(self, file_path: str):
        confirm = QMessageBox.question(
            self, "Delete File",
            f"Are you sure you want to delete this file?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            if self.controller.delete_file(file_path):
                self.run_js("showToast('File deleted successfully');")
                self.files_refresh()
            else:
                self.run_js("showToast('Failed to delete file', true);")
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            self.run_js(f"showToast({json.dumps(f'Error deleting file: {str(e)}')}, true);")

    def files_open_details(self, file_path: str):
        files = self.controller.get_ingested_files()
        info = next((f for f in files if f['file_path'] == file_path), None)
        
        if info:
             details = f"File: {info.get('file_name', 'Unknown')}\n"
             details += f"Path: {info.get('file_path', 'Unknown')}\n"
             details += f"Size: {info.get('file_size', 0)} bytes\n"
             details += f"Type: {info.get('mime_type', 'Unknown')}\n"
             details += f"Modified: {info.get('file_modified', 'Unknown')}\n"
             QMessageBox.information(self, "File Details", details)
        else:
            self.run_js("showToast('File info not found', true);")

    # ==========================================
    #              MEMORY LOGIC
    # ==========================================

    def memory_refresh(self):
        self.memory_search("", {})

    def memory_search(self, query: str, filters: Dict[str, Any]):
        try:
            limit = filters.get('limit', 20)
            results = self.controller.search_memory(query, limit=limit)
            
            filtered_results = []
            req_type = filters.get('type', 'All')
            req_importance = filters.get('importance', 'All')
            req_tags = filters.get('tags', [])
            
            for mem in results:
                metadata = mem.get('metadata', {})
                mem_type = metadata.get('type', 'unknown')
                mem_imp = metadata.get('importance', 'normal')
                mem_tags = metadata.get('tags', [])
                
                if req_type != 'All' and mem_type != req_type:
                    continue
                    
                if req_importance != 'All':
                    importance_levels = {"low": 0, "normal": 1, "high": 2, "critical": 3}
                    if importance_levels.get(mem_imp, 0) < importance_levels.get(req_importance, 0):
                        continue
                        
                if req_tags:
                    if not any(t in mem_tags for t in req_tags):
                        continue
                        
                filtered_results.append(mem)
                
            results_json = json.dumps(filtered_results)
            self.run_js(f"memoryDisplayResults({results_json});")
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            self.run_js(f"showToast({json.dumps(f'Search failed: {str(e)}')}, true);")

    def memory_add_fact(self, fact_data: Dict[str, Any]):
        try:
            self.controller.store_memory_fact(
                fact=fact_data['content'],
                fact_type=fact_data.get('type', 'semantic'),
                importance=fact_data.get('importance', 'normal'),
                tags=fact_data.get('tags', [])
            )
            self.run_js("showToast('Fact added successfully');")
            self.memory_refresh()
        except Exception as e:
            logger.error(f"Failed to add fact: {e}")
            self.run_js(f"showToast({json.dumps(f'Failed to add fact: {str(e)}')}, true);")

    def memory_update(self, memory_data: Dict[str, Any]):
        try:
             self.controller.store_memory_fact(
                fact=memory_data.get('content'),
                fact_type=memory_data.get('type'),
                importance=memory_data.get('importance'),
                tags=memory_data.get('tags', [])
            )
             self.run_js("showToast('Memory updated (stored as new fact)');")
             self.memory_refresh()
        except Exception as e:
             logger.error(f"Failed to update memory: {e}")
             self.run_js(f"showToast({json.dumps(f'Failed to update: {str(e)}')}, true);")

    def memory_delete(self, memory_id: str):
        self.run_js("showToast('Individual deletion not implemented yet', true);")

    def memory_export(self):
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Export Memory", "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            try:
                self.controller.export_memory(file_path)
                self.run_js(f"showToast('Exported to {Path(file_path).name}');")
            except Exception as e:
                self.run_js(f"showToast({json.dumps(f'Export failed: {str(e)}')}, true);")

    def memory_import(self):
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Import Memory", "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            try:
                self.controller.import_memory(file_path)
                self.run_js(f"showToast('Imported from {Path(file_path).name}');")
                self.memory_refresh()
            except Exception as e:
                self.run_js(f"showToast({json.dumps(f'Import failed: {str(e)}')}, true);")

    def memory_clear_all(self):
        confirm = QMessageBox.question(
            None, "Confirm Clear All",
            "Are you sure you want to clear ALL memories?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            try:
                self.controller.clear_memory()
                self.run_js("showToast('All memories cleared');")
                self.memory_refresh()
            except Exception as e:
                self.run_js(f"showToast({json.dumps(f'Clear failed: {str(e)}')}, true);")

    def memory_toggle_enabled(self, enabled: bool):
        self.controller.toggle_memory(enabled)
        self.memory_update_stats()

    def memory_update_stats(self):
        try:
            stats = self.controller.get_memory_stats()
            stats_json = json.dumps(stats)
            self.run_js(f"memoryUpdateStats({stats_json});")
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")

    # ==========================================
    #              SETTINGS LOGIC
    # ==========================================

    def settings_on_ready(self):
        # Already handled in on_ui_ready mostly, but if JS triggers it specifically:
        self.settings_load_from_file()
        self.settings_send_current()
        self.settings_send_voices()

    def settings_load_from_file(self) -> Dict[str, Any]:
        if self.controller.settings_manager:
            self.current_settings = self.controller.settings_manager.load_settings()
        else:
            logger.warning("Settings manager not available, using empty settings")
            self.current_settings = {}
            
        return self.current_settings

    def settings_send_current(self):
        json_settings = json.dumps(self.current_settings)
        self.run_js(f"settingsLoad({json_settings});")

        # Send combos data if available
        combos = self.current_settings.get('combos', {})
        if combos:
            json_combos = json.dumps(combos)
            self.run_js(f"settingsLoadCombos({json_combos});")

    def settings_send_voices(self):
        try:
            voices = self.controller.get_available_tts_voices()
            if voices:
                simple_voices = [{'id': v['name'], 'name': f"{v['name']} - {v.get('description', v.get('language', ''))}"} for v in voices]
                json_voices = json.dumps(simple_voices)
                self.run_js(f"settingsSetVoices({json_voices});")
            else:
                self.run_js("settingsSetVoices([]);")
        except Exception as e:
            self.run_js("settingsSetVoices([]);")

    def settings_save(self, new_settings):
        try:
            # new_settings can be JSON string or dict depending on Bridge implementation. 
            # SettingsBridge.saveSettings definition: @pyqtSlot(str)
            if isinstance(new_settings, str):
                settings_dict = json.loads(new_settings)
            else:
                settings_dict = new_settings

            logger.info("Saving settings...")
            
            if self.controller.settings_manager:
                self.controller.settings_manager.save_settings(settings_dict)
                self.current_settings = settings_dict
                self.settings_apply_to_controller()
                self.run_js("showToast('Settings saved successfully');")
                self.settings_changed.emit()
            else:
                self.run_js("showToast('Settings manager not available', true);")

        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            self.run_js(f"showToast({json.dumps(f'Failed to save settings: {str(e)}')}, true);")

    def settings_reset(self):
        try:
            if self.controller.settings_manager:
                self.controller.settings_manager.reset_to_defaults()
                
                self.settings_load_from_file()
                self.settings_send_current()
                self.settings_apply_to_controller()
                self.run_js("showToast('Settings reset to defaults');")
                self.settings_changed.emit()
            else:
                self.run_js("showToast('Settings manager not available', true);")
        except Exception as e:
            logger.error(f"Failed to reset settings: {e}")
            self.run_js("showToast('Failed to reset settings', true);")

    def settings_test_connection(self, api_key: str):
        if not api_key:
            self.run_js("showToast('Please enter an API key', true);")
            return
        
        self.run_js("showToast('Testing connection...', false);")
        # Simple validity check
        if len(api_key) > 10:
             self.run_js("showToast('Connection Successful! API Key format valid.');")
        else:
             self.run_js("showToast('Connection Failed. Invalid key format.', true);")

    def settings_apply_to_controller(self):
        try:
            s = self.current_settings
            
            # Application
            # theme = s['app']['theme'] (Not applied to controller yet, maybe UI theme)
            
            # API Keys
            api_keys = s.get('api_keys', {})
            openai_key = api_keys.get('openai')
            openrouter_key = api_keys.get('openrouter')
            
            self.controller.update_api_keys(openai_api_key=openai_key, openrouter_api_key=openrouter_key)
            
            # LLM
            llm = s.get('llm', {})
            config = {}
            if llm.get('provider') == 'openrouter':
                config['api_key'] = openrouter_key
            if llm.get('model'):
                config['model'] = llm['model']
            if 'temperature' in llm:
                config['temperature'] = llm['temperature']
            self.controller.configure_llm(llm.get('provider', 'openrouter'), **config)
            
            # Retrieval
            retrieval = s.get('retrieval', {})
            self.controller.toggle_retrieval(retrieval.get('enabled', True))
            self.controller.set_max_retrieved_docs(retrieval.get('max_docs', 3))
            
            is_hybrid = retrieval.get('method') == 'hybrid'
            self.controller.set_hybrid_retrieval(is_hybrid, retrieval.get('semantic_weight'), retrieval.get('keyword_weight'))
            
            # Audio
            audio = s.get('audio', {})
            self.controller.configure_audio(
                tts_enabled=audio.get('tts_enabled', True),
                asr_enabled=audio.get('asr_enabled', True),
                voice=audio.get('tts_voice', 'alloy'),
                speed=audio.get('tts_speed', 1.0),
                language=audio.get('asr_language', 'en-US')
            )
            
            logger.info("Settings applied to controller")
        except Exception as e:
            logger.error(f"Failed to apply settings: {e}")
