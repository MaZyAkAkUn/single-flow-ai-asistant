import json
import logging
import os
from typing import List, Dict, Any
from pathlib import Path
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QObject, QTimer
from PyQt6.QtWidgets import QMessageBox, QFileDialog

from .base_web_view import BaseWebView
from ..core.controller import AIController

logger = logging.getLogger(__name__)

class MemoryBridge(QObject):
    """Bridge for Memory Web View."""
    
    # Signals
    searchRequested = pyqtSignal(str, dict) # query, filters
    updateRequested = pyqtSignal(dict) # memory_data
    deleteRequested = pyqtSignal(str) # memory_id (or sufficient data to identify)
    addFactRequested = pyqtSignal(dict) # fact_data
    refreshRequested = pyqtSignal()
    exportRequested = pyqtSignal()
    importRequested = pyqtSignal()
    clearResult = pyqtSignal()
    toggleRequested = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

    @pyqtSlot(str, str)
    def requestSearch(self, query: str, filters_json: str):
        """Request memory search."""
        try:
            filters = json.loads(filters_json)
            self.searchRequested.emit(query, filters)
        except json.JSONDecodeError:
            logger.error("Invalid filters JSON")
            self.searchRequested.emit(query, {})

    @pyqtSlot(str)
    def requestUpdate(self, memory_json: str):
        """Request update memory."""
        try:
            memory = json.loads(memory_json)
            self.updateRequested.emit(memory)
        except json.JSONDecodeError:
            logger.error("Invalid memory JSON")

    @pyqtSlot(str)
    def requestDelete(self, memory_id: str):
        self.deleteRequested.emit(memory_id)
        
    @pyqtSlot(str)
    def requestAddFact(self, fact_json: str):
        try:
            fact_data = json.loads(fact_json)
            self.addFactRequested.emit(fact_data)
        except json.JSONDecodeError:
            logger.error("Invalid fact JSON")

    @pyqtSlot()
    def requestRefresh(self):
        self.refreshRequested.emit()

    @pyqtSlot()
    def requestExport(self):
        self.exportRequested.emit()
        
    @pyqtSlot()
    def requestImport(self):
        self.importRequested.emit()
        
    @pyqtSlot()
    def requestClear(self):
        self.clearResult.emit()
        
    @pyqtSlot(bool)
    def requestToggle(self, enabled: bool):
        self.toggleRequested.emit(enabled)

    @pyqtSlot(str)
    def log(self, message: str):
        logger.debug(f"JS [Memory]: {message}")


class MemoryWebWidget(BaseWebView):
    """
    QWebEngineView-based memory manager.
    """

    def __init__(self, controller: AIController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.bridge = MemoryBridge()
        
        self.init_web_channel({"backend": self.bridge})
        self.load_ui("memory.html")
        
        # Connect signals
        self.bridge.searchRequested.connect(self.search_memory)
        self.bridge.refreshRequested.connect(self.refresh_memory)
        self.bridge.updateRequested.connect(self.update_memory)
        self.bridge.addFactRequested.connect(self.add_fact)
        self.bridge.deleteRequested.connect(self.delete_memory)
        self.bridge.exportRequested.connect(self.export_memory)
        self.bridge.importRequested.connect(self.import_memory)
        self.bridge.clearResult.connect(self.clear_all_memory)
        self.bridge.toggleRequested.connect(self.toggle_memory_enabled)

    def on_ui_ready(self):
        """Called when JS is ready."""
        self.refresh_memory()
        self.update_stats()
        # Init toggle state
        enabled = True # Needs check from controller settings if exposed
        # We can infer from stats or settings. 
        # Using a direct call to controller settings if available or just default true.
        # self.run_js(f"setMemoryEnabled({str(enabled).lower()});") 

    def update_stats(self):
        try:
            stats = self.controller.get_memory_stats()
            stats_json = json.dumps(stats)
            self.run_js(f"updateStats({stats_json});")
            
            if 'enabled' in stats:
                self.run_js(f"setMemoryEnabled({str(stats['enabled']).lower()});")
                
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")

    def refresh_memory(self):
        # Refresh is just a search with empty query usually
        self.search_memory("", {})

    def search_memory(self, query: str, filters: Dict[str, Any]):
        try:
            # Native widget logic
            limit = filters.get('limit', 20)
            results = self.controller.search_memory(query, limit=limit)
            
            # Apply client-side filters (type, importance, tags)
            # Or implementing filter in controller would be better, but staying consistent with previous widget logic
            
            filtered_results = []
            req_type = filters.get('type', 'All')
            req_importance = filters.get('importance', 'All')
            req_tags = filters.get('tags', []) # List of strings
            
            for mem in results:
                metadata = mem.get('metadata', {})
                mem_type = metadata.get('type', 'unknown')
                mem_imp = metadata.get('importance', 'normal')
                mem_tags = metadata.get('tags', [])
                
                if req_type != 'All' and mem_type != req_type:
                    continue
                    
                if req_importance != 'All':
                    # Simplified importance check (exact match or gte?)
                    # Previous widget did gte logic.
                    importance_levels = {"low": 0, "normal": 1, "high": 2, "critical": 3}
                    if importance_levels.get(mem_imp, 0) < importance_levels.get(req_importance, 0):
                        continue
                        
                if req_tags:
                    if not any(t in mem_tags for t in req_tags):
                        continue
                        
                filtered_results.append(mem)
                
            results_json = json.dumps(filtered_results)
            self.run_js(f"displaySearchResults({results_json});")
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            self.run_js(f"showToast('Search failed: {str(e)}', true);")

    def add_fact(self, fact_data: Dict[str, Any]):
        try:
            self.controller.store_memory_fact(
                fact=fact_data['content'],
                fact_type=fact_data.get('type', 'semantic'),
                importance=fact_data.get('importance', 'normal'),
                tags=fact_data.get('tags', [])
            )
            self.run_js("showToast('Fact added successfully');")
            self.refresh_memory()
        except Exception as e:
            logger.error(f"Failed to add fact: {e}")
            self.run_js(f"showToast('Failed to add fact: {str(e)}', true);")

    def update_memory(self, memory_data: Dict[str, Any]):
        # Memori SDK might not support direct update easily without ID.
        # Native widget stored as NEW fact if content changed.
        # We'll follow suit for now.
        # But wait, we receive the full memory object being edited.
        
        try:
            # Treat as new fact 
             self.controller.store_memory_fact(
                fact=memory_data.get('content'),
                fact_type=memory_data.get('type'),
                importance=memory_data.get('importance'),
                tags=memory_data.get('tags', [])
            )
             self.run_js("showToast('Memory updated (stored as new fact)');")
             self.refresh_memory()
        except Exception as e:
             logger.error(f"Failed to update memory: {e}")
             self.run_js(f"showToast('Failed to update: {str(e)}', true);")

    def delete_memory(self, memory_id: str):
        self.run_js("showToast('Individual deletion not implemented yet', true);")

    def export_memory(self):
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Export Memory", "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            try:
                self.controller.export_memory(file_path)
                self.run_js(f"showToast('Exported to {Path(file_path).name}');")
            except Exception as e:
                self.run_js(f"showToast('Export failed: {str(e)}', true);")

    def import_memory(self):
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Import Memory", "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            try:
                self.controller.import_memory(file_path)
                self.run_js(f"showToast('Imported from {Path(file_path).name}');")
                self.refresh_memory()
            except Exception as e:
                self.run_js(f"showToast('Import failed: {str(e)}', true);")

    def clear_all_memory(self):
        confirm = QMessageBox.question(
            None, "Confirm Clear All",
            "Are you sure you want to clear ALL memories? This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            try:
                self.controller.clear_memory()
                self.run_js("showToast('All memories cleared');")
                self.refresh_memory()
            except Exception as e:
                self.run_js(f"showToast('Clear failed: {str(e)}', true);")

    def toggle_memory_enabled(self, enabled: bool):
        self.controller.toggle_memory(enabled)
        self.update_stats()
