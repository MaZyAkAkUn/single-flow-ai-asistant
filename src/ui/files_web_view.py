import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QObject, QThread
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from .base_web_view import BaseWebView
from ..core.controller import AIController

logger = logging.getLogger(__name__)

class FilesBridge(QObject):
    """Bridge for Files Web View."""
    
    # Signals
    uploadRequested = pyqtSignal()
    deleteRequested = pyqtSignal(str) # file_path
    openRequested = pyqtSignal(str) # file_path
    refreshRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    @pyqtSlot()
    def requestUpload(self):
        """Request file upload dialog."""
        self.uploadRequested.emit()

    @pyqtSlot(str)
    def requestDelete(self, file_path: str):
        """Request file deletion."""
        self.deleteRequested.emit(file_path)

    @pyqtSlot(str)
    def requestOpen(self, file_path_json: str):
        """Request opening file details."""
        # Note: Depending on JS call, this might need parsing if passed as JSON object
        # simpler to pass string ID or path
        self.openRequested.emit(file_path_json)

    @pyqtSlot()
    def requestRefresh(self):
        """Request file list refresh."""
        self.refreshRequested.emit()
    
    @pyqtSlot(str)
    def log(self, message: str):
        """Log message from JS."""
        logger.debug(f"JS [Files]: {message}")


class FileIngestionWorker(QThread):
    """Worker thread for file ingestion to avoid blocking UI."""
    # Signals matching the JS update expectations
    progress_updated = pyqtSignal(int, str)  # percentage, message
    ingestion_completed = pyqtSignal(dict)   # result summary
    error_occurred = pyqtSignal(str)

    def __init__(self, controller: AIController, file_paths: List[str]):
        super().__init__()
        self.controller = controller
        self.file_paths = file_paths

    def run(self):
        total_files = len(self.file_paths)
        successful = 0
        failed = 0

        for i, file_path in enumerate(self.file_paths):
            try:
                progress = int((i / total_files) * 100)
                name = Path(file_path).name
                self.progress_updated.emit(progress, f"Processing: {name}")

                result = self.controller.ingest_file(file_path)

                if result['status'] == 'success' or result['status'] == 'already_exists':
                    successful += 1
                else:
                    failed += 1
                    logger.warning(f"Failed to ingest {file_path}: {result.get('error', 'Unknown error')}")

            except Exception as e:
                failed += 1
                logger.error(f"Error ingesting {file_path}: {e}")

        final_msg = f"Completed: {successful} successful, {failed} failed"
        self.progress_updated.emit(100, final_msg)

        result = {
            'total': total_files,
            'successful': successful,
            'failed': failed
        }
        self.ingestion_completed.emit(result)


class FilesWebWidget(BaseWebView):
    """
    QWebEngineView-based files manager.
    """

    def __init__(self, controller: AIController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.bridge = FilesBridge()
        self.ingestion_worker = None
        
        self.init_web_channel({"backend": self.bridge})
        self.load_ui("files.html")
        
        # Connect signals
        self.bridge.uploadRequested.connect(self.upload_files)
        self.bridge.deleteRequested.connect(self.delete_file)
        self.bridge.openRequested.connect(self.open_file_details)
        self.bridge.refreshRequested.connect(self.refresh_files)

    def on_ui_ready(self):
        """Called when JS is ready."""
        self.refresh_files()

    def refresh_files(self):
        """Fetch files and update JS."""
        try:
            files = self.controller.get_ingested_files()
            # Sort by name
            files.sort(key=lambda x: x.get('file_name', '').lower())
            
            # Send to JS
            files_json = json.dumps(files)
            self.run_js(f"updateFileList({files_json});")
        except Exception as e:
            logger.error(f"Failed to refresh files: {e}")
            self.run_js(f"showToast('Error loading files', true);")

    def upload_files(self):
        """Open file dialog and ingest."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Documents (*.pdf *.docx *.doc *.txt *.md *.py *.js *.html *.css *.json)")

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.start_ingestion(selected_files)

    def start_ingestion(self, file_paths: List[str]):
        """Start ingestion worker."""
        self.run_js("setIngestionBusy(true);")
        
        self.ingestion_worker = FileIngestionWorker(self.controller, file_paths)
        self.ingestion_worker.progress_updated.connect(self._on_ingest_progress)
        self.ingestion_worker.ingestion_completed.connect(self._on_ingest_complete)
        self.ingestion_worker.error_occurred.connect(self._on_ingest_error)
        self.ingestion_worker.start()

    def _on_ingest_progress(self, percent: int, message: str):
        self.run_js(f"updateIngestionProgress({percent}, {json.dumps(message)});")

    def _on_ingest_complete(self, result: dict):
        self.run_js("setIngestionBusy(false);")
        msg = f"Ingested {result['successful']} files."
        if result['failed'] > 0:
            msg += f" {result['failed']} failed."
            self.run_js(f"showToast({json.dumps(msg)}, true);")
        else:
            self.run_js(f"showToast({json.dumps(msg)}, false);")
        
        self.refresh_files()

    def _on_ingest_error(self, error: str):
        self.run_js("setIngestionBusy(false);")
        self.run_js(f"showToast({json.dumps(str(error))}, true);")

    def delete_file(self, file_path: str):
        """Delete file via controller."""
        # Confirm logic could be here or in JS. JS can show confirm dialog.
        # Assuming JS already confirmed or we do it here.
        # Let's trust JS confirmation for now for cleaner UX flow (modal in web view).
        # OR better, simple confirm here if we want native feel, BUT we want unification.
        # Let's assume JS calls this AFTER confirmation.
        
        # NOTE: Native FileWidget had a confirmation box.
        # Ideally, we implement `delete_file` to first show a box, then proceed.
        # But `confirm` in JS is synchronous (blocking renderer), `QMessageBox` blocks Python.
        # Let's prompt in Python for safety, or pass a flag if confirmed.
        
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
                self.refresh_files()
            else:
                self.run_js("showToast('Failed to delete file', true);")
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            self.run_js(f"showToast('Error deleting file: {str(e)}', true);")

    def open_file_details(self, file_path: str):
        """Show file details."""
        # Find file info
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
