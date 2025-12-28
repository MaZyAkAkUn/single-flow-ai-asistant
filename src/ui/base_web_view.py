import logging
from pathlib import Path
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtCore import QUrl, QObject

logger = logging.getLogger(__name__)

class BaseWebView(QWebEngineView):
    """
    Base class for QWebEngineView-based widgets.
    Handles common setup for QWebChannel and asset loading.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_loaded = False
        self.pending_js_calls = []
        self._bridge = None
        self._channel = None

        # Developer tools references (to prevent garbage collection)
        self._dev_tools_window = None
        self._dev_tools_view = None

        # Configure web engine settings to allow loading external images
        # This enables local HTML files to load remote images and network access
        self.settings().setAttribute(QWebEngineSettings.WebAttribute.AllowRunningInsecureContent, True)
        self.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        self.settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        self.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

        # Try to disable web security entirely for local development
        try:
            self.settings().setAttribute(QWebEngineSettings.WebAttribute.WebSecurityEnabled, False)
        except AttributeError:
            # WebSecurityEnabled might not be available in this Qt version
            logger.debug("WebSecurityEnabled attribute not available")

        # Connect load finished signal
        self.loadFinished.connect(self._on_load_finished)

    def init_web_channel(self, objects: dict[str, QObject]):
        """Initialize the QWebChannel with the provided bridge objects."""
        self._channel = QWebChannel()
        for name, obj in objects.items():
            self._channel.registerObject(name, obj)
        self.page().setWebChannel(self._channel)

    def load_ui(self, html_filename: str):
        """Load the HTML file from the assets directory."""
        assets_path = Path(__file__).parent / "assets" / html_filename
        if not assets_path.exists():
            logger.error(f"Asset file not found: {assets_path}")
            return
            
        self.load(QUrl.fromLocalFile(str(assets_path.absolute())))

    def run_js(self, js_code: str):
        """Run JavaScript code, queuing it if the page isn't loaded yet."""
        if self.is_loaded:
            self.page().runJavaScript(js_code)
        else:
            self.pending_js_calls.append(js_code)

    def _on_load_finished(self, ok: bool):
        """Handle internal load finished event."""
        if ok:
            self.is_loaded = True
            logger.debug(f"Web view loaded successfully: {self.url().toString()}")
            # Execute pending JS calls
            for js_code in self.pending_js_calls:
                self.page().runJavaScript(js_code)
            self.pending_js_calls.clear()
            
            # Call hook for subclasses
            self.on_ui_ready()
        else:
            logger.error(f"Failed to load web view: {self.url().toString()}")

    def on_ui_ready(self):
        """Hook called when UI is fully loaded. Override in subclasses."""
        pass

    def open_dev_tools(self):
        """Open developer tools in a separate window."""
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtWidgets import QMainWindow

        # Check if dev tools are already open
        if self._dev_tools_window is not None and self._dev_tools_window.isVisible():
            # Bring to front
            self._dev_tools_window.raise_()
            self._dev_tools_window.activateWindow()
            return

        # Create a new window for dev tools
        self._dev_tools_window = QMainWindow()
        self._dev_tools_window.setWindowTitle("Developer Tools")
        self._dev_tools_window.setGeometry(100, 100, 800, 600)

        # Create dev tools view
        self._dev_tools_view = QWebEngineView()
        self._dev_tools_window.setCentralWidget(self._dev_tools_view)

        # Set this page as the inspected page
        self._dev_tools_view.page().setInspectedPage(self.page())

        # Connect close event to clear references
        self._dev_tools_window.closeEvent = lambda event: self._on_dev_tools_closed()

        # Show the window
        self._dev_tools_window.show()

    def _on_dev_tools_closed(self):
        """Handle dev tools window being closed."""
        self._dev_tools_window = None
        self._dev_tools_view = None
