"""
Main window for the Single-Flow Personal Assistant application.
Implements the primary UI layout with message timeline, input bar, and collapsible side panel.
"""

import sys
import asyncio
import time
import uuid
from typing import Optional, List, Dict, Any
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QScrollArea, QFrame, QTextEdit, QPushButton,
    QLineEdit, QLabel, QProgressBar, QSizePolicy, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QObject
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon, QKeySequence, QShortcut, QCloseEvent
from pathlib import Path
import json
from datetime import datetime

from ..core.logging_config import get_logger, setup_logging
from ..core.controller import AIController
from ..core.settings import PlainSettings
from ..core.file_ingestor import FileIngestor
from .prompt_web_view import PromptWebWidget
from .chat_web_view import ChatWebWidget
from .side_panel_web_view import SidePanelWebView

logger = get_logger(__name__)


class AIWorker(QObject):
    """Worker object for running AI processing in a separate thread with streaming support."""

    stream_started = pyqtSignal()  # Emitted when streaming begins
    token_received = pyqtSignal(str)  # Emitted for each token received
    status_changed = pyqtSignal(str, str)  # Emitted when status updates (status_type, message)
    stream_finished = pyqtSignal(dict)  # Emitted when streaming is complete with metadata
    error_occurred = pyqtSignal(str)  # Emitted when an error occurs
    finished = pyqtSignal(str)  # Legacy signal for backward compatibility
    progress = pyqtSignal(int)  # Legacy signal for progress updates
    prompt_received = pyqtSignal(dict)  # Emitted when a structured prompt is generated
    reasoning_received = pyqtSignal(str)  # Emitted when reasoning tokens are received
    tool_started = pyqtSignal(str, dict)  # Emitted when a tool starts execution (tool_name, input)
    tool_finished = pyqtSignal(str, str)  # Emitted when a tool finishes execution (tool_name, output)

    def __init__(self, controller: AIController):
        super().__init__()
        self.controller = controller
        self.loop = None
        self._is_running = False
        self._shutdown_event = asyncio.Event()

    def cancel_processing(self):
        """Cancel any ongoing processing by setting the shutdown event."""
        logger.info("Canceling AI worker processing tasks...")
        self._shutdown_event.set()

    def reset_shutdown_event(self):
        """Reset the shutdown event for new processing."""
        self._shutdown_event.clear()

    def process_message(self, message: str):
        """Process a message asynchronously (legacy)."""
        logger.info(f"AIWorker processing message: {message[:50]}...")
        if self._is_running:
            logger.warning("AI worker is already processing a message")
            return

        self._is_running = True
        try:
            # Check if there's already an event loop in this thread
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we need to create a new one
                    self.loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self.loop)
                else:
                    self.loop = loop
            except RuntimeError:
                # No event loop exists, create one
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

            # Run the async processing
            if asyncio.iscoroutinefunction(self.controller.process_message):
                logger.info("Using async process_message")
                result = self.loop.run_until_complete(self.controller.process_message(message))
            else:
                # Handle sync method
                logger.info("Using sync process_message_sync")
                result = self.controller.process_message_sync(message)

            self.finished.emit(result)

        except Exception as e:
            logger.error(f"Error in AI worker: {e}")
            self.error.emit(str(e))
        finally:
            self._is_running = False
            # Only close the loop if we created it
            if self.loop and self.loop != asyncio.get_event_loop():
                try:
                    # Clean up async generators before closing loop
                    if self.loop.is_running():
                         self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                    elif not self.loop.is_closed():
                         self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                    
                    self.loop.close()
                except Exception as e:
                    logger.warning(f"Failed to close event loop: {e}")

    async def process_message_stream(self, message: str, combo_id: Optional[str] = None):
        """Process a message with streaming support asynchronously."""
        try:
            logger.info(f"AIWorker streaming message: {message[:50]}... (combo: {combo_id})")

            # Check if controller has stream_message method
            if not hasattr(self.controller, 'stream_message'):
                logger.error("Controller does not support streaming")
                self.error_occurred.emit("Streaming not supported by controller")
                return

            # Create a task for the streaming operation
            stream_task = asyncio.create_task(self._stream_to_events(message, combo_id))

            try:
                # Wait for the streaming task with cancellation support
                await asyncio.wait_for(asyncio.shield(stream_task), timeout=None)
            except asyncio.CancelledError:
                logger.info("Streaming task was cancelled")
                stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass
                raise

        except asyncio.CancelledError:
            logger.info("Streaming operation cancelled due to shutdown")
            self.error_occurred.emit("Operation cancelled")
        except Exception as e:
            logger.error(f"Error in streaming AI worker: {e}")
            self.error_occurred.emit(str(e))

    async def _stream_to_events(self, message: str, combo_id: Optional[str] = None):
        """Stream from controller and emit events with immediate token delivery."""
        token_buffer = []
        last_flush_time = time.time()

        try:
            async for event in self.controller.stream_message(message, combo_id):
                # Check for cancellation
                if self._shutdown_event.is_set():
                    logger.info("Shutdown event detected, cancelling streaming")
                    break

                event_type = event.get("type")
                if event_type == "status":
                    # Flush any buffered tokens before status update
                    if token_buffer:
                        combined_tokens = ''.join(token_buffer)
                        self.token_received.emit(combined_tokens)
                        token_buffer.clear()
                    
                    self.status_changed.emit("processing", event.get("content", ""))
                elif event_type == "token":
                    # Buffer tokens for immediate delivery
                    token_buffer.append(event.get("content", ""))
                    current_time = time.time()
                    
                    # Flush conditions for real-time streaming:
                    # 1. Significant content accumulated
                    # 2. Buffer timeout reached (0.5 seconds)
                    # 3. Large token chunks
                    should_flush = (
                        len(token_buffer) >= 5 or  # Flush every 5 tokens
                        current_time - last_flush_time >= 0.1 or  # 0.1s timeout for UI responsiveness
                        len(event.get("content", "")) > 10  # Large chunks
                    )
                    
                    if should_flush:
                        combined_tokens = ''.join(token_buffer)
                        if combined_tokens:
                            self.token_received.emit(combined_tokens)
                        token_buffer.clear()
                        last_flush_time = current_time
                        
                elif event_type == "error":
                    # Flush remaining tokens on error
                    if token_buffer:
                        combined_tokens = ''.join(token_buffer)
                        self.token_received.emit(combined_tokens)
                        token_buffer.clear()
                    
                    self.error_occurred.emit(event.get("content", "Unknown error"))
                    break
                elif event_type == "done":
                    # Final flush and completion
                    if token_buffer:
                        combined_tokens = ''.join(token_buffer)
                        self.token_received.emit(combined_tokens)
                        token_buffer.clear()
                    
                    self.stream_finished.emit({})
                    break
                
                elif event_type == "reasoning":
                    # Emit reasoning content
                    self.reasoning_received.emit(event.get("content", ""))
                    
                elif event_type == "prompt":
                    # Emit prompt data
                    # event.get("content") is structured prompt string
                    # event.get("metadata") contains hash etc.
                    # We might need to construct the dict expected by prompt_received
                    prompt_data = {
                        'structured_prompt': event.get("content", ""),
                        'flattened_prompt': event.get("flattened", ""),
                        'timestamp': datetime.now().isoformat()
                    }
                    if event.get("metadata"):
                        prompt_data.update(event.get("metadata"))
                        
                    self.prompt_received.emit(prompt_data)
                
                elif event_type == "tool_start":
                    self.tool_started.emit(event.get("tool", "unknown"), event.get("input", {}))
                    
                elif event_type == "tool_end":
                    self.tool_finished.emit(event.get("tool", "unknown"), str(event.get("output", "")))
                    
                # Handle other event types as needed
                
        except Exception as e:
            logger.error(f"Streaming error in worker: {e}")
            # Flush any remaining tokens on exception
            if token_buffer:
                combined_tokens = ''.join(token_buffer)
                self.token_received.emit(combined_tokens)
                token_buffer.clear()
            
            self.error_occurred.emit(f"Streaming error: {str(e)}")
        finally:
            # Ensure any remaining tokens are flushed
            if token_buffer:
                combined_tokens = ''.join(token_buffer)
                self.token_received.emit(combined_tokens)
                token_buffer.clear()

    def stream_message(self, message: str):
        """Process a message with streaming support."""
        # Parse combo and message content
        combo_id = None
        message_content = message

        # Check if message contains combo separator
        if '|' in message:
            parts = message.split('|', 1)
            if len(parts) == 2:
                combo_id = parts[0] if parts[0].strip() else None
                message_content = parts[1]

        logger.info(f"AIWorker starting streaming: {message_content[:50]}... (combo: {combo_id})")
        if self._is_running:
            logger.warning("AI worker is already processing a message")
            return

        self._is_running = True
        # Reset shutdown event for new processing
        self.reset_shutdown_event()
        self.stream_started.emit()

        try:
            # Check if there's already an event loop in this thread
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we need to create a new one
                    self.loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self.loop)
                else:
                    self.loop = loop
            except RuntimeError:
                # No event loop exists, create one
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

            # Run the async streaming processing with combo_id
            self.loop.run_until_complete(self.process_message_stream(message_content, combo_id))

            # Emit finished signal for backward compatibility
            # Note: We don't want to emit this with "Streaming completed" as it creates a duplicate message
            # The stream_finished signal is sufficient
            # self.finished.emit("Streaming completed")

        except Exception as e:
            logger.error(f"Error in streaming worker: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self._is_running = False
            # Only close the loop if we created it
            if self.loop and self.loop != asyncio.get_event_loop():
                try:
                    # Clean up async generators before closing loop
                    if not self.loop.is_closed():
                        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                    self.loop.close()
                except Exception as e:
                    logger.warning(f"Failed to close event loop: {e}")


class TTSWorker(QThread):
    """Worker object for running TTS generation in a separate thread."""
    
    finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, controller: AIController, text: str):
        super().__init__()
        self.controller = controller
        self.text = text
        
    def run(self):
        try:
            # Generate audio file (blocking network I/O)
            file_path = self.controller.speak_text(self.text, save_to_file=True)
            if file_path:
                self.finished.emit(file_path)
            else:
                self.error_occurred.emit("Failed to generate audio")
        except Exception as e:
            self.error_occurred.emit(str(e))

class SidePanelWidget(QWidget):
    """Unified side panel containing the web view."""

    def __init__(self, controller: AIController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.is_collapsed = True
        self.init_ui()

    def init_ui(self):
        """Initialize the side panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Unified Web UI
        self.web_view = SidePanelWebView(self.controller)
        layout.addWidget(self.web_view)

        # Set minimum width
        self.setMinimumWidth(400)
        self.setMaximumWidth(400)


class MainWindow(QMainWindow):
    """Main application window."""

    # Signal to request streaming in worker thread
    start_streaming_requested = pyqtSignal(str)

    def __init__(self):
        setup_logging()
        super().__init__()
        
        self.plain_settings = PlainSettings()

        # Load API keys before initializing controller
        openai_api_key = self._load_openai_api_key()

        # Initialize controller with API key loaded from settings and settings manager
        self.ai_controller = AIController(openai_api_key=openai_api_key, settings_manager=self.plain_settings)

        # Initialize worker thread
        self.ai_thread = QThread()
        self.ai_worker = AIWorker(self.ai_controller)
        self.ai_worker.moveToThread(self.ai_thread)

        # Connect worker signals (legacy and new streaming)
        self.ai_worker.finished.connect(self.on_ai_response_received)
        self.ai_worker.error_occurred.connect(self.on_ai_streaming_error)
        self.ai_worker.stream_started.connect(self.on_stream_started)
        self.ai_worker.status_changed.connect(self.on_status_changed)
        self.ai_worker.token_received.connect(self.on_token_received)
        self.ai_worker.stream_finished.connect(self.on_stream_finished)
        self.ai_worker.progress.connect(self.update_progress)
        self.ai_worker.prompt_received.connect(self.on_prompt_received)
        self.ai_worker.reasoning_received.connect(self.on_reasoning_received)
        self.ai_worker.tool_started.connect(self.on_tool_started)
        self.ai_worker.tool_finished.connect(self.on_tool_finished)

        # Connect streaming request signal to worker
        self.start_streaming_requested.connect(self.ai_worker.stream_message)

        # Start the worker thread
        self.ai_thread.start()

        # Load and apply settings before creating GUI components
        self.load_and_apply_settings()

        self.init_ui()
        self.init_connections()

        # Load conversation history after UI is initialized
        self.load_conversation_history_on_startup()

    def load_conversation_history_on_startup(self):
        """Load conversation history from default location on startup."""
        try:
            # Try to load from default conversation file
            default_conversation_path = Path("./data/conversation.json")
            if default_conversation_path.exists():
                logger.info("Loading conversation history from default location")
                self.ai_controller.load_conversation_history(str(default_conversation_path))

                # Initialize UI with lazy loading state
                total_count = self.ai_controller.get_total_message_count()
                self.message_list.initialize_lazy_loading(total_count)

                # Load initial page of recent messages for immediate display
                if total_count > 0:
                    initial_page_size = 20  # Load last 20 messages initially
                    # For initial load, always get the most recent page (page 0)
                    recent_messages = self.ai_controller.get_messages_page(0, initial_page_size)

                    if recent_messages:
                        logger.info(f"Loading {len(recent_messages)} recent messages for initial display")

                        # Convert to UI format and add IDs
                        ui_messages = []
                        for msg in recent_messages:
                            ui_message = {
                                'id': str(uuid.uuid4()),
                                'role': msg.get('role', 'user'),
                                'content': msg.get('content', ''),
                                'timestamp': msg.get('timestamp', '')
                            }
                            ui_messages.append(ui_message)

                        # Add messages to UI (append at bottom for recent messages)
                        self.message_list.load_message_page(ui_messages, prepend=False)

                        logger.info(f"Displayed {len(ui_messages)} recent messages on startup")
                    else:
                        logger.warning("No recent messages found despite total_count > 0")

                # Restore current draft to input field if it exists
                current_draft = self.ai_controller.get_current_draft()
                if current_draft:
                    logger.info(f"Restoring draft to input field: {current_draft[:50]}...")
                    # Use a queued connection to ensure UI is ready
                    from PyQt6.QtCore import QTimer
                    QTimer.singleShot(100, lambda: self.message_list.set_input_value(current_draft))

                logger.info(f"Loaded {total_count} messages from conversation history (lazy loading enabled)")
            else:
                logger.info("No default conversation file found, starting with empty conversation")
                # Initialize with zero messages
                self.message_list.initialize_lazy_loading(0)

        except Exception as e:
            logger.error(f"Failed to load conversation history on startup: {e}")
            # Continue with empty conversation if loading fails
            self.message_list.initialize_lazy_loading(0)

    def _load_openai_api_key(self) -> Optional[str]:
        """Load OpenAI API key from plain settings."""
        try:
            api_key = self.plain_settings.get_api_key('openai')
            if api_key:
                logger.info("OpenAI API key loaded from plain settings")
                return api_key
            else:
                logger.warning("OpenAI API key not found in plain settings")
        except Exception as e:
            logger.error(f"Failed to load OpenAI API key: {e}")

        logger.warning("No OpenAI API key found, TTS and memory features may not work")
        return None

    def init_ui(self):
        """Initialize the main window UI."""
        self.setWindowTitle("Single-Flow Personal Assistant")
        self.setMinimumSize(800, 600)

        # Load window settings
        window_settings = self.plain_settings.get_setting('window')
        if window_settings:
            width = window_settings.get('width', 1400)
            height = window_settings.get('height', 800)
            maximized = window_settings.get('maximized', True)
            fullscreen = window_settings.get('fullscreen', False)

            if fullscreen:
                # True fullscreen mode (no window borders)
                self.showFullScreen()
            elif maximized:
                # Maximized windowed mode (fills screen with borders)
                screen = QApplication.primaryScreen()
                if screen:
                    screen_geometry = screen.availableGeometry()
                    self.resize(screen_geometry.width(), screen_geometry.height())
                    self.move(screen_geometry.left(), screen_geometry.top())
                else:
                    # Fallback to default size
                    self.resize(width, height)
            else:
                # Custom window size
                self.resize(width, height)
        else:
            # Fallback to default size
            self.resize(1400, 800)

        # Create menu bar
        self.create_menu_bar()

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create splitter for resizable panels
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side panel (collapsible)
        self.side_panel = SidePanelWidget(self.ai_controller)
        self.splitter.addWidget(self.side_panel)

        # Main content area (messages + input)
        main_content = QWidget()
        main_content_layout = QVBoxLayout(main_content)
        main_content_layout.setContentsMargins(0, 0, 0, 0)

        # Message list (now includes input bar and process indicator)
        self.message_list = ChatWebWidget()
        main_content_layout.addWidget(self.message_list)

        self.splitter.addWidget(main_content)

        # Right side panel (prompt panel, collapsible)
        self.prompt_panel = PromptWebWidget()
        self.splitter.addWidget(self.prompt_panel)

        # Set splitter proportions (both side panels start collapsed)
        self.splitter.setSizes([0, self.width(), 0])  # Both panels collapsed
        self.splitter.setCollapsible(0, True)  # Left panel collapsible
        self.splitter.setCollapsible(2, True)  # Right panel collapsible
        self.splitter.setStretchFactor(1, 1)  # Main content stretches

        main_layout.addWidget(self.splitter)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

        # Progress bar for operations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        # Export conversation
        export_action = file_menu.addAction('Export Conversation')
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_conversation)

        # Import conversation
        import_action = file_menu.addAction('Import Conversation')
        import_action.setShortcut('Ctrl+I')
        import_action.triggered.connect(self.import_conversation)

        file_menu.addSeparator()

        # Clear conversation
        clear_action = file_menu.addAction('Clear Conversation')
        clear_action.triggered.connect(self.clear_conversation)

        file_menu.addSeparator()

        # Exit
        exit_action = file_menu.addAction('Exit')
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)

        # Edit menu
        edit_menu = menubar.addMenu('Edit')

        # Search messages
        search_action = edit_menu.addAction('Search Messages')
        search_action.setShortcut('Ctrl+F')
        search_action.triggered.connect(self.search_messages)

        # View menu
        view_menu = menubar.addMenu('View')

        # Toggle left side panel
        toggle_panel_action = view_menu.addAction('Toggle Side Panel')
        toggle_panel_action.setShortcut('Ctrl+B')
        toggle_panel_action.triggered.connect(self.toggle_side_panel)

        # Toggle right prompt panel
        toggle_prompt_panel_action = view_menu.addAction('Toggle Prompt Panel')
        toggle_prompt_panel_action.triggered.connect(self.toggle_prompt_panel)

        # Developer Tools
        view_menu.addSeparator()
        dev_tools_action = view_menu.addAction('Developer Tools')
        dev_tools_action.setShortcut('F12')
        dev_tools_action.triggered.connect(self.open_developer_tools)

    def init_connections(self):
        """Initialize signal connections."""
        # Connect ChatWebWidget signals
        self.message_list.input_received.connect(self.on_send_message)
        self.message_list.audio_toggled.connect(self.on_toggle_audio)
        self.message_list.open_audio_settings_requested.connect(self.on_open_audio_settings)
        self.message_list.ui_ready.connect(self.on_chat_ui_ready)
        self.message_list.load_more_messages_requested.connect(self.on_load_more_messages)

        self.message_list.show_prompt_requested.connect(self.on_message_show_prompt)
        self.message_list.speak_text_requested.connect(self.on_message_speak_text)
        self.message_list.regenerate_requested.connect(self.on_message_regenerate)
        self.message_list.edit_requested.connect(self.on_message_edit)
        self.message_list.audio_control_received.connect(self.on_audio_control)
        self.message_list.draft_action_received.connect(self.on_draft_action)

        # Connect audio provider status signals to UI
        if self.ai_controller.audio_provider:
            self.ai_controller.audio_provider.playback_state_changed.connect(self.on_audio_state_changed)
            self.ai_controller.audio_provider.position_changed.connect(self.on_audio_position_changed)
            self.ai_controller.audio_provider.duration_changed.connect(self.on_audio_duration_changed)

        # Connect prompt panel editing signal
        self.prompt_panel.edit_requested.connect(self.on_edit_system_prompt)

        # Keyboard shortcut for side panel toggle is already set on the menu action

        # Add keyboard shortcut to toggle prompt panel (Ctrl+P)
        toggle_prompt_shortcut = QShortcut(QKeySequence("Ctrl+P"), self)
        toggle_prompt_shortcut.activated.connect(self.toggle_prompt_panel)

    def load_and_apply_settings(self):
        """Load settings from plain storage and apply them to the controller on startup."""
        try:
            settings = self.plain_settings.load_settings()
            logger.info("Settings loaded from plain storage")

            # Get API keys
            api_keys = settings.get('api_keys', {})
            openai_api_key = api_keys.get('openai', '')
            openrouter_api_key = api_keys.get('openrouter', '')

            # Update API keys in controller (this will re-initialize memory and audio providers)
            self.ai_controller.update_api_keys(
                openai_api_key=openai_api_key if openai_api_key else None,
                openrouter_api_key=openrouter_api_key if openrouter_api_key else None
            )

            # Apply embeddings settings and reconfigure file ingestor
            embeddings_settings = settings.get('embeddings', {})
            embedding_provider = embeddings_settings.get('provider', 'openrouter')
            embedding_model = embeddings_settings.get('model', 'text-embedding-3-small')

            # Get embedding API key
            embedding_api_key = api_keys.get(embedding_provider, '')

            # Get vector store settings
            vector_store_settings = settings.get('vector_store', {})
            vector_provider = vector_store_settings.get('provider', 'faiss')

            # Reconfigure file ingestor with correct settings
            self.ai_controller.reconfigure_file_ingestor(
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                api_key=embedding_api_key if embedding_api_key else None,
                vector_provider=vector_provider
            )

            # Apply LLM settings
            llm_settings = settings.get('llm', {})
            provider = llm_settings.get('provider', 'openrouter')

            # Get API key for LLM provider
            api_key = api_keys.get(provider, '')

            config = {'api_key': api_key if api_key else None}
            if llm_settings.get('model'):
                config['model'] = llm_settings['model']
            if 'temperature' in llm_settings:
                config['temperature'] = llm_settings['temperature']

            # Configure intent analysis
            intent_settings = settings.get('intent_analysis', {})
            logger.info(f"Intent analysis settings: enabled={intent_settings.get('enabled')}, combo={intent_settings.get('combo')}")
            intent_config = None
            if intent_settings.get('enabled'):
                intent_config = {
                    'enabled': True,
                    'combo': intent_settings.get('combo'),  # Use combo ID instead of direct provider/model
                    'api_key': api_keys.get('openrouter', '')  # Use default provider API key
                }
                logger.info(f"Created intent_config: {intent_config}")
            else:
                logger.info("Intent analysis disabled in settings")

            self.ai_controller.configure_llm(provider, intent_config=intent_config, **config)

            # Apply retrieval settings
            retrieval_settings = settings.get('retrieval', {})
            self.ai_controller.toggle_retrieval(retrieval_settings.get('enabled', True))
            self.ai_controller.set_max_retrieved_docs(retrieval_settings.get('max_docs', 3))

            # Apply hybrid retrieval settings
            method = retrieval_settings.get('method', 'semantic')
            is_hybrid = method == 'hybrid'
            semantic_weight = retrieval_settings.get('semantic_weight', 0.7)
            keyword_weight = retrieval_settings.get('keyword_weight', 0.3)
            self.ai_controller.set_hybrid_retrieval(is_hybrid, semantic_weight, keyword_weight)

            # Apply memory settings
            memory_settings = settings.get('memory', {})
            self.ai_controller.toggle_memory(memory_settings.get('enabled', True))
            self.ai_controller.set_max_retrieved_memories(memory_settings.get('max_memories', 3))

            # Apply audio settings
            audio_settings = settings.get('audio', {})
            self.ai_controller.configure_audio(
                tts_enabled=audio_settings.get('tts_enabled', True),
                asr_enabled=audio_settings.get('asr_enabled', True),
                voice=audio_settings.get('tts_voice', 'alloy'),
                speed=audio_settings.get('tts_speed', 1.0),
                language=audio_settings.get('asr_language', 'en-US')
            )

            logger.info("Settings loaded and applied to controller successfully")

        except Exception as e:
            logger.error(f"Failed to load and apply settings: {e}")
            # Continue with default settings if loading fails

    def on_toggle_audio(self):
        """Handle audio recording toggle."""
        # This connects to the AIController's audio status if available
        # or initiates voice input flow.
        logger.info("Toggle audio recording requested")
        
        # Check if we are currently recording (need a state flag or check controller)
        if hasattr(self, 'is_recording') and self.is_recording:
            # Stop recording
            self.stop_voice_input()
        else:
            # Start recording
            self.start_voice_input()

    def start_voice_input(self):
        """Start voice input recording."""
        self.is_recording = True
        self.message_list.update_audio_status(True)
        self.status_label.setText("Listening...")
        
        # Use QTimer to simulate or interact with AIController/AudioProvider
        # If AIController has async audio handling:
        # asyncio.create_task(self.ai_controller.start_listening()) 
        # For now, we simulate success after 3 seconds for testing UI flow
        # In real implementation, this would trigger actual ASR
        QTimer.singleShot(3000, self.finish_voice_input_simulation)

    def stop_voice_input(self):
        """Stop voice input recording."""
        self.is_recording = False
        self.message_list.update_audio_status(False)
        self.status_label.setText("Ready")

    def finish_voice_input_simulation(self):
        """Simulate finishing voice input."""
        if hasattr(self, 'is_recording') and self.is_recording:
            self.stop_voice_input()
            # Simulate transcribed text
            transcribed_text = "This is a simulated voice input."
            self.on_send_message(transcribed_text)

    def on_open_audio_settings(self):
        """Open audio settings."""
        logger.info("Open audio settings requested")
        # Switch side panel to settings tab via JS
        self.side_panel.web_view.run_js("switchMainTab('settings'); settingsSwitchTab('audio');")
        if self.side_panel.is_collapsed:
            self.toggle_side_panel()

    def on_send_message(self, message: str):
        """Handle sending a message."""
        # Parse combo and message content
        combo_id = None
        message_content = message

        # Check if message contains combo separator
        if '|' in message:
            parts = message.split('|', 1)
            if len(parts) == 2:
                combo_id = parts[0] if parts[0].strip() else None
                message_content = parts[1]

        logger.info(f"Sending message: {message_content[:50]}... (combo: {combo_id})")

        # Add user message to timeline
        user_message = {
            'role': 'user',
            'content': message_content,
            'timestamp': datetime.now().strftime("%H:%M")
        }
        self.message_list.add_message(user_message)

        # Disable input while processing
        self.message_list.set_input_enabled(False)
        self.message_list.set_process_state("thinking", "Initializing...")

        self.status_label.setText("Processing...")
        self.show_progress(True)

        # Process message with AI controller in a separate thread
        self.process_message_async(message_content, combo_id)

    def process_message_async(self, message: str, combo_id: Optional[str] = None):
        """Process message asynchronously using worker thread with streaming."""
        # Use streaming method for new behavior via signal/slot to run in worker thread
        # Pass combo_id to worker if provided
        if combo_id:
            # Emit with combo information
            self.start_streaming_requested.emit(f"{combo_id}|{message}")
        else:
            self.start_streaming_requested.emit(message)

    def on_ai_response_received(self, response: str):
        """Handle successful AI response from worker thread."""
        # Add assistant response to timeline
        assistant_message = {
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().strftime("%H:%M")
        }
        self.message_list.add_message(assistant_message)

        self.status_label.setText("Ready")
        self.message_list.set_process_state("idle")

        # Re-enable input
        self.message_list.set_input_enabled(True)
        self.show_progress(False)

        # Archive current draft if it exists (mark as sent) for legacy non-streaming responses
        if self.ai_controller.current_draft_id:
            logger.info(f"Archiving draft after successful message send (legacy): {self.ai_controller.current_draft_id[:8]}...")
            self.ai_controller.archive_current_draft()

    def on_ai_error_occurred(self, error_message: str):
        """Handle error from AI worker thread."""
        logger.error(f"AI processing error: {error_message}")

        # Add error message to timeline
        error_response = {
            'role': 'assistant',
            'content': f"Sorry, I encountered an error: {error_message}. Please check your API configuration.",
            'timestamp': datetime.now().strftime("%H:%M")
        }
        self.message_list.add_message(error_response)
        self.status_label.setText("Error - Check configuration")
        self.message_list.set_process_state("idle")

        # Re-enable input
        self.message_list.set_input_enabled(True)
        self.show_progress(False)

    # Streaming signal handlers
    def on_stream_started(self):
        """Handle the start of a streaming response."""
        logger.info("Streaming started")
        # Create an empty assistant message for streaming into
        self.streaming_message_data = {
            'role': 'assistant',
            'content': '',
            'reasoning': '',
            'timestamp': 'now'  # TODO: Add proper timestamp
        }
        self.message_list.add_message(self.streaming_message_data)

    def on_status_changed(self, status_type: str, message: str):
        """Handle status updates from streaming."""
        logger.info(f"Status changed: {status_type} - {message}")
        if status_type == "processing":
            self.status_label.setText(f"Processing: {message}")

            # Map status messages to animation states
            animation_state = None
            if message in ["Initializing processing...", "Analyzing user intent...", "Retrieving relevant context...", "Building optimized prompt..."]:
                animation_state = "thinking"
            elif "Executed" in message and "tool" in message:
                animation_state = "tool_execution"
            elif "Generation" in message or "Streaming" in message or "Generating response..." in message:
                animation_state = "streaming"

            if animation_state:
                self.message_list.set_process_state(animation_state, message)

    def on_token_received(self, token: str):
        """Handle incoming streaming tokens."""
        logger.debug(f"Token received: {token}")
        # Append the token to the current streaming message
        if hasattr(self, 'streaming_message_data'):
            self.streaming_message_data['content'] += token
            # Update the streaming message UI
            self._update_streaming_message_content()

    def on_reasoning_received(self, reasoning: str):
        """Handle incoming reasoning tokens."""
        # logger.debug(f"Reasoning received: {reasoning[:20]}...")
        if hasattr(self, 'streaming_message_data'):
            # Initialize reasoning if not present
            if 'reasoning' not in self.streaming_message_data:
                self.streaming_message_data['reasoning'] = ''
            
            # Append reasoning
            self.streaming_message_data['reasoning'] += reasoning
            
            # Update UI state to show reasoning is happening
            if not getattr(self, '_is_reasoning_animating', False):
                self.message_list.set_process_state("reasoning", "Thinking...")
                self._is_reasoning_animating = True
            
            # Update the message content (which includes reasoning)
            self._update_streaming_message_content()

    def on_stream_finished(self):
        """Handle the end of streaming."""
        logger.info("Streaming finished")
        self.status_label.setText("Ready")

        # Reset process indicator to idle
        self.message_list.set_process_state("idle")

        # Reset reasoning animation flag
        self._is_reasoning_animating = False

        # Re-enable input
        self.message_list.set_input_enabled(True)
        self.show_progress(False)

        # Archive current draft if it exists (mark as sent)
        if self.ai_controller.current_draft_id:
            logger.info(f"Archiving draft after successful message send: {self.ai_controller.current_draft_id[:8]}...")
            self.ai_controller.archive_current_draft()

        # Finalize the streaming message with gallery processing
        if hasattr(self, 'streaming_message_data'):
            # Apply citations and final formatting
            content = self.streaming_message_data['content']
            # Make sure citations are styled (though this could be done progressively)
            import re
            def style_citation(match):
                citation_num = match.group(1)
                return f'<span style="color: #3498db; font-weight: bold; text-decoration: underline;">[{citation_num}]</span>'
            styled_content = re.sub(r'\[(\d+)\]', style_citation, content)
            self.streaming_message_data['content'] = styled_content

            # Update the message with final content to trigger gallery finalization
            final_message_data = self.streaming_message_data.copy()
            final_message_data['content'] = styled_content
            self.message_list.update_message(final_message_data)

            # Clear the streaming state
            delattr(self, 'streaming_message_data')

    def _update_streaming_message_content(self):
        """Update the streaming message widget with new content."""
        if not hasattr(self, 'streaming_message_data'):
            return

        try:
            content = self.streaming_message_data.get('content', '')
            reasoning = self.streaming_message_data.get('reasoning', '')
            # Only pass reasoning if it's not empty, otherwise None
            reasoning_arg = reasoning if reasoning else None
            
            self.message_list.update_streaming_message(content, reasoning_arg)
        except Exception as e:
            logger.error(f"Error updating streaming message content: {e}")

    def on_prompt_received(self, prompt_data: dict):
        """Handle received structured prompt."""
        logger.info("Received structured prompt update")
        if hasattr(self, 'prompt_panel'):
            self.prompt_panel.set_current_prompt(prompt_data)
            # Auto-expand if collapsed
            # current_sizes = self.splitter.sizes()
            # if current_sizes[2] == 0:
            #     self.toggle_prompt_panel()

    def on_tool_started(self, tool_name: str, input_args: dict):
        """Handle tool start event."""
        logger.info(f"Tool started: {tool_name}")
        self.message_list.set_process_state("tool_execution", f"Using tool: {tool_name}...")
        
    def on_tool_finished(self, tool_name: str, output: str):
        """Handle tool finish event."""
        logger.info(f"Tool finished: {tool_name}")

        # Don't display web search tool results as separate messages to avoid duplication
        # The assistant will incorporate images and results into its final response
        if tool_name in ['tavily_search', 'exa_search', 'jina_search']:
            logger.info(f"Skipping display of {tool_name} results to prevent duplication - images will be shown in assistant response")
        else:
            # Display tool output as a visible message for other tools that provide user-visible results
            tool_message = {
                'role': 'system',
                'content': output,
                'timestamp': datetime.now().strftime("%H:%M"),
                'tool_name': tool_name
            }
            self.message_list.add_message(tool_message)
            logger.info(f"Displayed {tool_name} results in chat")

        # Continue with processing
        self.message_list.set_process_state("thinking", "Processing results...")

    def on_ai_streaming_error(self, error_message: str):
        """Handle streaming error."""
        logger.error(f"Streaming error: {error_message}")

        # Re-enable input
        self.message_list.set_input_enabled(True)
        self.message_list.set_process_state("idle")
        self.show_progress(False)

        # Add error message to timeline
        error_response = {
            'role': 'assistant',
            'content': f"Streaming error: {error_message}. Please check your API configuration.",
            'timestamp': datetime.now().strftime("%H:%M")
        }
        self.message_list.add_message(error_response)
        self.status_label.setText("Error - Check configuration")

    def toggle_side_panel(self):
        """Toggle the side panel visibility."""
        if self.side_panel.is_collapsed:
            # Expand
            self.splitter.setSizes([200, self.width() - 200])
            self.side_panel.is_collapsed = False
        else:
            # Collapse
            self.splitter.setSizes([0, self.width()])
            self.side_panel.is_collapsed = True

    def toggle_prompt_panel(self):
        """Toggle the prompt panel visibility."""
        current_sizes = self.splitter.sizes()
        if current_sizes[2] == 0:  # Panel is collapsed
            # Expand
            self.splitter.setSizes([current_sizes[0], current_sizes[1] - 300, 300])
        else:
            # Collapse
            self.splitter.setSizes([current_sizes[0], current_sizes[1] + current_sizes[2], 0])

    def show_progress(self, show: bool = True):
        """Show or hide the progress bar."""
        self.progress_bar.setVisible(show)
        if not show:
            self.progress_bar.setValue(0)

    def update_progress(self, value: int):
        """Update the progress bar value."""
        self.progress_bar.setValue(value)

    def on_message_regenerate(self, message_data: Dict[str, Any]):
        """Handle message regeneration request."""
        logger.info("Regenerating message response")

        # Find the message in the conversation and get the user message that prompted it
        message_index = None
        user_message = None

        for i, msg in enumerate(self.message_list.messages):
            if msg.get('id') == message_data.get('id'):
                message_index = i
                break

        # Find the preceding user message
        if message_index is not None:
            for i in range(message_index - 1, -1, -1):
                if self.message_list.messages[i].get('role') == 'user':
                    user_message = self.message_list.messages[i]
                    break

        if user_message:
            # Remove the old assistant message
            self.message_list.remove_message(message_data)

            # Re-send the user message to get a new response
            self.on_send_message(user_message['content'])
        else:
            logger.warning("Could not find user message to regenerate response")

    def on_message_edit(self, message_data: Dict[str, Any]):
        """Handle message edit request."""
        logger.info("Message edited")
        # For now, just log the edit. The message content has already been updated in the widget
        # In a full implementation, this might trigger re-processing if it was an assistant message

    def on_message_speak_text(self, message_data: Dict[str, Any]):
        """Handle message speak text request."""
        logger.info("Speaking message text")
        content = message_data.get('content', '')
        
        if not content:
            logger.warning("No content to speak in message")
            return
            
        # Create worker for TTS generation
        self.tts_worker = TTSWorker(self.ai_controller, content)
        
        def on_tts_finished(file_path: str):
            logger.info(f"TTS audio generated: {file_path}")
            # Play the audio file asynchronously in the main thread
            self.ai_controller.play_audio_file(file_path)
            
        def on_tts_error(error: str):
            logger.error(f"TTS generation failed: {error}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "TTS Error", f"Failed to speak message: {error}")
            
        self.tts_worker.finished.connect(on_tts_finished)
        self.tts_worker.error_occurred.connect(on_tts_error)
        
        # Start the worker and keep a reference to prevent GC
        self.tts_worker.start()

    def on_audio_control(self, command: str, value: float):
        """Handle audio control requests from UI."""
        if command == 'pause':
            self.ai_controller.pause_audio()
        elif command == 'resume':
            self.ai_controller.resume_audio()
        elif command == 'stop':
            self.ai_controller.stop_audio()
        elif command == 'seek':
            self.ai_controller.seek_audio(int(value))

    def on_audio_state_changed(self, state: str):
        """Handle audio playback state changes."""
        # We need to maintain state locally to send full update
        if not hasattr(self, '_audio_state'):
            self._audio_state = {'state': 'stopped', 'position': 0, 'duration': 0}
        self._audio_state['state'] = state
        self.message_list.update_audio_player_state(state, self._audio_state['position'], self._audio_state['duration'])

    def on_audio_position_changed(self, position: int):
        """Handle audio position changes."""
        if not hasattr(self, '_audio_state'):
            self._audio_state = {'state': 'stopped', 'position': 0, 'duration': 0}
        self._audio_state['position'] = position
        # Optimization: Only send update every 250ms or so if needed, but for now direct
        self.message_list.update_audio_player_state(self._audio_state['state'], position, self._audio_state['duration'])

    def on_audio_duration_changed(self, duration: int):
        """Handle audio duration changes."""
        if not hasattr(self, '_audio_state'):
            self._audio_state = {'state': 'stopped', 'position': 0, 'duration': 0}
        self._audio_state['duration'] = duration
        self.message_list.update_audio_player_state(self._audio_state['state'], self._audio_state['position'], duration)

    def on_draft_action(self, action: str, content: str):
        """Handle draft actions from UI."""
        logger.debug(f"Draft action received: {action}")
        if action == 'save':
            self.ai_controller.save_draft(content)
        elif action == 'archive':
            self.ai_controller.archive_current_draft()
        else:
            logger.warning(f"Unknown draft action: {action}")

    def on_chat_ui_ready(self):
        """Handle chat UI ready event - load combos now that UI is initialized."""
        logger.info("Chat UI ready, loading combos...")
        try:
            # Get default combo setting
            settings = self.plain_settings.load_settings()
            default_llm_combo = settings.get('llm', {}).get('combo', '')

            combos = self.plain_settings.get_combos()
            if combos:
                self.message_list.load_combos(combos, default_llm_combo)
                logger.info(f"Loaded {len(combos)} combos into chat UI after initialization with default: {default_llm_combo}")
            else:
                logger.info("No combos found to load")
        except Exception as e:
            logger.error(f"Failed to load combos on UI ready: {e}")

    def on_edit_system_prompt(self):
        """Handle system prompt edit request."""
        logger.info("Opening system prompt editor")
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Prompt Editor",
                              "System Prompt editing is not yet implemented in this version.\n"
                              "Please update the 'System Info' directly in the configuration file or wait for future updates.")

    def on_message_show_prompt(self, message_data: Dict[str, Any]):
        """Handle message show prompt request."""
        logger.info("Showing prompt for message")

        # If it's an assistant message, we need to find the preceding user message
        # because the prompt is associated with the user input that generated it.
        target_message = message_data
        if message_data.get('role') == 'assistant':
             # Find the message in the valid list
            message_index = None
            for i, msg in enumerate(self.message_list.messages):
                if msg.get('id') == message_data.get('id'):
                    message_index = i
                    break
            
            if message_index is not None:
                # Look backwards for the user message
                for i in range(message_index - 1, -1, -1):
                    if self.message_list.messages[i].get('role') == 'user':
                        target_message = self.message_list.messages[i]
                        logger.info(f"Resolved assistant message to user message id: {target_message.get('id')}")
                        break

        # Get the prompt for this message from the controller
        try:
            prompt = self.ai_controller.get_prompt_for_message(target_message)
            if prompt:
                # Set prompt on web widget
                self.prompt_panel.set_current_prompt({
                    'structured_prompt': prompt,
                    'user_input': message_data.get('content', ''),
                    'timestamp': message_data.get('timestamp', '')
                })

                # Auto-expand the prompt panel if it's collapsed
                current_sizes = self.splitter.sizes()
                if current_sizes[2] == 0:  # Panel is collapsed
                    self.toggle_prompt_panel()
            else:
                logger.warning("No prompt found for message")
                self.prompt_panel.display_prompt("No prompt available for this message.")
                if self.splitter.sizes()[2] == 0:  # Panel is collapsed
                    self.toggle_prompt_panel()
        except Exception as e:
            logger.error(f"Failed to get prompt for message: {e}")
            self.prompt_panel.display_prompt(f"Error retrieving prompt: {str(e)}")
            if self.splitter.sizes()[2] == 0:  # Panel is collapsed
                self.toggle_prompt_panel()

    def export_conversation(self):
        """Export the current conversation to a file."""
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Conversation", "", "JSON Files (*.json);;Text Files (*.txt);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Prepare conversation data
            conversation_data = {
                'metadata': {
                    'exported_at': 'now',  # TODO: Add proper timestamp
                    'message_count': len(self.message_list.messages),
                    'version': '0.2.0'
                },
                'messages': self.message_list.messages
            }

            # Export based on file extension
            if file_path.endswith('.txt'):
                # Export as plain text
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Single-Flow Personal Assistant Conversation Export\n")
                    f.write("=" * 50 + "\n\n")
                    for msg in self.message_list.messages:
                        role = "Assistant" if msg.get('role') == 'assistant' else "You"
                        content = msg.get('content', '')
                        f.write(f"{role}: {content}\n\n")
            else:
                # Export as JSON
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(conversation_data, f, indent=2, ensure_ascii=False)

            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Export Successful", f"Conversation exported to {file_path}")

        except Exception as e:
            logger.error(f"Failed to export conversation: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Export Error", f"Failed to export conversation: {str(e)}")

    def import_conversation(self):
        """Import a conversation from a file."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Conversation", "", "JSON Files (*.json);;Text Files (*.txt);;All Files (*)"
        )

        if not file_path:
            return

        try:
            if file_path.endswith('.txt'):
                # Import from text file (basic parsing)
                QMessageBox.warning(self, "Not Implemented", "Text file import is not yet implemented.")
                return
            else:
                # Import from JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                messages = data.get('messages', [])
                if not messages:
                    QMessageBox.warning(self, "Import Error", "No messages found in the file.")
                    return

                # Confirm import
                reply = QMessageBox.question(
                    self, "Import Conversation",
                    f"Import {len(messages)} messages? This will replace the current conversation.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )

                if reply == QMessageBox.StandardButton.Yes:
                    # Clear current conversation
                    self.message_list.clear_messages()

                    # Add imported messages
                    for msg in messages:
                        self.message_list.add_message(msg)

                    QMessageBox.information(self, "Import Successful", f"Imported {len(messages)} messages.")

        except Exception as e:
            logger.error(f"Failed to import conversation: {e}")
            QMessageBox.warning(self, "Import Error", f"Failed to import conversation: {str(e)}")

    def clear_conversation(self):
        """Clear all messages from the conversation."""
        from PyQt6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self, "Clear Conversation",
            "Are you sure you want to clear all messages? This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.message_list.clear_messages()
            # Clear conversation memory in controller
            self.ai_controller.clear_conversation_history()
            logger.info("Conversation cleared")

    def search_messages(self):
        """Open search dialog for messages."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QListWidget, QListWidgetItem, QLabel

        # Create search dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Search Messages")
        dialog.setMinimumSize(500, 400)

        layout = QVBoxLayout(dialog)

        # Search input
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search terms...")
        self.search_input.returnPressed.connect(lambda: self.perform_message_search(dialog))
        search_layout.addWidget(self.search_input)

        search_button = QPushButton("Search")
        search_button.clicked.connect(lambda: self.perform_message_search(dialog))
        search_layout.addWidget(search_button)

        layout.addLayout(search_layout)

        # Results list
        self.search_results = QListWidget()
        layout.addWidget(self.search_results)

        # Buttons
        buttons_layout = QHBoxLayout()

        view_button = QPushButton("View Message")
        view_button.clicked.connect(lambda: self.view_searched_message(dialog))
        buttons_layout.addWidget(view_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        buttons_layout.addWidget(close_button)

        layout.addLayout(buttons_layout)

        dialog.exec()

    def perform_message_search(self, dialog):
        """Perform search on messages."""
        query = self.search_input.text().strip().lower()
        if not query:
            return

        self.search_results.clear()

        for i, msg in enumerate(self.message_list.messages):
            content = msg.get('content', '').lower()
            if query in content:
                role = "Assistant" if msg.get('role') == 'assistant' else "You"
                preview = content[:100] + "..." if len(content) > 100 else content

                item_text = f"#{i+1} {role}: {preview}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, i)  # Store message index
                self.search_results.addItem(item)

        if self.search_results.count() == 0:
            self.search_results.addItem(QListWidgetItem("No messages found matching the search."))

    def view_searched_message(self, dialog):
        """View the selected message from search results."""
        current_item = self.search_results.currentItem()
        if not current_item:
            return

        message_index = current_item.data(Qt.ItemDataRole.UserRole)
        if message_index is None:
            return

        # Scroll to the message in the main window
        # For now, just close the dialog
        dialog.accept()

        # TODO: Implement scrolling to specific message

    def open_developer_tools(self):
        """Open developer tools for the chat web view."""
        logger.info("Opening developer tools for chat web view")
        if hasattr(self.message_list, 'open_dev_tools'):
            self.message_list.open_dev_tools()
        else:
            logger.warning("Chat web view does not have open_dev_tools method")

    def closeEvent(self, event: QCloseEvent):
        """Handle application close event to properly clean up threads and save conversation."""
        logger.info("Application closing, saving conversation and cleaning up threads...")

        # Save conversation before closing
        try:
            default_conversation_path = Path("./data/conversation.json")
            default_conversation_path.parent.mkdir(parents=True, exist_ok=True)
            self.ai_controller.save_conversation_history(str(default_conversation_path))
            logger.info("Conversation saved successfully")
        except Exception as e:
            logger.error(f"Failed to save conversation on close: {e}")

        # Stop the worker thread properly
        if hasattr(self, 'ai_thread') and self.ai_thread.isRunning():
            # Cancel any ongoing asyncio processing first
            logger.info("Cancelling AI worker processing...")
            self.ai_worker.cancel_processing()

            # Disconnect signals to prevent crashes during shutdown
            self.ai_worker.finished.disconnect()
            self.ai_worker.error_occurred.disconnect()
            self.ai_worker.progress.disconnect()

            # Request thread to quit and wait for it to finish
            self.ai_thread.quit()
            self.ai_thread.wait(3000)  # Wait up to 3 seconds

            if self.ai_thread.isRunning():
                logger.warning("Thread did not stop gracefully, terminating")
                self.ai_thread.terminate()
                self.ai_thread.wait(1000)

        # Cancel any remaining tasks if the loop still exists
        if hasattr(self, 'ai_worker') and self.ai_worker.loop and not self.ai_worker.loop.is_closed():
            try:
                # Cancel all pending tasks
                pending_tasks = [task for task in asyncio.all_tasks(self.ai_worker.loop) if not task.done()]
                if pending_tasks:
                    logger.info(f"Cancelling {len(pending_tasks)} pending asyncio tasks")
                    for task in pending_tasks:
                        task.cancel()
                    # Give a moment for cancellation to complete
                    try:
                        self.ai_worker.loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
                    except Exception:
                        pass  # Ignore errors during shutdown
            except Exception as e:
                logger.warning(f"Error cancelling tasks: {e}")
            finally:
                try:
                    self.ai_worker.loop.close()
                except Exception as e:
                    logger.warning(f"Error closing event loop: {e}")

    def on_load_more_messages(self):
        """Handle request to load more messages for lazy loading."""
        logger.info("Loading more messages for lazy loading")

        # Get current loaded message count from UI
        loaded_count = len(self.message_list.messages)
        total_count = self.ai_controller.get_total_message_count()

        if loaded_count >= total_count:
            logger.info("All messages already loaded")
            return

        # Load a page of older messages (20 messages at a time)
        page_size = 20
        page = (total_count - loaded_count) // page_size

        try:
            messages = self.ai_controller.get_messages_page(page, page_size)

            if messages:
                logger.info(f"Loaded {len(messages)} older messages (page {page})")

                # Convert to UI format and add IDs
                ui_messages = []
                for msg in messages:
                    ui_message = {
                        'id': str(uuid.uuid4()),
                        'role': msg.get('role', 'user'),
                        'content': msg.get('content', ''),
                        'timestamp': msg.get('timestamp', '')
                    }
                    ui_messages.append(ui_message)

                # Add messages to UI (prepended at top)
                self.message_list.load_message_page(ui_messages, prepend=True)

                logger.info(f"Added {len(ui_messages)} messages to UI")
            else:
                logger.info("No more messages to load")

        except Exception as e:
            logger.error(f"Failed to load more messages: {e}")


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Single-Flow Personal Assistant")
    app.setApplicationVersion("0.2.0")
    app.setOrganizationName("Personal Assistant")

    # Create and show main window
    window = MainWindow()
    window.show()

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
