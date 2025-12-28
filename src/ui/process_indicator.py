"""
Process Indicator Widget - Visual animations for AI processing phases.
Shows icons and animations for thinking, tool calls, and streaming phases.
"""

from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QFrame, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QTimer, QPoint, QPropertyAnimation, QSequentialAnimationGroup,
    QParallelAnimationGroup, pyqtSignal, QEasingCurve
)
from PyQt6.QtGui import QPalette, QColor, QFont, QFontDatabase
from PyQt6.QtWidgets import QGraphicsOpacityEffect

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class ProcessIndicatorWidget(QWidget):
    """Widget displaying animated indicators for different AI processing phases."""

    # Animation states
    STATE_IDLE = "idle"
    STATE_THINKING = "thinking"
    STATE_TOOL_EXEC = "tool_execution"
    STATE_STREAMING = "streaming"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_state = self.STATE_IDLE
        self.thinking_dots = 0
        self.streaming_phase = 0

        # Animation timers
        self.thinking_timer = QTimer(self)
        self.thinking_timer.timeout.connect(self._update_thinking_animation)
        self.thinking_timer.setInterval(500)  # 500ms interval

        self.streaming_timer = QTimer(self)
        self.streaming_timer.timeout.connect(self._update_streaming_animation)
        self.streaming_timer.setInterval(200)  # 200ms for wave effect

        # Animation objects
        self.icon_animation = None
        self.background_animation = None

        self.init_ui()
        logger.info("Process indicator widget initialized")

    def init_ui(self):
        """Initialize the widget UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(8)

        # Main container frame
        self.container = QFrame()
        self.container.setFrameStyle(QFrame.Shape.Box)
        self.container.setLineWidth(1)
        self.container.setStyleSheet("""
            QFrame {
                border: 1px solid #bdc3c7;
                border-radius: 8px;
                background-color: rgba(255, 255, 255, 0.9);
            }
        """)
        self.container.setVisible(False)  # Hidden by default

        container_layout = QHBoxLayout(self.container)
        container_layout.setContentsMargins(8, 4, 8, 4)
        container_layout.setSpacing(6)

        # Icon label
        self.icon_label = QLabel()
        # Use a system font that supports Unicode symbols well
        icon_font = QFont()
        icon_font.setFamily("Segoe UI, Arial")  # System fonts that support Unicode
        icon_font.setPixelSize(18)
        self.icon_label.setFont(icon_font)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setFixedSize(24, 24)

        # Status text label
        self.status_label = QLabel("Processing...")
        self.status_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Loading dots container
        self.dots_container = QWidget()
        dots_layout = QHBoxLayout(self.dots_container)
        dots_layout.setContentsMargins(0, 0, 0, 0)
        dots_layout.setSpacing(2)

        # Three dots for thinking animation
        self.dots = []
        for i in range(3):
            dot = QLabel("·")
            dot.setStyleSheet("color: #3498db; font-size: 18px; opacity: 0.3;")
            dot.setFixedSize(8, 20)
            dots_layout.addWidget(dot)
            self.dots.append(dot)

        # Wave elements for streaming animation
        self.wave_container = QWidget()
        wave_layout = QHBoxLayout(self.wave_container)
        wave_layout.setContentsMargins(0, 0, 0, 0)
        wave_layout.setSpacing(1)

        self.wave_elements = []
        for i in range(5):
            wave = QLabel("∼")
            wave.setStyleSheet("color: #9b59b6; font-size: 16px; opacity: 0.3;")
            wave.setFixedSize(10, 20)
            wave_layout.addWidget(wave)
            self.wave_elements.append(wave)

        # Initially hide animation elements
        self.dots_container.setVisible(False)
        self.wave_container.setVisible(False)

        # Add to container
        container_layout.addWidget(self.icon_label)
        container_layout.addWidget(self.status_label)
        container_layout.addWidget(self.dots_container)
        container_layout.addWidget(self.wave_container)
        container_layout.addStretch()

        layout.addWidget(self.container)
        layout.addStretch()

        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(40)

    def set_state(self, state: str, message: str = ""):
        """
        Set the current animation state.

        Args:
            state: One of STATE_* constants
            message: Optional status message to display
        """
        if state == self.current_state:
            return

        logger.debug(f"Process indicator state change: {self.current_state} → {state}")

        # Stop current animations
        self._stop_all_animations()

        # Update state
        self.current_state = state

        if state == self.STATE_IDLE:
            self._set_idle_state()

        elif state == self.STATE_THINKING:
            self._set_thinking_state(message)

        elif state == self.STATE_TOOL_EXEC:
            self._set_tool_execution_state(message)

        elif state == self.STATE_STREAMING:
            self._set_streaming_state(message)

        else:
            logger.warning(f"Unknown state: {state}")
            self._set_idle_state()

    def _set_idle_state(self):
        """Hide the indicator when idle."""
        self.container.setVisible(False)
        self.thinking_timer.stop()
        self.streaming_timer.stop()

    def _set_thinking_state(self, message: str):
        """Set up thinking animation with rotating dots."""
        self.container.setVisible(True)

        # Set icon and text (using Unicode symbol instead of emoji)
        self.icon_label.setText("●")
        self.status_label.setText(message or "Thinking...")

        # Show dots, hide wave
        self.dots_container.setVisible(True)
        self.wave_container.setVisible(False)

        # Start thinking animation
        self.thinking_dots = 0
        self.thinking_timer.start()

        # Pulse animation for the icon
        self._start_icon_pulse("#3498db")

    def _set_tool_execution_state(self, message: str):
        """Set up tool execution animation with bouncing icon."""
        self.container.setVisible(True)

        # Set icon and text (using Unicode symbol instead of emoji)
        self.icon_label.setText("⟲")
        self.status_label.setText(message or "Searching...")

        # Hide animation elements
        self.dots_container.setVisible(False)
        self.wave_container.setVisible(False)

        # Start bounce animation
        self._start_icon_bounce("#27ae60")

    def _set_streaming_state(self, message: str):
        """Set up streaming animation with wave effect."""
        self.container.setVisible(True)

        # Set icon and text (using Unicode symbol instead of emoji)
        self.icon_label.setText("⚡")
        self.status_label.setText(message or "Streaming...")

        # Show wave, hide dots
        self.dots_container.setVisible(False)
        self.wave_container.setVisible(True)

        # Start streaming animation
        self.streaming_phase = 0
        self.streaming_timer.start()

        # Subtle pulse for streaming icon
        self._start_icon_pulse("#9b59b6")

    def _start_icon_pulse(self, color: str):
        """Start a pulsing animation for the icon using stylesheet opacity."""
        # Remove any existing graphics effect
        if hasattr(self.icon_label, '_opacity_effect'):
            self.icon_label.setGraphicsEffect(None)
            delattr(self.icon_label, '_opacity_effect')

        # Create a simple pulsing effect using stylesheet changes
        # We'll use a timer-based approach instead of QPropertyAnimation
        if not hasattr(self, '_pulse_timer'):
            self._pulse_timer = QTimer(self)
            self._pulse_timer.timeout.connect(lambda: self._update_pulse_animation(color))

        # Initialize pulse state
        self._pulse_opacity = 0.7
        self._pulse_direction = 0.05  # Amount to change opacity each step

        # Set initial style
        self.icon_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 20px;
            }}
        """)

        # Start pulsing timer (60 FPS for smooth animation)
        self._pulse_timer.start(16)  # ~60 FPS

    def _update_pulse_animation(self, color: str):
        """Update the pulsing animation by changing stylesheet opacity."""
        self._pulse_opacity += self._pulse_direction

        # Reverse direction at boundaries
        if self._pulse_opacity >= 1.0:
            self._pulse_opacity = 1.0
            self._pulse_direction = -0.05
        elif self._pulse_opacity <= 0.3:
            self._pulse_opacity = 0.3
            self._pulse_direction = 0.05

        # Update stylesheet with new opacity
        self.icon_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 20px;
                opacity: {self._pulse_opacity:.2f};
            }}
        """)

    def _start_icon_bounce(self, color: str):
        """Start a bouncing animation for the icon."""
        if not self.icon_animation:
            self.icon_animation = QPropertyAnimation(self.icon_label, b"pos")
            self.icon_animation.setDuration(800)
            self.icon_animation.setLoopCount(-1)
            self.icon_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        # Get current position and create bounce effect
        start_pos = self.icon_label.pos()
        end_pos = QPoint(start_pos.x(), start_pos.y() - 8)  # Bounce up 8 pixels

        self.icon_animation.setStartValue(start_pos)
        self.icon_animation.setKeyValueAt(0.5, end_pos)
        self.icon_animation.setEndValue(start_pos)

        # Set color
        self.icon_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 20px;
            }}
        """)

        self.icon_animation.start()

    def _update_thinking_animation(self):
        """Update the thinking dots animation."""
        # Reset all dots to dim
        for dot in self.dots:
            dot.setStyleSheet("color: #3498db; font-size: 18px; opacity: 0.3;")

        # Highlight current dot
        if self.thinking_dots < len(self.dots):
            self.dots[self.thinking_dots].setStyleSheet("color: #3498db; font-size: 18px; opacity: 1.0;")

        # Cycle through dots
        self.thinking_dots = (self.thinking_dots + 1) % (len(self.dots) * 2)  # Extra cycle for pause effect

    def _update_streaming_animation(self):
        """Update the streaming wave animation."""
        # Create wave effect by varying opacity
        for i, wave in enumerate(self.wave_elements):
            # Calculate opacity based on position in wave
            phase_offset = (i / len(self.wave_elements)) * 2
            opacity = 0.3 + 0.7 * abs(((self.streaming_phase + phase_offset) % 2) - 1)
            wave.setStyleSheet(f"color: #9b59b6; font-size: 16px; opacity: {opacity:.2f};")

        # Update phase
        self.streaming_phase += 0.2
        if self.streaming_phase >= 2:
            self.streaming_phase = 0

    def _stop_all_animations(self):
        """Stop all running animations."""
        if self.icon_animation and self.icon_animation.state() == QPropertyAnimation.State.Running:
            self.icon_animation.stop()

        if self.background_animation and self.background_animation.state() == QPropertyAnimation.State.Running:
            self.background_animation.stop()

        # Stop pulse timer if it exists
        if hasattr(self, '_pulse_timer') and self._pulse_timer.isActive():
            self._pulse_timer.stop()

        self.thinking_timer.stop()
        self.streaming_timer.stop()

    def reset(self):
        """Reset to idle state."""
        self.set_state(self.STATE_IDLE)

    def cleanup(self):
        """Clean up timers and animations."""
        self._stop_all_animations()
        self.thinking_timer.deleteLater()
        self.streaming_timer.deleteLater()
