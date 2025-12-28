#!/usr/bin/env python3
"""
Main entry point for the Single-Flow Personal Assistant application.
"""

import sys
import os

# Add the parent directory to the Python path to make src a package
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure Qt WebEngine to allow network access from local files
# This is needed for loading external images in the chat interface
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--disable-web-security --allow-file-access-from-files --allow-running-insecure-content'

# Initialize settings and inject environment variables (e.g., for LangSmith tracing)
try:
    from src.core.settings import PlainSettings
    
    settings_manager = PlainSettings()
    settings = settings_manager.load_settings()
    tracing_config = settings.get('tracing', {})
    
    if tracing_config.get('enabled') and tracing_config.get('api_key'):
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = tracing_config.get('api_key')
        os.environ["LANGSMITH_PROJECT"] = tracing_config.get('project', 'default')
        os.environ["LANGSMITH_ENDPOINT"] = tracing_config.get('endpoint', 'https://api.smith.langchain.com')
        # print("LangSmith tracing enabled via environment variables")
        
except Exception as e:
    print(f"Failed to initialize tracing settings: {e}")

# Import the main function
try:
    from src.ui.main_window import main
except ImportError as e:
    logger = None
    try:
        # Try to import logging for error reporting
        import logging
        logger = logging.getLogger(__name__)
    except ImportError:
        pass

    # Fallback for when running from src directory
    try:
        from ui.main_window import main
    except ImportError as e2:
        error_msg = f"Failed to import main window module. Tried both 'src.ui.main_window' and 'ui.main_window'.\nOriginal error: {e}\nFallback error: {e2}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"CRITICAL ERROR: {error_msg}")
        print("Please ensure you're running from the correct directory and all dependencies are installed.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Application failed to start: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
