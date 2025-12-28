# Personal Assistant v0.3.0

A single-flow personal assistant application built with PyQt6 and LangChain, featuring advanced conversation management, multi-provider LLM support, vector search, memory integration, web search tools, and cross-platform support.

## 🚀 Key Features

### Core Functionality ✅
- **Single Conversation Flow**: Continuous chat interface with persistent context and topic tracking
- **Multi-LLM Support**: OpenRouter integration with provider switching and model combos
- **Vector Search & Retrieval**: Chat-with-files using FAISS/Chroma vector stores with hybrid search
- **Memory System**: Long-term memory with Memori SDK integration and episodic/semantic memory
- **Web Search Tools**: Integrated Tavily, Exa, and Jina search with automatic image galleries
- **Audio Processing**: TTS/ASR with provider abstraction and streaming playback
- **Topic Detection**: Automatic conversation topic tracking and context management
- **Tool Sets**: Configurable tool collections for different use cases
- **Draft Management**: Save and manage message drafts across sessions
- **Plain Settings**: Simple JSON-based configuration with secure API key management

### Technical Highlights ✅
- **Async Processing**: Non-blocking UI with proper threading and streaming responses
- **Modular Architecture**: Clean separation of UI, core logic, providers, and tools
- **Advanced Prompt Engineering**: Structured prompts with XML tags, memory injection, and context retrieval
- **Image Processing**: Automatic thumbnail generation and gallery display for web search results
- **Cross-Platform**: Windows, Linux, and macOS support with PyInstaller packaging
- **Comprehensive Testing**: Unit tests with pytest and Qt testing integration
- **Tool Integration**: Extensible tool system with web search, custom tools, and provider management

## 🏗️ Architecture

```
personal_assistant/
├── src/
│   ├── ui/                    # PyQt6 UI components with WebEngine integration
│   ├── core/                  # Business logic and orchestration
│   │   ├── controller.py      # AI Controller (central orchestrator)
│   │   ├── prompt_builder.py  # Advanced structured prompt composition
│   │   ├── memory_adapter.py  # Memori integration with memory types
│   │   ├── vector_manager.py  # Vector store management with hybrid search
│   │   ├── file_ingestor.py   # Multi-format file parsing and ingestion
│   │   ├── audio_provider.py  # TTS/ASR provider abstraction
│   │   ├── topic_tracker.py   # Conversation topic detection middleware
│   │   ├── project_manager.py # Project context management framework
│   │   └── settings.py        # Configuration and secure settings management
│   ├── langchain_adapters/    # LangChain provider wrappers with tool integration
│   └── storage/               # Data persistence layer (SQLite planned)
├── tests/                     # Unit and integration tests
├── packaging/                 # Build configurations and PyInstaller scripts
└── docs/                      # Documentation and implementation plans
```

## 🔧 Installation

### Prerequisites
- Python 3.9+
- pip

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd personal-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## 🚀 Usage

### First Run
1. Launch the application: `python src/main.py`
2. Configure your API keys in the settings panel (OpenAI, OpenRouter, web search providers)
3. Set up model combos in the settings for different use cases
4. Start chatting! The assistant will automatically detect topics and use appropriate tools

### Key Features
- **File Upload**: Drag & drop files for vector search integration
- **Voice Input**: Click microphone for speech-to-text with real-time feedback
- **Web Search**: Ask questions requiring current information - tools activate automatically
- **Image Galleries**: Web search results with automatic image download and thumbnail generation
- **Topic Tracking**: Conversations are organized by topics with automatic detection
- **Memory Management**: View and manage long-term conversation memories
- **Draft System**: Save message drafts and recover unsent messages across sessions
- **Model Combos**: Create and switch between different LLM provider/model combinations
- **Tool Sets**: Configure different tool collections for various tasks
- **Settings**: Comprehensive configuration for LLM providers, tools, audio, and memory

## 🔒 Security

- **Plain Storage**: API keys stored in plain JSON format (user responsibility for security)
- **Local Access**: All data stored locally on user's machine
- **Tool Isolation**: Web search tools use separate API keys with configurable permissions
- **No Telemetry**: No data collection or external tracking

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📦 Building for Distribution

### Using PyInstaller
```bash
python packaging/build.py
```

This creates a standalone executable in the `dist/` directory.

### Manual PyInstaller Build
```bash
pip install pyinstaller
pyinstaller --clean --noconfirm packaging/personal_assistant.spec
```

## 🔄 Recent Improvements (v0.3.0)

### Major New Features ✅
- **Topic Detection & Tracking**: Automatic conversation topic identification and context management
- **Provider/Model Combos**: Create reusable combinations of LLM providers and models for different use cases
- **Web Search Integration**: Tavily, Exa, and Jina search tools with automatic activation
- **Image Gallery System**: Automatic image download, thumbnail generation, and gallery display
- **Tool Sets Management**: Configurable tool collections with web search and custom tools
- **Draft Management**: Save, edit, and recover message drafts across application restarts
- **Enhanced Streaming**: Improved response streaming with better error handling
- **Markdown Rendering**: Beautiful markdown display with syntax highlighting
- **Project Context Framework**: Foundation for project-based conversation organization

### Technical Enhancements ✅
- **Structured Prompt System**: XML-tagged prompts with memory, context, and tool integration
- **Advanced Tool Architecture**: Extensible tool system with provider abstraction
- **Image Processing Pipeline**: Automatic thumbnail generation and local storage
- **Conversation Persistence**: Robust autosave with drafts and message history
- **Async Processing Fixes**: Resolved UI blocking issues with proper threading
- **Error Handling Improvements**: Better error recovery and user feedback
- **Memory System Integration**: Enhanced Memori SDK integration with multiple memory types
- **Hybrid Vector Search**: Semantic + keyword search for improved document retrieval

## 🗺️ Roadmap

### Phase 8a: Stability and Polish ✅ MOSTLY COMPLETE
- [x] Fix streaming response handling and UI blocking
- [x] Implement topic detection middleware
- [x] Add provider/model combo system
- [x] Integrate web search tools with automatic galleries
- [x] Implement draft management system
- [x] Add markdown rendering and syntax highlighting
- [ ] Complete project context management UI
- [ ] Add conversation export/import functionality

### Phase 8b: Advanced Features 🔄 PLANNED
- [ ] Email integration (notifications for new emails)
- [ ] Project-based conversation organization
- [ ] Advanced memory visualization and management
- [ ] Conversation branching and versioning
- [ ] Multi-language support
- [ ] Plugin system for custom tools

### Phase 8c: Performance and Scalability 🔄 PLANNED
- [ ] Database migration from JSON to SQLite
- [ ] Vector store optimization and indexing
- [ ] Memory consolidation and cleanup automation
- [ ] Large conversation history handling
- [ ] Background processing for heavy operations

### Phase 8d: Security and Privacy 🔄 PLANNED
- [ ] Encrypted API key storage
- [ ] Data export functionality
- [ ] Privacy controls and data deletion
- [ ] Security audit and penetration testing
- [ ] Secure communication protocols

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyQt6**: Modern Python GUI framework with WebEngine integration
- **LangChain**: LLM orchestration framework with extensive provider support
- **Memori**: Advanced long-term memory system for AI conversations
- **FAISS/Chroma**: High-performance vector search libraries
- **OpenRouter**: Multi-provider LLM API gateway
- **Tavily/Exa/Jina**: Web search APIs for current information retrieval
- **Cryptography**: Secure encryption for future security enhancements

---

**Status**: v0.3.0 - Feature Complete with Advanced Capabilities
**Last Updated**: December 2025
