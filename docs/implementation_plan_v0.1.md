# Implementation Plan v0.1: Single-Flow Personal Assistant (PyQt6 + LangChain)

## Project Overview

This plan outlines the development of a cross-platform desktop personal assistant application featuring a single continuous conversation flow, built with PyQt6 for the UI and LangChain for AI interactions. The application integrates vector-based file retrieval, advanced memory systems (using Memori), TTS/ASR capabilities, and secure API key management.

**Key Technologies:**
- **UI Framework:** PyQt6 (/websites/riverbankcomputing_static_pyqt6)
- **AI Framework:** LangChain (/websites/python_langchain_com-v0.2-api_reference-ollama-llms-langchain_ollama.llms.ollamallm.html)
- **Memory System:** Memori (/gibsonai/memori)
- **Vector Stores:** FAISS/Chroma (configurable)
- **Database:** SQLite for metadata
- **Packaging:** PyInstaller for cross-platform binaries

## Architecture Overview

The application follows a modular architecture with clear separation of concerns:

```
personal_assistant/
├── src/
│   ├── ui/                    # PyQt6 UI components
│   ├── core/                  # Business logic and orchestration
│   │   ├── controller.py      # AI Controller (central orchestrator)
│   │   ├── prompt_builder.py  # Prompt composition logic
│   │   ├── memory_adapter.py  # Memori integration
│   │   ├── vector_manager.py  # Vector store management
│   │   ├── file_ingestor.py   # File parsing and ingestion
│   │   ├── audio_manager.py   # TTS/ASR provider abstraction
│   │   └── settings.py        # Configuration and secrets
│   ├── langchain_adapters/    # LangChain provider wrappers
│   └── storage/               # Data persistence layer
├── tests/                     # Unit and integration tests
├── packaging/                 # Build configurations
└── docs/                      # Documentation
```

## Dependencies and Libraries

### Core Dependencies
- `PyQt6>=6.5.0` - UI framework
- `langchain>=0.2.0` - AI orchestration
- `langchain-openai` - OpenAI provider integration
- `langchain-community` - Community providers
- `faiss-cpu` or `chromadb` - Vector stores
- `memoripy` - Memory system (Memori SDK)
- `sqlite3` - Local database (built-in Python)
- `cryptography` - API key encryption
- `python-dotenv` - Environment management

### File Processing
- `pypdf2` or `PyMuPDF` - PDF parsing
- `python-docx` - Word document parsing
- `openpyxl` - Excel support
- `pytesseract` - OCR for images
- `beautifulsoup4` - HTML parsing

### Audio Processing
- `pyaudio` - Audio I/O
- `speechrecognition` - ASR abstraction
- `pyttsx3` - Local TTS fallback
- `gtts` - Google TTS
- `playsound` - Audio playback

### Development and Testing
- `pytest` - Testing framework
- `pytest-qt` - Qt testing
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking

## Implementation Phases

### Phase 1: Project Setup and Core Infrastructure (Week 1-2)

#### 1.1 Repository and Environment Setup
- Initialize Git repository with proper .gitignore
- Set up Python virtual environment
- Create basic project structure
- Configure development tools (black, flake8, mypy)
- Set up CI/CD pipeline (GitHub Actions for linting and basic tests)

#### 1.2 Core Dependencies and Configuration
- Install and verify all core dependencies
- Set up logging configuration (structured JSON logs)
- Implement basic settings management with encryption
- Create configuration schemas for providers and preferences

#### 1.3 Database Schema Design
- Design SQLite schema for messages, metadata, and settings
- Implement database migration system
- Create data access layer with proper error handling

### Phase 2: UI Skeleton and Basic LangChain Integration (Week 3-4)

#### 2.1 PyQt6 UI Foundation
- Implement main window with responsive layout
- Create message list widget (continuous timeline)
- Build input bar with text entry and basic controls
- Add collapsible side panel for files and settings
- Implement basic event handling and threading

#### 2.2 LangChain Plumbing
- Set up LangChain configuration for OpenRouter integration
- Implement basic LLM chain for simple text responses
- Create LangChain adapter layer for provider abstraction
- Add error handling for API failures and rate limits

#### 2.3 Basic Conversation Flow
- Implement message storage and retrieval
- Create conversation buffer management
- Add basic prompt templating system
- Integrate UI with LangChain responses

### Phase 3: File Ingestion and Vector Retrieval (Week 5-6)

#### 3.1 File Processing Pipeline
- Implement file type detection and validation
- Create text extraction for PDF, DOCX, TXT files
- Add chunking strategy with configurable overlap
- Integrate OCR for image files (optional)

#### 3.2 Vector Store Integration
- Set up FAISS/Chroma with LangChain wrappers
- Implement embedding generation (OpenAI or local)
- Create ingestion pipeline with metadata storage
- Add file management UI (upload, delete, status)

#### 3.3 Retrieval Integration
- Implement vector similarity search
- Create hybrid retrieval (keyword + semantic)
- Integrate retrieval results into prompt building
- Add source citation in UI responses

### Phase 4: Memory System Integration (Week 7-8)

#### 4.1 Memori Adapter Development
- Study Memori SDK integration patterns
- Implement memory adapter with LangChain compatibility
- Create memory storage and retrieval logic
- Add memory injection into prompts

#### 4.2 Memory UI and Management
- Add memory viewing interface
- Implement memory editing and deletion
- Create memory tagging and search
- Add "save to memory" functionality for messages

#### 4.3 Advanced Memory Features
- Implement episodic vs semantic memory separation
- Add memory importance scoring
- Create memory consolidation and cleanup
- Integrate memory with conversation context

### Phase 5: Audio Subsystem (TTS/ASR) (Week 9-10)

#### 5.1 Audio Provider Abstraction
- Design AudioProvider interface
- Implement OpenAI TTS integration via LangChain
- Add local TTS fallback (pyttsx3)
- Create audio playback controls

#### 5.2 Speech Recognition
- Implement ASR provider abstraction
- Integrate speech-to-text functionality
- Add voice input UI controls
- Handle audio permissions and device selection

#### 5.3 Audio UI Integration
- Add voice input button to input bar
- Implement TTS playback controls
- Create audio settings panel
- Add audio file export capabilities

### Phase 6: Settings, Security, and Polish (Week 11-12)

#### 6.1 Settings Management
- Implement comprehensive settings UI
- Add provider switching (LLM, TTS, ASR)
- Create vector store configuration
- Add memory and privacy settings

#### 6.2 Security Implementation
- Implement API key encryption with user passphrase
- Add secure key storage (OS keyring fallback)
- Create privacy controls and data export
- Implement telemetry opt-in system

#### 6.3 UI Polish and Features
- Add message controls (regenerate, edit, delete)
- Implement conversation export/import
- Add developer mode with prompt inspection
- Polish responsive design and theming

### Phase 7: Testing, Documentation, and Packaging (Week 13-14)

#### 7.1 Testing Suite
- Write unit tests for core components
- Create integration tests for LangChain chains
- Add UI tests with pytest-qt
- Implement mock providers for testing

#### 7.2 Documentation
- Create comprehensive README
- Document API and configuration
- Add user guide and troubleshooting
- Generate API documentation

#### 7.3 Packaging and Distribution
- Configure PyInstaller for Windows, Linux, macOS
- Create build scripts and CI/CD
- Implement auto-update mechanism (optional)
- Test cross-platform compatibility

## Detailed Component Specifications

### AI Controller
- Central orchestrator receiving user input
- Coordinates prompt building, memory retrieval, and LLM calls
- Manages conversation flow and context window
- Handles async operations and error recovery

### Prompt Builder
- Composes prompts from multiple sources:
  - System prompt (editable templates)
  - Retrieved documents (formatted with metadata)
  - Memory items (prioritized by relevance)
  - Conversation history (sliding window)
  - User input
- Implements token management and truncation logic
- Supports prompt presets and customization

### Memory Adapter
- Wraps Memori SDK for LangChain compatibility
- Handles memory storage, retrieval, and updates
- Supports different memory types (episodic, semantic, procedural)
- Provides UI integration for memory management

### Vector Manager
- Abstracts vector store operations (FAISS/Chroma)
- Manages embeddings generation and storage
- Implements efficient retrieval with filtering
- Handles vector store migration and backup

### Audio Manager
- Provider-agnostic audio processing
- Supports multiple TTS/ASR providers
- Manages audio I/O and device selection
- Implements streaming and file export

## Testing Strategy

### Unit Tests
- Core logic components (prompt builder, memory adapter)
- Data processing (file ingestion, chunking)
- Provider abstractions (LLM, TTS, ASR)
- Database operations

### Integration Tests
- End-to-end conversation flows
- File ingestion and retrieval
- Memory persistence across sessions
- Audio processing pipelines

### UI Tests
- Basic interaction flows
- Settings management
- File upload and management
- Audio controls

## Risk Assessment and Mitigation

### Technical Risks
- **LangChain Version Compatibility:** Regular updates and testing with latest versions
- **Provider API Changes:** Abstraction layers and fallback providers
- **Memory System Complexity:** Incremental implementation with thorough testing
- **Cross-Platform UI Consistency:** Extensive testing on target platforms

### Performance Risks
- **Large Vector Stores:** Implement indexing and caching strategies
- **Memory Usage:** Monitor and optimize for desktop constraints
- **Response Latency:** Async processing and progress indicators

### Security Risks
- **API Key Exposure:** Encrypted storage with user authentication
- **Data Privacy:** Local processing with user consent controls
- **Memory Content:** Sanitization and access controls

## Success Criteria

- Application launches successfully on Windows, Linux, and macOS
- Basic chat functionality with LLM responses
- File upload and retrieval-augmented responses
- Memory persistence across application restarts
- TTS/ASR functionality with provider switching
- Secure settings management
- Comprehensive test coverage (>80%)
- User-friendly installation packages

## Timeline and Milestones

- **Milestone 1 (End of Week 4):** Basic UI and LangChain integration
- **Milestone 2 (End of Week 8):** File ingestion and memory system
- **Milestone 3 (End of Week 12):** Audio features and settings
- **Milestone 4 (End of Week 14):** Testing, documentation, and packaging

## Next Steps

1. Begin Phase 1 implementation
2. Set up development environment
3. Create initial project structure
4. Implement basic PyQt6 window
5. Integrate first LangChain LLM call

This plan provides a comprehensive roadmap for building the personal assistant application, with clear phases, dependencies, and success criteria. Regular reviews and adjustments will be made based on implementation progress and discovered requirements.
