# Advanced Structured Prompt Building System - Implementation Plan

## Concept Overview

This implementation creates a sophisticated two-layer memory system (STM + LTM) + RAG retrieval layer + structured agent prompts following the LangChain/DeepAgents architecture. The system enables continuous dialogue, reduces token costs, and maintains agent context consistency.

## Architecture Layers

### 1. System Layer (SystemPrompt)
- Structured XML-like prompts with AgentMode, PersonaProfile, ConversationFrame, MemoryRules
- Provides model stability and memory writing rules
- Integration with LangChain agent chains

### 2. Short-Term Memory (STM)
- Stores last 10-20 messages for local coherence
- Used for planning current responses
- Small size to save tokens

### 3. Retriever/RAG Layer
- Vector DB with indexing for docs, project contexts, notes
- 2-step RAG: retrieve-then-read
- Chunking (300-1000 chars) with 10-20% overlap

### 4. Long-Term Memory (LTM)
- Stores personal preferences, project metadata, major decisions
- Only updated via triggers (explicit save, milestone completion)
- Uses LangChain/LangGraph memory stores

### 5. Agent Orchestration
- DeepAgent harness with planner
- Virtual filesystem and subagent spawning
- ProjectContext and AgentMode as state

## Core Components to Implement

### Core Classes
1. **StructuredPromptBuilder** - Main prompt building system
2. **ProjectContextManager** - Project state and context management
3. **UserProfileManager** - User personalization data
4. **ContextAggregator** - Smart context assembly
5. **AgentStateManager** - Global and local agent status tracking

### Data Models
1. **AgentState** - Current agent mode and goals
2. **ProjectContext** - Project metadata and full context
3. **UserProfile** - Personalization preferences
4. **MemoryRules** - Instructions for memory management

## Implementation Files Structure

```
src/core/
├── prompt_builder.py           # Core structured prompt builder
├── project_manager.py          # Project context management
├── user_profile_manager.py     # User personalization
├── agent_state_manager.py      # Agent status tracking
├── context_aggregator.py       # Smart context assembly
└── memory_management.py        # STM/LTM operations

src/data/
├── schemas/                    # Data schemas
│   ├── agent_state.py
│   ├── project_context.py
│   └── user_profile.py
└── templates/                  # Prompt templates
    ├── system_prompt_templates.py
    └── structured_prompt_base.py

src/langchain_adapters/
├── enhanced_llm_adapter.py     # Extended LLM adapter
└── deep_agent_integration.py   # DeepAgent compatibility
```

## Structured Prompt Format

```xml
<SystemPrompt>
    <SystemInfo>
        <AgentMode>
            <CurrentMode>ProjectWork</CurrentMode>
            <CurrentGoal>refactor project structure</CurrentGoal>
            <PersonaProfile>relaxed-bro style + detail-oriented expert</PersonaProfile>
        </AgentMode>
        <ConversationFrame>
            <Type>continuous_dialogue</Type>
            <MaintainTone>true</MaintainTone>
            <LongTermContinuity>true</LongTermContinuity>
        </ConversationFrame>
        <MemoryRules>
            <StoreLongTermPrefs>true</StoreLongTermPrefs>
            <StoreProjectMetadata>true</StoreProjectMetadata>
            <UpdateProjectContext>explicit_request_only</UpdateProjectContext>
            <SummarizeBeforeStore>true</SummarizeBeforeStore>
        </MemoryRules>
    </SystemInfo>
    
    <UserPersonalization>
        <CommunicationStyle>casual_technical</CommunicationStyle>
        <DetailPreference>comprehensive</DetailPreference>
        <OpinionRequested>true</OpinionRequested>
    </UserPersonalization>
    
    <RelatedMemories>
        <!-- Retrieved relevant memories with summaries -->
    </RelatedMemories>
    
    <UserProjects>
        <CurrentProject>
            <ProjectId>proj_001</ProjectId>
            <ProjectName>Self-Coded Client</ProjectName>
            <ProjectDescription>AI assistant with advanced prompt system</ProjectDescription>
            <ProjectStatus>Active Development</ProjectStatus>
            <ProjectContextReference>Full context available via tool call</ProjectContextReference>
        </CurrentProject>
        <!-- Other active projects with minimal info -->
    </UserProjects>
    
    <UserIntent>
        <Type>analysis_and_opinion</Type>
        <ExpectedDetailLevel>deep</ExpectedDetailLevel>
        <ExpectedStructure>arguments + clear breakdown</ExpectedStructure>
    </UserIntent>
    
    <UserQuery>
        <!-- Current user message -->
    </UserQuery>
    
    <MessageHistory>
        <!-- Last 10-20 messages -->
    </MessageHistory>
    
    <AdditionalContextRAG>
        <!-- Retrieved documents and context -->
    </AdditionalContextRAG>
</SystemPrompt>
```

## Workflow Implementation

### 1. Intent Interpretation
- Classify user intent (opinion/code/plan/quick-answer)
- Apply persona behavior rules
- Reduce style inconsistencies

### 2. Context Loading
- Load STM (last 10-20 messages)
- Load relevant LTM snippets
- Run vector retriever for related docs

### 3. DeepAgent Planning
- For complex queries: activate planner
- Delegate subtasks to sub-agents
- Handle tool calls (web search, code runner, summarizer)

### 4. Final Prompt Composition
- System prompt (AgentMode + Persona + MemoryRules)
- Synthesized context (STM + retrieved docs + LTM highlights)
- Token optimization with relevant summaries only

### 5. Response and Memory Update
- Generate response
- Update state if project context changed
- Trigger memory writing via MemoryRules

## Configuration Schema

```json
{
  "prompt_builder": {
    "structured_format_enabled": true,
    "xml_validation": true,
    "token_limits": {
      "project_context_reference": 200,
      "message_history": 1000,
      "retrieved_context": 800
    }
  },
  "memory_management": {
    "stm_buffer_size": 20,
    "ltm_triggers": ["explicit_save", "milestone_complete", "threshold_summarize"],
    "chunk_size": 500,
    "chunk_overlap": 0.15
  },
  "agent_state": {
    "global_status_persistence": true,
    "local_status_duration": "session",
    "auto_mode_detection": true
  },
  "projects": {
    "max_active_projects": 5,
    "context_caching": true,
    "lazy_loading": true
  }
}
```

## Implementation Steps

### Phase 1: Core Infrastructure
1. Create structured prompt builder class
2. Implement project context management
3. Build agent state tracking system
4. Create user profile management

### Phase 2: Memory System
1. Implement STM/LTM operations
2. Create context aggregation system
3. Build memory trigger mechanisms
4. Add summarization capabilities

### Phase 3: Integration
1. Extend LLM adapter for structured prompts
2. Integrate with existing controller
3. Add DeepAgent compatibility
4. Implement configuration system

### Phase 4: Advanced Features
1. Add middleware support
2. Implement prompt templates
3. Create validation and testing
4. Performance optimization

## Key Benefits

✅ **Context Continuity**: Two-layer memory maintains conversation flow
✅ **Token Efficiency**: Smart context management reduces costs
✅ **Agent Stability**: Structured prompts ensure consistent behavior
✅ **Project Awareness**: Integrated project state and context
✅ **Extensible Architecture**: Easy to add new sections and features
✅ **LangChain Compatible**: Uses proven agent creation patterns
✅ **Production Ready**: Handles edge cases and memory management

This implementation provides a sophisticated AI assistant capable of maintaining long-term project context while optimizing for token usage and response quality.
