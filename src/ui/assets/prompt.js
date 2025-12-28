/* Prompt JavaScript */

let backend;

// Initialize QWebChannel
document.addEventListener("DOMContentLoaded", () => {
    new QWebChannel(qt.webChannelTransport, (channel) => {
        backend = channel.objects.backend;
        backend.log("Prompt web view initialized (JSON Mode)");
        backend.requestRefresh();
    });
});

// Prompt Parsing & Rendering
function updatePrompt(payload) {
    const container = document.getElementById('promptContainer');
    container.innerHTML = '';
    
    if (!payload || !payload.structured_prompt) {
        container.innerHTML = '<div class="empty-state small"><p>No prompt generated yet</p></div>';
        return;
    }

    const rawPrompt = payload.structured_prompt;

    try {
        // Try parsing as JSON first
        let jsonData;
        if (typeof rawPrompt === 'string') {
            // Check if it looks like JSON
            if (rawPrompt.trim().startsWith('{')) {
                jsonData = JSON.parse(rawPrompt);
            } else {
                // Fallback to raw text if not JSON
                throw new Error("Not JSON");
            }
        } else {
            jsonData = rawPrompt;
        }

        // Render JSON Structure
        Object.keys(jsonData).forEach(key => {
            renderSection(container, key, jsonData[key]);
        });

    } catch (e) {
        // Fallback for non-JSON or Parse Errors (e.g. legacy XML or raw text)
        console.warn("JSON Parse Error or non-JSON content", e);
        renderRawSection(container, rawPrompt, "Raw Format");
    }
}

function renderSection(container, title, data) {
    // If empty or null, skip
    if (!data) return;

    const section = document.createElement('div');
    section.className = 'section expanded';
    
    // Format Title
    const displayTitle = convertCamelCase(title);
    const icon = getSectionIcon(title);
    
    const header = document.createElement('div');
    header.className = 'section-header';
    header.innerHTML = `
        <div class="section-title">
            <span class="section-icon">${icon}</span>
            ${displayTitle}
        </div>
        <button class="copy-btn" title="Copy Content">📋</button>
    `;
    
    // Copy functionality: strict JSON for objects, text for others
    const copyContent = typeof data === 'object' ? JSON.stringify(data, null, 2) : String(data);
    header.querySelector('.copy-btn').onclick = (e) => {
        e.stopPropagation();
        copyText(copyContent);
    };
    
    header.onclick = () => section.classList.toggle('expanded');
    
    const content = document.createElement('div');
    content.className = 'section-content';
    
    // Render content based on type
    if (typeof data === 'object' && data !== null) {
        if (Array.isArray(data)) {
            // Array Handling
            renderArray(content, data);
        } else {
            // Object Handling
            renderObject(content, data);
        }
    } else {
        // Primitive Value
        content.textContent = String(data);
        content.classList.add('text-block-content');
    }
    
    section.appendChild(header);
    section.appendChild(content);
    container.appendChild(section);
}

function renderObject(container, obj) {
    Object.keys(obj).forEach(key => {
        const value = obj[key];
        
        if (typeof value === 'object' && value !== null) {
            // Nested Object or Array -> Render as Subsection
            renderSubsection(container, key, value);
        } else {
            // Simple Key-Value
            renderKeyValue(container, key, value);
        }
    });
}

function renderArray(container, arr) {
    if (arr.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'kv-row';
        empty.innerHTML = '<span class="kv-value text-muted">Empty list</span>';
        container.appendChild(empty);
        return;
    }

    arr.forEach((item, index) => {
        if (typeof item === 'object' && item !== null) {
            // Object in Array (e.g., Memory Item, Message history)
            renderSubsection(container, `[${index + 1}]`, item);
        } else {
            // Primitive in Array
            const row = document.createElement('div');
            row.className = 'kv-row';
            row.innerHTML = `<span class="kv-key">[${index + 1}]</span> <span class="kv-value">${item}</span>`;
            container.appendChild(row);
        }
    });
}

function renderSubsection(container, title, data) {
    const wrapper = document.createElement('div');
    wrapper.className = 'subsection';
    
    const header = document.createElement('div');
    header.className = 'subsection-header';
    header.textContent = convertCamelCase(title);
    header.onclick = () => wrapper.classList.toggle('collapsed');
    
    const content = document.createElement('div');
    content.className = 'subsection-content';
    
    if (Array.isArray(data)) {
        renderArray(content, data);
    } else {
        renderObject(content, data);
    }
    
    wrapper.appendChild(header);
    wrapper.appendChild(content);
    container.appendChild(wrapper);
}

function renderKeyValue(container, key, value) {
    const row = document.createElement('div');
    row.className = 'kv-row';
    
    const keyName = convertCamelCase(key);
    row.innerHTML = `<span class="kv-key">${keyName}</span>`;
    
    const valSpan = document.createElement('span');
    valSpan.className = 'kv-value';
    
    // Handle specific value types logic or simple display
    valSpan.textContent = String(value);
    
    if (typeof value === 'boolean') {
        valSpan.classList.add(value ? 'bool-true' : 'bool-false');
    } else if (typeof value === 'number') {
        valSpan.classList.add('number');
    }
    
    row.appendChild(valSpan);
    container.appendChild(row);
}

function renderRawSection(container, text, label) {
    const section = document.createElement('div');
    section.className = 'section expanded';
    section.innerHTML = `
        <div class="section-header">
            <div class="section-title">${label || 'Raw Content'}</div>
        </div>
        <div class="section-content">
            <div class="text-block-content" style="white-space: pre-wrap;">${text}</div>
        </div>
    `;
    container.appendChild(section);
}

// History Mgt
function addToHistory(payload) {
    const list = document.getElementById('historyList');
    const countEl = document.getElementById('historyCount');
    
    if (!list || !countEl) return;

    const li = document.createElement('li');
    li.className = 'history-item';
    
    const time = payload.timestamp || 'Unknown';
    const preview = payload.user_input ? payload.user_input.substring(0, 50) + (payload.user_input.length > 50 ? '...' : '') : 'System Prompt';
    
    li.innerHTML = `
        <div class="history-time">🕒 ${time}</div>
        <div class="history-preview">"${preview}"</div>
    `;
    
    li.onclick = () => {
        document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
        li.classList.add('active');
        updatePrompt(payload);
    };
    
    // Insert top
    list.insertBefore(li, list.firstChild);
    
    // Limit history
    while (list.children.length > 20) {
        list.removeChild(list.lastChild);
    }
    
    countEl.textContent = list.children.length;
}

// Helpers
function requestEdit() {
    if (backend) backend.requestEdit();
}

function copyText(text) {
    if (backend) backend.copyToClipboard(text);
}

function convertCamelCase(str) {
    // Clean up key names (e.g. "user_personalization" -> "User Personalization")
    return str.replace(/_/g, ' ')
              .replace(/([A-Z])/g, ' $1')
              .replace(/^./, str => str.toUpperCase())
              .trim();
}

function getSectionIcon(name) {
    const map = {
        'system': '⚙️',
        'context': '🧠',
        'user': '👤',
        'ltm': '📚',
        'rag': '🔎',
        'task': '🎯',
        'history': '💬',
        'project': '📁'
    };
    // Fuzzy matching for keys
    const lower = name.toLowerCase();
    for (const key in map) {
        if (lower.includes(key)) return map[key];
    }
    return '📄';
}
