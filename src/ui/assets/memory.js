/* Memory JavaScript */

let backend;
let currentMemory = null; // Currently editing/viewing memory object

// Initialize QWebChannel
document.addEventListener("DOMContentLoaded", () => {
    new QWebChannel(qt.webChannelTransport, (channel) => {
        backend = channel.objects.backend;
        backend.log("Memory web view initialized");
        
        // Initial load will happen via backend calling refresh_memory() in on_ui_ready
    });
});

// UI Event Handlers

function requestSearch() {
    const query = document.getElementById('searchInput').value;
    const filters = {
        type: document.getElementById('typeFilter').value,
        importance: document.getElementById('importanceFilter').value,
        tags: document.getElementById('tagsFilter').value.split(',').map(t => t.trim()).filter(t => t),
        limit: parseInt(document.getElementById('limitInput').value) || 20
    };
    
    if (backend) {
        backend.requestSearch(query, JSON.stringify(filters));
    }
}

function handleSearchKey(event) {
    if (event.key === 'Enter') {
        requestSearch();
    }
}

function requestRefresh() {
    if (backend) backend.requestRefresh();
}

function showAddFact() {
    currentMemory = null;
    clearDetailView();
    
    document.getElementById('detailView').classList.remove('hidden');
    document.getElementById('emptyDetail').classList.add('hidden');
    
    // Set defaults
    document.getElementById('memType').value = 'semantic';
    document.getElementById('memImportance').value = 'normal';
    
    // Highlight none in list
    document.querySelectorAll('.memory-item').forEach(el => el.classList.remove('active'));
}

function selectMemory(element, memoryJson) {
    // Deserialize
    const memory = typeof memoryJson === 'string' ? JSON.parse(memoryJson) : memoryJson;
    currentMemory = memory;
    
    // Update List UI
    document.querySelectorAll('.memory-item').forEach(el => el.classList.remove('active'));
    element.classList.add('active');
    
    // Update Detail UI
    document.getElementById('detailView').classList.remove('hidden');
    document.getElementById('emptyDetail').classList.add('hidden');
    
    document.getElementById('memContent').value = memory.content || '';
    
    const metadata = memory.metadata || {};
    document.getElementById('memType').value = metadata.type || 'conversation';
    document.getElementById('memImportance').value = metadata.importance || 'normal';
    document.getElementById('memTags').value = (metadata.tags || []).join(', ');
    
    let metaInfo = '';
    if (metadata.timestamp) metaInfo += `Created: ${formatDate(metadata.timestamp)}<br>`;
    if (memory.score) metaInfo += `Score: ${memory.score.toFixed(4)}`;
    document.getElementById('memMetadata').innerHTML = metaInfo;
}

function saveMemory() {
    const content = document.getElementById('memContent').value.trim();
    if (!content) {
        showToast('Content cannot be empty', true);
        return;
    }
    
    const data = {
        content: content,
        type: document.getElementById('memType').value,
        importance: document.getElementById('memImportance').value,
        tags: document.getElementById('memTags').value.split(',').map(t => t.trim()).filter(t => t)
    };
    
    // Note: Backend treats edits as new facts currently due to SDK limitations mentioned in native widget
    if (currentMemory) {
        // We are "updating"
        backend.requestUpdate(JSON.stringify(data));
    } else {
        // New fact
        backend.requestAddFact(JSON.stringify(data));
    }
}

function deleteMemory() {
    // Not implemented fully on backend yet
    if (currentMemory) {
        // Extract ID if available
        const id = currentMemory.id || ''; // Assuming ID exists
        backend.requestDelete(id);
    }
}

function toggleMemory(enabled) {
    if (backend) backend.requestToggle(enabled);
}

// Backend Calls

function requestExport() { if (backend) backend.requestExport(); }
function requestImport() { if (backend) backend.requestImport(); }
function requestClear() { if (backend) backend.requestClear(); }


// Callbacks from Backend

function displaySearchResults(resultsJson) {
    const results = typeof resultsJson === 'string' ? JSON.parse(resultsJson) : resultsJson;
    const list = document.getElementById('memoryList');
    list.innerHTML = '';
    
    if (!results || results.length === 0) {
        list.innerHTML = '<li style="padding: 20px; text-align: center; color: #777;">No memories found</li>';
        return;
    }
    
    // Sort slightly? Native didn't sort beyond return order (relevance)
    
    results.forEach((mem, index) => {
        const item = document.createElement('li');
        item.className = 'memory-item';
        
        // Serialize for click handler
        // Using closure to avoid escaping hell
        item.onclick = () => selectMemory(item, mem);
        
        const metadata = mem.metadata || {};
        const content = mem.content || '';
        const type = metadata.type || 'unknown';
        const date = formatDate(metadata.timestamp);
        
        item.innerHTML = `
            <span class="mem-type">${type}</span>
            <div class="mem-preview">${escapeHtml(content)}</div>
            <div class="mem-meta">${date}</div>
        `;
        list.appendChild(item);
    });
}

function updateStats(statsJson) {
    const stats = typeof statsJson === 'string' ? JSON.parse(statsJson) : statsJson;
    // e.g. "Memory: Enabled | Database: mem.json"
    const el = document.getElementById('memoryStats');
    if (stats.error) {
        el.textContent = "Error loading stats";
        return;
    }
    
    const path = stats.database_path ? stats.database_path.split(/[\\/]/).pop() : 'Unknown';
    el.textContent = `Status: ${stats.enabled ? 'Active' : 'Disabled'} • DB: ${path}`;
}

function setMemoryEnabled(enabled) {
    document.getElementById('memoryEnabled').checked = enabled;
}

// Helpers

function clearDetailView() {
    document.getElementById('memContent').value = '';
    document.getElementById('memType').value = 'conversation';
    document.getElementById('memImportance').value = 'normal';
    document.getElementById('memTags').value = '';
    document.getElementById('memMetadata').innerHTML = '';
}

function formatDate(isoString) {
    if (!isoString) return '';
    try {
        const d = new Date(isoString);
        return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    } catch (e) { return isoString; }
}

function escapeHtml(text) {
    if (!text) return '';
    return text.replace(/&/g, "&")
               .replace(/</g, "<")
               .replace(/>/g, ">")
               .replace(/"/g, """)
               .replace(/'/g, "&#039;");
}

function showToast(message, isError = false) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.style.backgroundColor = isError ? '#e74c3c' : '#007acc';
    
    toast.classList.remove('hidden');
    
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}
