// Global Backend Objects
let filesBackend = null;
let memoryBackend = null;
let settingsBackend = null;

// --- Initialization ---

document.addEventListener("DOMContentLoaded", () => {
    initWebChannel();
    
    // Set default tab (files)
    switchMainTab('files');
    
    // Drag and Drop (Files)
    filesSetupDragAndDrop();
    
    // Initialize Settings Provider Models
    initSettingsConstants();
});

function initWebChannel() {
    if (typeof QWebChannel !== "undefined") {
        new QWebChannel(qt.webChannelTransport, (channel) => {
            filesBackend = channel.objects.files;
            memoryBackend = channel.objects.memory;
            settingsBackend = channel.objects.settings;
            
            // Files Initialization
            if (filesBackend) {
                // filesBackend.log("Files initialized");
                // Files usually refreshes on load via on_ui_ready in Python
            }
            
            // Memory Initialization
            if (memoryBackend) {
                // memoryBackend.log("Memory initialized");
            }
            
            // Settings Initialization
            if (settingsBackend) {
                // Settings usually sends current settings on load
                if (settingsBackend.onReady) settingsBackend.onReady();
            }
        });
    } else {
        console.error("QWebChannel not defined.");
    }
}

function switchMainTab(tabName) {
    // Check if tab is hidden
    const btn = document.getElementById(`btn-tab-${tabName}`);
    if (btn && btn.style.display === 'none') {
        console.warn(`Tab ${tabName} is hidden, cannot switch to it`);
        return;
    }

    // Update Tabs UI
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    if (btn) btn.classList.add('active');

    // Show View
    document.querySelectorAll('.tab-view').forEach(view => {
        view.classList.remove('active');
        view.classList.add('hidden');
    });

    const view = document.getElementById(`view-${tabName}`);
    if (view) {
        view.classList.remove('hidden');
        view.classList.add('active');
    }
}

function showMemoryTab() {
    const btn = document.getElementById('btn-tab-memory');
    const view = document.getElementById('view-memory');

    if (btn) btn.style.display = '';
    if (view) view.style.display = '';
}

function hideMemoryTab() {
    const btn = document.getElementById('btn-tab-memory');
    const view = document.getElementById('view-memory');

    // If memory tab is currently active, switch to files tab
    if (btn && btn.classList.contains('active')) {
        switchMainTab('files');
    }

    if (btn) btn.style.display = 'none';
    if (view) view.style.display = 'none';
}

// --- GLOBAL HELPERS ---

function showToast(message, isError = false) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.style.backgroundColor = isError ? '#e74c3c' : '#007acc';
    
    toast.classList.remove('hidden');
    // Force reflow
    void toast.offsetWidth;
    
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}

function escapeHtml(text) {
    if (!text) return '';
    return text.replace(/&/g, "&")
               .replace(/</g, "<")
               .replace(/>/g, ">")
               .replace(/"/g, "&quot")
               .replace(/'/g, "&#039;");
}

function getSafe(obj, path, defaultValue = '') {
    return path.split('.').reduce((o, p) => (o ? o[p] : undefined), obj) ?? defaultValue;
}


// ==========================================
//              FILES LOGIC
// ==========================================

function filesUpdateList(files) {
    const list = document.getElementById('fileList');
    list.innerHTML = '';

    if (!files || files.length === 0) {
        list.innerHTML = `
            <li class="empty-state">
                <div class="empty-icon">📂</div>
                <p>No documents uploaded yet</p>
                <button class="secondary-btn" onclick="filesRequestUpload()">Upload Files</button>
            </li>
        `;
        return;
    }

    files.forEach(file => {
        const item = filesCreateItem(file);
        list.appendChild(item);
    });
}

function filesCreateItem(file) {
    const li = document.createElement('li');
    li.className = 'file-item';
    li.onclick = (e) => {
        if (!e.target.closest('.action-icon')) {
            filesRequestOpen(file.file_path);
        }
    };

    const icon = filesGetIcon(file);
    const size = filesFormatSize(file.file_size);
    const date = filesFormatDate(file.file_modified);

    // Escape file path for onclick
    const safePath = file.file_path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");

    li.innerHTML = `
        <div class="file-icon">${icon}</div>
        <div class="file-info">
            <span class="file-name">${file.file_name}</span>
            <span class="file-meta">${size} • ${date}</span>
        </div>
        <div class="file-actions">
            <button class="action-icon" onclick="filesRequestOpen('${safePath}')" title="View">ℹ️</button>
            <button class="action-icon delete-icon" onclick="filesRequestDelete('${safePath}')" title="Delete">🗑️</button>
        </div>
    `;

    return li;
}

function filesGetIcon(file) {
    const name = file.file_name.toLowerCase();
    const type = file.mime_type || '';
    
    if (name.endsWith('.pdf') || type === 'application/pdf') return '📄';
    if (name.endsWith('.docx') || name.endsWith('.doc')) return '📝';
    if (name.endsWith('.txt') || name.endsWith('.md')) return '📃';
    if (name.endsWith('.py') || name.endsWith('.js') || name.endsWith('.json') || name.endsWith('.html') || name.endsWith('.css')) return '💻';
    return '📄';
}

function filesFormatSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function filesFormatDate(isoString) {
    if (!isoString) return 'Unknown date';
    try {
        const date = new Date(isoString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (e) {
        return 'Invalid date';
    }
}

function filesUpdateIngestionProgress(percent, message) {
    const status = document.getElementById('ingestionStatus');
    const bar = document.getElementById('progressBar');
    const text = document.getElementById('statusText');

    status.classList.remove('hidden');
    bar.style.width = percent + '%';
    text.textContent = message;
}

function filesSetIngestionBusy(isBusy) {
    const btn = document.getElementById('uploadBtn');
    const status = document.getElementById('ingestionStatus');
    
    if (isBusy) {
        btn.disabled = true;
        btn.style.opacity = '0.5';
        status.classList.remove('hidden');
    } else {
        btn.disabled = false;
        btn.style.opacity = '1';
        setTimeout(() => {
            status.classList.add('hidden');
            document.getElementById('progressBar').style.width = '0%';
        }, 3000);
    }
}

function filesRequestUpload() {
    if (filesBackend) filesBackend.requestUpload();
}

function filesRequestDelete(filePath) {
    if (filesBackend) filesBackend.requestDelete(filePath);
}

function filesRequestOpen(filePath) {
    if (filesBackend) filesBackend.requestOpen(filePath);
}

function filesSetupDragAndDrop() {
    const dropZone = document.getElementById('dropZone');
    // Only set up if elements exist (which they do in side_panel.html)
    if (!dropZone) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.body.addEventListener(eventName, e => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        document.body.addEventListener(eventName, () => dropZone.classList.remove('hidden'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('hidden'), false);
    });
    
    // Note: Actual data handling is done via Python drag/drop override or just ignored here
    // as we can't get paths easily in JS.
}


// ==========================================
//              MEMORY LOGIC
// ==========================================

let memoryCurrent = null;

function memoryRequestSearch() {
    const query = document.getElementById('memSearchInput').value;
    const filters = {
        type: document.getElementById('memTypeFilter').value,
        importance: document.getElementById('memImportanceFilter').value,
        tags: document.getElementById('memTagsFilter').value.split(',').map(t => t.trim()).filter(t => t),
        limit: parseInt(document.getElementById('memLimitInput').value) || 20
    };
    
    if (memoryBackend) {
        memoryBackend.requestSearch(query, JSON.stringify(filters));
    }
}

function handleMemSearchKey(event) {
    if (event.key === 'Enter') {
        memoryRequestSearch();
    }
}

function memoryRequestRefresh() {
    if (memoryBackend) memoryBackend.requestRefresh();
}

function memoryShowAddFact() {
    memoryCurrent = null;
    memoryClearDetailView();
    
    document.getElementById('memDetailView').classList.remove('hidden');
    document.getElementById('memEmptyDetail').classList.add('hidden');
    
    document.getElementById('memType').value = 'semantic';
    document.getElementById('memImportance').value = 'normal';
    
    document.querySelectorAll('.memory-item').forEach(el => el.classList.remove('active'));
}

function memorySelect(element, memoryJson) {
    const memory = typeof memoryJson === 'string' ? JSON.parse(memoryJson) : memoryJson;
    memoryCurrent = memory;
    
    document.querySelectorAll('.memory-item').forEach(el => el.classList.remove('active'));
    element.classList.add('active');
    
    document.getElementById('memDetailView').classList.remove('hidden');
    document.getElementById('memEmptyDetail').classList.add('hidden');
    
    document.getElementById('memContent').value = memory.content || '';
    
    const metadata = memory.metadata || {};
    document.getElementById('memType').value = metadata.type || 'conversation';
    document.getElementById('memImportance').value = metadata.importance || 'normal';
    document.getElementById('memTags').value = (metadata.tags || []).join(', ');
    
    let metaInfo = '';
    if (metadata.timestamp) metaInfo += `Created: ${filesFormatDate(metadata.timestamp)}<br>`;
    if (memory.score) metaInfo += `Score: ${memory.score.toFixed(4)}`;
    document.getElementById('memMetadata').innerHTML = metaInfo;
}

function memorySave() {
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
    
    if (memoryCurrent) {
        if (memoryBackend) memoryBackend.requestUpdate(JSON.stringify(data));
    } else {
        if (memoryBackend) memoryBackend.requestAddFact(JSON.stringify(data));
    }
}

function memoryToggle(enabled) {
    if (memoryBackend) memoryBackend.requestToggle(enabled);
}

function memoryRequestExport() { if (memoryBackend) memoryBackend.requestExport(); }
function memoryRequestImport() { if (memoryBackend) memoryBackend.requestImport(); }
function memoryRequestClear() { if (memoryBackend) memoryBackend.requestClear(); }

function memoryDisplayResults(resultsJson) {
    const results = typeof resultsJson === 'string' ? JSON.parse(resultsJson) : resultsJson;
    const list = document.getElementById('memoryList');
    list.innerHTML = '';
    
    if (!results || results.length === 0) {
        list.innerHTML = '<li style="padding: 20px; text-align: center; color: #777;">No memories found</li>';
        return;
    }
    
    results.forEach((mem) => {
        const item = document.createElement('li');
        item.className = 'memory-item';
        
        item.onclick = () => memorySelect(item, mem);
        
        const metadata = mem.metadata || {};
        const content = mem.content || '';
        const type = metadata.type || 'unknown';
        const date = filesFormatDate(metadata.timestamp);
        
        item.innerHTML = `
            <span class="mem-type">${type}</span>
            <div class="mem-preview">${escapeHtml(content)}</div>
            <div class="mem-meta">${date}</div>
        `;
        list.appendChild(item);
    });
}

function memoryUpdateStats(statsJson) {
    const stats = typeof statsJson === 'string' ? JSON.parse(statsJson) : statsJson;
    const el = document.getElementById('memoryStats');
    if (stats.error) {
        el.textContent = "Error loading stats";
        return;
    }
    const path = stats.database_path ? stats.database_path.split(/[\\/]/).pop() : 'Unknown';
    el.textContent = `Status: ${stats.enabled ? 'Active' : 'Disabled'} • DB: ${path}`;

    // Note: Memory enable/disable is now controlled via Settings, not this checkbox
    // The checkbox was removed from the memory tab UI
}

function memorySetEnabled(enabled) { // Legacy direct call - now does nothing
    // Memory enable/disable is now controlled via Settings
    // This function is kept for backward compatibility but does nothing
}

function memoryClearDetailView() {
    document.getElementById('memContent').value = '';
    document.getElementById('memType').value = 'conversation';
    document.getElementById('memImportance').value = 'normal';
    document.getElementById('memTags').value = '';
    document.getElementById('memMetadata').innerHTML = '';
}


// ==========================================
//              SETTINGS LOGIC
// ==========================================

let PROVIDER_MODELS = {};
let currentCombos = {};
let cachedSettings = {};

function initSettingsConstants() {
     PROVIDER_MODELS = {
        'openrouter': [],
        'openai': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
        'ollama': ['llama3', 'mistral', 'gemma'],
        'anthropic': ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku']
    };
}

// --- Combo Management ---
function settingsLoadCombos(combos) {
    currentCombos = combos || {};
    // Small delay to ensure layout is ready before updating selectors
    setTimeout(() => {
        settingsUpdateComboSelectors();
        settingsUpdateCombosList();
    }, 50);
}

function settingsUpdateComboSelectors() {
    // Update all combo selectors with current combos
    const selectors = ['llm-combo', 'emb-combo', 'intent-combo'];

    selectors.forEach(selectorId => {
        const select = document.getElementById(selectorId);
        if (select) {
            select.innerHTML = '';
            Object.keys(currentCombos).forEach(comboId => {
                const combo = currentCombos[comboId];
                const option = document.createElement('option');
                option.value = comboId;
                option.textContent = combo.name;
                select.appendChild(option);
            });

            // Apply pending value if exists
            const pendingValue = select.dataset.pendingValue;
            if (pendingValue && currentCombos[pendingValue]) {
                select.value = pendingValue;
                delete select.dataset.pendingValue;
            }

            // Add event listener for combo selection changes
            select.onchange = function() {
                settingsApplyComboSelection(selectorId, this.value);
            };
        }
    });
}

function settingsApplyComboSelection(selectorId, comboId) {
    // Apply the selected combo to the appropriate use case
    if (settingsBackend) {
        const settings = settingsBackend.getCurrentSettings();
        if (!settings) return;

        // Update the specific use case with the selected combo
        if (selectorId === 'llm-combo') {
            settings.llm.combo = comboId;
        } else if (selectorId === 'emb-combo') {
            settings.embeddings.combo = comboId;
        } else if (selectorId === 'intent-combo') {
            settings.intent_analysis.combo = comboId;
        }

        // Save the updated settings
        settingsBackend.saveSettings(JSON.stringify(settings));
        showToast(`Applied combo ${comboId} to ${selectorId.replace('-combo', '')}`);
    }
}

function settingsUpdateCombosList() {
    const listDiv = document.getElementById('combos-list');
    if (!listDiv) return;

    listDiv.innerHTML = '';

    if (Object.keys(currentCombos).length === 0) {
        listDiv.innerHTML = '<p class="hint">No combos created yet.</p>';
        return;
    }

    Object.keys(currentCombos).forEach(comboId => {
        const combo = currentCombos[comboId];
        const comboDiv = document.createElement('div');
        comboDiv.className = 'combo-item';
        comboDiv.innerHTML = `
            <div class="combo-info">
                <strong>${combo.name}</strong><br>
                <small>${combo.provider} / ${combo.model}</small>
            </div>
            <div class="combo-actions">
                <button class="secondary-btn small" onclick="settingsEditCombo('${comboId}')">Edit</button>
                <button class="danger-btn small" onclick="settingsDeleteCombo('${comboId}')">Delete</button>
            </div>
        `;
        listDiv.appendChild(comboDiv);
    });
}

function settingsCreateCombo() {
    const name = document.getElementById('combo-name').value.trim();
    const provider = document.getElementById('combo-provider').value;
    const model = document.getElementById('combo-model').value.trim();

    if (!name || !provider || !model) {
        showToast('Please fill in all fields', true);
        return;
    }

    // Generate a simple ID from the name
    const comboId = name.toLowerCase().replace(/[^a-z0-9]/g, '_');

    // Check if ID already exists
    if (currentCombos[comboId]) {
        showToast('A combo with this name already exists', true);
        return;
    }

    // Create combo and send to backend
    currentCombos[comboId] = { name, provider, model };

    // Update UI
    settingsUpdateComboSelectors();
    settingsUpdateCombosList();

    // Clear form
    document.getElementById('combo-name').value = '';
    document.getElementById('combo-model').value = '';

    showToast('Combo created successfully');
}

function settingsEditCombo(comboId) {
    const combo = currentCombos[comboId];
    if (!combo) return;

    // Populate form with existing values
    document.getElementById('combo-name').value = combo.name;
    document.getElementById('combo-provider').value = combo.provider;
    document.getElementById('combo-model').value = combo.model;

    // Change button to update mode
    const btn = document.querySelector('#tab-combos .primary-btn');
    if (btn) {
        btn.textContent = 'Update Combo';
        btn.onclick = () => settingsUpdateCombo(comboId);
    }

    // Add cancel button
    if (!document.getElementById('cancel-edit')) {
        const cancelBtn = document.createElement('button');
        cancelBtn.id = 'cancel-edit';
        cancelBtn.className = 'secondary-btn';
        cancelBtn.textContent = 'Cancel';
        cancelBtn.onclick = settingsCancelEdit;
        if (btn && btn.parentNode) {
            btn.parentNode.appendChild(cancelBtn);
        }
    }
}

function settingsUpdateCombo(comboId) {
    const name = document.getElementById('combo-name').value.trim();
    const provider = document.getElementById('combo-provider').value;
    const model = document.getElementById('combo-model').value.trim();

    if (!name || !provider || !model) {
        showToast('Please fill in all fields', true);
        return;
    }

    // Update combo
    currentCombos[comboId] = { name, provider, model };

    // Update UI
    settingsUpdateComboSelectors();
    settingsUpdateCombosList();

    // Reset form
    settingsCancelEdit();
    showToast('Combo updated successfully');
}

function settingsDeleteCombo(comboId) {
    if (!confirm('Are you sure you want to delete this combo?')) return;

    // Check if combo is currently in use
    if (settingsBackend) {
        const settings = settingsBackend.getCurrentSettings();
        if (settings) {
            const inUseBy = [];

            // Check if combo is used in LLM settings
            if (settings.llm && settings.llm.combo === comboId) {
                inUseBy.push('LLM configuration');
            }

            // Check if combo is used in embeddings settings
            if (settings.embeddings && settings.embeddings.combo === comboId) {
                inUseBy.push('embeddings configuration');
            }

            // Check if combo is used in intent analysis settings
            if (settings.intent_analysis && settings.intent_analysis.combo === comboId) {
                inUseBy.push('intent analysis configuration');
            }

            if (inUseBy.length > 0) {
                showToast(`Cannot delete combo: it is currently in use by ${inUseBy.join(', ')}`, true);
                return;
            }
        }
    }

    delete currentCombos[comboId];

    // Update UI
    settingsUpdateComboSelectors();
    settingsUpdateCombosList();

    showToast('Combo deleted successfully');
}

function settingsCancelEdit() {
    // Clear form
    document.getElementById('combo-name').value = '';
    document.getElementById('combo-provider').value = 'openrouter';
    document.getElementById('combo-model').value = '';

    // Reset button
    const btn = document.querySelector('#tab-combos .primary-btn');
    if (btn) {
        btn.textContent = 'Create Combo';
        btn.onclick = settingsCreateCombo;
    }

    // Remove cancel button
    const cancelBtn = document.getElementById('cancel-edit');
    if (cancelBtn) {
        cancelBtn.remove();
    }
}

function settingsSwitchTab(tabId) {
    document.querySelectorAll('.settings-sidebar .nav-links li').forEach(li => {
        li.classList.remove('active');
        if (li.id === `nav-${tabId}` || li.getAttribute('onclick').includes(tabId)) {
            li.classList.add('active');
        }
    });

    document.querySelectorAll('.settings-tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    const selectedTab = document.getElementById(`tab-${tabId}`);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }
}

function toggleVisibility(inputId) {
    const input = document.getElementById(inputId);
    if (input.type === "password") {
        input.type = "text";
    } else {
        input.type = "password";
    }
}

function settingsToggleIntentSettings() {
    const enabled = document.getElementById('intent-enabled').checked;
    const settingsDiv = document.getElementById('intent-settings');
    if (enabled) settingsDiv.classList.remove('disabled');
    else settingsDiv.classList.add('disabled');
}

function settingsToggleRagSettings() {
    const enabled = document.getElementById('rag-enabled').checked;
    const settingsDiv = document.getElementById('rag-settings');
    if (enabled) settingsDiv.classList.remove('disabled');
    else settingsDiv.classList.add('disabled');
}

function settingsToggleTtsSettings() {
    const enabled = document.getElementById('tts-enabled').checked;
    const settingsDiv = document.getElementById('tts-settings');
    if (enabled) settingsDiv.classList.remove('disabled');
    else settingsDiv.classList.add('disabled');
}

function settingsToggleHybridWeights() {
    const method = document.getElementById('rag-method').value;
    const weightsDiv = document.getElementById('hybrid-weights');
    if (method === 'hybrid') weightsDiv.classList.remove('hidden');
    else weightsDiv.classList.add('hidden');
}

function settingsUpdateWeights(changed) {
    const semInput = document.getElementById('sem-weight');
    const keyInput = document.getElementById('key-weight');
    let semVal = parseInt(semInput.value);
    let keyVal = parseInt(keyInput.value);

    if (changed === 'sem') {
        keyVal = 100 - semVal;
        keyInput.value = keyVal;
    } else {
        semVal = 100 - keyVal;
        semInput.value = semVal;
    }
    document.getElementById('sem-weight-val').textContent = semVal;
    document.getElementById('key-weight-val').textContent = keyVal;
}

function settingsOnProviderChanged() {
    const provider = document.getElementById('llm-provider').value;
    settingsUpdateModelInput(provider);
}

function settingsUpdateModelInput(provider, currentModel = '') {
    const container = document.getElementById('llm-model-container');
    container.innerHTML = ''; 

    if (provider === 'openrouter') {
        const input = document.createElement('input');
        input.type = 'text';
        input.id = 'llm-model';
        input.placeholder = 'select or type model...';
        input.value = currentModel;
        container.appendChild(input);
    } else {
        const select = document.createElement('select');
        select.id = 'llm-model';
        const models = PROVIDER_MODELS[provider] || [];
        if (models.length === 0) {
             const input = document.createElement('input');
            input.type = 'text';
            input.id = 'llm-model';
            input.placeholder = 'Enter model name...';
            input.value = currentModel;
            container.appendChild(input);
            return;
        }
        let found = false;
        models.forEach(m => {
            const option = document.createElement('option');
            option.value = m;
            option.textContent = m;
            if (m === currentModel) {
                option.selected = true;
                found = true;
            }
            select.appendChild(option);
        });
        if (currentModel && !found) {
             const option = document.createElement('option');
            option.value = currentModel;
            option.textContent = currentModel;
            option.selected = true;
            select.appendChild(option);
        }
        container.appendChild(select);
    }
}

function settingsLoad(settings) {
    cachedSettings = settings;

    document.getElementById('openrouter-key').value = getSafe(settings, 'api_keys.openrouter');
    document.getElementById('openai-key').value = getSafe(settings, 'api_keys.openai');
    document.getElementById('tavily-key').value = getSafe(settings, 'api_keys.tools.tavily');
    document.getElementById('exa-key').value = getSafe(settings, 'api_keys.tools.exa');
    document.getElementById('jina-key').value = getSafe(settings, 'api_keys.tools.jina');

    // Load combo IDs instead of provider/model
    const llmCombo = getSafe(settings, 'llm.combo', 'default_llm');
    const embCombo = getSafe(settings, 'embeddings.combo', 'default_embeddings');
    const intentCombo = getSafe(settings, 'intent_analysis.combo', 'default_intent');

    // Set combo values (will be applied when combos are loaded)
    document.getElementById('llm-combo').dataset.pendingValue = llmCombo;
    document.getElementById('emb-combo').dataset.pendingValue = embCombo;
    document.getElementById('intent-combo').dataset.pendingValue = intentCombo;

    const temp = getSafe(settings, 'llm.temperature', 0.7);
    document.getElementById('llm-temperature').value = temp;
    document.getElementById('temp-value').textContent = temp;

    // Intent Analysis
    const intentEnabled = getSafe(settings, 'intent_analysis.enabled', false);
    document.getElementById('intent-enabled').checked = intentEnabled;
    settingsToggleIntentSettings();

    const ragEnabled = getSafe(settings, 'retrieval.enabled', true);
    document.getElementById('rag-enabled').checked = ragEnabled;
    settingsToggleRagSettings();

    document.getElementById('rag-max-docs').value = getSafe(settings, 'retrieval.max_docs', 3);
    const ragMethod = getSafe(settings, 'retrieval.method', 'semantic');
    document.getElementById('rag-method').value = ragMethod;
    settingsToggleHybridWeights();

    const semWeight = Math.round(getSafe(settings, 'retrieval.semantic_weight', 0.7) * 100);
    const keyWeight = Math.round(getSafe(settings, 'retrieval.keyword_weight', 0.3) * 100);

    document.getElementById('sem-weight').value = semWeight;
    document.getElementById('sem-weight-val').textContent = semWeight;
    document.getElementById('key-weight').value = keyWeight;
    document.getElementById('key-weight-val').textContent = keyWeight;

    document.getElementById('vector-provider').value = getSafe(settings, 'vector_store.provider', 'faiss');
    document.getElementById('vector-path').value = getSafe(settings, 'vector_store.path', './data/vector_store');

    const ttsEnabled = getSafe(settings, 'audio.tts_enabled', true);
    document.getElementById('tts-enabled').checked = ttsEnabled;
    settingsToggleTtsSettings();

    document.getElementById('tts-speed').value = getSafe(settings, 'audio.tts_speed', 1.0);
    document.getElementById('tts-speed-val').textContent = getSafe(settings, 'audio.tts_speed', 1.0);

    document.getElementById('asr-enabled').checked = getSafe(settings, 'audio.asr_enabled', true);
    document.getElementById('asr-language').value = getSafe(settings, 'audio.asr_language', 'en-US');

    document.getElementById('app-history').value = getSafe(settings, 'app.max_history', 50);
    document.getElementById('app-theme').value = getSafe(settings, 'app.theme', 'dark');
}

function settingsCollect() {
    // Start with a deep copy of cached settings to preserve fields not in UI (like memory, tracing, etc.)
    const s = cachedSettings ? JSON.parse(JSON.stringify(cachedSettings)) : {};

    // Ensure structure exists for fields we modify
    if (!s.api_keys) s.api_keys = {};
    if (!s.api_keys.tools) s.api_keys.tools = {};
    if (!s.llm) s.llm = {};
    if (!s.embeddings) s.embeddings = {};
    if (!s.intent_analysis) s.intent_analysis = {};
    if (!s.retrieval) s.retrieval = {};
    if (!s.vector_store) s.vector_store = {};
    if (!s.audio) s.audio = {};
    if (!s.app) s.app = {};

    // Update with UI values
    s.api_keys.openrouter = document.getElementById('openrouter-key').value;
    s.api_keys.openai = document.getElementById('openai-key').value;
    s.api_keys.tools.tavily = document.getElementById('tavily-key').value;
    s.api_keys.tools.exa = document.getElementById('exa-key').value;
    s.api_keys.tools.jina = document.getElementById('jina-key').value;

    s.combos = currentCombos;

    s.llm.combo = document.getElementById('llm-combo').value;
    s.llm.temperature = parseFloat(document.getElementById('llm-temperature').value);

    s.embeddings.combo = document.getElementById('emb-combo').value;

    s.intent_analysis.enabled = document.getElementById('intent-enabled').checked;
    s.intent_analysis.combo = document.getElementById('intent-combo').value;

    s.retrieval.enabled = document.getElementById('rag-enabled').checked;
    s.retrieval.max_docs = parseInt(document.getElementById('rag-max-docs').value);
    s.retrieval.method = document.getElementById('rag-method').value;
    s.retrieval.semantic_weight = parseInt(document.getElementById('sem-weight').value) / 100;
    s.retrieval.keyword_weight = parseInt(document.getElementById('key-weight').value) / 100;

    s.vector_store.provider = document.getElementById('vector-provider').value;
    s.vector_store.path = document.getElementById('vector-path').value;

    s.audio.tts_enabled = document.getElementById('tts-enabled').checked;
    s.audio.tts_voice = document.getElementById('tts-voice').value;
    s.audio.tts_speed = parseFloat(document.getElementById('tts-speed').value);
    s.audio.asr_enabled = document.getElementById('asr-enabled').checked;
    s.audio.asr_language = document.getElementById('asr-language').value;

    s.app.max_history = parseInt(document.getElementById('app-history').value);
    s.app.theme = document.getElementById('app-theme').value;

    return s;
}

function settingsSave() {
    if (settingsBackend) {
        const settings = settingsCollect();
        settingsBackend.saveSettings(JSON.stringify(settings)); // Backend expects JSON string usually? Or dict? Original was dict in signal but sent as string if simplified. Original: `saveSettings(settings_json)`
    }
}

function settingsReset() {
    if (confirm("Are you sure you want to reset all settings to defaults?")) {
        if (settingsBackend) settingsBackend.resetSettings();
    }
}

function settingsTestConnection() {
    if (settingsBackend) {
        const key = document.getElementById('openrouter-key').value;
        settingsBackend.testConnection(key);
    }
}

function settingsSetVoices(voicesJson) {
    const voices = typeof voicesJson === 'string' ? JSON.parse(voicesJson) : voicesJson;
    const select = document.getElementById('tts-voice');
    select.innerHTML = '';
    
    if (voices.length === 0) {
        const opt = document.createElement('option');
        opt.textContent = "No voices available";
        select.appendChild(opt);
        return;
    }

    voices.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v.id;
        opt.textContent = v.name;
        select.appendChild(opt);
    });
}
