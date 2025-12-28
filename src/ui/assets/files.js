/* Files JavaScript */

let backend;

// Initialize QWebChannel
document.addEventListener("DOMContentLoaded", () => {
    new QWebChannel(qt.webChannelTransport, (channel) => {
        backend = channel.objects.backend;
        
        // Signal connection logging (optional)
        backend.log("Files web view initialized");
        
        // Notify backend ready is handled by backend waiting for loadFinished
    });

    setupDragAndDrop();
});

// UI Functions

function updateFileList(files) {
    const list = document.getElementById('fileList');
    list.innerHTML = '';

    if (!files || files.length === 0) {
        list.innerHTML = `
            <li class="empty-state">
                <div class="empty-icon">📂</div>
                <p>No documents uploaded yet</p>
                <button class="secondary-btn" onclick="requestUpload()">Upload Files</button>
            </li>
        `;
        return;
    }

    files.forEach(file => {
        const item = createFileItem(file);
        list.appendChild(item);
    });
}

function createFileItem(file) {
    const li = document.createElement('li');
    li.className = 'file-item';
    li.onclick = (e) => {
        // Only open if not clicking an action button directly
        if (!e.target.closest('.action-icon')) {
            requestOpen(file.file_path);
        }
    };

    const icon = getFileIcon(file);
    const size = formatFileSize(file.file_size);
    const date = formatDate(file.file_modified);

    li.innerHTML = `
        <div class="file-icon">${icon}</div>
        <div class="file-info">
            <span class="file-name">${file.file_name}</span>
            <span class="file-meta">${size} • ${date}</span>
        </div>
        <div class="file-actions">
            <button class="action-icon" onclick="requestOpen('${file.file_path.replace(/\\/g, '\\\\')}')" title="View Details">ℹ️</button>
            <button class="action-icon delete-icon" onclick="requestDelete('${file.file_path.replace(/\\/g, '\\\\')}')" title="Delete">🗑️</button>
        </div>
    `;

    return li;
}

function getFileIcon(file) {
    const name = file.file_name.toLowerCase();
    const type = file.mime_type || '';
    
    if (name.endsWith('.pdf') || type === 'application/pdf') return '📄';
    if (name.endsWith('.docx') || name.endsWith('.doc')) return '📝';
    if (name.endsWith('.txt') || name.endsWith('.md')) return '📃';
    if (name.endsWith('.py') || name.endsWith('.js') || name.endsWith('.json') || name.endsWith('.html') || name.endsWith('.css')) return '💻';
    return '📄';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function formatDate(isoString) {
    if (!isoString) return 'Unknown date';
    try {
        const date = new Date(isoString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (e) {
        return 'Invalid date';
    }
}

// Ingestion Progress

function updateIngestionProgress(percent, message) {
    const status = document.getElementById('ingestionStatus');
    const bar = document.getElementById('progressBar');
    const text = document.getElementById('statusText');

    status.classList.remove('hidden');
    bar.style.width = percent + '%';
    text.textContent = message;
}

function setIngestionBusy(isBusy) {
    const btn = document.getElementById('uploadBtn');
    const status = document.getElementById('ingestionStatus');
    
    if (isBusy) {
        btn.disabled = true;
        btn.style.opacity = '0.5';
        status.classList.remove('hidden');
    } else {
        btn.disabled = false;
        btn.style.opacity = '1';
        // Keep status visible for a moment? Backend handles clearing/toast
        setTimeout(() => {
            status.classList.add('hidden');
            document.getElementById('progressBar').style.width = '0%';
        }, 3000);
    }
}

// Backend Calls

function requestUpload() {
    if (backend) backend.requestUpload();
}

function requestDelete(filePath) {
    if (backend) backend.requestDelete(filePath);
}

function requestOpen(filePath) {
    if (backend) backend.requestOpen(filePath);
}

// Drag and Drop

function setupDragAndDrop() {
    const dropZone = document.getElementById('dropZone');
    
    // Prevent default behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone
    ['dragenter', 'dragover'].forEach(eventName => {
        document.body.addEventListener(eventName, () => dropZone.classList.remove('hidden'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('hidden'), false);
    });

    // Handle Drop
    dropZone.addEventListener('drop', handleDrop, false);
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;

    if (files.length > 0) {
       // Since web view can't access local file paths directly safely without input
       // We might need to handle this differently.
       // Actually QtWebEngine supports drag and drop but we want to trigger our ingestion.
       // Default drop behavior might navigate away or do nothing.
       
       // PROBLEM: Browser JS File API gives 'File' objects, but usually hides full path for security.
       // HOWEVER, in QtWebEngine, we might be able to intercept the drop event in Python or use a trick.
       // BUT, the native FileWidget handled drops on the widget itself.
       // The WebEngineView also accepts drops.
       
       // Solution: We trigger the upload dialog via backend for simplicity for now, 
       // OR we accept that drag/drop might need native handling in `FilesWebWidget` class (overriding `dropEvent`).
       
       // Let's implement `dropEvent` on the Python side `FilesWebWidget` instead of JS. 
       // JS drag and drop is mostly for file content which isn't what we want (we want paths).
       // So I will remove JS drag/drop logic or use it just for visual feedback if Python handles the drop.
       // Actually, if I override `dropEvent` in Python, the event won't reach JS unless I pass it.
       // So let's handle Drop in Python `FilesWebWidget`.
       
       // I'll keep the JS visual overlay but trigger nothing here? 
       // Wait, if Python handles drop, I can just call backend.requestDrop(files)? No, JS doesn't have paths.
       
       // OK, I'll rely on the Python side `FilesWebWidget` `dropEvent` to handle the file paths,
       // and maybe it can call JS to show "Processing..." status.
       // So, remove JS handling of files, just UI.
    }
}

// Toast
function showToast(message, isError = false) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.style.backgroundColor = isError ? '#e74c3c' : '#007acc';
    
    toast.classList.remove('hidden');
    
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}
