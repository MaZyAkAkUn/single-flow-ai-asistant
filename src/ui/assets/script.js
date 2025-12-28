// Global variables
let backend = null;
let md = null;

// Lazy loading state
let lazyLoadingState = {
    totalMessageCount: 0,
    loadedMessageCount: 0,
    isLoading: false,
    hasMoreMessages: true
};

// Animation States
const STATE_IDLE = "idle";
const STATE_THINKING = "thinking";
const STATE_REASONING = "reasoning";
const STATE_TOOL_EXEC = "tool_execution";
const STATE_STREAMING = "streaming";

// Reasoning display settings
let showReasoning = true; // Default to showing reasoning

// Audio Player State
let audioState = {
    isPlaying: false,
    position: 0,
    duration: 0
};

// Draft management
let draftTimer = null;
const DRAFT_SAVE_INTERVAL = 7000; // 7 seconds

// Initialize Logic
document.addEventListener("DOMContentLoaded", function () {
    initMarkdown();
    initWebChannel();
    initInputHandling();
    initAudioPlayer();
});

// --- Initialization ---

function initMarkdown() {
    // Check if markdown-it is loaded
    if (window.markdownit) {
        md = window.markdownit({
            html: true,
            linkify: true,
            typographer: true,
            breaks: true,
            highlight: function (str, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    try {
                        return '<pre><button class="copy-code-btn" onclick="copyCode(this)">Copy</button><code>' +
                            hljs.highlight(str, { language: lang, ignoreIllegals: true }).value +
                            '</code></pre>';
                    } catch (__) { }
                }
                return '<pre><button class="copy-code-btn" onclick="copyCode(this)">Copy</button><code>' + 
                       md.utils.escapeHtml(str) + '</code></pre>';
            }
        });
        
        // Add KaTeX support if available
        if (window.texmath && window.katex) {
             md.use(window.texmath, { engine: window.katex, delimiters: 'dollars' });
        }
    } else {
        console.error("Markdown-it library not found!");
    }
}

function initWebChannel() {
    if (typeof QWebChannel !== "undefined") {
        new QWebChannel(qt.webChannelTransport, function (channel) {
            backend = channel.objects.backend;
        });
    }
}

function initInputHandling() {
    const input = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');

    // Auto-resize textarea
    input.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value === '') {
            this.style.height = '';
        }

        // Enable/disable send button
        sendBtn.disabled = this.value.trim() === '';

        // Start draft timer on input
        startDraftTimer();
    });

    // Handle Enter key
    input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            if (e.shiftKey) {
                // Shift+Enter: allow default behavior (new line)
                return;
            } else {
                // Enter: send message
                e.preventDefault();
                sendMessage();
            }
        }
    });

    // Disable send initially
    sendBtn.disabled = true;

    // Initialize scroll detection for lazy loading
    initScrollDetection();
}

function initAudioPlayer() {
    // Inject Audio Player HTML
    const playerHtml = `
    <div id="audio-player" class="hidden">
        <div class="audio-player-content">
            <div class="audio-controls-row">
                <button class="audio-btn" onclick="controlAudio('seek', -10000)" title="-10s">⏪</button>
                <button class="audio-btn play-pause" id="audio-play-pause-btn" onclick="toggleAudioPlayback()" title="Play/Pause">▶</button>
                <button class="audio-btn" onclick="controlAudio('seek', 10000)" title="+10s">⏩</button>
                <button class="audio-btn" onclick="controlAudio('stop')" title="Stop" style="color: #e74c3c;">⏹</button>
            </div>
            <div class="audio-time-row">
                <span id="audio-current-time">0:00</span>
                <div class="audio-progress-container" onclick="seekAudioFromBar(event)">
                    <div class="audio-progress-bar" id="audio-progress-bar"></div>
                </div>
                <span id="audio-total-time">0:00</span>
            </div>
        </div>
    </div>`;
    
    document.body.insertAdjacentHTML('beforeend', playerHtml);
}

// --- Public API called from Python ---

function appendMessage(id, role, content, pinned, timestamp, reasoning = null) {
    const container = document.getElementById('chat-container');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role} ${pinned ? 'pinned' : ''}`;
    msgDiv.id = `msg-${id}`;
    msgDiv.dataset.id = id;
    msgDiv.dataset.role = role;

    // Header
    const header = document.createElement('div');
    header.className = 'message-header';
    let roleDisplayName = 'You';
    if (role === 'assistant') {
        roleDisplayName = 'Assistant';
    } else if (role === 'system') {
        roleDisplayName = 'System';
    }
    header.innerHTML = `
        <span class="role-name">${roleDisplayName}</span>
        <span class="timestamp">${timestamp}</span>
    `;
    msgDiv.appendChild(header);

    // Reasoning (only for assistant messages with reasoning)
    if (role === 'assistant' && reasoning && showReasoning) {
        const reasoningDiv = document.createElement('div');
        reasoningDiv.className = 'reasoning';
        reasoningDiv.innerHTML = `
            <div class="reasoning-header">
                <span class="reasoning-icon">🧠</span>
                <span class="reasoning-label">Thinking</span>
                <button class="reasoning-toggle" onclick="toggleReasoning('${id}')" title="Toggle reasoning visibility">▼</button>
            </div>
            <div class="reasoning-content" id="reasoning-${id}">
                ${renderReasoning(reasoning)}
            </div>
        `;
        msgDiv.appendChild(reasoningDiv);
    }

    // Content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    contentDiv.innerHTML = renderContent(content);
    msgDiv.appendChild(contentDiv);

    // Controls (Footer)
    if (role === 'assistant') {
        const controls = document.createElement('div');
        controls.className = 'message-controls';
        controls.innerHTML = `
            <button class="control-btn" onclick="sendAction('${id}', 'show_prompt')" title="Show Prompt">🔎</button>
            <button class="control-btn" onclick="sendAction('${id}', 'copy')" title="Copy Content">📋</button>
            <button class="control-btn" onclick="sendAction('${id}', 'speak')" title="Speak">🔊</button>
            <button class="control-btn" onclick="sendAction('${id}', 'regenerate')" title="Regenerate">🔄</button>
            <button class="control-btn" onclick="sendAction('${id}', 'edit')" title="Edit">✏️</button>
        `;
        msgDiv.appendChild(controls);
    }

    container.appendChild(msgDiv);

    // Finalize gallery rendering for historical messages
    // This ensures loaded conversation messages get proper gallery formatting
    finalizeGalleryRendering(contentDiv);

    scrollToBottom();
}

function updateLastMessage(content, reasoning = null) {
    const container = document.getElementById('chat-container');
    if (container.lastElementChild) {
        const contentDiv = container.lastElementChild.querySelector('.content');
        if (contentDiv) {
            contentDiv.innerHTML = renderContent(content);
        }

        // Update or add reasoning if provided
        if (reasoning !== null && showReasoning) {
            const msgDiv = container.lastElementChild;
            const existingReasoning = msgDiv.querySelector('.reasoning');

            if (existingReasoning) {
                // Update existing reasoning
                const reasoningContent = existingReasoning.querySelector('.reasoning-content');
                if (reasoningContent) {
                    reasoningContent.innerHTML = renderReasoning(reasoning);
                }
            } else if (reasoning && msgDiv.dataset.role === 'assistant') {
                // Add new reasoning section
                const reasoningDiv = document.createElement('div');
                reasoningDiv.className = 'reasoning';
                const msgId = msgDiv.dataset.id;
                reasoningDiv.innerHTML = `
                    <div class="reasoning-header">
                        <span class="reasoning-icon">🧠</span>
                        <span class="reasoning-label">Thinking</span>
                        <button class="reasoning-toggle" onclick="toggleReasoning('${msgId}')" title="Toggle reasoning visibility">▼</button>
                    </div>
                    <div class="reasoning-content" id="reasoning-${msgId}">
                        ${renderReasoning(reasoning)}
                    </div>
                `;

                // Insert before content
                const contentDiv = msgDiv.querySelector('.content');
                if (contentDiv) {
                    msgDiv.insertBefore(reasoningDiv, contentDiv);
                }
            }
        }

        scrollToBottom();
    }
}

function updateMessage(id, content) {
    const msgDiv = document.getElementById(`msg-${id}`);
    if (msgDiv) {
        const contentDiv = msgDiv.querySelector('.content');
        if (contentDiv) {
            contentDiv.innerHTML = renderContent(content);
            // Finalize gallery rendering for complete messages
            finalizeGalleryRendering(contentDiv);
        }
    }
}

function removeMessage(id) {
    const msgDiv = document.getElementById(`msg-${id}`);
    if (msgDiv) {
        msgDiv.remove();
    }
}

function clearChat() {
    document.getElementById('chat-container').innerHTML = '';
}

// Re-process all galleries in existing messages (useful after conversation load)
function reprocessGalleries() {
    const allMessages = document.querySelectorAll('.message .content');
    allMessages.forEach(contentDiv => {
        // Only process if it doesn't already have galleries
        if (!contentDiv.querySelector('.image-gallery')) {
            finalizeGalleryRendering(contentDiv);
        }
    });
}

function setProcessState(state, message) {
    const indicator = document.getElementById('process-indicator');
    const icon = document.getElementById('process-icon');
    const text = document.getElementById('process-text');
    const anim = document.getElementById('process-anim');
    
    // Reset classes
    icon.className = '';
    anim.innerHTML = '';
    
    if (state === STATE_IDLE) {
        indicator.classList.add('hidden');
        return;
    }
    
    indicator.classList.remove('hidden');
    text.textContent = message || "Processing...";
    
    switch(state) {
        case STATE_THINKING:
            icon.textContent = "●";
            icon.className = 'anim-pulse';
            anim.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
            break;

        case STATE_REASONING:
            icon.textContent = "🧠";
            icon.className = 'anim-pulse';
            anim.innerHTML = '<span class="wave-bar"></span><span class="wave-bar"></span><span class="wave-bar"></span>';
            break;
            
        case STATE_TOOL_EXEC:
            icon.textContent = "🛠️";
            icon.className = 'anim-spin';
            // Show tool details if provided in message
            break;
            
        case STATE_STREAMING:
            icon.textContent = "⚡";
            anim.innerHTML = `
                <span class="wave-bar"></span>
                <span class="wave-bar"></span>
                <span class="wave-bar"></span>
                <span class="wave-bar"></span>
                <span class="wave-bar"></span>
            `;
            break;
    }
}

function setInputEnabled(enabled) {
    document.getElementById('message-input').disabled = !enabled;
    document.getElementById('send-btn').disabled = !enabled;
}

function setInputValue(value) {
    const input = document.getElementById('message-input');
    input.value = value;
    // Trigger input event to update UI state (resize, enable/disable send button)
    input.dispatchEvent(new Event('input', { bubbles: true }));
}

function updateAudioStatus(isRecording) {
    const btn = document.getElementById('voice-btn');
    const indicator = btn.querySelector('.recording-indicator');
    
    if (isRecording) {
        btn.classList.add('active');
        indicator.classList.remove('hidden');
    } else {
        btn.classList.remove('active');
        indicator.classList.add('hidden');
    }
}

function updateAudioPlayerState(state, position, duration) {
    const player = document.getElementById('audio-player');
    const playPauseBtn = document.getElementById('audio-play-pause-btn');
    const progressBar = document.getElementById('audio-progress-bar');
    const currentTimeEl = document.getElementById('audio-current-time');
    const totalTimeEl = document.getElementById('audio-total-time');
    
    if (state === 'stopped') {
        player.classList.add('hidden');
        audioState.isPlaying = false;
        return;
    }
    
    player.classList.remove('hidden');
    
    // Update local state
    audioState.position = position;
    audioState.duration = duration;
    
    // Update Play/Pause Icon
    if (state === 'playing') {
        audioState.isPlaying = true;
        playPauseBtn.textContent = '⏸';
    } else {
        audioState.isPlaying = false;
        playPauseBtn.textContent = '▶';
    }
    
    // Update Time Text
    currentTimeEl.textContent = formatTime(position);
    totalTimeEl.textContent = formatTime(duration);
    
    // Update Progress Bar
    if (duration > 0) {
        const percent = (position / duration) * 100;
        progressBar.style.width = `${percent}%`;
    } else {
        progressBar.style.width = '0%';
    }
}

// --- Internal Logic ---

// --- Animation Functions ---

function animateScrollForNewMessage(callback) {
    // Calculate how much to scroll - approximately viewport height to hide history
    const viewportHeight = window.innerHeight;
    const currentScroll = window.scrollY;
    const targetScroll = currentScroll + viewportHeight;

    // Use smooth scrolling with a promise-based approach
    window.scrollTo({
        top: targetScroll,
        behavior: 'smooth'
    });

    // Wait for scroll animation to complete (smooth scroll takes about 300-500ms)
    // We use a timeout as fallback since scroll events can be unreliable
    setTimeout(() => {
        if (callback) {
            callback();
        }
    }, 600); // Slightly longer than typical smooth scroll duration
}

function renderContent(content) {
    if (md) {
        // Basic markdown rendering with KaTeX support
        let html = md.render(content);
        console.log('Rendered HTML:', html); // Debug: log rendered HTML
        return html;
    }
    // Fallback: simple text escape
    var escaped = content;
    escaped = escaped.replace(/&/g, '&');
    escaped = escaped.replace(/</g, '<');
    escaped = escaped.replace(/>/g, '>');
    escaped = escaped.replace(/"/g, '"');
    escaped = escaped.replace(/'/g, '&#39;');
    escaped = escaped.replace(/\n/g, '<br>');
    return escaped;
}



// Finalize gallery rendering for complete messages
function finalizeGalleryRendering(contentDiv) {
    // First, handle automatic gallery detection for multiple images
    createAutomaticGalleries(contentDiv);

    // Then handle legacy gallery format (with "Image Gallery:" header)
    processLegacyGalleries(contentDiv);

    // Finally, add click handlers for single images
    setupImageClickHandlers(contentDiv);
}

// Create automatic galleries for multiple images in a message
function createAutomaticGalleries(contentDiv) {
    const allImages = contentDiv.querySelectorAll('img');

    if (allImages.length >= 2) {
        // Group consecutive images (within the same paragraph or adjacent elements)
        const imageGroups = groupConsecutiveImages(contentDiv, allImages);

        imageGroups.forEach(group => {
            if (group.length >= 2) {
                createGalleryFromImages(contentDiv, group);
            }
        });
    }
}

// Group consecutive images that are close to each other
function groupConsecutiveImages(contentDiv, allImages) {
    const groups = [];
    let currentGroup = [];

    for (let i = 0; i < allImages.length; i++) {
        const img = allImages[i];

        // Check if this image is part of a gallery group
        if (shouldIncludeInGallery(img)) {
            currentGroup.push(img);
        } else {
            // Finish current group if it has images
            if (currentGroup.length > 0) {
                groups.push([...currentGroup]);
                currentGroup = [];
            }
        }
    }

    // Add final group
    if (currentGroup.length > 0) {
        groups.push(currentGroup);
    }

    return groups;
}

// Determine if an image should be included in automatic gallery
function shouldIncludeInGallery(img) {
    // Skip images that are already in galleries
    if (img.closest('.image-gallery')) {
        return false;
    }

    // Include images with data-full-src (processed by our system)
    if (img.hasAttribute('data-full-src')) {
        return true;
    }

    // Include images that look like search results or external images
    const src = img.getAttribute('src') || '';
    if (src.includes('http') || src.includes('search') || img.alt.includes('result')) {
        return true;
    }

    return false;
}

// Create a gallery from a group of images
function createGalleryFromImages(contentDiv, images) {
    // Use carousel layout for better navigation experience
    const galleryContainer = createCarouselGallery(images);

    // Replace the first image with the gallery, remove others
    const firstImage = images[0];
    const parent = firstImage.parentNode;
    parent.replaceChild(galleryContainer, firstImage);

    // Remove remaining images from the group
    for (let i = 1; i < images.length; i++) {
        const img = images[i];
        const imgParent = img.parentNode;
        if (imgParent && imgParent.tagName === 'P' && imgParent.children.length === 1) {
            // Remove entire paragraph if it only contains the image
            imgParent.remove();
        } else {
            // Just remove the image
            img.remove();
        }
    }
}

// Create a carousel-style gallery with main image and thumbnail navigation
function createCarouselGallery(images) {
    const galleryContainer = document.createElement('div');
    galleryContainer.className = 'image-gallery carousel-gallery auto-gallery';

    const galleryHeader = document.createElement('div');
    galleryHeader.className = 'gallery-header';

    const title = document.createElement('span');
    title.className = 'gallery-title';
    title.textContent = 'Image Gallery';

    const counter = document.createElement('span');
    counter.className = 'gallery-counter';
    counter.textContent = `1 / ${images.length}`;

    galleryHeader.appendChild(title);
    galleryHeader.appendChild(counter);

    // Main image display area
    const mainImageArea = document.createElement('div');
    mainImageArea.className = 'gallery-main-image';

    // Navigation controls
    const navControls = document.createElement('div');
    navControls.className = 'gallery-nav-controls';

    const prevBtn = document.createElement('button');
    prevBtn.className = 'gallery-nav-btn prev-btn';
    prevBtn.innerHTML = '‹';
    prevBtn.title = 'Previous image';
    prevBtn.onclick = () => navigateGallery(galleryContainer, -1);

    const nextBtn = document.createElement('button');
    nextBtn.className = 'gallery-nav-btn next-btn';
    nextBtn.innerHTML = '›';
    nextBtn.title = 'Next image';
    nextBtn.onclick = () => navigateGallery(galleryContainer, 1);

    navControls.appendChild(prevBtn);
    navControls.appendChild(nextBtn);

    // Main image container
    const mainImageContainer = document.createElement('div');
    mainImageContainer.className = 'gallery-main-container';

    const mainImg = document.createElement('img');
    mainImg.className = 'gallery-main-img';
    const firstImage = images[0];
    const mainSrc = firstImage.getAttribute('data-full-src') || firstImage.getAttribute('src');
    mainImg.src = mainSrc;
    mainImg.alt = firstImage.getAttribute('alt') || 'Gallery image';
    mainImg.loading = 'lazy';
    mainImg.onerror = () => handleImageError(mainImg);
    mainImg.onload = () => handleImageLoad(mainImg);

    const mainLoading = document.createElement('div');
    mainLoading.className = 'image-loading main-loading';
    mainLoading.textContent = 'Loading...';

    mainImageContainer.appendChild(mainImg);
    mainImageContainer.appendChild(mainLoading);

    mainImageArea.appendChild(navControls);
    mainImageArea.appendChild(mainImageContainer);

    // Thumbnail navigation strip
    const thumbnailStrip = document.createElement('div');
    thumbnailStrip.className = 'gallery-thumbnails';

    // Store image data for navigation
    galleryContainer.dataset.images = JSON.stringify(images.map(img => ({
        thumbnail: img.getAttribute('src'),
        full: img.getAttribute('data-full-src') || img.getAttribute('src'),
        alt: img.getAttribute('alt') || 'Gallery image'
    })));
    galleryContainer.dataset.currentIndex = '0';

    // Create thumbnails
    images.forEach((img, index) => {
        const thumbContainer = document.createElement('div');
        thumbContainer.className = `gallery-thumb-container ${index === 0 ? 'active' : ''}`;
        thumbContainer.onclick = () => setGalleryImage(galleryContainer, index);

        const thumbImg = document.createElement('img');
        thumbImg.src = img.getAttribute('src');
        thumbImg.alt = img.getAttribute('alt') || `Thumbnail ${index + 1}`;
        thumbImg.loading = 'lazy';
        thumbImg.onerror = () => handleImageError(thumbImg);
        thumbImg.onload = () => handleImageLoad(thumbImg);

        const thumbLoading = document.createElement('div');
        thumbLoading.className = 'image-loading thumb-loading';
        thumbLoading.textContent = 'Loading...';

        thumbContainer.appendChild(thumbImg);
        thumbContainer.appendChild(thumbLoading);
        thumbnailStrip.appendChild(thumbContainer);
    });

    galleryContainer.appendChild(galleryHeader);
    galleryContainer.appendChild(mainImageArea);
    galleryContainer.appendChild(thumbnailStrip);

    // Add keyboard navigation
    galleryContainer.tabIndex = 0;
    galleryContainer.addEventListener('keydown', (e) => handleGalleryKeydown(e, galleryContainer));

    return galleryContainer;
}

// Navigate gallery (prev/next)
function navigateGallery(galleryContainer, direction) {
    const images = JSON.parse(galleryContainer.dataset.images || '[]');
    const currentIndex = parseInt(galleryContainer.dataset.currentIndex || '0');
    const newIndex = Math.max(0, Math.min(images.length - 1, currentIndex + direction));

    if (newIndex !== currentIndex) {
        setGalleryImage(galleryContainer, newIndex);
    }
}

// Set gallery to show specific image
function setGalleryImage(galleryContainer, index) {
    const images = JSON.parse(galleryContainer.dataset.images || '[]');
    if (index < 0 || index >= images.length) return;

    // Update current index
    galleryContainer.dataset.currentIndex = index.toString();

    // Update counter
    const counter = galleryContainer.querySelector('.gallery-counter');
    if (counter) {
        counter.textContent = `${index + 1} / ${images.length}`;
    }

    // Update main image
    const mainImg = galleryContainer.querySelector('.gallery-main-img');
    if (mainImg) {
        // Add loading state
        const mainContainer = mainImg.parentElement;
        const existingLoading = mainContainer.querySelector('.main-loading');
        if (existingLoading) {
            existingLoading.style.display = 'flex';
        }

        mainImg.src = images[index].full;
        mainImg.alt = images[index].alt;
    }

    // Update thumbnail active state
    const thumbnails = galleryContainer.querySelectorAll('.gallery-thumb-container');
    thumbnails.forEach((thumb, i) => {
        thumb.classList.toggle('active', i === index);
    });

    // Scroll thumbnail into view
    const activeThumb = galleryContainer.querySelector('.gallery-thumb-container.active');
    if (activeThumb) {
        activeThumb.scrollIntoView({
            behavior: 'smooth',
            block: 'nearest',
            inline: 'center'
        });
    }
}

// Handle keyboard navigation for gallery
function handleGalleryKeydown(event, galleryContainer) {
    switch (event.key) {
        case 'ArrowLeft':
            event.preventDefault();
            navigateGallery(galleryContainer, -1);
            break;
        case 'ArrowRight':
            event.preventDefault();
            navigateGallery(galleryContainer, 1);
            break;
        case 'Home':
            event.preventDefault();
            setGalleryImage(galleryContainer, 0);
            break;
        case 'End':
            event.preventDefault();
            const images = JSON.parse(galleryContainer.dataset.images || '[]');
            setGalleryImage(galleryContainer, images.length - 1);
            break;
    }
}

// Create a gallery thumbnail from an image element
function createGalleryThumbnail(img) {
    const thumbnail = document.createElement('div');
    thumbnail.className = 'gallery-image';

    // Get the full-size URL (prefer data-full-src, fallback to src)
    const fullSrc = img.getAttribute('data-full-src') || img.getAttribute('src');
    thumbnail.onclick = () => openImageModal(fullSrc);

    // Clone the image for the thumbnail
    const imgElement = document.createElement('img');
    imgElement.src = img.getAttribute('src'); // Use thumbnail src
    imgElement.alt = img.getAttribute('alt') || 'Gallery image';
    imgElement.loading = 'lazy';
    imgElement.onerror = () => handleImageError(imgElement);
    imgElement.onload = () => handleImageLoad(imgElement);

    // Create loading indicator
    const loading = document.createElement('div');
    loading.className = 'image-loading';
    loading.textContent = 'Loading...';

    thumbnail.appendChild(imgElement);
    thumbnail.appendChild(loading);

    return thumbnail;
}

// Process legacy galleries with "Image Gallery:" headers
function processLegacyGalleries(contentDiv) {
    // Find all paragraphs in the content
    const paragraphs = contentDiv.querySelectorAll('p');

    for (let i = 0; i < paragraphs.length; i++) {
        const p = paragraphs[i];

        // Check if this paragraph contains the gallery header
        if (p.querySelector('strong') && p.textContent.includes('Image Gallery:')) {
            // Found gallery header, collect all subsequent image paragraphs
            const galleryImages = [];
            let currentIndex = i + 1;

            // Look for image paragraphs after the header
            while (currentIndex < paragraphs.length) {
                const nextP = paragraphs[currentIndex];
                const images = nextP.querySelectorAll('img[alt="Image"]');

                if (images.length > 0) {
                    // Convert regular images to gallery thumbnails
                    images.forEach(img => {
                        const src = img.getAttribute('src');
                        if (src) {
                            const thumbnail = createGalleryThumbnail(img);
                            galleryImages.push(thumbnail);
                        }
                    });

                    // Remove the processed paragraph
                    nextP.remove();
                } else {
                    // No more images, stop looking
                    break;
                }

                currentIndex++;
            }

            // Create gallery container if we found images
            if (galleryImages.length > 0) {
                const galleryContainer = document.createElement('div');
                galleryContainer.className = 'image-gallery';

                const galleryHeader = document.createElement('div');
                galleryHeader.className = 'gallery-header';

                const title = document.createElement('span');
                title.className = 'gallery-title';
                title.textContent = 'Image Gallery';

                const count = document.createElement('span');
                count.className = 'gallery-count';
                count.textContent = `(${galleryImages.length} images)`;

                galleryHeader.appendChild(title);
                galleryHeader.appendChild(count);

                const galleryGrid = document.createElement('div');
                galleryGrid.className = 'gallery-grid';

                // Add images to grid (limit to 10)
                galleryImages.slice(0, 10).forEach(thumbnail => {
                    galleryGrid.appendChild(thumbnail);
                });

                galleryContainer.appendChild(galleryHeader);
                galleryContainer.appendChild(galleryGrid);

                // Replace the header paragraph with the gallery
                p.parentNode.replaceChild(galleryContainer, p);
            }

            // Exit the loop since we processed the gallery
            break;
        }
    }
}

// Setup click handlers for single images
function setupImageClickHandlers(contentDiv) {
    const images = contentDiv.querySelectorAll('img:not(.gallery-image img)');

    images.forEach(img => {
        // Skip images that already have click handlers or are in galleries
        if (img.onclick || img.closest('.image-gallery')) {
            return;
        }

        // Add click handler for image expansion
        img.onclick = () => handleImageClick(img);

        // Mark large images
        if (img.naturalHeight > 400 || img.naturalWidth > 600) {
            img.classList.add('large-image');
        }
    });
}

// Handle single image clicks
function handleImageClick(img) {
    const fullSrc = img.getAttribute('data-full-src') || img.getAttribute('src');

    if (img.classList.contains('expanded')) {
        // Collapse the image
        img.classList.remove('expanded');
        img.classList.add('large-image');
    } else {
        // Expand the image or open modal
        if (img.naturalHeight > 800 || img.naturalWidth > 1200) {
            // Very large image - open in modal
            openImageModal(fullSrc);
        } else {
            // Medium image - expand in place
            img.classList.add('expanded');
            img.classList.remove('large-image');
        }
    }
}

// Handle image loading errors
function handleImageError(img) {
    img.style.display = 'none';
    const container = img.parentElement;
    const loading = container.querySelector('.image-loading');
    if (loading) {
        loading.style.display = 'none';
    }
    container.innerHTML = '<div class="image-error">⚠️ Image failed to load</div>';
}

// Handle successful image loading
function handleImageLoad(img) {
    const container = img.parentElement;
    const loading = container.querySelector('.image-loading');
    if (loading) {
        loading.style.display = 'none';
    }
}

// Open image modal for full-size viewing
function openImageModal(src) {
    // Close any existing modal first
    closeImageModal();

    // Create modal HTML
    const modalHtml = `
        <div id="image-modal" class="image-modal" onclick="closeImageModal()">
            <div class="modal-content" onclick="event.stopPropagation()">
                <button class="modal-close" onclick="closeImageModal()" title="Close">×</button>
                <img src="${src}" alt="Full size image" class="modal-image">
            </div>
        </div>
    `;

    // Add to body
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Prevent body scrolling and improve focus handling
    document.body.style.overflow = 'hidden';

    // Add keyboard event listener for Escape key
    const modal = document.getElementById('image-modal');
    if (modal) {
        // Focus the modal for keyboard navigation
        modal.focus();

        // Add event listener for Escape key
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                closeImageModal();
                document.removeEventListener('keydown', handleEscape);
            }
        };
        document.addEventListener('keydown', handleEscape);

        // Store the handler for cleanup
        modal._escapeHandler = handleEscape;
    }
}

// Close image modal
function closeImageModal() {
    const modal = document.getElementById('image-modal');
    if (modal) {
        // Clean up event listeners
        if (modal._escapeHandler) {
            document.removeEventListener('keydown', modal._escapeHandler);
        }

        // Remove modal
        modal.remove();

        // Restore body scrolling
        document.body.style.overflow = '';

        // Return focus to the document body
        document.body.focus();
    }
}

function sendMessage() {
    const input = document.getElementById('message-input');
    const text = input.value.trim();
    const comboSelector = document.getElementById('combo-selector');
    const selectedCombo = comboSelector.value;

    if (text) {
        // Archive current draft before sending
        archiveCurrentDraft();

        // Start scroll animation to hide history
        animateScrollForNewMessage(() => {
            // Animation completed, now send message to backend
            if (backend) {
                // Send message with combo information
                // Format: "combo_id|message_text" if combo is selected, otherwise just message_text
                let messageData = text;
                if (selectedCombo && selectedCombo !== '') {
                    messageData = `${selectedCombo}|${text}`;
                }
                backend.receiveTextInput(messageData);
            }
            input.value = '';
            input.style.height = '';
            document.getElementById('send-btn').disabled = true;

            // Clear draft timer
            clearDraftTimer();
        });
    }
}

function toggleVoiceInput() {
    if (backend) {
        backend.toggleAudioRecording();
    }
}

function openAudioSettings() {
    if (backend) {
        backend.receiveMessage('global', 'open_audio_settings', "");
    }
}

function sendAction(id, action) {
    if (backend) {
        backend.receiveMessage(id, action, "");
    }
}

function copyCode(btn) {
    const pre = btn.parentElement;
    const code = pre.querySelector('code').innerText;
    
    if (backend) {
        backend.copyToClipboard(code);
        
        const originalText = btn.innerText;
        btn.innerText = "Copied!";
        setTimeout(() => {
            btn.innerText = originalText;
        }, 2000);
    }
}

function scrollToBottom() {
    window.scrollTo(0, document.body.scrollHeight);
}

// --- Audio Control Functions ---

function toggleAudioPlayback() {
    if (audioState.isPlaying) {
        controlAudio('pause');
    } else {
        controlAudio('resume');
    }
}

function controlAudio(command, value = 0) {
    if (backend) {
        backend.controlAudio(command, value);
    }
}

function seekAudioFromBar(event) {
    if (audioState.duration <= 0) return;
    
    const container = event.currentTarget;
    const rect = container.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const percent = clickX / rect.width;
    
    // Calculate seek offset relative to current position
    // Or send absolute position if backend supports it.
    // Our backend seek_audio expects OFFSET.
    // Let's implement absolute seek in logic:
    // target = duration * percent
    // offset = target - current
    
    const targetPos = audioState.duration * percent;
    const offset = targetPos - audioState.position;
    
    controlAudio('seek', offset);
}

function formatTime(ms) {
    if (isNaN(ms) || ms < 0) return "0:00";
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

// --- Reasoning Functions ---

function renderReasoning(reasoning) {
    if (!reasoning) return '';

    // Simple text rendering for reasoning (can be enhanced with markdown if needed)
    return reasoning.replace(/&/g, "&")
        .replace(/</g, "<")
        .replace(/>/g, ">")
        .replace(/"/g, "&quot")
        .replace(/'/g, "&#039;")
        .replace(/\n/g, '<br>');
}

function toggleReasoning(messageId) {
    const reasoningContent = document.getElementById(`reasoning-${messageId}`);
    const toggleBtn = document.querySelector(`#msg-${messageId} .reasoning-toggle`);

    if (reasoningContent && toggleBtn) {
        const isCollapsed = reasoningContent.classList.contains('collapsed');

        if (isCollapsed) {
            reasoningContent.classList.remove('collapsed');
            toggleBtn.textContent = '▼';
            toggleBtn.title = 'Collapse reasoning';
        } else {
            reasoningContent.classList.add('collapsed');
            toggleBtn.textContent = '▶';
            toggleBtn.title = 'Expand reasoning';
        }
    }
}

function toggleGlobalReasoning() {
    showReasoning = !showReasoning;

    // Update all existing reasoning sections
    const reasoningSections = document.querySelectorAll('.reasoning');
    reasoningSections.forEach(section => {
        if (showReasoning) {
            section.style.display = 'block';
        } else {
            section.style.display = 'none';
        }
    });

    // Store preference (could be persisted to localStorage)
    if (typeof localStorage !== 'undefined') {
        localStorage.setItem('showReasoning', showReasoning);
    }
}

// --- Draft Management Functions ---

function startDraftTimer() {
    // Clear existing timer
    clearDraftTimer();

    // Start new timer
    draftTimer = setTimeout(() => {
        saveDraft();
    }, DRAFT_SAVE_INTERVAL);
}

function clearDraftTimer() {
    if (draftTimer) {
        clearTimeout(draftTimer);
        draftTimer = null;
    }
}

function saveDraft() {
    const input = document.getElementById('message-input');
    const content = input.value.trim();

    if (content && backend) {
        // Send draft to backend for saving
        backend.receiveMessage('draft', 'save', content);
    }
}

function archiveCurrentDraft() {
    if (backend) {
        backend.receiveMessage('draft', 'archive');
    }
}

// Load reasoning preference on init
function loadReasoningPreference() {
    if (typeof localStorage !== 'undefined') {
        const saved = localStorage.getItem('showReasoning');
        if (saved !== null) {
            showReasoning = saved === 'true';
        }
    }
}

// --- Combo Management Functions ---

function loadCombos(combos, defaultComboId = '') {
    const comboSelector = document.getElementById('combo-selector');
    if (!comboSelector) return;

    // Clear existing options except the first one
    comboSelector.innerHTML = '';

    // Add default option (use default settings)
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Default';
    comboSelector.appendChild(defaultOption);

    // Add combo options
    if (combos) {
        Object.keys(combos).forEach(comboId => {
            const combo = combos[comboId];
            const option = document.createElement('option');
            option.value = comboId;
            option.textContent = combo.name || comboId;
            comboSelector.appendChild(option);
        });
    }

    // Set default selection
    if (defaultComboId && combos && combos[defaultComboId]) {
        comboSelector.value = defaultComboId;
    } else {
        comboSelector.value = '';
    }
}

function initCombos() {
    // Request combos from backend when ready
    if (backend) {
        // For now, we'll call this from Python side after UI is loaded
        // The backend will call loadCombos with the combo data
    }
}

// --- Gallery Configuration Functions ---

// Set gallery thumbnail size
function setGalleryThumbnailSize(size) {
    if (typeof size === 'number' && size > 0) {
        document.documentElement.style.setProperty('--gallery-thumbnail-size', `${size}px`);
    }
}

// Get current gallery thumbnail size
function getGalleryThumbnailSize() {
    const computed = getComputedStyle(document.documentElement);
    const size = computed.getPropertyValue('--gallery-thumbnail-size');
    return parseInt(size) || 150;
}

// Reset gallery to default settings
function resetGallerySettings() {
    document.documentElement.style.removeProperty('--gallery-thumbnail-size');
    document.documentElement.style.removeProperty('--gallery-max-height');
}

// --- Accessibility Improvements ---

// Add ARIA labels and keyboard navigation
function enhanceGalleryAccessibility() {
    // Add ARIA labels to gallery images
    document.querySelectorAll('.gallery-image img').forEach(img => {
        if (!img.getAttribute('alt')) {
            img.setAttribute('alt', 'Gallery image');
        }
    });

    // Add keyboard navigation for image modals
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeImageModal();
        }
    });
}

// --- Lazy Loading Functions ---

function initializeLazyLoading(totalCount) {
    lazyLoadingState.totalMessageCount = totalCount;
    lazyLoadingState.loadedMessageCount = 0;
    lazyLoadingState.hasMoreMessages = totalCount > 0;

    console.log(`Initialized lazy loading with ${totalCount} total messages`);
}

function loadMessagePage(messages, prepend = false) {
    if (!messages || messages.length === 0) {
        console.log('No messages to load');
        return;
    }

    const container = document.getElementById('chat-container');

    if (prepend) {
        // Add messages at the top (older messages)
        const fragment = document.createDocumentFragment();

        messages.forEach(msg => {
            const msgDiv = createMessageElement(msg);
            fragment.appendChild(msgDiv);
        });

        // Insert before first child or append if no children
        if (container.firstChild) {
            container.insertBefore(fragment, container.firstChild);
        } else {
            container.appendChild(fragment);
        }

        // Update loaded count
        lazyLoadingState.loadedMessageCount += messages.length;

        console.log(`Prepended ${messages.length} older messages, total loaded: ${lazyLoadingState.loadedMessageCount}`);
    } else {
        // Add messages at the bottom (normal case for new messages or initial load)
        messages.forEach(msg => {
            appendMessage(msg.id, msg.role, msg.content, false, msg.timestamp);
        });

        // Update loaded count for initial load
        lazyLoadingState.loadedMessageCount += messages.length;

        console.log(`Appended ${messages.length} messages, total loaded: ${lazyLoadingState.loadedMessageCount}`);
    }

    // Check if we've loaded all messages
    if (lazyLoadingState.loadedMessageCount >= lazyLoadingState.totalMessageCount) {
        lazyLoadingState.hasMoreMessages = false;
        console.log('All messages loaded');
    }
}

function createMessageElement(msg) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${msg.role}`;
    msgDiv.id = `msg-${msg.id}`;
    msgDiv.dataset.id = msg.id;
    msgDiv.dataset.role = msg.role;

    // Header
    const header = document.createElement('div');
    header.className = 'message-header';
    let roleDisplayName = 'You';
    if (msg.role === 'assistant') {
        roleDisplayName = 'Assistant';
    } else if (msg.role === 'system') {
        roleDisplayName = 'System';
    }
    header.innerHTML = `
        <span class="role-name">${roleDisplayName}</span>
        <span class="timestamp">${msg.timestamp || 'unknown'}</span>
    `;
    msgDiv.appendChild(header);

    // Content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    contentDiv.innerHTML = renderContent(msg.content);
    msgDiv.appendChild(contentDiv);

    // Finalize gallery rendering for loaded messages
    finalizeGalleryRendering(contentDiv);

    return msgDiv;
}

function initScrollDetection() {
    // Throttle scroll events for performance
    let scrollTimeout = null;

    window.addEventListener('scroll', function() {
        if (scrollTimeout) {
            clearTimeout(scrollTimeout);
        }

        scrollTimeout = setTimeout(function() {
            checkScrollForLazyLoading();
        }, 100); // Check every 100ms max
    });

    console.log('Scroll detection initialized');
}

function checkScrollForLazyLoading() {
    // Only trigger if we have more messages to load and not currently loading
    if (!lazyLoadingState.hasMoreMessages || lazyLoadingState.isLoading) {
        return;
    }

    // Check if user has scrolled near the top (within 200px of top)
    const scrollTop = window.scrollY;

    if (scrollTop < 200) {
        // Prevent multiple simultaneous requests
        lazyLoadingState.isLoading = true;

        console.log('Triggering lazy load - user scrolled near top');

        // Request more messages from backend
        if (backend && backend.requestLoadMoreMessages) {
            backend.requestLoadMoreMessages();
        } else {
            console.error('Backend not available or requestLoadMoreMessages not implemented');
            lazyLoadingState.isLoading = false;
        }
    }
}

// Called by Python when lazy loading request completes
function onLazyLoadComplete(success = true) {
    lazyLoadingState.isLoading = false;

    if (success) {
        console.log('Lazy loading completed successfully');
    } else {
        console.log('Lazy loading failed');
    }
}

// --- Initialization ---

// Initialize reasoning preference and combos
document.addEventListener("DOMContentLoaded", function () {
    loadReasoningPreference();
    initCombos();
    enhanceGalleryAccessibility();
});
