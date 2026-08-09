document.addEventListener('DOMContentLoaded', () => {
    const state = { file: null, previewTimers: new Set() };
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const selection = document.getElementById('selection');
    const fileName = document.getElementById('file-name');
    const fileMeta = document.getElementById('file-meta');
    const clearFile = document.getElementById('clear-file');
    const searchButton = document.getElementById('search-button');
    const results = document.getElementById('results');
    const resultSummary = document.getElementById('result-summary');
    const resultTemplate = document.getElementById('result-template');
    const finalOutput = document.getElementById('final-output');
    const outputVideo = document.getElementById('output-video');
    const outputDescription = document.getElementById('output-description');
    const downloadOutput = document.getElementById('download-output');
    const candidatesHeading = document.getElementById('candidates-heading');

    function clearSelection(resetResults = false) {
        state.file = null;
        fileInput.value = '';
        selection.hidden = true;
        dropZone.hidden = false;
        searchButton.disabled = true;
        if (resetResults) {
            stopPreviews();
            outputVideo.removeAttribute('src');
            finalOutput.hidden = true;
            candidatesHeading.hidden = true;
            resultSummary.textContent = 'Waiting for your video';
            results.className = 'results empty-state';
            results.innerHTML = '<div class="empty-graphic" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span></div><strong>Your finished video will appear here</strong><p>Upload a silent video to find and attach its soundtrack.</p>';
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    }

    function selectFile(file) {
        if (!file) return;
        if (file.type && !file.type.startsWith('video/')) {
            showError('Please choose a video file.');
            return;
        }
        state.file = file;
        fileName.textContent = file.name;
        fileMeta.textContent = `${formatBytes(file.size)} · silent video`;
        dropZone.hidden = true;
        selection.hidden = false;
        searchButton.disabled = false;
    }

    function formatBytes(bytes) {
        if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    fileInput.addEventListener('change', () => selectFile(fileInput.files[0]));
    clearFile.addEventListener('click', () => clearSelection(false));
    document.getElementById('new-video').addEventListener('click', () => clearSelection(true));

    ['dragenter', 'dragover'].forEach((name) => dropZone.addEventListener(name, (event) => {
        event.preventDefault();
        dropZone.classList.add('dragging');
    }));
    ['dragleave', 'drop'].forEach((name) => dropZone.addEventListener(name, (event) => {
        event.preventDefault();
        dropZone.classList.remove('dragging');
    }));
    dropZone.addEventListener('drop', (event) => selectFile(event.dataTransfer.files[0]));

    searchButton.addEventListener('click', async () => {
        if (!state.file) return;
        setLoading(true);
        const body = new FormData();
        body.append('type', 'video');
        body.append('file', state.file);

        try {
            const response = await fetch('/api/retrieve', { method: 'POST', body });
            const contentType = response.headers.get('content-type') || '';
            const payload = contentType.includes('application/json') ? await response.json() : null;
            if (!response.ok || !payload?.success) {
                throw new Error(payload?.error || `Server returned HTTP ${response.status}`);
            }
            if (!payload.output) throw new Error('The server did not create a synchronized video.');
            renderOutput(payload);
        } catch (error) {
            showError(error.message || 'Could not reach the composition service.');
        } finally {
            setLoading(false);
        }
    });

    function setLoading(loading) {
        searchButton.disabled = loading;
        searchButton.classList.toggle('loading', loading);
        searchButton.querySelector('span').textContent = loading ? 'Matching + syncing…' : 'Create synced video';
        if (loading) {
            finalOutput.hidden = true;
            candidatesHeading.hidden = true;
            resultSummary.textContent = 'Finding audio and rendering MP4';
            results.className = 'results loading-state';
            results.innerHTML = '<div class="loader"><i></i><i></i><i></i><i></i><i></i></div><strong>Creating your final video</strong><p>Matching the scene, fitting the audio, and encoding the output.</p>';
        }
    }

    function renderOutput(payload) {
        outputVideo.src = `${payload.output.url}?v=${Date.now()}`;
        downloadOutput.href = `${payload.output.download_url}&name=${encodeURIComponent(payload.output.filename)}`;
        downloadOutput.download = payload.output.filename;
        outputDescription.textContent = `Matched with “${payload.output.audio.title}” (${payload.output.audio.score.toFixed(1)}% similarity). The audio now runs for the full video duration.`;
        finalOutput.hidden = false;
        candidatesHeading.hidden = false;
        resultSummary.textContent = 'Ready to preview and download';
        renderCandidates(payload.results);
        document.getElementById('results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function renderCandidates(items) {
        stopPreviews();
        results.className = 'results results-list compact-results';
        results.innerHTML = '';
        items.forEach((item, index) => {
            const card = resultTemplate.content.firstElementChild.cloneNode(true);
            card.style.setProperty('--delay', `${index * 70}ms`);
            card.querySelector('.rank').textContent = String(item.rank).padStart(2, '0');
            card.querySelector('.class-label').textContent = item.rank === 1 ? 'Selected audio' : `Candidate ${item.rank}`;
            card.querySelector('.score').textContent = `${item.score.toFixed(1)}% similarity`;
            card.querySelector('h3').textContent = item.title;
            card.querySelector('.result-id').textContent = item.id;
            const image = card.querySelector('img');
            image.src = item.frames[0];
            image.alt = `${item.title} reference frame`;
            card.querySelector('.media-kind').textContent = 'audio';
            card.querySelector('audio').src = item.audio_url;
            card.querySelector('.open-video').href = item.video_url;
            let frameIndex = 0;
            const preview = card.querySelector('.media-preview');
            preview.addEventListener('mouseenter', () => {
                const timer = setInterval(() => {
                    frameIndex = (frameIndex + 1) % item.frames.length;
                    image.src = item.frames[frameIndex];
                }, 180);
                state.previewTimers.add(timer);
                preview.dataset.timer = String(timer);
            });
            preview.addEventListener('mouseleave', () => {
                const timer = Number(preview.dataset.timer);
                clearInterval(timer);
                state.previewTimers.delete(timer);
                frameIndex = 0;
                image.src = item.frames[0];
            });
            results.appendChild(card);
        });
    }

    function stopPreviews() {
        state.previewTimers.forEach(clearInterval);
        state.previewTimers.clear();
    }

    function showError(message) {
        finalOutput.hidden = true;
        candidatesHeading.hidden = true;
        resultSummary.textContent = 'Request needs attention';
        results.className = 'results error-state';
        results.innerHTML = '<span aria-hidden="true">!</span><strong>Unable to create the video</strong><p></p>';
        results.querySelector('p').textContent = message;
    }

    async function checkHealth() {
        const status = document.getElementById('system-status');
        try {
            const response = await fetch('/api/health');
            if (!response.ok) throw new Error();
            const health = await response.json();
            status.classList.add('online');
            status.querySelector('span:last-child').textContent = `Ready · ${health.device.toUpperCase()} · ${health.version}`;
            document.getElementById('gallery-count').textContent = health.gallery_size;
        } catch {
            status.classList.add('offline');
            status.querySelector('span:last-child').textContent = 'Service unavailable';
        }
    }

    checkHealth();
});
