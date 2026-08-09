document.addEventListener('DOMContentLoaded', () => {
    const videoInput = document.getElementById('video-input');
    const fileNameDisplay = document.getElementById('file-name-display');
    const uploadBtn = document.getElementById('upload-btn');
    const classifyBtn = document.getElementById('classify-btn');
    const resultsContainer = document.getElementById('results-container');
    const queryInfo = document.getElementById('query-info');

    let selectedFile = null;

    videoInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            fileNameDisplay.innerText = selectedFile.name;
            uploadBtn.disabled = false;
        } else {
            selectedFile = null;
            fileNameDisplay.innerText = 'No file selected';
            uploadBtn.disabled = true;
        }
    });

    async function uploadAndClassify() {
        if (!selectedFile) return;

        uploadBtn.innerHTML = '<span class="spinner"></span> Uploading & Classifying...';
        uploadBtn.disabled = true;
        classifyBtn.disabled = true;
        resultsContainer.innerHTML = '<div class="placeholder-text"><span class="spinner"></span> Extracting frames & audio and running models...</div>';
        queryInfo.innerText = '';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/api/classify', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.success) {
                renderClassification(data);
            } else {
                resultsContainer.innerHTML = `<div class="placeholder-text" style="color: #ff4444;">Error processing video: ${data.error}</div>`;
            }
        } catch (error) {
            resultsContainer.innerHTML = `<div class="placeholder-text" style="color: #ff4444;">Connection error. Backend server might be starting up.</div>`;
        } finally {
            uploadBtn.innerHTML = 'Upload & Classify MP4';
            uploadBtn.disabled = false;
            classifyBtn.disabled = false;
        }
    }

    async function runSampleClassification() {
        classifyBtn.innerHTML = '<span class="spinner"></span> Running Sample...';
        classifyBtn.disabled = true;
        uploadBtn.disabled = true;
        resultsContainer.innerHTML = '<div class="placeholder-text"><span class="spinner"></span> Classifying sample video...</div>';
        queryInfo.innerText = '';

        try {
            const response = await fetch('/api/classify', {
                method: 'POST'
            });
            const data = await response.json();

            if (data.success) {
                renderClassification(data);
            } else {
                resultsContainer.innerHTML = `<div class="placeholder-text" style="color: #ff4444;">Error: ${data.error}</div>`;
            }
        } catch (error) {
            resultsContainer.innerHTML = `<div class="placeholder-text" style="color: #ff4444;">Connection error. Backend server might be starting up.</div>`;
        } finally {
            classifyBtn.innerHTML = 'Classify Sample Test Video';
            classifyBtn.disabled = false;
            uploadBtn.disabled = selectedFile ? false : true;
        }
    }

    uploadBtn.addEventListener('click', uploadAndClassify);
    classifyBtn.addEventListener('click', runSampleClassification);

    function renderClassification(data) {
        queryInfo.innerText = `| File: ${data.video_id}`;
        resultsContainer.innerHTML = '';

        // Media Player Section for Uploaded Video / Audio
        if (data.video_url || data.audio_url) {
            const playerBox = document.createElement('div');
            playerBox.className = 'media-player-box glass-panel';
            playerBox.style.gridColumn = '1 / -1';
            playerBox.style.padding = '20px';
            playerBox.style.marginBottom = '15px';
            playerBox.style.borderRadius = '12px';

            let videoHtml = data.video_url ? `
                <div style="flex: 1; min-width: 280px;">
                    <div style="font-size: 0.85rem; color: #a0aec0; margin-bottom: 8px; font-weight: 500;">🎬 Uploaded Video Playback</div>
                    <video controls style="width: 100%; max-height: 220px; border-radius: 8px; background: #000; border: 1px solid rgba(255,255,255,0.1);">
                        <source src="${data.video_url}" type="video/mp4">
                        Your browser does not support HTML5 video playback.
                    </video>
                </div>` : '';

            let audioHtml = data.audio_url ? `
                <div style="flex: 1; min-width: 280px;">
                    <div style="font-size: 0.85rem; color: #a0aec0; margin-bottom: 8px; font-weight: 500;">🔊 Extracted 16kHz Audio Playback</div>
                    <audio controls style="width: 100%; margin-top: 10px;">
                        <source src="${data.audio_url}" type="audio/wav">
                        Your browser does not support HTML5 audio playback.
                    </audio>
                </div>` : '';

            playerBox.innerHTML = `<div style="display: flex; gap: 20px; flex-wrap: wrap; align-items: center;">${videoHtml}${audioHtml}</div>`;
            resultsContainer.appendChild(playerBox);
        }

        // 3 Model Prediction Cards
        data.results.forEach(res => {
            const card = document.createElement('div');
            card.className = 'result-card';

            const hasFrames = data.frames && data.frames.length > 0;
            const initialBg = hasFrames ? `background-image: url('${data.frames[0]}'); background-size: cover; background-position: center;` : `background: linear-gradient(135deg, #1a1a2e, #16213e);`;

            card.innerHTML = `
                <div class="card-thumbnail" style="${initialBg} position: relative; height: 160px; border-radius: 8px;">
                    ${hasFrames ? `
                    <div class="play-overlay" style="position:absolute; top:0; left:0; right:0; bottom:0; display:flex; align-items:center; justify-content:center; background:rgba(0,0,0,0.4); opacity:0; transition:opacity 0.2s;">
                        <span style="font-size: 13px; color: white; background: rgba(0,0,0,0.7); padding: 5px 10px; border-radius: 4px;">▶ Hover for 8 Frames</span>
                    </div>` : ''}
                    <div class="match-badge" style="position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.75); padding: 4px 8px; border-radius: 4px; border: 1px solid var(--neon-cyan);">
                        ${res.confidence}% Conf.
                    </div>
                </div>
                <div style="padding: 15px 5px;">
                    <div style="font-size: 13px; color: #a0aec0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">${res.modality}</div>
                    <div class="card-title" style="font-size: 18px; font-weight: bold; margin-bottom: 8px; color: #ffffff;">${res.predicted_class}</div>
                    <div style="font-size: 14px; margin-bottom: 12px; color: #00ff88; font-weight: 500;">Confidence: ${res.confidence}%</div>
                </div>
            `;

            if (hasFrames) {
                const thumbnail = card.querySelector('.card-thumbnail');
                const overlay = card.querySelector('.play-overlay');
                let frameInterval;

                thumbnail.addEventListener('mouseenter', () => {
                    if (overlay) overlay.style.opacity = '1';
                    let currentFrame = 0;
                    frameInterval = setInterval(() => {
                        currentFrame = (currentFrame + 1) % data.frames.length;
                        thumbnail.style.backgroundImage = `url('${data.frames[currentFrame]}')`;
                    }, 180);
                });

                thumbnail.addEventListener('mouseleave', () => {
                    if (overlay) overlay.style.opacity = '0';
                    clearInterval(frameInterval);
                    thumbnail.style.backgroundImage = `url('${data.frames[0]}')`;
                });
            }

            resultsContainer.appendChild(card);
        });
    }
});
