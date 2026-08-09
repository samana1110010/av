document.addEventListener('DOMContentLoaded', () => {
    const videoZone = document.getElementById('video-upload');
    const audioZone = document.getElementById('audio-upload');
    const videoInput = videoZone.querySelector('.file-input');
    const audioInput = audioZone.querySelector('.file-input');
    const startBtn = document.getElementById('start-btn');
    const resultsContainer = document.getElementById('results-container');
    const queryInfo = document.getElementById('query-info');

    let selectedFile = null;
    let selectedType = null;

    // Helper to handle zone activation
    function setActiveZone(type) {
        if (type === 'video') {
            videoZone.querySelector('.drop-zone').classList.add('active-zone');
            audioZone.querySelector('.drop-zone').classList.remove('active-zone');
        } else {
            audioZone.querySelector('.drop-zone').classList.add('active-zone');
            videoZone.querySelector('.drop-zone').classList.remove('active-zone');
        }
        startBtn.disabled = false;
    }

    // Input change events
    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            selectedType = 'video';
            setActiveZone('video');
            videoZone.querySelector('p').innerText = selectedFile.name;
            audioZone.querySelector('p').innerHTML = "Drag & Drop Audio or<br>Click to Upload (MP3, WAV)";
        }
    });

    audioInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            selectedType = 'audio';
            setActiveZone('audio');
            audioZone.querySelector('p').innerText = selectedFile.name;
            videoZone.querySelector('p').innerHTML = "Drag & Drop Video or<br>Click to Upload (MP4, AVI)";
        }
    });

    // Handle retrieval
    startBtn.addEventListener('click', async () => {
        if (!selectedType) return;

        // UI Loading State
        startBtn.innerHTML = '<span class="spinner"></span> Processing...';
        startBtn.disabled = true;
        resultsContainer.innerHTML = '<div class="placeholder-text"><span class="spinner"></span> Running inference...</div>';
        queryInfo.innerText = '';

        const formData = new FormData();
        if (selectedFile) {
            formData.append('file', selectedFile);
        }
        formData.append('type', selectedType);

        try {
            const response = await fetch('/api/retrieve', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.success) {
                renderResults(data.query, data.results);
            } else {
                resultsContainer.innerHTML = `<div class="placeholder-text" style="color: #ff4444;">Error: ${data.error}</div>`;
            }
        } catch (error) {
            resultsContainer.innerHTML = `<div class="placeholder-text" style="color: #ff4444;">Connection error. Backend might be down.</div>`;
        } finally {
            startBtn.innerHTML = 'Start Retrieval';
            startBtn.disabled = false;
        }
    });

    function renderResults(query, results) {
        queryInfo.innerText = `- ${query.title}`;
        resultsContainer.innerHTML = '';

        results.forEach(res => {
            const card = document.createElement('div');
            card.className = 'result-card';
            
            // Neon color based on type
            const neonClass = res.type === 'video' ? 'neon-purple' : 'neon-cyan';
            
            card.innerHTML = `
                <div class="card-thumbnail" data-frames='${JSON.stringify(res.frames)}' style="background-image: url('${res.frames[0]}'); background-size: cover; background-position: center; position: relative; cursor: pointer;">
                    <div class="play-overlay" style="position:absolute; top:0; left:0; right:0; bottom:0; display:flex; align-items:center; justify-content:center; background:rgba(0,0,0,0.4); opacity:0; transition:opacity 0.2s;">
                        <span style="font-size: 32px; color: white;">▶ Hover to Play</span>
                    </div>
                    <div class="match-badge" style="border-color: var(--${neonClass}); color: var(--${neonClass});">
                        ${res.score}% Match
                    </div>
                </div>
                <div class="card-title">${res.title}</div>
                <div class="card-type">${res.type} Source</div>
                <audio controls style="width: 100%; margin-top: 10px; height: 35px; border-radius: 5px;" onplay="this.parentElement.querySelector('.card-thumbnail').dispatchEvent(new Event('mouseenter'))" onpause="this.parentElement.querySelector('.card-thumbnail').dispatchEvent(new Event('mouseleave'))">
                    <source src="${res.audio_url}" type="audio/wav">
                </audio>
            `;
            
            // Add hover effect to play video frames
            const thumbnail = card.querySelector('.card-thumbnail');
            const overlay = card.querySelector('.play-overlay');
            let frameInterval;
            
            thumbnail.addEventListener('mouseenter', () => {
                overlay.style.opacity = '1';
                let currentFrame = 0;
                frameInterval = setInterval(() => {
                    currentFrame = (currentFrame + 1) % res.frames.length;
                    thumbnail.style.backgroundImage = \`url('\${res.frames[currentFrame]}')\`;
                }, 100); // 10 fps
            });
            
            thumbnail.addEventListener('mouseleave', () => {
                overlay.style.opacity = '0';
                clearInterval(frameInterval);
                thumbnail.style.backgroundImage = \`url('\${res.frames[0]}')\`; // Reset to first frame
            });
            
            resultsContainer.appendChild(card);
        });
    }
});
