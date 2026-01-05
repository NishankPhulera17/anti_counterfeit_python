document.addEventListener('DOMContentLoaded', () => {
    // Navigation active state
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-item').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
            const pageTitle = link.querySelector('span').textContent;
            const titleElement = document.getElementById('page-title');
            if (titleElement) titleElement.textContent = pageTitle;
        }
    });

    // Check availability of APIs
    if (typeof fetch === 'undefined') {
        alert('Your browser does not support fetch API. Please upgrade.');
    }
});

// Generator Logic
const setupGenerator = () => {
    const form = document.getElementById('generate-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const productId = document.getElementById('product-id').value;
        const resultDiv = document.getElementById('generate-result');
        const qrImage = document.getElementById('qr-image');
        const btn = form.querySelector('button');

        // Reset state
        const originalBtnText = btn.innerHTML;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating...';
        btn.disabled = true;

        try {
            const response = await fetch('/generate_qr_cdp', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ product_id: productId })
            });

            const data = await response.json();

            if (data.status === 'success') {
                resultDiv.style.display = 'block';
                qrImage.src = `data:image/png;base64,${data.qrCdpImage}`;

                // Fill details
                document.getElementById('res-product-id').textContent = productId;
                document.getElementById('res-serial-id').textContent = data.serial_id;
                document.getElementById('res-cdp-id').textContent = data.cdp_id;
            } else {
                alert('Error: ' + data.message);
            }
        } catch (err) {
            console.error(err);
            alert('Failed to generate QR code');
        } finally {
            btn.innerHTML = originalBtnText;
            btn.disabled = false;
        }
    });
};

// Verification Logic
const setupVerifier = () => {
    const video = document.getElementById('video-element');
    const verifyBtn = document.getElementById('verify-btn');
    const resultContainer = document.getElementById('verification-result');

    if (!video || !verifyBtn) return;

    let stream = null;
    let isScanning = false;
    let frames = [];
    const MAX_FRAMES = 5;

    // Start Camera
    const startCamera = async () => {
        try {
            // Using constraints for better quality on phones (facingMode: environment)
            const constraints = {
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                }
            };
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
            video.play();
        } catch (err) {
            console.error("Camera access denied:", err);
            alert("Could not access camera. Please allow permissions.");
            verifyBtn.disabled = true;
        }
    };

    startCamera();

    // Verification handler
    verifyBtn.addEventListener('click', async () => {
        if (isScanning) return;
        isScanning = true;

        const originalText = verifyBtn.innerHTML;
        verifyBtn.innerHTML = '<i class="fa-solid fa-camera fa-spin"></i> Scanning...';
        verifyBtn.disabled = true;
        resultContainer.style.display = 'none';

        frames = [];

        // Capture 3 frames with small delay (for liveness)
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');

        for (let i = 0; i < 3; i++) {
            ctx.drawImage(video, 0, 0);
            const base64 = canvas.toDataURL('image/png').split(',')[1];
            frames.push(base64);
            await new Promise(r => setTimeout(r, 200)); // 200ms delay between frames
        }

        // Use the middle frame as the main CDP image
        const cdpImage = frames[1];

        // Get selected conditions
        const labelCondition = document.querySelector('#label-group .toggle-btn.selected')?.dataset.value || null;
        const lightingCondition = document.querySelector('#lighting-group .toggle-btn.selected')?.dataset.value || null;

        try {
            const response = await fetch('/verify_cdp', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    cdp_image: cdpImage,
                    video_frames: frames,
                    label_condition: labelCondition,
                    lighting_condition: lightingCondition
                })
            });

            const data = await response.json();
            displayResult(data);
        } catch (err) {
            console.error("Verification error:", err);
            alert("Verification failed. Check console for details.");
        } finally {
            isScanning = false;
            verifyBtn.innerHTML = originalText;
            verifyBtn.disabled = false;
        }
    });
};

// Toggle Buttons Logic
const setupToggleControls = () => {
    document.querySelectorAll('.toggle-group').forEach(group => {
        group.addEventListener('click', (e) => {
            if (e.target.classList.contains('toggle-btn')) {
                // Deselect all others in group
                group.querySelectorAll('.toggle-btn').forEach(btn => btn.classList.remove('selected'));
                // Select clicked
                e.target.classList.add('selected');
            }
        });
    });
};

const displayResult = (data) => {
    const container = document.getElementById('verification-result');
    const icon = container.querySelector('.result-icon');
    const title = container.querySelector('.result-title');
    const score = document.getElementById('result-score');
    const liveness = document.getElementById('result-liveness');
    const message = document.getElementById('result-message');

    container.style.display = 'block';

    // Remove previous classes
    container.classList.remove('success', 'error');

    if (data.is_authentic) {
        container.classList.add('success');
        icon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
        title.textContent = 'Authentic Product';
        title.style.color = 'var(--accent-green)';
    } else {
        container.classList.add('error');
        icon.innerHTML = '<i class="fa-solid fa-circle-xmark"></i>';
        title.textContent = 'Verification Failed';
        title.style.color = 'var(--accent-red)';
    }

    score.textContent = (data.similarity_score * 100).toFixed(1) + '%';
    liveness.textContent = data.liveness_passed ? 'Passed' : 'Failed';
    message.textContent = data.message;

    // Also show metrics if available
    if (data.pattern_size) {
        console.log("Pattern Size:", data.pattern_size);
    }
};

// Initialize based on page
if (document.getElementById('generate-form')) setupGenerator();
if (document.getElementById('video-element')) {
    setupVerifier();
    setupToggleControls();
}
