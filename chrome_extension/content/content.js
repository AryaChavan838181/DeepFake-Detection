// content/content.js

// Main script injected into pages to scan media elements
console.log("DFDetective active on:", window.location.href);

// Configure ONNX Web
ort.env.wasm.wasmPaths = chrome.runtime.getURL("lib/");
ort.env.wasm.numThreads = 1;

function initScanner() {
    chrome.storage.sync.get(['realTimeProtection'], function(data) {
        if (!data.realTimeProtection) return;
        
        scanImages();
        scanVideos();
    });
}

async function scanImages() {
    const images = document.querySelectorAll('img');
    for (let img of images) {
        // Prevent re-scanning and ignore pure base64 small icons
        if (!img.src || img.src.startsWith('data:') || img.dataset.dfdetectiveScanned) continue;
        
        img.dataset.dfdetectiveScanned = "true";
        
        try {
            // Check hash from background (to bypass heavy CORS limits on random websites)
            chrome.runtime.sendMessage({ action: 'processImage', url: img.src }, async (bgResp) => {
                if (chrome.runtime.lastError) {
                    console.error("DFDetective: Background script communication error:", chrome.runtime.lastError);
                    return;
                }
                
                if (!bgResp || !bgResp.success) {
                    console.warn(`DFDetective: Skipping unfetchable image (CORS/Private): ${img.src.substring(0,60)}...`);
                    return;
                }

                const hash = bgResp.hash;
                const ext = img.src.split('.').pop().split('?')[0] || 'jpg';
                
                console.log("DFDetective checking hash:", hash, "for image size:", bgResp.dataUrl.length);

                chrome.runtime.sendMessage({
                    action: 'checkHash',
                    hash: hash
                }, async (dbResp) => {
                    if (dbResp && dbResp.found) {
                        console.log("DFDetective: Found DB MATCH!", dbResp);
                        processDatabaseResponse(img, dbResp.status, hash);
                    } else {
                        console.log("DFDetective: DB miss, evaluating local AI model locally for:", hash);
                        
                        // Not in DB: Run WebGL AI Check using the safe CORS-free Base64 dataUrl we got from background script
                        const isFake = await runModelCheckLocally(bgResp.dataUrl);
                        const status = isFake ? 'fake' : 'real';
                        
                        console.log("DFDetective: Local Model predicted [", status, "] for image hash:", hash);
                        
                        // Display & Update DB
                        processDatabaseResponse(img, status, hash);
                        
                        chrome.runtime.sendMessage({
                            action: 'reportHash',
                            hash: hash,
                            ext: ext,
                            url: window.location.href,
                            status: status
                        });
                    }
                });
            });
        } catch (e) { console.error("DFDetective: Error processing image", img.src, e); }
    }
}

async function scanVideos() {
    const videos = document.querySelectorAll('video');
    if (videos.length > 0) {
        const allowFrameCapture = confirm("DFDetective wants to capture video frames for deepfake analysis. Allow?");
        
        if (!allowFrameCapture) {
            chrome.storage.sync.set({ realTimeProtection: false });
            chrome.runtime.sendMessage({ action: 'disableProtection' });
            alert("Real-time protection disabled for this site.");
            return;
        }
        
        for(let vid of videos) {
            // (Video logic here: capture frames on canvas, hash, and run model)
            console.log("Monitoring video:", vid.src);
        }
    }
}

function processDatabaseResponse(element, status, hash) {
    if (status === 'fake') {
        element.style.border = "4px solid #d32f2f";
        element.title = "DFDetective Warning: AI Generated/Deepfake Media Detected";
        
        // Notify popup script through storage and build log
        chrome.storage.local.get(['fakeCount', 'fakeLogs'], function(result) {
            let count = (result.fakeCount || 0) + 1;
            let logs = result.fakeLogs || [];
            
            // Only add to log if we didn't already log this element recently
            if(!logs.some(l => l.hash === hash && l.url === window.location.href)) {
                logs.push({
                    url: window.location.href,
                    src: element.src,
                    hash: hash || 'N/A',
                    time: new Date().toLocaleString()
                });
            }
            
            chrome.storage.local.set({fakeCount: count, fakeLogs: logs});
            chrome.runtime.sendMessage({action: 'updateBadge', count: count});
        });
    } else {
        element.title = "DFDetective Verified: Authentic Human Media";
    }
}

// Run Model local inference logic via ONNX runtime web
async function runModelCheckLocally(src) {
    console.log("Running local GPU model prediction for", src);
    
    try {
        // Load the image onto a canvas to format tensors
        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.src = src;
        await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = reject;
        });

        // The model (EfficientNet-B0) expects exactly 226x226 input size!
        const SIZE = 226;
        const canvas = document.createElement('canvas');
        canvas.width = SIZE;
        canvas.height = SIZE;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, SIZE, SIZE);
        const imgData = ctx.getImageData(0, 0, SIZE, SIZE).data;
        
        // Prepare Float32Array for tensor (RGB PyTorch normalization)
        const float32Data = new Float32Array(3 * SIZE * SIZE);
        
        for (let i = 0; i < SIZE * SIZE; i++) {
            let r = imgData[i * 4] / 255.0;
            let g = imgData[i * 4 + 1] / 255.0;
            let b = imgData[i * 4 + 2] / 255.0;
            
            float32Data[i] = r;                   // R channel
            float32Data[i + SIZE*SIZE] = g;       // G channel
            float32Data[i + 2*SIZE*SIZE] = b;     // B channel
        }
        
        const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, SIZE, SIZE]);

        // Fetch model from manual hostinger CDN
        const session = await ort.InferenceSession.create('https://api.bhakarwadi.team/models/image_model.onnx', {
            executionProviders: ['webgl', 'wasm']
        });
        
        const feeds = {};
        feeds[session.inputNames[0]] = inputTensor;
        
        const results = await session.run(feeds);
        const outputTensor = results[session.outputNames[0]];
        
        console.log("DFDetective ONNX Raw Output Tensor:", outputTensor.data);
        
        // As per real_predictor.py, the model outputs 1=Real and 0=Fake
        // So if the score is < 0.5, it is a Deepfake.
        let isFake = false;
        if (outputTensor.data.length >= 2) {
            isFake = outputTensor.data[0] < outputTensor.data[1]; 
        } else {
            // Because the pytorch model output is raw logits from nn.Linear, we apply Sigmoid safely if it's not already
            // Actually simply checking if < 0.0 for raw logits or < 0.5 for sigmoid logits
            let score = outputTensor.data[0];
            // If output doesn't seem to be in 0-1, apply sigmoid
            if (score < 0 || score > 1) {
                score = 1 / (1 + Math.exp(-score));
            }
            
            console.log("DFDetective Normalized Probability (1=Real, 0=Fake):", score);
            isFake = score < 0.5; // From real_predictor.py: prob_image < 0.5
        }
        
        return isFake;
        
    } catch (e) {
        console.error("Local Model execution failed, doing safe fallback", e);
        return false;
    }
}

window.addEventListener('load', initScanner);