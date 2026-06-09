// background/background.js
const API_BASE = "https://api.bhakarwadi.team";
const SALT = "your_super_secret_salt_here_make_it_long"; // Shared secret

// Function to generate an HMAC-SHA256 signature equivalent for security
async function generateSignature(dataString) {
    const encoder = new TextEncoder();
    const keyData = encoder.encode(SALT);
    const key = await crypto.subtle.importKey("raw", keyData, { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
    const signatureBuffer = await crypto.subtle.sign("HMAC", key, encoder.encode(dataString));
    return Array.from(new Uint8Array(signatureBuffer)).map(b => b.toString(16).padStart(2, '0')).join('');
}

// Intercept messages from the content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'processImage') {
        processImageBg(request.url).then(sendResponse);
        return true;
    }
    else if (request.action === 'checkHash') {
        checkReport(request.hash).then(sendResponse);
        return true; 
    } 
    else if (request.action === 'reportHash') {
        sendReport(request.hash, request.ext, request.url, request.status).then(sendResponse);
        return true;
    }
    else if (request.action === 'updateBadge') {
        chrome.action.setBadgeText({ text: request.count.toString(), tabId: sender.tab.id });
        chrome.action.setBadgeBackgroundColor({ color: '#d32f2f' });
    }
    else if (request.action === 'disableProtection') {
        chrome.action.setIcon({ path: '../icons/icon_gray16.png', tabId: sender.tab.id });
    }
});

async function checkReport(hash) {
    try {
        console.log("Background checking DB for hash:", hash);
        const response = await fetch(`${API_BASE}/api.php?request=check&hash=${hash}`);
        const result = await response.json();
        console.log("Background check DB result:", result);
        return result;
    } catch(e) {
        console.error("Background API Check Error:", e);
        return { found: false };
    }
}

async function sendReport(hash, ext, url, status) {
    console.log("Background sending report to DB for hash:", hash, "Status:", status);
    // Include database setup API key directly as fallback for stripped headers
    const payload = JSON.stringify({ hash, ext, url, status, token: "ext_key_9a8b7c6d5e4f3g2h1i0j" });
    const signature = await generateSignature(payload);
    
    try {
        const response = await fetch(`${API_BASE}/api.php?request=report`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Signature': signature
            },
            body: payload
        });
        const result = await response.json();
        console.log("Background report saved DB result:", result);
        return result;
    } catch(e) {
        console.error("Background API Report Error:", e);
        return { error: 'Failed' };
    }
}

async function processImageBg(url) {
    console.log("Background processing image to bypass CORS:", url);
    try {
        // Fetch runs in the background script to bypass webpage CORS limits
        const resp = await fetch(url);
        const buffer = await resp.arrayBuffer();
        
        // Hash it here
        const hashBuf = await crypto.subtle.digest('SHA-256', buffer);
        const hash = Array.from(new Uint8Array(hashBuf)).map(b => b.toString(16).padStart(2, '0')).join('');
        
        // Convert to Base64 dataUrl so content.js ML canvas won't be tainted
        const type = resp.headers.get('content-type') || 'image/jpeg';
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        
        return { success: true, hash: hash, dataUrl: `data:${type};base64,${btoa(binary)}` };
    } catch(e) {
        console.error("Background CORS fetch error for url:", url, e);
        return { success: false, error: e.toString() };
    }
}
