// utils/hasher.js
async function generateFileHash(url) {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    res = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    return res;
}
