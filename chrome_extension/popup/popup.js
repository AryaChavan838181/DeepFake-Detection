// popup/popup.js
document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('protectionToggle');
    const threatCount = document.getElementById('threatCount');
    const viewReportBtn = document.getElementById('viewReportBtn');

    // Init values
    chrome.storage.local.get(['fakeCount'], (result) => {
        threatCount.textContent = result.fakeCount || 0;
    });

    chrome.storage.sync.get(['realTimeProtection'], (data) => {
        toggle.checked = data.realTimeProtection !== false; // Default true
    });

    // Update listeners
    toggle.addEventListener('change', () => {
        chrome.storage.sync.set({ realTimeProtection: toggle.checked });
        
        let path = toggle.checked ? "icons/icon16.png" : "icons/icon_gray16.png"; 
        
        chrome.action.setIcon({ path: '../' + path });
    });

    viewReportBtn.addEventListener('click', () => {
        chrome.tabs.create({ url: chrome.runtime.getURL('popup/report.html') });
    });
});
