document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('reportContainer');
    const clearBtn = document.getElementById('clearBtn');

    function loadLogs() {
        chrome.storage.local.get(['fakeLogs'], (result) => {
            const logs = result.fakeLogs || [];
            
            if (logs.length === 0) {
                container.innerHTML = "<p>No deepfakes detected yet.</p>";
                return;
            }

            container.innerHTML = "";
            logs.reverse().forEach(log => {
                const div = document.createElement('div');
                div.className = 'report-item';
                div.innerHTML = `
                    <p style="margin: 0 0 5px 0; font-weight: bold; font-size: 12px;">Detected: ${log.time}</p>
                    <img src="${log.src}" alt="Flagged Media" onerror="this.style.display='none'">
                    <div class="url-box"><strong>Page:</strong> <a href="${log.url}" target="_blank">${log.url}</a></div>
                    <div class="hash-box"><strong>Hash:</strong> ${log.hash}</div>
                `;
                container.appendChild(div);
            });
        });
    }

    clearBtn.addEventListener('click', () => {
        chrome.storage.local.set({ fakeLogs: [], fakeCount: 0 }, () => {
            chrome.action.setBadgeText({ text: '' });
            loadLogs();
        });
    });

    loadLogs();
});