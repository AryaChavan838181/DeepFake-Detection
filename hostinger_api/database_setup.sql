CREATE TABLE IF NOT EXISTS scan_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_hash VARCHAR(255) NOT NULL,
    media_extension VARCHAR(10) NOT NULL,
    referrer_url TEXT NOT NULL,
    status ENUM('real', 'fake') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_hash (file_hash)
);

CREATE TABLE IF NOT EXISTS api_keys (
    id INT AUTO_INCREMENT PRIMARY KEY,
    api_key VARCHAR(64) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert a default API key for the extension to use
INSERT INTO api_keys (api_key) VALUES ('ext_key_9a8b7c6d5e4f3g2h1i0j');
