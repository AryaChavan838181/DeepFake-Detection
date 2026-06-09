<?php
// hostinger_api/config.php
define('DB_HOST', 'localhost'); // Usually localhost on Hostinger
define('DB_USER', 'u131025371_dfdetective');
define('DB_PASS', 'Dfdetective@12345');
define('DB_NAME', 'u131025371_dfdetective');
define('SECRET_SALT', 'your_super_secret_salt_here_make_it_long');

$conn = new mysqli(DB_HOST, DB_USER, DB_PASS, DB_NAME);

if ($conn->connect_error) {
    die(json_encode(["error" => "Database connection failed."]));
}

// Basic Security: Validate HMAC Signature to prevent Burp/Wireshark replay formatting easily
function validate_signature($payload, $provided_signature) {
    $expected_signature = hash_hmac('sha256', $payload, SECRET_SALT);
    return hash_equals($expected_signature, $provided_signature);
}
?>