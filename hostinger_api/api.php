<?php
// hostinger_api/api.php
header("Content-Type: application/json; charset=UTF-8");
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: POST, GET, OPTIONS");
header("Access-Control-Max-Age: 3600");
header("Access-Control-Allow-Headers: Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With, X-Signature");

require_once 'config.php';

$request = isset($_GET['request']) ? explode('/', trim($_GET['request'],'/')) : [];
$endpoint = isset($request[0]) ? $request[0] : null;

// Handle simple tokens (can be an API key passed in header)
$headers = apache_request_headers();
$api_key = isset($headers['Authorization']) ? str_replace('Bearer ', '', $headers['Authorization']) : '';
$signature = isset($headers['X-Signature']) ? $headers['X-Signature'] : '';

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

$raw_input = file_get_contents("php://input");
$data = json_decode($raw_input, true);

// Hostinger often strips custom headers like Authorization or X-Signature. a fast workaround is checking the payload directly:
$valid_auth = false;
if (isset($data['token'])) {
    $stmt = $conn->prepare("SELECT id FROM api_keys WHERE api_key = ? LIMIT 1");
    if ($stmt) {
        $stmt->bind_param("s", $data['token']);
        $stmt->execute();
        if ($stmt->get_result()->num_rows > 0) $valid_auth = true;
        $stmt->close();
    }
}

if ($_SERVER['REQUEST_METHOD'] === 'POST' && !$valid_auth && !validate_signature($raw_input, $signature)) {
    http_response_code(403);
    echo json_encode(["error" => "Forbidden: Invalid Signature or Forged Payload", "received_sig" => $signature]);
    exit();
}

switch ($endpoint) {
    case 'check':
        // GET /check?hash=abc
        if ($_SERVER['REQUEST_METHOD'] !== 'GET') exit;
        
        $hash = $conn->real_escape_string($_GET['hash']);
        $stmt = $conn->prepare("SELECT status FROM scan_results WHERE file_hash = ? LIMIT 1");
        $stmt->bind_param("s", $hash);
        $stmt->execute();
        $result = $stmt->get_result();
        
        if ($result->num_rows > 0) {
            $row = $result->fetch_assoc();
            echo json_encode(["found" => true, "status" => $row['status']]);
        } else {
            echo json_encode(["found" => false]);
        }
        $stmt->close();
        break;

    case 'report':
        // POST /report
        if ($_SERVER['REQUEST_METHOD'] !== 'POST') exit;
        
        $hash = $conn->real_escape_string($data['hash']);
        $ext = $conn->real_escape_string($data['ext']);
        $url = $conn->real_escape_string($data['url']);
        $status = $conn->real_escape_string($data['status']);

        // Insert or update on conflict
        $stmt = $conn->prepare("INSERT INTO scan_results (file_hash, media_extension, referrer_url, status) VALUES (?, ?, ?, ?) ON DUPLICATE KEY UPDATE status=? ");
        $stmt->bind_param("sssss", $hash, $ext, $url, $status, $status);
        if($stmt->execute()) {
            http_response_code(201);
            echo json_encode(["message" => "Report created."]);
        } else {
            http_response_code(500);
            echo json_encode(["error" => "Failed to create report."]);
        }
        $stmt->close();
        break;

    default:
        http_response_code(404);
        echo json_encode(["error" => "Endpoint not found."]);
        break;
}
$conn->close();
?>