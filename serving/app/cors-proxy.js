const http = require('http');
const https = require('https');
const url = require('url');

const PORT = 3000;
const TARGET_HOST = 'localhost';
const TARGET_PORT = 8080;

const server = http.createServer((req, res) => {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  // Proxy the request
  const options = {
    hostname: TARGET_HOST,
    port: TARGET_PORT,
    path: req.url,
    method: req.method,
    headers: req.headers
  };

  const proxyReq = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res, { end: true });
  });

  proxyReq.on('error', (err) => {
    console.error('Proxy error:', err);
    res.writeHead(500);
    res.end('Proxy error');
  });

  req.pipe(proxyReq, { end: true });
});

server.listen(PORT, () => {
  console.log(`CORS proxy server running on http://localhost:${PORT}`);
  console.log(`Proxying requests to http://${TARGET_HOST}:${TARGET_PORT}`);
});