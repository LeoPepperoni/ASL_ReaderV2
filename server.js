// On EC2 instance run:
// sudo apt-get update
// sudo apt-get install nodejs npm
// npm install ws

// Make sure the port (e.g., 8080) is open in your EC2 security group settings to allow inbound WebSocket connections.

// Run the WebSocket server on your EC2 instance:
// node server.js


const WebSocket = require('ws');

// Set up the WebSocket server on port 8080
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', (ws) => {
  console.log('Client connected');

  // Handle messages from clients
  ws.on('message', (message) => {
    // Broadcast the message to all connected clients
    wss.clients.forEach((client) => {
      if (client !== ws && client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    });
  });

  ws.on('close', () => {
    console.log('Client disconnected');
  });
});

console.log('WebSocket server is running on ws://localhost:8080');
