// Initialize THREE.js scene
const canvas = document.getElementById("pointcloud-canvas");
const renderer = new THREE.WebGLRenderer({ canvas });
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5;

// Add axes helper
scene.add(new THREE.AxesHelper(5));

// Function to render point cloud
function renderPointCloud(points) {
  // Remove old point cloud
  while (scene.children.length > 1) {
    const oldObject = scene.children.pop();
    oldObject.geometry.dispose();
    oldObject.material.dispose();
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(points, 3));

  const material = new THREE.PointsMaterial({
    color: 0x00ff00,
    size: 0.05,
  });

  const pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);
}

// WebSocket connection
function connectWebSocket() {
  const socket = new WebSocket("ws://localhost:8080"); // Change localhost to server IP if needed

  socket.onopen = () => {
    console.log("WebSocket connected");
  };

  socket.onmessage = (event) => {
    const buffer = event.data; // Received binary data
    const points = new Float32Array(buffer); // Convert to Float32Array
    renderPointCloud(points); // Render the point cloud
  };

  socket.onclose = () => {
    console.log("WebSocket disconnected");
  };

  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
  };
}

// Start WebSocket connection
connectWebSocket();

// Render loop
function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();