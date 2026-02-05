import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

let renderer, scene, camera, controls;
let meshes_dict = new Map();
let previousKey = null;
let currentObject = null;
let isMeshVisible = true;
let toggleMeshBtn;

// Shared materials (avoid recreating)
const sharedPointsMat = new THREE.PointsMaterial({
  size: 2.0,              // big enough to see
  sizeAttenuation: false, // stay visible regardless of distance
  vertexColors: true
});
const sharedMeshMat = new THREE.MeshStandardMaterial({
  color: 0xaaaaaa,
  side: THREE.DoubleSide
});

function addLights() {
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(3, 5, 2);
  scene.add(dir);
}

function frameObject(obj) 
{
  const box = new THREE.Box3().setFromObject(obj);
  if (box.isEmpty()) return;

  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);

  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const fov = (camera.fov * Math.PI) / 180;
  const dist = (maxDim / (2 * Math.tan(fov / 2))) * 1.2;

  controls.target.copy(center);
  camera.position.set(center.x, center.y, center.z + dist);
  camera.near = dist / 100;
  camera.far = dist * 100;
  camera.updateProjectionMatrix();
  controls.update();
}

/* -------------------- PLY loading -------------------- */
function readPLY(key) {
  return new Promise((resolve, reject) => {
    const entry = meshes_dict.get(key);
    if (!entry || !entry.file) {
      reject(new Error("Missing file for key " + key));
      return;
    }

    const reader = new FileReader();
    const loader = new PLYLoader();

    reader.onload = (e) => {
      try {
        const geometry = loader.parse(e.target.result);

        // Ensure bounds exist (helps framing and general correctness)
        geometry.computeBoundingBox();
        geometry.computeBoundingSphere();

        const idx = geometry.getIndex();
        const isMesh = !!(idx && idx.count > 0);

        meshes_dict.set(key, { ...entry, geometry, isMesh });
        resolve();
      } catch (err) {
        reject(err);
      }
    };

    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(entry.file);
  });
}

/* -------------------- Switch/display -------------------- */
function switchMesh(key) {
  const entry = meshes_dict.get(key);
  if (!entry || !entry.geometry) return;

  if (currentObject) scene.remove(currentObject);

  if (!entry.group) {
    // If mesh needs normals
    if (entry.isMesh && !entry.geometry.attributes.normal) {
      entry.geometry.computeVertexNormals();
    }

    // Base points material; white fallback if no vertex colors
    const hasColor = !!entry.geometry.attributes.color;
    const pointsMat = hasColor ? sharedPointsMat : new THREE.PointsMaterial({
      size: 2.0,
      sizeAttenuation: false,
      color: 0xffffff
    });

    const group = new THREE.Group();

    if (entry.isMesh) {
      const mesh = new THREE.Mesh(entry.geometry, sharedMeshMat);
      mesh.name = "mesh";
      const points = new THREE.Points(entry.geometry, pointsMat);
      points.name = "points";
      group.add(mesh);
      group.add(points);
    } else {
      const points = new THREE.Points(entry.geometry, pointsMat);
      points.name = "points";
      group.add(points);
    }

    entry.group = group;
    meshes_dict.set(key, entry);
  }

  currentObject = entry.group;
  currentObject.visible = true;
  const meshChild = currentObject.getObjectByName("mesh");
  if (meshChild) {
    meshChild.visible = isMeshVisible;
  }
  scene.add(currentObject);
  frameObject(currentObject);
  previousKey = key;
}

function handleMeshChange(event) {
  const key = event.target.value;
  if (key === previousKey) return;

  const entry = meshes_dict.get(key);
  if (!entry) return;

  if (!entry.geometry) {
    readPLY(key).then(() => switchMesh(key)).catch(console.error);
  } else {
    switchMesh(key);
  }
}

/* -------------------- Folder selection -------------------- */
function handleFolder(e) {
  const files = Array.from(e.target.files || [])
    .filter(f => /\.ply$/i.test(f.name))
    .sort((a, b) => a.name.localeCompare(b.name));

  if (files.length === 0) {
    alert("No PLY files found.");
    return;
  }

  const dropdown = document.getElementById("fileList");
  dropdown.innerHTML = "";
  meshes_dict.clear();
  previousKey = null;

  // Clear current object from scene (don’t dispose: cached objects were cleared)
  if (currentObject) {
    scene.remove(currentObject);
    currentObject = null;
  }

  files.forEach((file, i) => {
    const key = "Mesh" + i;
    meshes_dict.set(key, { file });

    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = file.name;
    dropdown.appendChild(opt);
  });

  dropdown.onchange = handleMeshChange;

  // ✅ auto-load first file so you see something immediately
  dropdown.selectedIndex = 0;
  dropdown.dispatchEvent(new Event("change"));
}

/* -------------------- Init -------------------- */
function init() {
  const container = document.getElementById("viewport");
  if (!container) throw new Error("Missing #viewport");

  const folderInput = document.getElementById("folderInput");
  const loadFolderBtn = document.getElementById("loadFolderBtn");
  toggleMeshBtn = document.getElementById("toggleMeshBtn");

  loadFolderBtn.addEventListener("click", () => {
    folderInput.value = "";
    folderInput.click();
  });

  folderInput.addEventListener("change", handleFolder);

  function setMeshVisibility(nextVisible) {
    isMeshVisible = nextVisible;
    if (currentObject) {
      const meshChild = currentObject.getObjectByName("mesh");
      if (meshChild) meshChild.visible = isMeshVisible;
    }

    if (toggleMeshBtn) {
      toggleMeshBtn.classList.toggle("is-active", isMeshVisible);
      toggleMeshBtn.textContent = isMeshVisible ? "Hide Mesh (M)" : "Show Mesh (M)";
    }
  }

  toggleMeshBtn?.addEventListener("click", () => {
    setMeshVisibility(!isMeshVisible);
  });

  document.addEventListener("keydown", (evt) => {
    if (evt.key === "m" || evt.key === "M") {
      setMeshVisibility(!isMeshVisible);
    }
  });

  function bindRangeValue(inputId, valueId, formatter = (v) => v) {
    const input = document.getElementById(inputId);
    const value = document.getElementById(valueId);
    if (!input || !value) return;
    const update = () => {
      value.textContent = formatter(input.value);
    };
    input.addEventListener("input", update);
    update();
  }

  bindRangeValue("hMin", "hMinValue");
  bindRangeValue("hMax", "hMaxValue");
  bindRangeValue("pc2Percentile", "pc2PercentileValue");
  bindRangeValue("inflationDistance", "inflationDistanceValue", (v) => Number(v).toFixed(1));

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  addLights(); // ✅ needed for MeshStandardMaterial

  camera = new THREE.PerspectiveCamera(
    60,
    container.clientWidth / container.clientHeight,
    0.01,
    100000
  );
  camera.position.set(0, 0, 3);

  scene.add(new THREE.AxesHelper(1));

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  setMeshVisibility(true);

  window.addEventListener("resize", () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  });
}

document.addEventListener("DOMContentLoaded", init);