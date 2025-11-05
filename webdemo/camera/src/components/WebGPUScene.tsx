import { useLayoutEffect, useRef, useState, useCallback } from 'preact/hooks';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SceneControls } from './SceneControls';
import { CameraObject, CameraParameters } from './CameraObject';
import './WebGPUScene.css';

interface WebGPUSceneProps {
  className?: string;
}

export function WebGPUScene({ className }: WebGPUSceneProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    gizmo: THREE.Group;
    gridHelper: THREE.GridHelper;
    cameraObject: CameraObject;
    animationId?: number;
    initialCameraPosition: THREE.Vector3;
  } | null>(null);
  const [isInitialized, setIsInitialized] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  console.log('WebGPUScene render - mountRef.current:', mountRef.current);

  const resetCamera = useCallback(() => {
    if (!sceneRef.current) return;
    
    const { camera, controls, initialCameraPosition } = sceneRef.current;
    camera.position.copy(initialCameraPosition);
    controls.target.set(0, 0, 0);
    controls.update();
  }, []);

  const toggleWireframe = useCallback((enabled: boolean) => {
    if (!sceneRef.current) return;
    
    sceneRef.current.gizmo.traverse((child) => {
      if (child instanceof THREE.Mesh && child.material instanceof THREE.MeshLambertMaterial) {
        child.material.wireframe = enabled;
      }
    });
  }, []);

  const toggleGrid = useCallback((enabled: boolean) => {
    if (!sceneRef.current) return;
    
    sceneRef.current.gridHelper.visible = enabled;
  }, []);

  const updateCameraFocal = useCallback((fx: number, fy: number) => {
    if (!sceneRef.current) return;
    
    const params = sceneRef.current.cameraObject.getParameters();
    sceneRef.current.cameraObject.updateIntrinsics(
      fx, fy, params.cx, params.cy, params.width, params.height
    );
  }, []);

  const rotateCameraObject = useCallback((rx: number, ry: number, rz: number) => {
    if (!sceneRef.current) return;
    
    const params = sceneRef.current.cameraObject.getParameters();
    sceneRef.current.cameraObject.updateExtrinsics(
      params.position,
      new THREE.Euler(rx, ry, rz)
    );
  }, []);

  // Function to create default camera parameters
  const createDefaultCameraParameters = (): CameraParameters => {
    return {
      // Intrinsic parameters (typical camera)
      fx: 800,          // focal length x
      fy: 800,          // focal length y  
      cx: 320,          // principal point x (center)
      cy: 240,          // principal point y (center)
      width: 640,       // image width
      height: 480,      // image height
      
      // Extrinsic parameters (pose)
      position: new THREE.Vector3(2, 2, 2),
      rotation: new THREE.Euler(-0.3, 0.4, 0), // slight rotation to show orientation
      
      // Visual parameters
      frustumLength: 2.0, // how far to draw the FOV lines
    };
  };

  useLayoutEffect(() => {
    console.log('useLayoutEffect running - mountRef.current:', mountRef.current);
    let mounted = true;

    const initScene = () => {
      console.log('Starting scene initialization...');
      console.log('mountRef.current in initScene:', mountRef.current);
      
      if (!mountRef.current) {
        console.error('Mount ref is null');
        return false;
      }

      // Check if element has dimensions
      const rect = mountRef.current.getBoundingClientRect();
      console.log('Element rect:', rect);
      
      if (rect.width === 0 || rect.height === 0) {
        console.error('Mount element has no dimensions:', rect);
        return false;
      }

      console.log('Mount ref available with dimensions:', rect);

      try {
        // Scene setup
        console.log('Creating scene...');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a1a);

        // Camera setup
        console.log('Creating camera...');
        const camera = new THREE.PerspectiveCamera(
          75,
          rect.width / rect.height,
          0.1,
          1000
        );
        const initialCameraPosition = new THREE.Vector3(5, 5, 5);
        camera.position.copy(initialCameraPosition);

        // WebGL Renderer setup (more compatible than WebGPU for now)
        console.log('Creating renderer...');
        const renderer = new THREE.WebGLRenderer({ 
          antialias: true,
          alpha: true,
          powerPreference: "high-performance"
        });
        
        renderer.setSize(rect.width, rect.height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        
        console.log('Appending renderer to DOM...');
        mountRef.current.appendChild(renderer.domElement);

        // Orbit controls
        console.log('Creating orbit controls...');
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.enableZoom = true;
        controls.autoRotate = false;
        controls.target.set(0, 0, 0);
        controls.minDistance = 2;
        controls.maxDistance = 20;
        controls.enablePan = true;
        controls.panSpeed = 0.8;
        controls.rotateSpeed = 0.8;
        controls.zoomSpeed = 1.2;

        // Create gizmo
        console.log('Creating gizmo...');
        const gizmo = createGizmo();
        scene.add(gizmo);

        // Create camera object
        console.log('Creating camera object...');
        const cameraParameters = createDefaultCameraParameters();
        const cameraObject = new CameraObject(cameraParameters);
        scene.add(cameraObject.getGroup());

        // Lighting
        console.log('Adding lighting...');
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(5, 5, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.1;
        directionalLight.shadow.camera.far = 50;
        directionalLight.shadow.camera.left = -10;
        directionalLight.shadow.camera.right = 10;
        directionalLight.shadow.camera.top = 10;
        directionalLight.shadow.camera.bottom = -10;
        scene.add(directionalLight);

        // Grid helper
        console.log('Adding grid helper...');
        const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
        scene.add(gridHelper);

        // Store scene objects
        console.log('Storing scene objects...');
        sceneRef.current = {
          scene,
          camera,
          renderer,
          controls,
          gizmo,
          gridHelper,
          cameraObject,
          initialCameraPosition,
        };

        console.log('Setting initialized to true...');
        setIsInitialized(true);

        // Animation loop
        console.log('Starting animation loop...');
        const animate = () => {
          if (!mounted || !sceneRef.current) return;

          sceneRef.current.animationId = requestAnimationFrame(animate);
          
          // Update controls
          sceneRef.current.controls.update();
          
          // Render
          sceneRef.current.renderer.render(sceneRef.current.scene, sceneRef.current.camera);
        };

        animate();
        console.log('Scene initialization complete!');
        return true;

      } catch (err) {
        console.error('3D Scene initialization failed:', err);
        setError(err instanceof Error ? err.message : 'Unknown error occurred');
        return false;
      }
    };

    // Try immediate initialization first
    console.log('Trying immediate initialization...');
    if (initScene()) {
      console.log('Immediate initialization successful!');
      return () => {
        mounted = false;
        if (sceneRef.current) {
          if (sceneRef.current.animationId) {
            cancelAnimationFrame(sceneRef.current.animationId);
          }
          sceneRef.current.controls.dispose();
          sceneRef.current.renderer.dispose();
          sceneRef.current.cameraObject.dispose();
          if (mountRef.current && sceneRef.current.renderer.domElement) {
            mountRef.current.removeChild(sceneRef.current.renderer.domElement);
          }
        }
      };
    }

    // If immediate initialization fails, try with retries
    let attempts = 0;
    const maxAttempts = 10;
    
    const tryInit = () => {
      attempts++;
      console.log(`Initialization attempt ${attempts}/${maxAttempts}`);
      
      if (initScene()) {
        console.log('Initialization successful!');
        return;
      }
      
      if (attempts < maxAttempts) {
        setTimeout(tryInit, 200);
      } else {
        console.error('Failed to initialize after maximum attempts');
        setError('Failed to initialize 3D scene - element not ready');
      }
    };

    // Start retry initialization
    const timer = setTimeout(tryInit, 100);

    // Handle resize
    const handleResize = () => {
      if (!sceneRef.current || !mountRef.current) return;

      const { camera, renderer } = sceneRef.current;
      
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      mounted = false;
      clearTimeout(timer);
      window.removeEventListener('resize', handleResize);
      
      if (sceneRef.current) {
        if (sceneRef.current.animationId) {
          cancelAnimationFrame(sceneRef.current.animationId);
        }
        
        sceneRef.current.controls.dispose();
        sceneRef.current.renderer.dispose();
        sceneRef.current.cameraObject.dispose();
        
        if (mountRef.current && sceneRef.current.renderer.domElement) {
          mountRef.current.removeChild(sceneRef.current.renderer.domElement);
        }
      }
    };
  }, []);

  console.log('About to render, isInitialized:', isInitialized, 'error:', error);

  return (
    <div className={`threejs-scene ${className || ''}`}>
      <div 
        ref={mountRef} 
        className="scene-viewport"
        style={{ width: '100%', height: '100%' }}
      />
      
      {error && (
        <div className="scene-fallback-overlay">
          <div className="error-message">
            <h2>3D Scene Error</h2>
            <p>{error}</p>
            <p>Please refresh the page or try a different browser.</p>
          </div>
        </div>
      )}
      
      {!isInitialized && !error && (
        <div className="scene-loading-overlay">
          <div className="loading-message">
            <h2>Initializing 3D Scene...</h2>
            <div className="spinner"></div>
          </div>
        </div>
      )}
      
      {isInitialized && (
        <SceneControls
          onResetCamera={resetCamera}
          onToggleWireframe={toggleWireframe}
          onToggleGrid={toggleGrid}
          onUpdateCameraFocal={updateCameraFocal}
          onRotateCameraObject={rotateCameraObject}
        />
      )}
    </div>
  );
}

function createGizmo(): THREE.Group {
  const gizmo = new THREE.Group();

  // Create axes
  const axisLength = 2;
  const axisRadius = 0.05;

  // X-axis (Red)
  const xGeometry = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLength, 8);
  const xMaterial = new THREE.MeshLambertMaterial({ color: 0xff0000 });
  const xAxis = new THREE.Mesh(xGeometry, xMaterial);
  xAxis.rotation.z = -Math.PI / 2;
  xAxis.position.x = axisLength / 2;
  xAxis.castShadow = true;
  xAxis.receiveShadow = true;
  gizmo.add(xAxis);

  // X-axis arrow
  const xArrowGeometry = new THREE.ConeGeometry(axisRadius * 2, axisRadius * 4, 8);
  const xArrow = new THREE.Mesh(xArrowGeometry, xMaterial);
  xArrow.rotation.z = -Math.PI / 2;
  xArrow.position.x = axisLength + axisRadius * 2;
  xArrow.castShadow = true;
  gizmo.add(xArrow);

  // Y-axis (Green)
  const yGeometry = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLength, 8);
  const yMaterial = new THREE.MeshLambertMaterial({ color: 0x00ff00 });
  const yAxis = new THREE.Mesh(yGeometry, yMaterial);
  yAxis.position.y = axisLength / 2;
  yAxis.castShadow = true;
  yAxis.receiveShadow = true;
  gizmo.add(yAxis);

  // Y-axis arrow
  const yArrowGeometry = new THREE.ConeGeometry(axisRadius * 2, axisRadius * 4, 8);
  const yArrow = new THREE.Mesh(yArrowGeometry, yMaterial);
  yArrow.position.y = axisLength + axisRadius * 2;
  yArrow.castShadow = true;
  gizmo.add(yArrow);

  // Z-axis (Blue)
  const zGeometry = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLength, 8);
  const zMaterial = new THREE.MeshLambertMaterial({ color: 0x0000ff });
  const zAxis = new THREE.Mesh(zGeometry, zMaterial);
  zAxis.rotation.x = Math.PI / 2;
  zAxis.position.z = axisLength / 2;
  zAxis.castShadow = true;
  zAxis.receiveShadow = true;
  gizmo.add(zAxis);

  // Z-axis arrow
  const zArrowGeometry = new THREE.ConeGeometry(axisRadius * 2, axisRadius * 4, 8);
  const zArrow = new THREE.Mesh(zArrowGeometry, zMaterial);
  zArrow.rotation.x = Math.PI / 2;
  zArrow.position.z = axisLength + axisRadius * 2;
  zArrow.castShadow = true;
  gizmo.add(zArrow);

  // Center sphere
  const centerGeometry = new THREE.SphereGeometry(axisRadius * 1.5, 16, 16);
  const centerMaterial = new THREE.MeshLambertMaterial({ color: 0xffffff });
  const centerSphere = new THREE.Mesh(centerGeometry, centerMaterial);
  centerSphere.castShadow = true;
  centerSphere.receiveShadow = true;
  gizmo.add(centerSphere);

  return gizmo;
} 