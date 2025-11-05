import * as THREE from 'three';

export interface CameraParameters {
  // Intrinsic parameters
  fx: number; // focal length x
  fy: number; // focal length y
  cx: number; // principal point x
  cy: number; // principal point y
  width: number; // image width
  height: number; // image height
  
  // Extrinsic parameters (pose)
  position: THREE.Vector3;
  rotation: THREE.Euler;
  
  // Visual parameters
  frustumLength: number; // how far to draw the FOV lines
}

export class CameraObject {
  private group: THREE.Group;
  private parameters: CameraParameters;
  private intrinsicMatrix: THREE.Matrix3;
  private extrinsicMatrix: THREE.Matrix4;
  
  // Visual components
  private cameraBody: THREE.LineSegments;
  private frustumLines: THREE.LineSegments;
  private coordinateAxes: THREE.Group;
  private imagePlane: THREE.LineSegments;

  constructor(parameters: CameraParameters) {
    this.parameters = { ...parameters };
    this.group = new THREE.Group();
    this.intrinsicMatrix = new THREE.Matrix3();
    this.extrinsicMatrix = new THREE.Matrix4();
    
    this.updateMatrices();
    this.createVisualRepresentation();
    this.updateVisualFromMatrices();
  }

  private updateMatrices(): void {
    // Update intrinsic matrix K
    // K = [[fx,  0, cx],
    //      [ 0, fy, cy],
    //      [ 0,  0,  1]]
    this.intrinsicMatrix.set(
      this.parameters.fx, 0, this.parameters.cx,
      0, this.parameters.fy, this.parameters.cy,
      0, 0, 1
    );

    // Update extrinsic matrix (rotation + translation)
    const rotationMatrix = new THREE.Matrix4();
    rotationMatrix.makeRotationFromEuler(this.parameters.rotation);
    
    const translationMatrix = new THREE.Matrix4();
    translationMatrix.makeTranslation(
      this.parameters.position.x,
      this.parameters.position.y,
      this.parameters.position.z
    );
    
    this.extrinsicMatrix.multiplyMatrices(translationMatrix, rotationMatrix);
  }

  private createVisualRepresentation(): void {
    // Create camera body (simple box outline)
    this.createCameraBody();
    
    // Create coordinate axes (local gizmo)
    this.createCoordinateAxes();
    
    // Create frustum lines (FOV representation)
    this.createFrustumLines();
    
    // Create image plane
    this.createImagePlane();
    
    // Add all components to the group
    this.group.add(this.cameraBody);
    this.group.add(this.coordinateAxes);
    this.group.add(this.frustumLines);
    this.group.add(this.imagePlane);
  }

  private createCameraBody(): void {
    // Simple box representing camera body
    const size = 0.3;
    const depth = 0.2;
    
    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array([
      // Front face
      -size, -size, 0,   size, -size, 0,
       size, -size, 0,   size,  size, 0,
       size,  size, 0,  -size,  size, 0,
      -size,  size, 0,  -size, -size, 0,
      
      // Back face
      -size, -size, -depth,   size, -size, -depth,
       size, -size, -depth,   size,  size, -depth,
       size,  size, -depth,  -size,  size, -depth,
      -size,  size, -depth,  -size, -size, -depth,
      
      // Connect front to back
      -size, -size, 0,  -size, -size, -depth,
       size, -size, 0,   size, -size, -depth,
       size,  size, 0,   size,  size, -depth,
      -size,  size, 0,  -size,  size, -depth,
    ]);
    
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    
    const material = new THREE.LineBasicMaterial({ color: 0xffffff });
    this.cameraBody = new THREE.LineSegments(geometry, material);
  }

  private createCoordinateAxes(): void {
    this.coordinateAxes = new THREE.Group();
    const axisLength = 0.5;
    
    // X-axis (Red)
    const xGeometry = new THREE.BufferGeometry();
    xGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      0, 0, 0,  axisLength, 0, 0
    ]), 3));
    const xMaterial = new THREE.LineBasicMaterial({ color: 0xff0000 });
    const xAxis = new THREE.Line(xGeometry, xMaterial);
    this.coordinateAxes.add(xAxis);
    
    // Y-axis (Green)
    const yGeometry = new THREE.BufferGeometry();
    yGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      0, 0, 0,  0, axisLength, 0
    ]), 3));
    const yMaterial = new THREE.LineBasicMaterial({ color: 0x00ff00 });
    const yAxis = new THREE.Line(yGeometry, yMaterial);
    this.coordinateAxes.add(yAxis);
    
    // Z-axis (Blue) - points forward (negative Z for camera convention)
    const zGeometry = new THREE.BufferGeometry();
    zGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      0, 0, 0,  0, 0, axisLength
    ]), 3));
    const zMaterial = new THREE.LineBasicMaterial({ color: 0x0000ff });
    const zAxis = new THREE.Line(zGeometry, zMaterial);
    this.coordinateAxes.add(zAxis);
  }

  private createFrustumLines(): void {
    // Calculate frustum corners based on intrinsic parameters
    const corners = this.calculateFrustumCorners();
    
    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array([
      // Lines from camera center to frustum corners
      0, 0, 0,  corners[0].x, corners[0].y, corners[0].z, // to top-left
      0, 0, 0,  corners[1].x, corners[1].y, corners[1].z, // to top-right
      0, 0, 0,  corners[2].x, corners[2].y, corners[2].z, // to bottom-right
      0, 0, 0,  corners[3].x, corners[3].y, corners[3].z, // to bottom-left
    ]);
    
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    
    const material = new THREE.LineBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.6 });
    this.frustumLines = new THREE.LineSegments(geometry, material);
  }

  private createImagePlane(): void {
    // Create rectangle representing the image plane at frustum end
    const corners = this.calculateFrustumCorners();
    
    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array([
      // Rectangle outline
      corners[0].x, corners[0].y, corners[0].z,  corners[1].x, corners[1].y, corners[1].z, // top
      corners[1].x, corners[1].y, corners[1].z,  corners[2].x, corners[2].y, corners[2].z, // right
      corners[2].x, corners[2].y, corners[2].z,  corners[3].x, corners[3].y, corners[3].z, // bottom
      corners[3].x, corners[3].y, corners[3].z,  corners[0].x, corners[0].y, corners[0].z, // left
    ]);
    
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    
    const material = new THREE.LineBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.8 });
    this.imagePlane = new THREE.LineSegments(geometry, material);
  }

  private calculateFrustumCorners(): THREE.Vector3[] {
    const { width, height, fx, fy, cx, cy, frustumLength } = this.parameters;
    
    // Convert image coordinates to normalized camera coordinates
    // Image coordinates: (0,0) is top-left, (width, height) is bottom-right
    // Principal point (cx, cy) is the projection center
    const left = (0 - cx) / fx;           // Left edge of image
    const right = (width - cx) / fx;      // Right edge of image  
    const top = (0 - cy) / fy;            // Top edge of image
    const bottom = (height - cy) / fy;    // Bottom edge of image
    
    // Scale by frustum length (depth)
    const z = frustumLength;
    
    return [
      new THREE.Vector3(left * z, top * z, z),      // top-left
      new THREE.Vector3(right * z, top * z, z),     // top-right
      new THREE.Vector3(right * z, bottom * z, z),  // bottom-right
      new THREE.Vector3(left * z, bottom * z, z),   // bottom-left
    ];
  }

  private updateVisualFromMatrices(): void {
    // Apply extrinsic matrix transformation to the entire group
    this.group.position.copy(this.parameters.position);
    this.group.rotation.copy(this.parameters.rotation);
    
    // Update frustum and image plane based on intrinsic parameters
    this.updateFrustumGeometry();
    this.updateImagePlaneGeometry();
  }

  private updateFrustumGeometry(): void {
    const corners = this.calculateFrustumCorners();
    const vertices = new Float32Array([
      0, 0, 0,  corners[0].x, corners[0].y, corners[0].z,
      0, 0, 0,  corners[1].x, corners[1].y, corners[1].z,
      0, 0, 0,  corners[2].x, corners[2].y, corners[2].z,
      0, 0, 0,  corners[3].x, corners[3].y, corners[3].z,
    ]);
    
    this.frustumLines.geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    this.frustumLines.geometry.attributes.position.needsUpdate = true;
  }

  private updateImagePlaneGeometry(): void {
    const corners = this.calculateFrustumCorners();
    const vertices = new Float32Array([
      corners[0].x, corners[0].y, corners[0].z,  corners[1].x, corners[1].y, corners[1].z,
      corners[1].x, corners[1].y, corners[1].z,  corners[2].x, corners[2].y, corners[2].z,
      corners[2].x, corners[2].y, corners[2].z,  corners[3].x, corners[3].y, corners[3].z,
      corners[3].x, corners[3].y, corners[3].z,  corners[0].x, corners[0].y, corners[0].z,
    ]);
    
    this.imagePlane.geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    this.imagePlane.geometry.attributes.position.needsUpdate = true;
  }

  // Public methods to update camera parameters
  public updateIntrinsics(fx: number, fy: number, cx: number, cy: number, width: number, height: number): void {
    this.parameters.fx = fx;
    this.parameters.fy = fy;
    this.parameters.cx = cx;
    this.parameters.cy = cy;
    this.parameters.width = width;
    this.parameters.height = height;
    
    this.updateMatrices();
    this.updateVisualFromMatrices();
  }

  public updateExtrinsics(position: THREE.Vector3, rotation: THREE.Euler): void {
    this.parameters.position.copy(position);
    this.parameters.rotation.copy(rotation);
    
    this.updateMatrices();
    this.updateVisualFromMatrices();
  }

  public updateFromMatrices(intrinsicMatrix: THREE.Matrix3, extrinsicMatrix: THREE.Matrix4): void {
    // Extract parameters from matrices
    const K = intrinsicMatrix.elements;
    this.parameters.fx = K[0];
    this.parameters.fy = K[4];
    this.parameters.cx = K[2];
    this.parameters.cy = K[5];
    
    // Extract position and rotation from extrinsic matrix
    const position = new THREE.Vector3();
    const rotation = new THREE.Euler();
    const scale = new THREE.Vector3();
    
    const tempMatrix = extrinsicMatrix.clone();
    tempMatrix.decompose(position, new THREE.Quaternion().setFromEuler(rotation), scale);
    
    this.parameters.position.copy(position);
    this.parameters.rotation.copy(rotation);
    
    this.intrinsicMatrix.copy(intrinsicMatrix);
    this.extrinsicMatrix.copy(extrinsicMatrix);
    
    this.updateVisualFromMatrices();
  }

  // Getters
  public getGroup(): THREE.Group {
    return this.group;
  }

  public getIntrinsicMatrix(): THREE.Matrix3 {
    return this.intrinsicMatrix.clone();
  }

  public getExtrinsicMatrix(): THREE.Matrix4 {
    return this.extrinsicMatrix.clone();
  }

  public getParameters(): CameraParameters {
    return { ...this.parameters };
  }

  // Cleanup
  public dispose(): void {
    this.cameraBody.geometry.dispose();
    this.frustumLines.geometry.dispose();
    this.imagePlane.geometry.dispose();
    
    this.coordinateAxes.children.forEach(child => {
      if (child instanceof THREE.Line) {
        child.geometry.dispose();
      }
    });
  }
} 