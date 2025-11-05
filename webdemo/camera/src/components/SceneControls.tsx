import { useState } from 'preact/hooks';
import './SceneControls.css';

interface SceneControlsProps {
  onResetCamera?: () => void;
  onToggleWireframe?: (enabled: boolean) => void;
  onToggleGrid?: (enabled: boolean) => void;
  onUpdateCameraFocal?: (fx: number, fy: number) => void;
  onRotateCameraObject?: (rx: number, ry: number, rz: number) => void;
  className?: string;
}

export function SceneControls({ 
  onResetCamera, 
  onToggleWireframe, 
  onToggleGrid,
  onUpdateCameraFocal,
  onRotateCameraObject,
  className 
}: SceneControlsProps) {
  const [wireframeEnabled, setWireframeEnabled] = useState(false);
  const [gridEnabled, setGridEnabled] = useState(true);
  const [isExpanded, setIsExpanded] = useState(false);
  const [focalLength, setFocalLength] = useState(800);
  const [cameraRotation, setCameraRotation] = useState({ x: -0.3, y: 0.4, z: 0 });

  const handleResetCamera = () => {
    onResetCamera?.();
  };

  const handleToggleWireframe = () => {
    const newState = !wireframeEnabled;
    setWireframeEnabled(newState);
    onToggleWireframe?.(newState);
  };

  const handleToggleGrid = () => {
    const newState = !gridEnabled;
    setGridEnabled(newState);
    onToggleGrid?.(newState);
  };

  const handleFocalLengthChange = (value: number) => {
    setFocalLength(value);
    onUpdateCameraFocal?.(value, value); // Same focal length for both axes
  };

  const handleCameraRotationChange = (axis: 'x' | 'y' | 'z', value: number) => {
    const newRotation = { ...cameraRotation, [axis]: value };
    setCameraRotation(newRotation);
    onRotateCameraObject?.(newRotation.x, newRotation.y, newRotation.z);
  };

  return (
    <div className={`scene-controls ${isExpanded ? 'expanded' : ''} ${className || ''}`}>
      <button 
        className="controls-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-label="Toggle controls"
      >
        ⚙️
      </button>
      
      {isExpanded && (
        <div className="controls-panel">
          <div className="controls-header">
            <h3>Scene Controls</h3>
          </div>
          
          <div className="controls-section">
            <button 
              className="control-button"
              onClick={handleResetCamera}
              title="Reset camera to initial position"
            >
              🎯 Reset Camera
            </button>
            
            <label className="control-checkbox">
              <input
                type="checkbox"
                checked={wireframeEnabled}
                onChange={handleToggleWireframe}
              />
              <span>Wireframe Mode</span>
            </label>
            
            <label className="control-checkbox">
              <input
                type="checkbox"
                checked={gridEnabled}
                onChange={handleToggleGrid}
              />
              <span>Show Grid</span>
            </label>
          </div>

          <div className="controls-section">
            <div className="controls-subheader">
              <h4>Camera Object</h4>
            </div>
            
            <div className="control-slider">
              <label>Focal Length: {focalLength}px</label>
              <input
                type="range"
                min="200"
                max="1500"
                step="50"
                value={focalLength}
                onChange={(e) => handleFocalLengthChange(Number(e.currentTarget.value))}
              />
            </div>

            <div className="control-slider">
              <label>Rotation X: {cameraRotation.x.toFixed(2)} rad</label>
              <input
                type="range"
                min="-1.57"
                max="1.57"
                step="0.1"
                value={cameraRotation.x}
                onChange={(e) => handleCameraRotationChange('x', Number(e.currentTarget.value))}
              />
            </div>

            <div className="control-slider">
              <label>Rotation Y: {cameraRotation.y.toFixed(2)} rad</label>
              <input
                type="range"
                min="-1.57"
                max="1.57"
                step="0.1"
                value={cameraRotation.y}
                onChange={(e) => handleCameraRotationChange('y', Number(e.currentTarget.value))}
              />
            </div>

            <div className="control-slider">
              <label>Rotation Z: {cameraRotation.z.toFixed(2)} rad</label>
              <input
                type="range"
                min="-1.57"
                max="1.57"
                step="0.1"
                value={cameraRotation.z}
                onChange={(e) => handleCameraRotationChange('z', Number(e.currentTarget.value))}
              />
            </div>
          </div>
          
          <div className="controls-info">
            <div className="info-item">
              <strong>Controls:</strong>
            </div>
            <div className="info-item">
              • Left click + drag: Rotate
            </div>
            <div className="info-item">
              • Right click + drag: Pan
            </div>
            <div className="info-item">
              • Scroll: Zoom in/out
            </div>
            <div className="info-item">
              <strong>Camera Object:</strong>
            </div>
            <div className="info-item">
              • White box: Camera body
            </div>
            <div className="info-item">
              • RGB axes: Local coordinate system
            </div>
            <div className="info-item">
              • Cyan rectangle: Image plane (FOV)
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 