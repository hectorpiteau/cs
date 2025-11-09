import numpy as np
import ctypes
import OpenGL.GL as GL
from pyglm import glm


def _build_frustum_lines(fov_y_deg: float, aspect: float, near_d: float, far_d: float,
                         color=(1.0, 1.0, 0.2), include_image_plane: bool = True) -> np.ndarray:
    """Build frustum rays from camera origin to image plane at distance far_d, plus image plane rectangle.
    Camera looks along -Z in its local space.
    """
    fov = np.radians(fov_y_deg)
    hf = np.tan(fov / 2.0) * far_d
    wf = hf * aspect

    # Image plane (far plane) corners
    tl = (-wf,  hf, -far_d)
    tr = ( wf,  hf, -far_d)
    br = ( wf, -hf, -far_d)
    bl = (-wf, -hf, -far_d)

    verts = []
    # Rays from origin to corners (frustum edges)
    for p in (tl, tr, br, bl):
        verts.extend((0.0, 0.0, 0.0)); verts.extend(color)
        verts.extend(p);                 verts.extend(color)

    if include_image_plane:
        # Image plane rectangle (wireframe) in white
        plane_color = (1.0, 1.0, 1.0)
        for a, b in ((tl, tr), (tr, br), (br, bl), (bl, tl)):
            verts.extend(a); verts.extend(plane_color)
            verts.extend(b); verts.extend(plane_color)

    return np.array(verts, dtype=np.float32)

def _build_frustum_lines_world(pos: glm.vec3, target: glm.vec3, up: glm.vec3,
                               fov_y_deg: float, aspect: float, length: float,
                               color=(1.0, 1.0, 0.2), include_image_plane: bool = True) -> np.ndarray:
    """Build frustum edges and image plane rectangle directly in world coordinates."""
    fov = np.radians(fov_y_deg)
    hf = np.tan(fov / 2.0) * length
    wf = hf * aspect
    # Local camera basis
    fwd = glm.normalize(target - pos)
    right = glm.normalize(glm.cross(fwd, up))
    upv = glm.cross(right, fwd)
    # Local far-plane corners in camera space (z forward)
    tl = (-wf,  hf, length)
    tr = ( wf,  hf, length)
    br = ( wf, -hf, length)
    bl = (-wf, -hf, length)
    def to_world(p):
        x,y,z = p
        w = pos + right * x + upv * y + fwd * z
        return (float(w.x), float(w.y), float(w.z))
    TL, TR, BR, BL = map(to_world, (tl, tr, br, bl))
    origin = (float(pos.x), float(pos.y), float(pos.z))
    verts = []
    # Rays
    for p in (TL, TR, BR, BL):
        verts.extend(origin); verts.extend(color)
        verts.extend(p);      verts.extend(color)
    # Image plane rectangle
    if include_image_plane:
        plane_color = (1.0, 1.0, 1.0)
        for a,b in ((TL,TR),(TR,BR),(BR,BL),(BL,TL)):
            verts.extend(a); verts.extend(plane_color)
            verts.extend(b); verts.extend(plane_color)
    return np.array(verts, dtype=np.float32)
