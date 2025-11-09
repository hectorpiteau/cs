import OpenGL.GL as GL
import numpy as np
from pyglm import glm
from imgui_bundle import imgui, immapp, hello_imgui, ImVec2
import time
import random
import ctypes
import sys
from shaders.loader import load_vertex_fragment
from primitives.cube import create_cube_vao, create_cube
from primitives.grid import create_grid_vao
from primitives.gizmo import create_gizmo_vao
from primitives.sphere import create_sphere_vao
from shaders.utils import create_shader_program

from primitives.frustum import _build_frustum_lines_world
from primitives.lines import _create_lines_vao_from_vertices

def _model_from_pos_target_up(pos: glm.vec3, target: glm.vec3, up: glm.vec3) -> np.ndarray:
    """Build a model matrix from position, target, and up vector."""
    forward = glm.normalize(target - pos)
    right = glm.normalize(glm.cross(forward, up))
    real_up = glm.cross(right, forward)
    # Columns are right, up, -forward
    m = np.array([
        [ right.x,  right.y,  right.z, 0.0],
        [ real_up.x, real_up.y, real_up.z, 0.0],
        [-forward.x,-forward.y,-forward.z, 0.0],
        [ pos.x,     pos.y,     pos.z,     1.0],
    ], dtype=np.float32)
    return m

def _build_gizmo_lines_world(pos: glm.vec3, target: glm.vec3, up: glm.vec3,
                             length: float = 0.6, head: float = 0.08) -> np.ndarray:
    """Build axis gizmo lines (with small arrowheads) in world coordinates starting at camera pos.
    Colors: X=red, Y=green, Z=blue. Uses OpenGL convention (+Y up).
    """
    fwd = glm.normalize(target - pos)
    right = glm.normalize(glm.cross(fwd, up))
    upv = glm.cross(right, fwd)
    p = pos
    # Axis endpoints
    px = p + right * length
    py = p + upv * length
    pz = p + fwd * length
    # Arrowhead offsets
    x_left  = px - right * head + upv * (head * 0.6)
    x_right = px - right * head - upv * (head * 0.6)
    y_left  = py - upv * head + right * (head * 0.6)
    y_right = py - upv * head - right * (head * 0.6)
    z_left  = pz - fwd * head + right * (head * 0.6)
    z_right = pz - fwd * head - right * (head * 0.6)
    r, g, b = (1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (0.1, 0.3, 1.0)
    def pack(v, c):
        return [float(v.x), float(v.y), float(v.z), *c]
    verts = []
    # X axis + arrow
    verts += pack(p, r);  verts += pack(px, r)
    verts += pack(px, r); verts += pack(x_left, r)
    verts += pack(px, r); verts += pack(x_right, r)
    # Y axis + arrow
    verts += pack(p, g);  verts += pack(py, g)
    verts += pack(py, g); verts += pack(y_left, g)
    verts += pack(py, g); verts += pack(y_right, g)
    # Z axis + arrow
    verts += pack(p, b);  verts += pack(pz, b)
    verts += pack(pz, b); verts += pack(z_left, b)
    verts += pack(pz, b); verts += pack(z_right, b)
    return np.array(verts, dtype=np.float32)


class AppState:
    def __init__(self):
        self.shader_program = None
        self.vao = None
        self.vertex_count = 36  # 6 faces * 2 triangles * 3 vertices
        
        # Uniform locations
        self.loc_model = None
        self.loc_view = None
        self.loc_projection = None
        self.loc_light_dir = None
        self.loc_color = None
        self.loc_ambient_sky = None
        self.loc_ambient_ground = None
        self.loc_ambient_intensity = None
        
        # Scene parameters
        self.rotation_speed = glm.vec3(20.0, 30.0, 15.0)
        self.cube_color = glm.vec3(0.2, 0.7, 0.3)
        self.light_dir = glm.vec3(1.0, 1.0, -1.0)
        self.last_time = time.time()
        self.angle = glm.vec3(0.0, 0.0, 0.0)

        # Hemisphere ambient lighting parameters
        self.ambient_sky = glm.vec3(0.6, 0.7, 1.0)
        self.ambient_ground = glm.vec3(0.25, 0.22, 0.20)
        self.ambient_intensity = 0.5

        # Offscreen framebuffer (FBO) resources
        self.fbo = None
        self.fbo_color_tex = None
        self.fbo_depth_rb = None
        self.fbo_width = 0
        self.fbo_height = 0

        # Camera (orbit) controls
        self.auto_rotate = False
        self.cam_yaw_deg = 45.0
        self.cam_pitch_deg = 25.0
        self.cam_distance = 6.0

        # Frustum visualization settings
        self.frustum_length = 1.0

        # Optional geometry VAOs
        self.sphere_vao = None
        self.sphere_vertex_count = 0

        # Line shader & helpers (grid, gizmo)
        self.line_shader = None
        self.loc_line_mvp = None
        self.grid_vao = None
        self.grid_vertex_count = 0
        self.gizmo_vao = None
        self.gizmo_vertex_count = 0

        # Debug cameras (frusta + gizmos)
        self.debug_cameras = []  # list of dicts: {vao, vbo, count, model}
        # Link line connecting debug cameras (world-space)
        self.cam_link_vao = None
        self.cam_link_vbo = None
        self.cam_link_count = 0

        # Image viewer (numpy -> GL texture)
        self.image_tex = None
        self.image_width = 0
        self.image_height = 0
        self.image_channels = 0

        # Point cloud resources & params
        self.point_shader = None
        self.loc_point_mvp = None
        self.loc_point_size = None
        self.point_vao = None
        self.point_vbo = None
        self.point_count = 0
        self.point_spacing = 0.05
        self.point_half_extent = 2.0
        self.point_height_sigma = 0.1
        self.point_size_px = 3.0
        self.point_color = glm.vec3(0.1, 0.8, 1.0)

    def add_camera_representation(self, pos: glm.vec3, target: glm.vec3, up: glm.vec3,
                                  fov_y_deg: float, aspect: float,
                                  near_d: float, length: float,
                                  color=(1.0, 1.0, 0.2)):
        verts_world = _build_frustum_lines_world(pos, target, up, fov_y_deg, aspect, length,
                                                 color=color, include_image_plane=True)
        vao, vbo, count = _create_lines_vao_from_vertices(verts_world)
        # Build a dedicated gizmo VAO in world space for this camera
        gizmo_verts = _build_gizmo_lines_world(pos, target, up, length=0.5, head=0.07)
        gizmo_vao, gizmo_vbo, gizmo_count = _create_lines_vao_from_vertices(gizmo_verts)
        model = _model_from_pos_target_up(pos, target, up)
        self.debug_cameras.append({
            'vao': vao,
            'vbo': vbo,
            'count': count,
            'gizmo_vao': gizmo_vao,
            'gizmo_vbo': gizmo_vbo,
            'gizmo_count': gizmo_count,
            'model': model,
            'fov': float(fov_y_deg),
            'aspect': float(aspect),
            'near': float(near_d),
            'length': float(length),
            'color': tuple(color),
            'pos': glm.vec3(pos),
            'target': glm.vec3(target),
            'up': glm.vec3(up),
        })
        # Update the camera link line whenever a new camera is added
        self.update_camera_link_line()

    def update_camera_link_line(self):
        """Rebuild a white line connecting all debug camera positions in sequence."""
        # Destroy previous if exists
        if self.cam_link_vao:
            try:
                GL.glDeleteVertexArrays(1, [self.cam_link_vao])
            except Exception:
                pass
            self.cam_link_vao = None
        if self.cam_link_vbo:
            try:
                GL.glDeleteBuffers(1, [self.cam_link_vbo])
            except Exception:
                pass
            self.cam_link_vbo = None
        self.cam_link_count = 0
        # Need at least 2 cameras to draw a link
        if len(self.debug_cameras) < 2:
            return
        # Build interleaved pos+color for GL_LINES segments between consecutive cameras
        white = (1.0, 1.0, 1.0)
        verts = []
        for i in range(len(self.debug_cameras) - 1):
            a = self.debug_cameras[i]['pos']
            b = self.debug_cameras[i + 1]['pos']
            verts.extend([float(a.x), float(a.y), float(a.z), *white])
            verts.extend([float(b.x), float(b.y), float(b.z), *white])
        verts_np = np.array(verts, dtype=np.float32)
        # Upload
        self.cam_link_vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.cam_link_vao)
        self.cam_link_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.cam_link_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts_np.nbytes, verts_np, GL.GL_DYNAMIC_DRAW)
        stride = 6 * 4
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        GL.glEnableVertexAttribArray(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        self.cam_link_count = len(verts_np) // 6
    
    def add_current_camera_to_scene(self):
        """Get the current orbit camera view and add it to the list of debug cameras."""
        # Calculate camera position from orbit controls
        yaw_r = glm.radians(self.cam_yaw_deg)
        pitch_r = glm.radians(self.cam_pitch_deg)
        cx = self.cam_distance * glm.cos(pitch_r) * glm.cos(yaw_r)
        cy = self.cam_distance * glm.sin(pitch_r)
        cz = self.cam_distance * glm.cos(pitch_r) * glm.sin(yaw_r)
        cam_pos = glm.vec3(cx, cy, cz)
        
        target = glm.vec3(0.0, 0.0, 0.0)
        up = glm.vec3(0.0, 1.0, 0.0)
        
        # Use FBO aspect ratio if available, otherwise default
        aspect = 1.6
        if self.fbo_height > 0:
            aspect = self.fbo_width / self.fbo_height
            
        # Other projection parameters from main camera
        fov_y_deg = 45.0
        near_d = 0.1
        length = self.frustum_length
        
        # Assign a random bright color
        color = (random.random() * 0.5 + 0.5, 
                 random.random() * 0.5 + 0.5, 
                 random.random() * 0.5 + 0.5)

        self.add_camera_representation(
            pos=cam_pos, target=target, up=up,
            fov_y_deg=fov_y_deg, aspect=aspect,
            near_d=near_d, length=length, color=color
        )
    
    def init_3d_resources(self):
        """Initialize OpenGL resources"""
        print("Initializing 3D resources...")
        
        # Create shader program from external files
        lambert_vsrc, lambert_fsrc = load_vertex_fragment("simple_lambert.vertex.shader", "simple_lambert.fragment.shader")
        self.shader_program = create_shader_program(lambert_vsrc, lambert_fsrc)
        # Line shader
        line_vsrc, line_fsrc = load_vertex_fragment("line.vertex.shader", "line.fragment.shader")
        self.line_shader = create_shader_program(line_vsrc, line_fsrc)
        self.loc_line_mvp = GL.glGetUniformLocation(self.line_shader, "u_mvp")
        
        # Create cube VAO
        self.vao = create_cube_vao()
        # Create helpers: grid and gizmo
        self.grid_vao, self.grid_vertex_count = create_grid_vao()
        self.gizmo_vao, self.gizmo_vertex_count = create_gizmo_vao()

        # Point shader
        pts_vsrc, pts_fsrc = load_vertex_fragment("points.vertex.shader", "points.fragment.shader")
        self.point_shader = create_shader_program(pts_vsrc, pts_fsrc)
        self.loc_point_mvp = GL.glGetUniformLocation(self.point_shader, "u_mvp")
        self.loc_point_size = GL.glGetUniformLocation(self.point_shader, "u_point_size")
        self.generate_point_cloud()

        # Add a test camera at (4,4,4) looking at (5,4,4)
        # self.add_camera_representation(
        #     pos=glm.vec3(4.0, 4.0, 4.0),
        #     target=glm.vec3(0.0, 4.0, 0.0),
        #     up=glm.vec3(0.0, 1.0, 0.0),
        #     fov_y_deg=45.0,
        #     aspect=1.6,
        #     near_d=0.1,
        #     length=self.frustum_length,
        #     color=(1.0, 0.85, 0.2)
        # )
        
        # self.add_camera_representation(
        #     pos=glm.vec3(10.0, 4.0, 4.0),
        #     target=glm.vec3(0.0, 0.0, 0.0),
        #     up=glm.vec3(0.0, 1.0, 0.0),
        #     fov_y_deg=45.0,
        #     aspect=1.6,
        #     near_d=0.1,
        #     length=self.frustum_length,
        #     color=(1.0, 0.85, 0.2)
        # )
        
        # Get uniform locations
        self.loc_model = GL.glGetUniformLocation(self.shader_program, "u_model")
        self.loc_view = GL.glGetUniformLocation(self.shader_program, "u_view")
        self.loc_projection = GL.glGetUniformLocation(self.shader_program, "u_projection")
        self.loc_light_dir = GL.glGetUniformLocation(self.shader_program, "u_light_dir")
        self.loc_color = GL.glGetUniformLocation(self.shader_program, "u_color")
        self.loc_ambient_sky = GL.glGetUniformLocation(self.shader_program, "u_ambient_sky")
        self.loc_ambient_ground = GL.glGetUniformLocation(self.shader_program, "u_ambient_ground")
        self.loc_ambient_intensity = GL.glGetUniformLocation(self.shader_program, "u_ambient_intensity")
        
        print(f"Shader program: {self.shader_program}")
        print(f"VAO: {self.vao}")
        print(f"Uniform locations: model={self.loc_model}, view={self.loc_view}, proj={self.loc_projection}")
        print("3D resources initialized!")
    
    def destroy_3d_resources(self):
        """Clean up OpenGL resources"""
        self.destroy_fbo()
        if self.vao:
            GL.glDeleteVertexArrays(1, [self.vao])
        if self.shader_program:
            GL.glDeleteProgram(self.shader_program)
        if self.line_shader:
            GL.glDeleteProgram(self.line_shader)
        if self.point_shader:
            GL.glDeleteProgram(self.point_shader)
        if self.point_vao:
            try:
                GL.glDeleteVertexArrays(1, [self.point_vao])
            except Exception:
                pass
            self.point_vao = None
        if self.point_vbo:
            try:
                GL.glDeleteBuffers(1, [self.point_vbo])
            except Exception:
                pass
            self.point_vbo = None
        # Destroy debug camera VAOs/VBOs
        for cam in self.debug_cameras:
            try:
                if cam.get('vao'):
                    GL.glDeleteVertexArrays(1, [cam['vao']])
                if cam.get('vbo'):
                    GL.glDeleteBuffers(1, [cam['vbo']])
                if cam.get('gizmo_vao'):
                    GL.glDeleteVertexArrays(1, [cam['gizmo_vao']])
                if cam.get('gizmo_vbo'):
                    GL.glDeleteBuffers(1, [cam['gizmo_vbo']])
            except Exception:
                pass
        self.debug_cameras.clear()
        # Destroy camera link resources
        if self.cam_link_vao:
            try:
                GL.glDeleteVertexArrays(1, [self.cam_link_vao])
            except Exception:
                pass
            self.cam_link_vao = None
        if self.cam_link_vbo:
            try:
                GL.glDeleteBuffers(1, [self.cam_link_vbo])
            except Exception:
                pass
            self.cam_link_vbo = None
        self.cam_link_count = 0
        # Destroy image texture
        if self.image_tex:
            try:
                GL.glDeleteTextures(1, [self.image_tex])
            except Exception:
                pass
            self.image_tex = None

    def set_image_numpy(self, image_np: np.ndarray):
        """Upload a numpy uint8 image to an OpenGL texture for display.
        Accepts HxW, HxWx1, HxWx3, HxWx4 arrays (uint8). Other dtypes will be converted to uint8.
        """
        if image_np is None:
            return
        arr = np.asarray(image_np)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack((arr, arr, arr), axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            pass
        else:
            # Unsupported shape; try to interpret last dim as channels
            raise ValueError("Unsupported image shape for viewer: expected HxW, HxWx1, HxWx3, or HxWx4")

        h, w, c = arr.shape
        self.image_width, self.image_height, self.image_channels = w, h, c

        # Ensure contiguous
        arr = np.ascontiguousarray(arr)

        # Prepare GL texture
        if not self.image_tex:
            self.image_tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        else:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_tex)

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        if c == 3:
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, arr)
        else:  # c == 4
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, arr)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def generate_point_cloud(self):
        """Generate a 2D grid on XZ with Gaussian height on Y and upload to GPU."""
        # Build grid coordinates
        half = float(self.point_half_extent)
        step = float(self.point_spacing)
        xs = np.arange(-half, half + 1e-6, step, dtype=np.float32)
        zs = np.arange(-half, half + 1e-6, step, dtype=np.float32)
        X, Z = np.meshgrid(xs, zs)
        Y = np.random.normal(loc=0.0, scale=float(self.point_height_sigma), size=X.shape).astype(np.float32)
        pts = np.stack([X, Y, Z], axis=2).reshape(-1, 3)
        # Normalize to [0,1] for colors based on position
        min_xyz = pts.min(axis=0)
        max_xyz = pts.max(axis=0)
        rng = np.maximum(max_xyz - min_xyz, 1e-6)
        cols = (pts - min_xyz) / rng
        cols = np.clip(cols, 0.0, 1.0).astype(np.float32)
        interleaved = np.concatenate([pts, cols], axis=1)
        self.point_count = pts.shape[0]

        # Upload to GPU
        interleaved = np.ascontiguousarray(interleaved, dtype=np.float32)
        if not self.point_vao:
            self.point_vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.point_vao)
        if not self.point_vbo:
            self.point_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.point_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL.GL_STATIC_DRAW)
        stride = 6 * 4
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        GL.glEnableVertexAttribArray(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

    def destroy_fbo(self):
        if self.fbo:
            GL.glDeleteFramebuffers(1, [self.fbo])
            self.fbo = None
        if self.fbo_color_tex:
            GL.glDeleteTextures(1, [self.fbo_color_tex])
            self.fbo_color_tex = None
        if self.fbo_depth_rb:
            GL.glDeleteRenderbuffers(1, [self.fbo_depth_rb])
            self.fbo_depth_rb = None
        self.fbo_width = 0
        self.fbo_height = 0

    def ensure_fbo(self, width: int, height: int):
        if width <= 0 or height <= 0:
            return
        if self.fbo and self.fbo_width == width and self.fbo_height == height:
            return
        # Recreate FBO
        self.destroy_fbo()
        self.fbo_width, self.fbo_height = width, height

        # Color texture
        self.fbo_color_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.fbo_color_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        # Depth renderbuffer
        self.fbo_depth_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.fbo_depth_rb)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, width, height)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)

        # Framebuffer
        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.fbo_color_tex, 0)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.fbo_depth_rb)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            print(f"FBO incomplete: status=0x{status:x}")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def render_scene_to_fbo(self, width: int, height: int):
        # Update rotation timing here so widget controls the animation cadence
        current = time.time()
        dt = current - self.last_time
        self.last_time = current
        if self.auto_rotate:
            self.angle += self.rotation_speed * dt

        self.ensure_fbo(width, height)
        if not self.fbo:
            return

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.1, 0.1, 0.12, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)

        GL.glUseProgram(self.shader_program)

        # Matrices
        model = glm.mat4(1.0)
        model = glm.rotate(model, glm.radians(self.angle.x), glm.vec3(1, 0, 0))
        model = glm.rotate(model, glm.radians(self.angle.y), glm.vec3(0, 1, 0))
        model = glm.rotate(model, glm.radians(self.angle.z), glm.vec3(0, 0, 1))
        # Orbit camera from yaw/pitch/distance around origin
        yaw_r = glm.radians(self.cam_yaw_deg)
        pitch_r = glm.radians(self.cam_pitch_deg)
        cx = self.cam_distance * glm.cos(pitch_r) * glm.cos(yaw_r)
        cy = self.cam_distance * glm.sin(pitch_r)
        cz = self.cam_distance * glm.cos(pitch_r) * glm.sin(yaw_r)
        cam_pos = glm.vec3(cx, cy, cz)
        view = glm.lookAt(cam_pos, glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        aspect = width / height if height > 0 else 1.0
        projection = glm.perspective(glm.radians(45.0), aspect, 0.1, 100.0)

        model_array = np.array(model, dtype='float32')
        view_array = np.array(view, dtype='float32')
        projection_array = np.array(projection, dtype='float32')

        GL.glUniformMatrix4fv(self.loc_model, 1, GL.GL_TRUE, model_array)
        GL.glUniformMatrix4fv(self.loc_view, 1, GL.GL_TRUE, view_array)
        GL.glUniformMatrix4fv(self.loc_projection, 1, GL.GL_TRUE, projection_array)
        GL.glUniform3f(self.loc_light_dir, self.light_dir.x, self.light_dir.y, self.light_dir.z)
        GL.glUniform3f(self.loc_color, self.cube_color.x, self.cube_color.y, self.cube_color.z)
        if self.loc_ambient_sky is not None and self.loc_ambient_sky != -1:
            GL.glUniform3f(self.loc_ambient_sky, self.ambient_sky.x, self.ambient_sky.y, self.ambient_sky.z)
        if self.loc_ambient_ground is not None and self.loc_ambient_ground != -1:
            GL.glUniform3f(self.loc_ambient_ground, self.ambient_ground.x, self.ambient_ground.y, self.ambient_ground.z)
        if self.loc_ambient_intensity is not None and self.loc_ambient_intensity != -1:
            GL.glUniform1f(self.loc_ambient_intensity, float(self.ambient_intensity))

        # Draw solid geometry (comment/uncomment to switch)
        self.draw_cube()
        # self.draw_sphere()

        # Draw helpers (grid and gizmo) with line shader and depth test ON
        mvp = projection_array @ view_array @ model_array
        GL.glUseProgram(self.line_shader)
        GL.glUniformMatrix4fv(self.loc_line_mvp, 1, GL.GL_TRUE, mvp)

        # Grid
        if self.grid_vao and self.grid_vertex_count > 0:
            GL.glBindVertexArray(self.grid_vao)
            GL.glLineWidth(1.0)
            GL.glDrawArrays(GL.GL_LINES, 0, self.grid_vertex_count)
            GL.glBindVertexArray(0)

        # Gizmo (draw slightly on top by disabling depth test or using polygon offset)
        GL.glDisable(GL.GL_DEPTH_TEST)
        if self.gizmo_vao and self.gizmo_vertex_count > 0:
            GL.glBindVertexArray(self.gizmo_vao)
            GL.glLineWidth(1.0)
            GL.glDrawArrays(GL.GL_LINES, 0, self.gizmo_vertex_count)
            GL.glBindVertexArray(0)
        GL.glEnable(GL.GL_DEPTH_TEST)

        # Draw point cloud (world space)
        self.draw_point_cloud(projection_array, view_array)

        # Draw debug camera frusta and their gizmos
        # For frusta in world space, model is identity
        mvp_lines = projection_array @ view_array
        GL.glUseProgram(self.line_shader)
        for cam in self.debug_cameras:
            # If frustum length changed, rebuild this camera's frustum VBO in world space
            if abs(cam.get('length', self.frustum_length) - self.frustum_length) > 1e-6:
                verts_new = _build_frustum_lines_world(cam['pos'], cam['target'], cam['up'],
                                                       cam['fov'], cam['aspect'], self.frustum_length,
                                                       color=cam['color'], include_image_plane=True)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, cam['vbo'])
                GL.glBufferData(GL.GL_ARRAY_BUFFER, verts_new.nbytes, verts_new, GL.GL_DYNAMIC_DRAW)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                cam['count'] = len(verts_new) // 6
                cam['length'] = float(self.frustum_length)
            # Frustum (world space)
            GL.glUniformMatrix4fv(self.loc_line_mvp, 1, GL.GL_TRUE, mvp_lines)
            GL.glBindVertexArray(cam['vao'])
            GL.glLineWidth(1.0)
            GL.glDrawArrays(GL.GL_LINES, 0, cam['count'])
            GL.glBindVertexArray(0)

            # Camera gizmo (pre-built in world space; lines start at camera.pos)
            if cam.get('gizmo_vao') and cam.get('gizmo_count', 0) > 0:
                GL.glUniformMatrix4fv(self.loc_line_mvp, 1, GL.GL_TRUE, mvp_lines)
                GL.glBindVertexArray(cam['gizmo_vao'])
                GL.glDrawArrays(GL.GL_LINES, 0, cam['gizmo_count'])
                GL.glBindVertexArray(0)

        # Draw link line connecting cameras (if available)
        if self.cam_link_vao and self.cam_link_count > 0:
            GL.glUniformMatrix4fv(self.loc_line_mvp, 1, GL.GL_TRUE, mvp_lines)
            GL.glBindVertexArray(self.cam_link_vao)
            GL.glLineWidth(1.0)
            GL.glDrawArrays(GL.GL_LINES, 0, self.cam_link_count)
            GL.glBindVertexArray(0)

        GL.glUseProgram(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def draw_cube(self):
        return
        if not self.vao:
            return
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)
        GL.glBindVertexArray(0)

    def draw_sphere(self):
        if not self.sphere_vao:
            self.sphere_vao, self.sphere_vertex_count = create_sphere_vao()
        GL.glBindVertexArray(self.sphere_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.sphere_vertex_count)
        GL.glBindVertexArray(0)

    def draw_point_cloud(self, projection_array, view_array):
        if not self.point_shader or not self.point_vao or self.point_count <= 0:
            return
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        mvp = projection_array @ view_array
        GL.glUseProgram(self.point_shader)
        GL.glUniformMatrix4fv(self.loc_point_mvp, 1, GL.GL_TRUE, mvp)
        GL.glUniform1f(self.loc_point_size, float(self.point_size_px))
        GL.glBindVertexArray(self.point_vao)
        GL.glDrawArrays(GL.GL_POINTS, 0, self.point_count)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)


def scaled_display_size():
    """Get display size in pixels (accounting for retina displays)"""
    io = imgui.get_io()
    return ImVec2(
        io.display_size.x * io.display_framebuffer_scale.x,
        io.display_size.y * io.display_framebuffer_scale.y
    )


def custom_background(app_state: AppState):
    """Clear background only; cube is rendered inside the widget FBO"""
    display_size = scaled_display_size()
    GL.glViewport(0, 0, int(display_size.x), int(display_size.y))
    GL.glClearColor(0.05, 0.05, 0.06, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)


def gui(app_state: AppState):
    """GUI controls"""
    # 3D View window with the rendered texture
    imgui.set_next_window_pos(ImVec2(10, 10), imgui.Cond_.appearing)
    imgui.set_next_window_size(ImVec2(800, 600), imgui.Cond_.appearing)
    imgui.begin("3D View", None, imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse)
    avail = imgui.get_content_region_avail()
    width, height = int(max(1.0, avail.x)), int(max(1.0, avail.y))
    if app_state.shader_program is not None:
        # Mouse interactions when hovering the 3D view
        hovered = imgui.is_window_hovered()
        io = imgui.get_io()
        if hovered:
            # Rotate with Left Mouse Button drag
            if imgui.is_mouse_down(0):
                sensitivity = 0.3
                app_state.cam_yaw_deg += io.mouse_delta.x * sensitivity
                # Invert Y: moving mouse up rotates camera up
                app_state.cam_pitch_deg += io.mouse_delta.y * sensitivity
                # Clamp pitch to avoid gimbal lock
                app_state.cam_pitch_deg = max(-89.0, min(89.0, app_state.cam_pitch_deg))
            # Zoom with Mouse Wheel
            if io.mouse_wheel != 0.0:
                zoom_factor = 1.0 - io.mouse_wheel * 0.1
                zoom_factor = max(0.1, min(10.0, zoom_factor))
                app_state.cam_distance *= zoom_factor
                app_state.cam_distance = max(1.0, min(50.0, app_state.cam_distance))

        app_state.render_scene_to_fbo(width, height)
        if app_state.fbo_color_tex:
            tex_ref = imgui.ImTextureRef(int(app_state.fbo_color_tex))
            # Flip vertically when showing
            imgui.image(tex_ref, ImVec2(avail.x, avail.y), ImVec2(0, 1), ImVec2(1, 0))
        else:
            imgui.text("FBO not ready")
    else:
        imgui.text("Shader not initialized yet")
    imgui.end()

    # Controls window
    imgui.set_next_window_pos(ImVec2(820, 10), imgui.Cond_.appearing)
    imgui.set_next_window_size(ImVec2(320, 340), imgui.Cond_.appearing)
    imgui.begin("Cube Controls")
    
    imgui.text("Rotation Speed (deg/s):")
    _, app_state.rotation_speed.x = imgui.slider_float("Speed X", app_state.rotation_speed.x, 0.0, 360.0)
    _, app_state.rotation_speed.y = imgui.slider_float("Speed Y", app_state.rotation_speed.y, 0.0, 360.0)
    _, app_state.rotation_speed.z = imgui.slider_float("Speed Z", app_state.rotation_speed.z, 0.0, 360.0)
    _, app_state.auto_rotate = imgui.checkbox("Auto rotate", app_state.auto_rotate)
    
    imgui.separator()
    imgui.text("Light Direction:")
    _, light_list = imgui.slider_float3("Light", 
        [app_state.light_dir.x, app_state.light_dir.y, app_state.light_dir.z], -1.0, 1.0)
    app_state.light_dir = glm.vec3(light_list[0], light_list[1], light_list[2])
    
    imgui.separator()
    imgui.text("Cube Color:")
    _, color_list = imgui.color_edit3("Color", 
        [app_state.cube_color.x, app_state.cube_color.y, app_state.cube_color.z])
    app_state.cube_color = glm.vec3(color_list[0], color_list[1], color_list[2])
    
    imgui.separator()
    imgui.text("Ambient (hemisphere) lighting:")
    sky_list = [app_state.ambient_sky.x, app_state.ambient_sky.y, app_state.ambient_sky.z]
    _, sky_list = imgui.color_edit3("Sky", sky_list)
    app_state.ambient_sky = glm.vec3(sky_list[0], sky_list[1], sky_list[2])
    ground_list = [app_state.ambient_ground.x, app_state.ambient_ground.y, app_state.ambient_ground.z]
    _, ground_list = imgui.color_edit3("Ground", ground_list)
    app_state.ambient_ground = glm.vec3(ground_list[0], ground_list[1], ground_list[2])
    _, app_state.ambient_intensity = imgui.slider_float("Ambient Intensity", float(app_state.ambient_intensity), 0.0, 2.0)

    imgui.separator()
    imgui.text("Camera:")
    _, app_state.cam_yaw_deg = imgui.slider_float("Yaw", float(app_state.cam_yaw_deg), -360.0, 360.0)
    _, app_state.cam_pitch_deg = imgui.slider_float("Pitch", float(app_state.cam_pitch_deg), -89.0, 89.0)
    _, app_state.cam_distance = imgui.slider_float("Distance", float(app_state.cam_distance), 1.0, 50.0)
    _, app_state.frustum_length = imgui.slider_float("Frustum length", float(app_state.frustum_length), 0.1, 10.0)
    if imgui.button("Reset Camera"):
        app_state.cam_yaw_deg = 45.0
        app_state.cam_pitch_deg = 25.0
        app_state.cam_distance = 6.0
    if imgui.button("Add current view to scene"):
        app_state.add_current_camera_to_scene()

    imgui.separator()
    imgui.text("Point Cloud:")
    changed, app_state.point_spacing = imgui.slider_float("Spacing", float(app_state.point_spacing), 0.01, 0.2)
    changed2, app_state.point_half_extent = imgui.slider_float("Half extent", float(app_state.point_half_extent), 0.5, 10.0)
    changed3, app_state.point_height_sigma = imgui.slider_float("Height sigma", float(app_state.point_height_sigma), 0.0, 0.5)
    changed4, app_state.point_size_px = imgui.slider_float("Point size", float(app_state.point_size_px), 1.0, 10.0)
    if imgui.button("Rebuild point cloud") or changed or changed2 or changed3:
        app_state.generate_point_cloud()

    imgui.separator()
    imgui.text(f"Current Angle:")
    imgui.text(f"  X: {app_state.angle.x:.1f}°")
    imgui.text(f"  Y: {app_state.angle.y:.1f}°")
    imgui.text(f"  Z: {app_state.angle.z:.1f}°")
    imgui.text(f"FPS: {hello_imgui.frame_rate():.1f}")
    
    imgui.end()

    # Image viewer window
    imgui.set_next_window_pos(ImVec2(820, 360), imgui.Cond_.appearing)
    imgui.set_next_window_size(ImVec2(320, 280), imgui.Cond_.appearing)
    imgui.begin("Image Viewer")
    if app_state.image_tex:
        avail2 = imgui.get_content_region_avail()
        # Keep aspect ratio
        if app_state.image_width > 0 and app_state.image_height > 0:
            aspect = app_state.image_width / app_state.image_height
            draw_w = avail2.x
            draw_h = draw_w / aspect
            if draw_h > avail2.y:
                draw_h = avail2.y
                draw_w = draw_h * aspect
        else:
            draw_w = avail2.x
            draw_h = avail2.y
        tex_ref = imgui.ImTextureRef(int(app_state.image_tex))
        # Flip vertically so that numpy row 0 appears at top
        imgui.image(tex_ref, ImVec2(draw_w, draw_h), ImVec2(0, 1), ImVec2(1, 0))
        # Overlay: draw a 5px-radius circle outline at the image center
        p0 = imgui.get_item_rect_min()
        p1 = imgui.get_item_rect_max()
        cx = (p0.x + p1.x) * 0.5
        cy = (p0.y + p1.y) * 0.5
        draw_list = imgui.get_window_draw_list()
        draw_list.add_circle(ImVec2(cx, cy), 5.0, 0xff00ff00, 32, 2.0)
        imgui.text(f"{app_state.image_width}x{app_state.image_height}x{app_state.image_channels}")
    else:
        imgui.text("No image uploaded. Use app_state.set_image_numpy(np_image)")
        if imgui.button("Load test image"):
            # Demo: upload a simple gradient test image
            h, w = 240, 320
            y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
            x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
            test = np.stack([x.repeat(h,0), y.repeat(w,1), np.full((h,w),128, np.uint8)], axis=2)
            app_state.set_image_numpy(test)
    imgui.end()


def main():
    # Create app state
    app_state = AppState()
    
    # Configure runner parameters
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Rotating Cube with ImGui + OpenGL"
    runner_params.app_window_params.window_geometry.size = (1200, 800)
    runner_params.fps_idling.enable_idling = False  # Run at full speed
    
    # Don't create default ImGui window
    runner_params.imgui_window_params.default_imgui_window_type = \
        hello_imgui.DefaultImGuiWindowType.no_default_window
    
    # Set callbacks
    runner_params.callbacks.post_init = app_state.init_3d_resources
    runner_params.callbacks.before_exit = app_state.destroy_3d_resources
    runner_params.callbacks.custom_background = lambda: custom_background(app_state)
    runner_params.callbacks.show_gui = lambda: gui(app_state)
    
    # Run the app
    immapp.run(runner_params)


if __name__ == "__main__":
    main()
