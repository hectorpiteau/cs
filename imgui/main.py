import OpenGL.GL as GL
import numpy as np
from pyglm import glm
from imgui_bundle import imgui, immapp, hello_imgui, ImVec2
import time
import ctypes
import sys

# Vertex & fragment shaders (simple Lambert shading)
VERTEX_SHADER_SOURCE = '''
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_normal;
out vec3 v_world_pos;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = mat3(u_model) * in_normal;
    gl_Position = u_projection * u_view * world_pos;
}
'''

FRAGMENT_SHADER_SOURCE = '''
#version 330 core
in vec3 v_normal;
in vec3 v_world_pos;

uniform vec3 u_light_dir;
uniform vec3 u_color;
uniform vec3 u_ambient_sky;
uniform vec3 u_ambient_ground;
uniform float u_ambient_intensity;

out vec4 f_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    float lambert = max(dot(N, L), 0.0);
    vec3 diffuse = lambert * u_color;

    // Hemisphere ambient (simple GI approximation):
    // N.y in [-1, 1] mixes ground (down) and sky (up)
    float hemi = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 hemi_ambient = mix(u_ambient_ground, u_ambient_sky, hemi);
    vec3 ambient = u_ambient_intensity * hemi_ambient * u_color;

    f_color = vec4(diffuse + ambient, 1.0);
}
'''

def create_cube():
    """Create cube vertex data (position + normal)"""
    pos = [
        (-1, -1, -1), (+1, -1, -1), (+1, +1, -1), (-1, +1, -1),
        (-1, -1, +1), (+1, -1, +1), (+1, +1, +1), (-1, +1, +1),
    ]
    vertices = []
    for face in [
        (0,1,2,3, (0,0,-1)),
        (4,5,6,7, (0,0,1)),
        (0,1,5,4, (0,-1,0)),
        (2,3,7,6, (0,1,0)),
        (0,3,7,4, (-1,0,0)),
        (1,2,6,5, (1,0,0)),
    ]:
        i0,i1,i2,i3, n = face
        for tri in [(i0,i1,i2),(i0,i2,i3)]:
            for i in tri:
                vertices.extend(pos[i])
                vertices.extend(n)
    return np.array(vertices, dtype='float32')

# Shader utilities (from demo_bg.py)
def fail_on_shader_compile_error(shader: int) -> None:
    shader_compile_success = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if not shader_compile_success:
        info_log = GL.glGetShaderInfoLog(shader)
        print(f"ERROR::SHADER::COMPILATION_FAILED\n{info_log}", file=sys.stderr)
        assert shader_compile_success, "Shader compilation failed"

def fail_on_shader_link_error(shader_program: int) -> None:
    is_linked = GL.glGetProgramiv(shader_program, GL.GL_LINK_STATUS)
    if not is_linked:
        info_log = GL.glGetProgramInfoLog(shader_program)
        print(f"ERROR::SHADER::PROGRAM::LINKING_FAILED\n{info_log}", file=sys.stderr)
        assert is_linked, "Shader program linking failed"

def compile_shader(shader_type: int, source: str) -> int:
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    fail_on_shader_compile_error(shader)
    return shader

def create_shader_program(vertex_shader_source: str, fragment_shader_source: str) -> int:
    vertex_shader = compile_shader(GL.GL_VERTEX_SHADER, vertex_shader_source)
    fragment_shader = compile_shader(GL.GL_FRAGMENT_SHADER, fragment_shader_source)
    
    shader_program = GL.glCreateProgram()
    GL.glAttachShader(shader_program, vertex_shader)
    GL.glAttachShader(shader_program, fragment_shader)
    GL.glLinkProgram(shader_program)
    fail_on_shader_link_error(shader_program)
    
    GL.glDeleteShader(vertex_shader)
    GL.glDeleteShader(fragment_shader)
    
    return shader_program

def create_cube_vao():
    """Create VAO for the cube"""
    vertices = create_cube()
    
    # Generate VAO and VBO
    vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    
    vbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
    
    # Position attribute (location 0)
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 6 * 4, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(0)
    
    # Normal attribute (location 1)
    GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
    GL.glEnableVertexAttribArray(1)
    
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)
    
    return vao

def create_sphere_vao(stacks: int = 20, slices: int = 32):
    """Create a UV sphere VAO with positions and normals (triangles)."""
    vertices = []
    for i in range(stacks):
        phi0 = np.pi * i / stacks
        phi1 = np.pi * (i + 1) / stacks
        for j in range(slices):
            theta0 = 2 * np.pi * j / slices
            theta1 = 2 * np.pi * (j + 1) / slices

            # Four points on the sphere
            p00 = np.array([
                np.sin(phi0) * np.cos(theta0),
                np.cos(phi0),
                np.sin(phi0) * np.sin(theta0)
            ], dtype=np.float32)
            p01 = np.array([
                np.sin(phi0) * np.cos(theta1),
                np.cos(phi0),
                np.sin(phi0) * np.sin(theta1)
            ], dtype=np.float32)
            p10 = np.array([
                np.sin(phi1) * np.cos(theta0),
                np.cos(phi1),
                np.sin(phi1) * np.sin(theta0)
            ], dtype=np.float32)
            p11 = np.array([
                np.sin(phi1) * np.cos(theta1),
                np.cos(phi1),
                np.sin(phi1) * np.sin(theta1)
            ], dtype=np.float32)

            # Two triangles per quad
            for tri in [(p00, p10, p11), (p00, p11, p01)]:
                for p in tri:
                    n = p / np.linalg.norm(p)
                    vertices.extend(p.tolist())
                    vertices.extend(n.tolist())

    vertices_np = np.array(vertices, dtype=np.float32)

    vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices_np.nbytes, vertices_np, GL.GL_STATIC_DRAW)
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 6 * 4, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
    GL.glEnableVertexAttribArray(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)

    vertex_count = len(vertices_np) // 6
    return vao, vertex_count

LINE_VERTEX_SHADER = '''
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
uniform mat4 u_mvp;
out vec3 v_color;
void main() {
    v_color = in_color;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
'''

LINE_FRAGMENT_SHADER = '''
#version 330 core
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
'''

def create_grid_vao(size: int = 10, step: int = 1):
    """Create a grid on XZ plane centered at origin using GL_LINES with per-vertex color."""
    verts = []
    color_main = (0.3, 0.3, 0.35)
    for i in range(-size, size + 1, step):
        # Lines parallel to X (vary z)
        z = float(i)
        verts.extend([-size, 0.0, z, *color_main])
        verts.extend([ size, 0.0, z, *color_main])
        # Lines parallel to Z (vary x)
        x = float(i)
        verts.extend([x, 0.0, -size, *color_main])
        verts.extend([x, 0.0,  size, *color_main])
    verts_np = np.array(verts, dtype=np.float32)

    vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, verts_np.nbytes, verts_np, GL.GL_STATIC_DRAW)
    stride = 6 * 4
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
    GL.glEnableVertexAttribArray(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)
    count = len(verts_np) // 6
    return vao, count

def create_gizmo_vao(length: float = 1.5, head: float = 0.15):
    """Create simple axis gizmo with lines and V-shaped arrowheads for X(red), Y(green), Z(blue)."""
    r,g,b = (1.0,0.1,0.1), (0.1,1.0,0.1), (0.1,0.3,1.0)
    verts = []
    # X axis
    verts.extend([0,0,0, *r]); verts.extend([length,0,0, *r])
    verts.extend([length,0,0, *r]); verts.extend([length-head, +head*0.5, 0, *r])
    verts.extend([length,0,0, *r]); verts.extend([length-head, -head*0.5, 0, *r])
    # Y axis
    verts.extend([0,0,0, *g]); verts.extend([0,length,0, *g])
    verts.extend([0,length,0, *g]); verts.extend([+head*0.5, length-head, 0, *g])
    verts.extend([0,length,0, *g]); verts.extend([-head*0.5, length-head, 0, *g])
    # Z axis
    verts.extend([0,0,0, *b]); verts.extend([0,0,length, *b])
    verts.extend([0,0,length, *b]); verts.extend([0, +head*0.5, length-head, *b])
    verts.extend([0,0,length, *b]); verts.extend([0, -head*0.5, length-head, *b])

    verts_np = np.array(verts, dtype=np.float32)
    vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, verts_np.nbytes, verts_np, GL.GL_STATIC_DRAW)
    stride = 6 * 4
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
    GL.glEnableVertexAttribArray(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)
    count = len(verts_np) // 6
    return vao, count

def _create_lines_vao_from_vertices(verts_np: np.ndarray):
    vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, verts_np.nbytes, verts_np, GL.GL_DYNAMIC_DRAW)
    stride = 6 * 4
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
    GL.glEnableVertexAttribArray(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)
    count = len(verts_np) // 6
    return vao, vbo, count

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

def _model_from_pos_target_up(pos: glm.vec3, target: glm.vec3, up: glm.vec3) -> np.ndarray:
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

    def add_camera_representation(self, pos: glm.vec3, target: glm.vec3, up: glm.vec3,
                                  fov_y_deg: float, aspect: float,
                                  near_d: float, length: float,
                                  color=(1.0, 1.0, 0.2)):
        verts_world = _build_frustum_lines_world(pos, target, up, fov_y_deg, aspect, length,
                                                 color=color, include_image_plane=True)
        vao, vbo, count = _create_lines_vao_from_vertices(verts_world)
        model = _model_from_pos_target_up(pos, target, up)
        self.debug_cameras.append({
            'vao': vao,
            'vbo': vbo,
            'count': count,
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
    
    def init_3d_resources(self):
        """Initialize OpenGL resources"""
        print("Initializing 3D resources...")
        
        # Create shader program
        self.shader_program = create_shader_program(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)
        # Line shader
        self.line_shader = create_shader_program(LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER)
        self.loc_line_mvp = GL.glGetUniformLocation(self.line_shader, "u_mvp")
        
        # Create cube VAO
        self.vao = create_cube_vao()
        # Create helpers: grid and gizmo
        self.grid_vao, self.grid_vertex_count = create_grid_vao()
        self.gizmo_vao, self.gizmo_vertex_count = create_gizmo_vao()

        # Add a test camera at (4,4,4) looking at (5,4,4)
        self.add_camera_representation(
            pos=glm.vec3(4.0, 4.0, 4.0),
            target=glm.vec3(5.0, 4.0, 4.0),
            up=glm.vec3(0.0, 1.0, 0.0),
            fov_y_deg=45.0,
            aspect=1.6,
            near_d=0.1,
            length=self.frustum_length,
            color=(1.0, 0.85, 0.2)
        )
        
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
        # Destroy debug camera VAOs/VBOs
        for cam in self.debug_cameras:
            try:
                if cam.get('vao'):
                    GL.glDeleteVertexArrays(1, [cam['vao']])
                if cam.get('vbo'):
                    GL.glDeleteBuffers(1, [cam['vbo']])
            except Exception:
                pass
        self.debug_cameras.clear()

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

        # Draw debug camera frusta and their gizmos
        # For frusta in world space, model is identity
        mvp_lines = projection_array @ view_array
        GL.glUniformMatrix4fv(self.loc_line_mvp, 1, GL.GL_TRUE, mvp_lines)
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
            GL.glBindVertexArray(cam['vao'])
            GL.glLineWidth(1.0)
            GL.glDrawArrays(GL.GL_LINES, 0, cam['count'])
            GL.glBindVertexArray(0)

            # Camera gizmo at its position/orientation (scaled small)
            model_cam = np.array(cam['model'], dtype=np.float32)
            s = 0.5
            scale_m = np.array([
                [s,0,0,0],
                [0,s,0,0],
                [0,0,s,0],
                [0,0,0,1],
            ], dtype=np.float32)
            mvp_gz = projection_array @ view_array @ (model_cam @ scale_m)
            GL.glUniformMatrix4fv(self.loc_line_mvp, 1, GL.GL_TRUE, mvp_gz)
            if self.gizmo_vao and self.gizmo_vertex_count > 0:
                GL.glBindVertexArray(self.gizmo_vao)
                GL.glDrawArrays(GL.GL_LINES, 0, self.gizmo_vertex_count)
                GL.glBindVertexArray(0)

        GL.glUseProgram(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def draw_cube(self):
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

    imgui.separator()
    imgui.text(f"Current Angle:")
    imgui.text(f"  X: {app_state.angle.x:.1f}°")
    imgui.text(f"  Y: {app_state.angle.y:.1f}°")
    imgui.text(f"  Z: {app_state.angle.z:.1f}°")
    imgui.text(f"FPS: {hello_imgui.frame_rate():.1f}")
    
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
