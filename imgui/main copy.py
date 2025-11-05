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

out vec4 f_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    float lambert = max(dot(N, L), 0.0);
    vec3 diffuse = lambert * u_color;
    vec3 ambient = 0.1 * u_color;
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
        
        # Scene parameters
        self.rotation_speed = glm.vec3(20.0, 30.0, 15.0)
        self.cube_color = glm.vec3(0.2, 0.7, 0.3)
        self.light_dir = glm.vec3(1.0, 1.0, -1.0)
        self.last_time = time.time()
        self.angle = glm.vec3(0.0, 0.0, 0.0)
    
    def init_3d_resources(self):
        """Initialize OpenGL resources"""
        print("Initializing 3D resources...")
        
        # Create shader program
        self.shader_program = create_shader_program(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)
        
        # Create cube VAO
        self.vao = create_cube_vao()
        
        # Get uniform locations
        self.loc_model = GL.glGetUniformLocation(self.shader_program, "u_model")
        self.loc_view = GL.glGetUniformLocation(self.shader_program, "u_view")
        self.loc_projection = GL.glGetUniformLocation(self.shader_program, "u_projection")
        self.loc_light_dir = GL.glGetUniformLocation(self.shader_program, "u_light_dir")
        self.loc_color = GL.glGetUniformLocation(self.shader_program, "u_color")
        
        print(f"Shader program: {self.shader_program}")
        print(f"VAO: {self.vao}")
        print(f"Uniform locations: model={self.loc_model}, view={self.loc_view}, proj={self.loc_projection}")
        print("3D resources initialized!")
    
    def destroy_3d_resources(self):
        """Clean up OpenGL resources"""
        if self.vao:
            GL.glDeleteVertexArrays(1, [self.vao])
        if self.shader_program:
            GL.glDeleteProgram(self.shader_program)


def scaled_display_size():
    """Get display size in pixels (accounting for retina displays)"""
    io = imgui.get_io()
    return ImVec2(
        io.display_size.x * io.display_framebuffer_scale.x,
        io.display_size.y * io.display_framebuffer_scale.y
    )


def custom_background(app_state: AppState):
    """Render the 3D cube scene as the background"""
    if app_state.shader_program is None:
        return
    
    # Update timing and rotation
    current = time.time()
    dt = current - app_state.last_time
    app_state.last_time = current
    app_state.angle += app_state.rotation_speed * dt
    
    # Get display size
    display_size = scaled_display_size()
    
    # Set viewport and clear
    GL.glViewport(0, 0, int(display_size.x), int(display_size.y))
    GL.glClearColor(0.1, 0.1, 0.12, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    
    # Enable depth testing, disable culling (some faces may be wound oppositely)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glDisable(GL.GL_CULL_FACE)
    
    # Use shader program
    GL.glUseProgram(app_state.shader_program)
    
    # Build transformation matrices
    model = glm.mat4(1.0)
    model = glm.rotate(model, glm.radians(app_state.angle.x), glm.vec3(1, 0, 0))
    model = glm.rotate(model, glm.radians(app_state.angle.y), glm.vec3(0, 1, 0))
    model = glm.rotate(model, glm.radians(app_state.angle.z), glm.vec3(0, 0, 1))
    
    view = glm.lookAt(glm.vec3(3, 3, 3), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    
    aspect = display_size.x / display_size.y if display_size.y > 0 else 1.0
    projection = glm.perspective(glm.radians(45.0), aspect, 0.1, 100.0)
    
    # Upload uniforms
    model_array = np.array(model, dtype='float32')
    view_array = np.array(view, dtype='float32')
    projection_array = np.array(projection, dtype='float32')
    
    # Transpose True because numpy arrays are row-major; OpenGL expects column-major
    GL.glUniformMatrix4fv(app_state.loc_model, 1, GL.GL_TRUE, model_array)
    GL.glUniformMatrix4fv(app_state.loc_view, 1, GL.GL_TRUE, view_array)
    GL.glUniformMatrix4fv(app_state.loc_projection, 1, GL.GL_TRUE, projection_array)
    
    GL.glUniform3f(app_state.loc_light_dir, 
                   app_state.light_dir.x, app_state.light_dir.y, app_state.light_dir.z)
    GL.glUniform3f(app_state.loc_color,
                   app_state.cube_color.x, app_state.cube_color.y, app_state.cube_color.z)
    
    # Render the cube
    GL.glBindVertexArray(app_state.vao)
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, app_state.vertex_count)
    GL.glBindVertexArray(0)
    
    # Cleanup
    GL.glUseProgram(0)


def gui(app_state: AppState):
    """GUI controls"""
    imgui.set_next_window_pos(ImVec2(10, 10), imgui.Cond_.appearing)
    imgui.set_next_window_size(ImVec2(300, 280), imgui.Cond_.appearing)
    
    imgui.begin("Cube Controls")
    
    imgui.text("Rotation Speed (deg/s):")
    _, app_state.rotation_speed.x = imgui.slider_float("Speed X", app_state.rotation_speed.x, 0.0, 360.0)
    _, app_state.rotation_speed.y = imgui.slider_float("Speed Y", app_state.rotation_speed.y, 0.0, 360.0)
    _, app_state.rotation_speed.z = imgui.slider_float("Speed Z", app_state.rotation_speed.z, 0.0, 360.0)
    
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
