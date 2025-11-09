import OpenGL.GL as GL
import sys

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