import numpy as np
import ctypes
import OpenGL.GL as GL


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
