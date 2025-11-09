import numpy as np
import ctypes
import OpenGL.GL as GL


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
