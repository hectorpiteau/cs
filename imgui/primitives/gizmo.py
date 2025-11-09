import numpy as np
import ctypes
import OpenGL.GL as GL


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