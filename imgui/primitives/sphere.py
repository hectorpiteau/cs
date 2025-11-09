import numpy as np
import ctypes
import OpenGL.GL as GL


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