import numpy as np
import ctypes
import OpenGL.GL as GL

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