import os


def _read_text(full_path: str) -> str:
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


def load_shader_source(filename: str) -> str:
    """Load a shader source file located next to this loader.

    filename: e.g., "simple_lambert.vertex.shader"
    Returns the file contents as a string.
    """
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, filename)
    return _read_text(full_path)


def load_vertex_fragment(vertex_filename: str, fragment_filename: str) -> tuple[str, str]:
    """Load a pair of vertex/fragment shader sources from files in this folder."""
    vsrc = load_shader_source(vertex_filename)
    fsrc = load_shader_source(fragment_filename)
    return vsrc, fsrc
