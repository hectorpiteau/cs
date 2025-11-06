#version 330 core
in vec3 v_color;
out vec4 f_color;
void main() {
    // Circular point sprite mask (optional); comment to use square points
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    if (dot(coord, coord) > 1.0) discard;
    f_color = vec4(v_color, 1.0);
}