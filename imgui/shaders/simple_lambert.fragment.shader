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