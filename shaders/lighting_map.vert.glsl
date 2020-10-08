#version 330 core

struct Camera {
    // The transformation converting from camera space to 
    // the canonical view volume.
    mat4 projection;
    // The coordinate transformation for converting from
    // world space to camera space.
    mat4 view;
};

struct FragData {
    // The vertex position for a vertex in camera space.
    vec3 position_eye;
    // The texture coordinates for a vertex.
    vec2 tex_coords;
    // The normal vector for a fragment in camera space.
    vec3 normal_eye;
};

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;
layout (location = 2) in vec3 aNormal;

// The coordinate transformation placing an object from model 
// space to world space.
uniform mat4 model;
uniform Camera camera;

out FragData vertex_data;


void main() {
    vertex_data.position_eye = vec3(camera.view * model * vec4(aPos, 1.0));
    vertex_data.tex_coords = aTexCoords;
    vertex_data.normal_eye = vec3(camera.view * model * vec4(aNormal, 0.0));

    gl_Position = camera.projection * vec4(vertex_data.position_eye, 1.0);
}
