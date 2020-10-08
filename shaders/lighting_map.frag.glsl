#version 330 core
const int NUM_LIGHTS = 3;

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

// Material properties for the Blinn-Phong shader model.
struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float specular_exponent;
};

// A point light with specular, diffuse, and ambient components. Each component is 
// specified in units of 'intensity' which is an unspecified unit of the light's radiant
// exitance on the interval [0, 1]. The three vectors approximate the spectral dependence
// of light 'intensity' in terms of R, G, and B channels.
struct Light {
    // The position of the light in world space.
    vec3 position_world;
    // The ambient component of the point light.
    vec3 ambient;
    // The diffuse component of the point light.
    vec3 diffuse;
    // The specular component of the point light.
    vec3 specular;
};

in FragData vertex_data;

uniform mat4 model;
uniform Camera camera;
uniform Material material;
uniform Light lights[NUM_LIGHTS];

out vec4 frag_color;


void main() {
    vec3 frag_result = vec3(0.0, 0.0, 0.0);
    for (int i = 0; i < NUM_LIGHTS; i++) {
        // Calculate the ambient part of the lighting model.
        vec3 frag_ambient = lights[i].ambient * vec3(texture(material.diffuse, vertex_data.tex_coords));

        // Calculate the diffuse part of the lighting model.
        vec3 norm_eye = normalize(vertex_data.normal_eye);
        vec3 light_position_eye = vec3(camera.view * vec4(lights[i].position_world, 1.0));
        vec3 light_dir_eye = normalize(light_position_eye - vertex_data.position_eye);
        float diff = max(dot(norm_eye, light_dir_eye), 0.0);
        vec3 frag_diffuse = lights[i].diffuse * diff * vec3(texture(material.diffuse, vertex_data.tex_coords));

        // Calculate the specular part of the lighting model.
        vec3 view_dir_eye = normalize(-vertex_data.position_eye);
        vec3 half_vec_eye = normalize(view_dir_eye + light_dir_eye);
        float dot_specular = max(dot(half_vec_eye, norm_eye), 0.0);
        float specular_factor = pow(dot_specular, material.specular_exponent);
        vec3 frag_specular = lights[i].specular * vec3(texture(material.specular, vertex_data.tex_coords)) * specular_factor;

        frag_result += frag_ambient + frag_diffuse + frag_specular;
    }

    frag_color = vec4(frag_result, 1.0);
}
