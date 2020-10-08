extern crate glfw;
extern crate cglinalg;
extern crate cgperspective;
extern crate image;
extern crate log;
extern crate file_logger;
extern crate mini_obj;


mod gl {
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));
}

mod backend;
mod camera;
mod light;
mod material;
mod texture;


use material::Material;
use cglinalg::{
    Angle,
    Degrees,
    Matrix4,
    Radians,
    Array,
    Vector3,
    Identity,
    Unit,
};
use cgperspective::{
    SimpleCameraMovement,
    CameraMovement,
    CameraAttitudeSpec,
    PerspectiveFovSpec,
    PerspectiveFovProjection,
    FreeKinematics,
    FreeKinematicsSpec,
    Camera
};
use glfw::{
    Action, 
    Context, 
    Key
};
use gl::types::{
    GLfloat,
    GLint,
    GLuint, 
    GLvoid, 
    GLsizeiptr,
};
use log::{
    info
};
use mini_obj::{
    ObjMesh
};

use crate::backend::{
    OpenGLContext,
};
use crate::camera::{
    PerspectiveFovCamera,
};
use crate::texture::{
    LightingMap,
    TextureImage2D,
};
use crate::light::*;

use std::io;
use std::mem;
use std::ptr;


// OpenGL extension constants.
const GL_TEXTURE_MAX_ANISOTROPY_EXT: u32 = 0x84FE;
const GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT: u32 = 0x84FF;

// Default value for the color buffer.
const CLEAR_COLOR: [f32; 4] = [0.1_f32, 0.1_f32, 0.1_f32, 1.0_f32];
// Default value for the depth buffer.
const CLEAR_DEPTH: [f32; 4] = [1.0_f32, 1.0_f32, 1.0_f32, 1.0_f32];

const SCREEN_WIDTH: u32 = 800;
const SCREEN_HEIGHT: u32 = 600;


fn create_box_mesh() -> ObjMesh {
    let points: Vec<[f32; 3]> = vec![
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5],
        [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5], [-0.5, -0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5],  
        [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5, -0.5,  0.5],
        [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5], [-0.5, -0.5, -0.5], 
        [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], 
        [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5], 
        [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5],
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5],  
        [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5], [-0.5, -0.5, -0.5],
        [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5], 
        [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],  
    ];
    let tex_coords = vec![
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
        [1.0, 1.0], [0.0, 1.0], [0.0, 0.0],
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
        [1.0, 1.0], [0.0, 1.0], [0.0, 0.0],
        [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        [0.0, 1.0], [0.0, 0.0], [1.0, 0.0],
        [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        [0.0, 1.0], [0.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [1.0, 0.0],
        [1.0, 0.0], [0.0, 0.0], [0.0, 1.0],
        [0.0, 1.0], [1.0, 1.0], [1.0, 0.0],
        [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]
    ];
    let normals = vec![
        [ 0.0,  0.0, -1.0], [ 0.0,  0.0, -1.0], [ 0.0,  0.0, -1.0],
        [ 0.0,  0.0, -1.0], [ 0.0,  0.0, -1.0], [ 0.0,  0.0, -1.0],
        [ 0.0,  0.0,  1.0], [ 0.0,  0.0,  1.0], [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0,  1.0], [ 0.0,  0.0,  1.0], [ 0.0,  0.0,  1.0],
        [-1.0,  0.0,  0.0], [-1.0,  0.0,  0.0], [-1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0], [-1.0,  0.0,  0.0], [-1.0,  0.0,  0.0],
        [ 1.0,  0.0,  0.0], [ 1.0,  0.0,  0.0], [ 1.0,  0.0,  0.0],
        [ 1.0,  0.0,  0.0], [ 1.0,  0.0,  0.0], [ 1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0], [ 0.0, -1.0,  0.0], [ 0.0, -1.0,  0.0],
        [ 0.0, -1.0,  0.0], [ 0.0, -1.0,  0.0], [ 0.0, -1.0,  0.0],
        [ 0.0,  1.0,  0.0], [ 0.0,  1.0,  0.0], [ 0.0,  1.0,  0.0],
        [ 0.0,  1.0,  0.0], [ 0.0,  1.0,  0.0], [ 0.0,  1.0,  0.0],
    ];

    ObjMesh::new(points, tex_coords, normals)
}

fn create_box_positions() -> Vec<Vector3<f32>> {
    vec![
        Vector3::new( 0.0,  0.0,  0.0),
        Vector3::new( 2.0,  5.0, -15.0),
        Vector3::new(-1.5, -2.2, -2.5),
        Vector3::new(-3.8, -2.0, -12.3),
        Vector3::new( 2.4, -0.4, -3.5),
        Vector3::new(-1.7,  3.0, -7.5),
        Vector3::new( 1.3, -2.0, -2.5),
        Vector3::new( 1.5,  2.0, -2.5),
        Vector3::new( 1.5,  0.2, -1.5),
        Vector3::new(-1.3,  1.0, -1.5)
    ]
}

fn create_box_model_matrices(box_positions: &[Vector3<f32>]) -> Vec<Matrix4<f32>> {
    let mut box_model_matrices = vec![];
    for i in 0..box_positions.len() {
        let translation_i = Matrix4::from_affine_translation(&box_positions[i]);
        let angle_i = Radians(20.0 * (i as f32));
        let axis_i = Unit::from_value(Vector3::new(1.0, 0.3, 0.5));
        let rotation_i = Matrix4::from_affine_axis_angle(&axis_i, angle_i);
        let model_i = rotation_i * translation_i;
        box_model_matrices.push(model_i);
    }

    debug_assert_eq!(box_model_matrices.len(), box_positions.len());

    box_model_matrices
}

fn create_camera(width: u32, height: u32) -> PerspectiveFovCamera<f32> {
    let near = 0.1;
    let far = 100.0;
    let fovy = Degrees(72.0);
    let aspect = width as f32 / height as f32;
    let model_spec = PerspectiveFovSpec::new(
        fovy, 
        aspect, 
        near, 
        far
    );
    let position = Vector3::new(0.0, 0.0, 3.0);
    let forward = Vector3::new(0.0, 0.0, 1.0);
    let right = Vector3::new(1.0, 0.0, 0.0);
    let up  = Vector3::new(0.0, 1.0, 0.0);
    let axis = Vector3::new(0.0, 0.0, -1.0);
    let attitude_spec = CameraAttitudeSpec::new(
        position,
        forward,
        right,
        up,
        axis,
    );
    let movement_speed = 5.0;
    let rotation_speed = Degrees(50.0);
    let kinematics_spec = FreeKinematicsSpec::new(
        movement_speed, 
        rotation_speed
    );

    Camera::new(&model_spec, &attitude_spec, &kinematics_spec)
}

fn create_cube_lights() -> [PointLight<f32>; 4] {
    let position_0 = Vector3::new(0.7, 0.2, 2.0);
    let ambient_0 = Vector3::new(0.2, 0.2, 0.2);
    let diffuse_0 = Vector3::new(0.5, 0.5, 0.5);
    let specular_0 = Vector3::new(1.0, 1.0, 1.0);
    let constant_0 = 1.0;
    let linear_0 = 0.09;
    let quadratic_0 = 0.032;
    let light_0 = PointLight::new(
        position_0,
        constant_0,
        linear_0,
        quadratic_0,
        ambient_0, 
        diffuse_0, 
        specular_0
    );

    let position_1 = Vector3::new(2.3, -3.3, -4.0);
    let ambient_1 = Vector3::new(0.2, 0.2, 0.2);   
    let diffuse_1 = Vector3::new(0.5, 0.5, 0.5);
    let specular_1 = Vector3::new(1.0, 1.0, 1.0);
    let constant_1 = 1.0;
    let linear_1 = 0.09;
    let quadratic_1 = 0.032;
    let light_1 = PointLight::new(
        position_1,
        constant_1,
        linear_1,
        quadratic_1,
        ambient_1,
        diffuse_1,
        specular_1
    );


    let position_2 = Vector3::new(-4.0, 2.0, -12.0);
    let ambient_2 = Vector3::new(0.2, 0.2, 0.2);
    let diffuse_2 = Vector3::new(0.5, 0.5, 0.5);
    let specular_2 = Vector3::new(1.0, 1.0, 1.0);
    let constant_2 = 1.0;
    let linear_2 = 0.09;
    let quadratic_2 = 0.032;
    let light_2 = PointLight::new(
        position_2,
        constant_2,
        linear_2,
        quadratic_2,
        ambient_2, 
        diffuse_2, 
        specular_2
    );

    let position_3 = Vector3::new(0.0, 0.0, -3.0);
    let ambient_3 = Vector3::new(0.05, 0.05, 0.05);
    let diffuse_3 = Vector3::new(0.8, 0.8, 0.8);
    let specular_3 = Vector3::new(1.0, 1.0, 1.0);
    let constant_3 = 1.0;
    let linear_3 = 0.09;
    let quadratic_3 = 0.032;
    let light_3 = PointLight::new(
        position_3,
        constant_3,
        linear_3,
        quadratic_3,
        ambient_3,
        diffuse_3,
        specular_3
    );

    [light_0, light_1, light_2, light_3]
}

fn create_directional_light() -> DirLight<f32> {
    let direction = Vector3::new(-0.2, -1.0, -0.3);
    let ambient = Vector3::new(0.05, 0.05, 0.05);
    let diffuse = Vector3::new(0.4, 0.4, 0.4);
    let specular = Vector3::new(0.5, 0.5, 0.5);

    DirLight::new(direction, ambient, diffuse, specular)
}

fn create_spotlight(camera: &PerspectiveFovCamera<f32>) -> SpotLight<f32> {
    let position = camera.position();
    let direction = camera.forward_axis();
    let ambient = Vector3::new(0.0, 0.0, 0.0);
    let diffuse = Vector3::new(1.0, 1.0, 1.0);
    let specular = Vector3::new(1.0, 1.0, 1.0);
    let constant = 1.0;
    let linear = 0.09;
    let quadratic = 0.032;
    let cutoff = Degrees(12.5).cos();
    let outer_cutoff = Degrees(15.0).cos();

    SpotLight::new(
        position,
        direction,
        cutoff,
        outer_cutoff,
        constant,
        linear,
        quadratic,
        ambient,
        diffuse,
        specular
    )
}

fn create_lighting_map() -> LightingMap {
    let diffuse_buffer = include_bytes!("../assets/container2_diffuse.png");
    let specular_buffer = include_bytes!("../assets/container2_specular.png");
    let emission_buffer = include_bytes!("../assets/matrix.png");
    
    texture::load_lighting_map(diffuse_buffer, specular_buffer, emission_buffer)
}

fn send_to_gpu_uniforms_cube_light_mesh(shader: GLuint, model_mat: &Matrix4<f32>) {
    let model_mat_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("model").as_ptr())
    };
    debug_assert!(model_mat_loc > -1);
    
    unsafe {
        gl::UseProgram(shader);
        gl::UniformMatrix4fv(model_mat_loc, 1, gl::FALSE, model_mat.as_ptr());
    }
}

fn send_to_gpu_uniforms_camera(
    shader: GLuint, camera: &Camera<f32, PerspectiveFovProjection<f32>, FreeKinematics<f32>>) {
    
    let camera_proj_mat_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("camera.projection").as_ptr())
    };
    debug_assert!(camera_proj_mat_loc > -1);
    let camera_view_mat_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("camera.view").as_ptr())
    };
    debug_assert!(camera_view_mat_loc > -1);

    unsafe {
        gl::UseProgram(shader);
        gl::UniformMatrix4fv(
            camera_proj_mat_loc, 
            1, 
            gl::FALSE, 
            camera.projection().as_ptr()
        );
        gl::UniformMatrix4fv(
            camera_view_mat_loc, 
            1, 
            gl::FALSE, 
            camera.view_matrix().as_ptr()
        );
    }
}

fn send_to_gpu_uniforms_dir_light(shader: GLuint, light: &DirLight<f32>) {
    let light_direction_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("dirLight.direction").as_ptr())
    };
    debug_assert!(light_direction_loc > -1);
    let light_ambient_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("dirLight.ambient").as_ptr())
    };
    debug_assert!(light_ambient_loc > -1);
    let light_diffuse_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("dirLight.diffuse").as_ptr())
    };
    debug_assert!(light_diffuse_loc > -1);
    let light_specular_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("dirLight.specular").as_ptr())
    };
    debug_assert!(light_specular_loc > -1);

    unsafe {
        gl::UseProgram(shader);
        gl::Uniform3fv(light_direction_loc, 1, light.direction.as_ptr());
        gl::Uniform3fv(light_ambient_loc, 1, light.ambient.as_ptr());
        gl::Uniform3fv(light_diffuse_loc, 1, light.diffuse.as_ptr());
        gl::Uniform3fv(light_specular_loc, 1, light.specular.as_ptr());
    }
}

/// Send the uniforms for the lighting data to the GPU for the mesh.
/// Note that in order to render multiple lights in the shader, we define an 
/// array of structs. In OpenGL, each elementary member of a struct is 
/// considered to be a uniform variable, and each struct is a struct of uniforms. 
/// Consequently, if every element of an array of struct uniforms is not used in 
/// the shader, OpenGL will optimize those uniform locations out at runtime. This
/// will cause OpenGL to return a `GL_INVALID_VALUE` on a call to 
/// `glGetUniformLocation`.
fn send_to_gpu_uniforms_point_lights(shader: GLuint, lights: &[PointLight<f32>; 4]) {
    let light_position_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[0].position").as_ptr())
    };
    debug_assert!(light_position_loc > -1);
    let light_constant_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[0].constant").as_ptr())
    };
    debug_assert!(light_constant_loc > -1);
    let light_linear_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[0].linear").as_ptr())
    };
    debug_assert!(light_linear_loc > -1);
    let light_quadratic_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[0].quadratic").as_ptr())
    };
    debug_assert!(light_linear_loc > -1);
    let light_ambient_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[0].ambient").as_ptr())
    };
    debug_assert!(light_ambient_loc > -1);
    let light_diffuse_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[0].diffuse").as_ptr())
    };
    debug_assert!(light_diffuse_loc > -1);
    let light_specular_loc = unsafe { 
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[0].specular").as_ptr())
    };
    debug_assert!(light_specular_loc > -1);

    unsafe {
        gl::UseProgram(shader);
        gl::Uniform3fv(light_position_loc, 1, lights[0].position.as_ptr());
        gl::Uniform1f(light_constant_loc, lights[0].constant);
        gl::Uniform1f(light_linear_loc, lights[0].linear);
        gl::Uniform1f(light_quadratic_loc, lights[0].quadratic);
        gl::Uniform3fv(light_ambient_loc, 1, lights[0].ambient.as_ptr());
        gl::Uniform3fv(light_diffuse_loc, 1, lights[0].diffuse.as_ptr());
        gl::Uniform3fv(light_specular_loc, 1, lights[0].specular.as_ptr());
    }

    let light_position_world_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[1].position").as_ptr())
    };
    debug_assert!(light_position_world_loc > -1);
    let light_constant_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[1].constant").as_ptr())
    };
    debug_assert!(light_constant_loc > -1);
    let light_linear_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[1].linear").as_ptr())
    };
    debug_assert!(light_linear_loc > -1);
    let light_quadratic_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[1].quadratic").as_ptr())
    };
    debug_assert!(light_linear_loc > -1);
    let light_ambient_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[1].ambient").as_ptr())
    };
    debug_assert!(light_ambient_loc > -1);
    let light_diffuse_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[1].diffuse").as_ptr())
    };
    debug_assert!(light_diffuse_loc > -1);
    let light_specular_loc = unsafe { 
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[1].specular").as_ptr())
    };
    debug_assert!(light_specular_loc > -1);

    unsafe {
        gl::UseProgram(shader);
        gl::Uniform3fv(light_position_world_loc, 1, lights[1].position.as_ptr());
        gl::Uniform1f(light_constant_loc, lights[1].constant);
        gl::Uniform1f(light_linear_loc, lights[1].linear);
        gl::Uniform1f(light_quadratic_loc, lights[1].quadratic);
        gl::Uniform3fv(light_ambient_loc, 1, lights[1].ambient.as_ptr());
        gl::Uniform3fv(light_diffuse_loc, 1, lights[1].diffuse.as_ptr());
        gl::Uniform3fv(light_specular_loc, 1, lights[1].specular.as_ptr());
    }

    let light_position_world_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[2].position").as_ptr())
    };
    debug_assert!(light_position_world_loc > -1);
    let light_constant_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[2].constant").as_ptr())
    };
    debug_assert!(light_constant_loc > -1);
    let light_linear_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[2].linear").as_ptr())
    };
    debug_assert!(light_linear_loc > -1);
    let light_quadratic_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[2].quadratic").as_ptr())
    };
    debug_assert!(light_linear_loc > -1);
    let light_ambient_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[2].ambient").as_ptr())
    };
    debug_assert!(light_ambient_loc > -1);
    let light_diffuse_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[2].diffuse").as_ptr())
    };
    debug_assert!(light_diffuse_loc > -1);
    let light_specular_loc = unsafe { 
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[2].specular").as_ptr())
    };
    debug_assert!(light_specular_loc > -1);

    unsafe {
        gl::UseProgram(shader);
        gl::Uniform3fv(light_position_world_loc, 1, lights[2].position.as_ptr());
        gl::Uniform1f(light_constant_loc, lights[2].constant);
        gl::Uniform1f(light_linear_loc, lights[2].linear);
        gl::Uniform1f(light_quadratic_loc, lights[2].quadratic);
        gl::Uniform3fv(light_ambient_loc, 1, lights[2].ambient.as_ptr());
        gl::Uniform3fv(light_diffuse_loc, 1, lights[2].diffuse.as_ptr());
        gl::Uniform3fv(light_specular_loc, 1, lights[2].specular.as_ptr());
    }

    let light_position_world_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[3].position").as_ptr())
    };
    debug_assert!(light_position_world_loc > -1);
    let light_constant_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[3].constant").as_ptr())
    };
    debug_assert!(light_constant_loc > -1);
    let light_linear_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[3].linear").as_ptr())
    };
    debug_assert!(light_linear_loc > -1);
    let light_quadratic_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[3].quadratic").as_ptr())
    };
    debug_assert!(light_linear_loc > -1);
    let light_ambient_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[3].ambient").as_ptr())
    };
    debug_assert!(light_ambient_loc > -1);
    let light_diffuse_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[3].diffuse").as_ptr())
    };
    debug_assert!(light_diffuse_loc > -1);
    let light_specular_loc = unsafe { 
        gl::GetUniformLocation(shader, backend::gl_str("pointLights[3].specular").as_ptr())
    };
    debug_assert!(light_specular_loc > -1);

    unsafe {
        gl::UseProgram(shader);
        gl::Uniform3fv(light_position_world_loc, 1, lights[3].position.as_ptr());
        gl::Uniform1f(light_constant_loc, lights[3].constant);
        gl::Uniform1f(light_linear_loc, lights[3].linear);
        gl::Uniform1f(light_quadratic_loc, lights[3].quadratic);
        gl::Uniform3fv(light_ambient_loc, 1, lights[3].ambient.as_ptr());
        gl::Uniform3fv(light_diffuse_loc, 1, lights[3].diffuse.as_ptr());
        gl::Uniform3fv(light_specular_loc, 1, lights[3].specular.as_ptr());
    }
}

/// Send the uniforms for the lighting data to the GPU for the mesh.
/// Note that in order to render multiple lights in the shader, we define an 
/// array of structs. In OpenGL, each elementary member of a struct is 
/// considered to be a uniform variable, and each struct is a struct of uniforms. 
/// Consequently, if every element of an array of struct uniforms is not used in 
/// the shader, OpenGL will optimize those uniform locations out at runtime. This
/// will cause OpenGL to return a `GL_INVALID_VALUE` on a call to 
/// `glGetUniformLocation`.
fn send_to_gpu_uniforms_spotlight(shader: GLuint, light: &SpotLight<f32>) {
    let light_position_world_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("spotLight.position").as_ptr())
    };
    debug_assert!(light_position_world_loc > -1);
    let light_direction_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("spotLight.direction").as_ptr())
    };
    debug_assert!(light_direction_loc > -1);
    let light_cutoff_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("spotLight.cutOff").as_ptr())
    };
    debug_assert!(light_cutoff_loc > -1);
    let light_outer_cutoff_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("spotLight.outerCutOff").as_ptr())
    };
    debug_assert!(light_outer_cutoff_loc > -1);
    let light_ambient_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("spotLight.ambient").as_ptr())
    };
    debug_assert!(light_ambient_loc > -1);
    let light_diffuse_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("spotLight.diffuse").as_ptr())
    };
    debug_assert!(light_diffuse_loc > -1);
    let light_specular_loc = unsafe { 
        gl::GetUniformLocation(shader, backend::gl_str("spotLight.specular").as_ptr())
    };
    debug_assert!(light_specular_loc > -1);
    let light_constant_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("spotLight.constant").as_ptr())
    };
    debug_assert!(light_constant_loc > -1);
    let light_linear_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("spotLight.linear").as_ptr())
    };
    debug_assert!(light_linear_loc > -1);
    let light_quadratic_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("spotLight.quadratic").as_ptr())
    };
    debug_assert!(light_quadratic_loc > -1);

    unsafe {
        gl::UseProgram(shader);
        gl::Uniform3fv(light_position_world_loc, 1, light.position.as_ptr());
        gl::Uniform3fv(light_direction_loc, 1, light.direction.as_ptr());
        gl::Uniform1f(light_cutoff_loc, light.cutoff);
        gl::Uniform1f(light_outer_cutoff_loc, light.outer_cutoff);
        gl::Uniform3fv(light_ambient_loc, 1, light.ambient.as_ptr());
        gl::Uniform3fv(light_diffuse_loc, 1, light.diffuse.as_ptr());
        gl::Uniform3fv(light_specular_loc, 1, light.specular.as_ptr());
        gl::Uniform1f(light_constant_loc, light.constant);
        gl::Uniform1f(light_linear_loc, light.linear);
        gl::Uniform1f(light_quadratic_loc, light.quadratic);
    }
}

fn send_to_gpu_textures_material(lighting_map: &LightingMap) -> (GLuint, GLuint) {
    let diffuse_tex = send_to_gpu_texture(&lighting_map.diffuse, gl::REPEAT).unwrap();
    let specular_tex = send_to_gpu_texture(&lighting_map.specular, gl::REPEAT).unwrap();

    (diffuse_tex, specular_tex)
}

#[derive(Copy, Clone)]
struct MaterialUniforms<'a> {
    diffuse_index: i32,
    specular_index: i32,
    material: &'a Material<f32>,
}

fn send_to_gpu_uniforms_material(shader: GLuint, uniforms: MaterialUniforms) {
    let material_diffuse_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("material.diffuse").as_ptr())
    };
    debug_assert!(material_diffuse_loc > -1);
    let material_specular_loc = unsafe {
        gl::GetUniformLocation(shader, backend::gl_str("material.specular").as_ptr())
    };
    debug_assert!(material_specular_loc > -1);
    let material_specular_exponent_loc = unsafe { 
        gl::GetUniformLocation(shader, backend::gl_str("material.specular_exponent").as_ptr())
    };
    debug_assert!(material_specular_exponent_loc > -1);

    unsafe {
        gl::UseProgram(shader);
        gl::Uniform1i(material_diffuse_loc, uniforms.diffuse_index);
        gl::Uniform1i(material_specular_loc, uniforms.specular_index);
        gl::Uniform1f(material_specular_exponent_loc, uniforms.material.specular_exponent);
    }
}

fn send_to_gpu_mesh(shader: GLuint, mesh: &ObjMesh) -> (GLuint, GLuint, GLuint, GLuint) {
    let v_pos_loc = unsafe {
        gl::GetAttribLocation(shader, backend::gl_str("aPos").as_ptr())
    };
    debug_assert!(v_pos_loc > -1);
    let v_pos_loc = v_pos_loc as u32;

    let v_tex_loc = unsafe {
        gl::GetAttribLocation(shader, backend::gl_str("aTexCoords").as_ptr())
    };
    debug_assert!(v_tex_loc > -1);
    let v_tex_loc = v_tex_loc as u32;

    let v_norm_loc = unsafe {
        gl::GetAttribLocation(shader, backend::gl_str("aNormal").as_ptr())
    };
    debug_assert!(v_norm_loc > -1);
    let v_norm_loc = v_norm_loc as u32;

    let mut v_pos_vbo = 0;
    unsafe {
        gl::GenBuffers(1, &mut v_pos_vbo);
    }
    debug_assert!(v_pos_vbo > 0);
    unsafe {
        gl::BindBuffer(gl::ARRAY_BUFFER, v_pos_vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            mesh.points.len_bytes() as GLsizeiptr,
            mesh.points.as_ptr() as *const GLvoid,
            gl::STATIC_DRAW
        );
    }

    let mut v_tex_vbo = 0;
    unsafe {
        gl::GenBuffers(1, &mut v_tex_vbo);
    }
    debug_assert!(v_tex_vbo > 0);
    unsafe {
        gl::BindBuffer(gl::ARRAY_BUFFER, v_tex_vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            mesh.tex_coords.len_bytes() as GLsizeiptr,
            mesh.tex_coords.as_ptr() as *const GLvoid,
            gl::STATIC_DRAW
        )
    }

    let mut v_norm_vbo = 0;
    unsafe {
        gl::GenBuffers(1, &mut v_norm_vbo);
    }
    debug_assert!(v_norm_vbo > 0);
    unsafe {
        gl::BindBuffer(gl::ARRAY_BUFFER, v_norm_vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            mesh.normals.len_bytes() as GLsizeiptr,
            mesh.normals.as_ptr() as *const GLvoid,
            gl::STATIC_DRAW
        );
    }

    let mut vao = 0;
    unsafe {
        gl::GenVertexArrays(1, &mut vao);
        gl::BindVertexArray(vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, v_pos_vbo);
        gl::VertexAttribPointer(v_pos_loc, 3, gl::FLOAT, gl::FALSE, 0, ptr::null());
        gl::BindBuffer(gl::ARRAY_BUFFER, v_tex_vbo);
        gl::VertexAttribPointer(v_tex_loc, 2, gl::FLOAT, gl::FALSE, 0, ptr::null());
        gl::BindBuffer(gl::ARRAY_BUFFER, v_norm_vbo);
        gl::VertexAttribPointer(v_norm_loc, 3, gl::FLOAT, gl::FALSE, 0, ptr::null());
        gl::EnableVertexAttribArray(v_pos_loc);
        gl::EnableVertexAttribArray(v_tex_loc);
        gl::EnableVertexAttribArray(v_norm_loc);
    }
    debug_assert!(vao > 0);

    (vao, v_pos_vbo, v_tex_vbo, v_norm_vbo)
}

fn send_to_gpu_light_mesh(shader: GLuint, mesh: &ObjMesh) -> (GLuint, GLuint) {
    let v_pos_loc = unsafe {
        gl::GetAttribLocation(shader, backend::gl_str("v_pos").as_ptr())
    };
    debug_assert!(v_pos_loc > -1);
    let v_pos_loc = v_pos_loc as u32;

    let mut v_pos_vbo = 0;
    unsafe {
        gl::GenBuffers(1, &mut v_pos_vbo);
        gl::BindBuffer(gl::ARRAY_BUFFER, v_pos_vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (3 * mem::size_of::<GLfloat>() * mesh.points.len()) as GLsizeiptr,
            mesh.points.as_ptr() as *const GLvoid,
            gl::STATIC_DRAW
        );
    }
    debug_assert!(v_pos_vbo > 0);

    let mut vao = 0;
    unsafe {
        gl::GenVertexArrays(1, &mut vao);
        gl::BindVertexArray(vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, v_pos_vbo);
        gl::VertexAttribPointer(v_pos_loc, 3, gl::FLOAT, gl::FALSE, 0, ptr::null());
        gl::EnableVertexAttribArray(v_pos_loc);
    }
    debug_assert!(vao > 0);

    (vao, v_pos_vbo)
}

/// Load texture image into the GPU.
/// TODO: Move this function into the backend module.
fn send_to_gpu_texture(texture_image: &TextureImage2D, wrapping_mode: GLuint) -> Result<GLuint, String> {
    let mut tex = 0;
    unsafe {
        gl::GenTextures(1, &mut tex);
    }
    debug_assert!(tex > 0);
    unsafe {
        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(gl::TEXTURE_2D, tex);
        gl::TexImage2D(
            gl::TEXTURE_2D, 0, gl::RGBA as i32, texture_image.width as i32, texture_image.height as i32, 0,
            gl::RGBA, gl::UNSIGNED_BYTE,
            texture_image.as_ptr() as *const GLvoid
        );
        gl::GenerateMipmap(gl::TEXTURE_2D);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, wrapping_mode as GLint);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, wrapping_mode as GLint);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as GLint);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR_MIPMAP_LINEAR as GLint);
    }

    let mut max_aniso = 0.0;
    unsafe {
        gl::GetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &mut max_aniso);
        // Set the maximum!
        gl::TexParameterf(gl::TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, max_aniso);
    }

    Ok(tex)
}

#[derive(Copy, Clone)]
struct ShaderSource {
    vert_name: &'static str,
    vert_source: &'static str,
    frag_name: &'static str,
    frag_source: &'static str,
}

fn create_mesh_shader_source() -> ShaderSource {
    let vert_source = include_str!("../shaders/multiple_lights.vert.glsl");
    let frag_source = include_str!("../shaders/multiple_lights.frag.glsl");
    
    ShaderSource {
        vert_name: "multiple_lights.vert.glsl",
        vert_source: vert_source,
        frag_name: "multiple_lights.frag.glsl",
        frag_source: frag_source,
    }
}

fn create_cube_light_shader_source() -> ShaderSource {
    let vert_source = include_str!("../shaders/lighting_cube.vert.glsl");
    let frag_source = include_str!("../shaders/lighting_cube.frag.glsl");

    ShaderSource {
        vert_name: "lighting_cube.vert.glsl",
        vert_source: vert_source,
        frag_name: "lighting_cube.frag.glsl",
        frag_source: frag_source,
    }
}

fn send_to_gpu_shaders(_context: &mut backend::OpenGLContext, source: ShaderSource) -> GLuint {
    let mut vert_reader = io::Cursor::new(source.vert_source);
    let mut frag_reader = io::Cursor::new(source.frag_source);
    let result = backend::compile_from_reader(
        &mut vert_reader, source.vert_name,
        &mut frag_reader, source.frag_name
    );
    let shader = match result {
        Ok(value) => value,
        Err(e) => {
            panic!("Could not compile shaders. Got error: {}", e);
        }
    };
    debug_assert!(shader > 0);

    shader
}

/// Initialize the logger.
fn init_logger(log_file: &str) {
    file_logger::init(log_file).expect("Failed to initialize logger.");
}

/// Create and OpenGL context.
fn init_gl(width: u32, height: u32) -> backend::OpenGLContext {
    let context = match backend::start_opengl(width, height) {
        Ok(val) => val,
        Err(e) => {
            panic!("Failed to Initialize OpenGL context. Got error: {}", e);
        }
    };

    context
}

/// The GLFW frame buffer size callback function. This is normally set using 
/// the GLFW `glfwSetFramebufferSizeCallback` function, but instead we explicitly
/// handle window resizing in our state updates on the application side. Run this function 
/// whenever the size of the viewport changes.
fn framebuffer_size_callback(context: &mut OpenGLContext, width: u32, height: u32) {
    context.width = width;
    context.height = height;
    unsafe {
        gl::Viewport(0, 0, width as i32, height as i32);
    }
}

fn process_input(context: &mut OpenGLContext) -> CameraMovement {
    match context.window.get_key(Key::Escape) {
        Action::Press | Action::Repeat => {
            context.window.set_should_close(true);
        }
        _ => {}
    }

    let mut movement = CameraMovement::new();
    match context.window.get_key(Key::A) {
        Action::Press | Action::Repeat => {
            movement += SimpleCameraMovement::MoveLeft;
        }
        _ => {}
        }
    match context.window.get_key(Key::D) {
        Action::Press | Action::Repeat => {
            movement += SimpleCameraMovement::MoveRight;
        }
        _ => {}
    }
    match context.window.get_key(Key::Q) {
        Action::Press | Action::Repeat => {
            movement += SimpleCameraMovement::MoveUp;
        }
        _ => {}
    }
    match context.window.get_key(Key::E) {
        Action::Press | Action::Repeat => {
            movement += SimpleCameraMovement::MoveDown;
        }
        _ => {}
    }
    match context.window.get_key(Key::W) {
        Action::Press | Action::Repeat => {
            movement += SimpleCameraMovement::MoveForward;
        }
        _ => {}
    }
    match context.window.get_key(Key::S) {
        Action::Press | Action::Repeat => {
            movement += SimpleCameraMovement::MoveBackward;
        }
        _ => {}
    }
    match context.window.get_key(Key::Left) {
        Action::Press | Action::Repeat => {
            movement += SimpleCameraMovement::YawLeft;
        }
        _ => {}
    }
    match context.window.get_key(Key::Right) {
        Action::Press | Action::Repeat => {
            movement += SimpleCameraMovement::YawRight;
        }
        _ => {}
    }
    match context.window.get_key(Key::Up) {
        Action::Press | Action::Repeat => {
            movement += SimpleCameraMovement::PitchUp;
        }
        _ => {}
    }
    match context.window.get_key(Key::Down) {
        Action::Press | Action::Repeat => {
            movement += SimpleCameraMovement::PitchDown;
        }
        _ => {}
    }
    match context.window.get_key(Key::Z) {
        Action::Press | Action::Repeat => {
            movement += SimpleCameraMovement::RollCounterClockwise;
        }
        _ => {}
    }
    match context.window.get_key(Key::C) {
        Action::Press | Action::Repeat => {
            movement += SimpleCameraMovement::RollClockwise;
        }
        _ => {}
    }

    movement
}

fn main() {
    let mesh = create_box_mesh();
    let box_positions = create_box_positions();
    let box_model_matrices = create_box_model_matrices(&box_positions);
    let light_mesh = create_box_mesh();
    init_logger("opengl_demo.log");
    info!("BEGIN LOG");
    let mut camera = create_camera(SCREEN_WIDTH, SCREEN_HEIGHT);
    let cube_lights= create_cube_lights();
    let mut spotlight = create_spotlight(&camera);
    let dir_light = create_directional_light();
    let material_diffuse_index = 0;
    let material_specular_index = 1;
    let material = material::sgi_material_table()["chrome"];
    let material_uniforms = MaterialUniforms { 
        diffuse_index: material_diffuse_index, 
        specular_index: material_specular_index,
        material: &material,
    };
    let lighting_map = create_lighting_map();
    let mut context = init_gl(SCREEN_WIDTH, SCREEN_HEIGHT);

    // The model matrix for the cube light shader and the flashlight shader.
    let mesh_model_mat = Matrix4::identity();

    // Load the lighting maps data for the flashlight and cubelight shaders.
    let (diffuse_tex, specular_tex) = send_to_gpu_textures_material(&lighting_map);

    //  Load the model data for the cube light shader..
    let mesh_shader_source = create_mesh_shader_source();
    let mesh_shader = send_to_gpu_shaders(&mut context, mesh_shader_source);
    let (
        mesh_vao, 
        _mesh_v_pos_vbo,
        _mesh_v_tex_vbo,
        _mesh_v_norm_vbo) = send_to_gpu_mesh(mesh_shader, &mesh);
    send_to_gpu_uniforms_cube_light_mesh(mesh_shader, &mesh_model_mat);
    send_to_gpu_uniforms_camera(mesh_shader, &camera);
    send_to_gpu_uniforms_material(mesh_shader, material_uniforms);
    send_to_gpu_uniforms_point_lights(mesh_shader, &cube_lights);
    send_to_gpu_uniforms_spotlight(mesh_shader, &spotlight);
    send_to_gpu_uniforms_dir_light(mesh_shader, &dir_light);


    // Load the lighting cube model.
    let light_shader_source = create_cube_light_shader_source();
    let light_shader = send_to_gpu_shaders(&mut context, light_shader_source);
    let (
        light_vao,
        _light_v_pos_vbo) = send_to_gpu_light_mesh(light_shader, &light_mesh);

    unsafe {
        gl::Enable(gl::DEPTH_TEST);
        gl::DepthFunc(gl::LESS);
        gl::ClearBufferfv(gl::COLOR, 0, &CLEAR_COLOR[0] as *const GLfloat);
        gl::ClearBufferfv(gl::DEPTH, 0, &CLEAR_DEPTH[0] as *const GLfloat);
        gl::Viewport(0, 0, context.width as GLint, context.height as GLint);
    }

    while !context.window.should_close() {
        let elapsed_seconds = context.update_timers();
        context.update_fps_counter();
        context.glfw.poll_events();
        let (width, height) = context.window.get_framebuffer_size();
        if (width != context.width as i32) && (height != context.height as i32) {
            camera.update_viewport(width as usize, height as usize);
            framebuffer_size_callback(&mut context, width as u32, height as u32);
        }

        let delta_movement = process_input(&mut context);
        camera.update_movement(delta_movement, elapsed_seconds as f32);
        spotlight.update(&camera.position(), &camera.forward_axis());

        send_to_gpu_uniforms_camera(mesh_shader, &camera);
        send_to_gpu_uniforms_camera(light_shader, &camera);
        send_to_gpu_uniforms_point_lights(mesh_shader, &cube_lights);
        send_to_gpu_uniforms_spotlight(mesh_shader, &spotlight);
        send_to_gpu_uniforms_dir_light(mesh_shader, &dir_light);
    
        unsafe {
            gl::ClearBufferfv(gl::COLOR, 0, &CLEAR_COLOR[0] as *const GLfloat);
            gl::ClearBufferfv(gl::DEPTH, 0, &CLEAR_DEPTH[0] as *const GLfloat);
            gl::Viewport(0, 0, context.width as GLint, context.height as GLint);
            gl::UseProgram(mesh_shader);
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, diffuse_tex);
            gl::ActiveTexture(gl::TEXTURE1);
            gl::BindTexture(gl::TEXTURE_2D, specular_tex);
            gl::BindVertexArray(mesh_vao);
            gl::DrawArrays(gl::TRIANGLES, 0, mesh.len() as i32);
        }

        // Illuminate the boxes.
        for model_matrix in box_model_matrices.iter() {
            unsafe {
                send_to_gpu_uniforms_cube_light_mesh(mesh_shader, &model_matrix);
                gl::DrawArrays(gl::TRIANGLES, 0, mesh.len() as i32);
            }
        }

        // Render the cube lights.
        let light_model_mat = cube_lights[0].model_matrix() * Matrix4::from_affine_scale(0.2);
        send_to_gpu_uniforms_cube_light_mesh(light_shader, &light_model_mat);
        unsafe {
            gl::UseProgram(light_shader);
            gl::BindVertexArray(light_vao);
            gl::DrawArrays(gl::TRIANGLES, 0, light_mesh.len() as i32);
        }
        let light_model_mat = cube_lights[1].model_matrix() * Matrix4::from_affine_scale(0.2);
        send_to_gpu_uniforms_cube_light_mesh(light_shader, &light_model_mat);
        unsafe {
            gl::UseProgram(light_shader);
            gl::BindVertexArray(light_vao);
            gl::DrawArrays(gl::TRIANGLES, 0, light_mesh.len() as i32);
        }
        
        let light_model_mat = cube_lights[2].model_matrix() * Matrix4::from_affine_scale(0.2);
        send_to_gpu_uniforms_cube_light_mesh(light_shader, &light_model_mat);
        unsafe {
            gl::UseProgram(light_shader);
            gl::BindVertexArray(light_vao);
            gl::DrawArrays(gl::TRIANGLES, 0, light_mesh.len() as i32);
        }

        context.window.swap_buffers();
    }

    info!("END LOG");
}
