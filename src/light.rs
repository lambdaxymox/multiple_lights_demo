use cglinalg::{
    Quaternion,
    Magnitude,
    Matrix4,
    Radians,
    Vector3,
    Unit,
    ScalarFloat,
};
use cgilluminate::{
    PointLightModel,
    PointLight,
    SpotLight,
    SpotLightModel,
};
use crate::camera::{
    PerspectiveFovCamera,
};


pub struct OrbitalKinematicsSpec<S> {
    scene_center: Vector3<S>,
    radial_speed: S,
    center_of_oscillation: Vector3<S>,
    radius_of_oscillation: S,
    orbital_axis: Vector3<S>,
    orbital_speed: S,
}

impl<S> OrbitalKinematicsSpec<S> {
    pub fn new(
        scene_center: Vector3<S>,
        radial_speed: S,
        center_of_oscillation: Vector3<S>,
        radius_of_oscillation: S,
        orbital_axis: Vector3<S>,
        orbital_speed: S) -> Self
    {
        OrbitalKinematicsSpec {
            scene_center: scene_center,
            radial_speed: radial_speed,
            center_of_oscillation: center_of_oscillation,
            radius_of_oscillation: radius_of_oscillation,
            orbital_axis: orbital_axis,
            orbital_speed: orbital_speed,
        }
    }
}

pub struct OrbitalKinematics<S> {
    scene_center: Vector3<S>,
    radial_speed: S,
    center_of_oscillation: Vector3<S>,
    radius_of_oscillation: S,
    position: Vector3<S>,
    radial_unit_velocity: S,
    orbital_axis: Vector3<S>,
    orbital_speed: S,
}

impl<S> OrbitalKinematics<S> where S: ScalarFloat {    
    pub fn from_spec(spec: &OrbitalKinematicsSpec<S>) -> Self {
        let radial_unit_velocity = S::one();
        let position = spec.center_of_oscillation;

        OrbitalKinematics {
            scene_center: spec.scene_center,
            radial_speed: spec.radial_speed,
            center_of_oscillation: spec.center_of_oscillation,
            radius_of_oscillation: spec.radius_of_oscillation,
            position: position,
            radial_unit_velocity: radial_unit_velocity,
            orbital_axis: spec.orbital_axis.normalize(),
            orbital_speed: spec.orbital_speed,
        }
    }

    pub fn update(&mut self, elapsed: S) -> &Vector3<S> {
        self.radial_unit_velocity = if self.radial_unit_velocity < S::zero() {
            -S::one() 
        } else { 
            S::one()
        };
        let radius_center_of_oscillation = 
            (self.center_of_oscillation - self.scene_center).magnitude();
        let radial_vector: Vector3<S> = (self.position - self.scene_center).normalize();
        let radius_perihelion = radius_center_of_oscillation - self.radius_of_oscillation;
        let radius_aphelion = radius_center_of_oscillation + self.radius_of_oscillation;
        let mut distance_from_scene_center = (self.position - self.scene_center).magnitude();
        distance_from_scene_center = 
            distance_from_scene_center + 
            (self.radial_speed * elapsed) * self.radial_unit_velocity;
        if distance_from_scene_center < radius_perihelion {
            distance_from_scene_center = radius_perihelion;
            self.radial_unit_velocity = S::one();
        } else if distance_from_scene_center > radius_aphelion {
            distance_from_scene_center = radius_aphelion;
            self.radial_unit_velocity = -S::one();
        }
    
        let orbital_axis = Unit::from_value(self.orbital_axis);
        let q = Quaternion::from_axis_angle(
            &orbital_axis, Radians(self.orbital_speed * elapsed)
        );
        let rot_mat = Matrix4::from(q);
        let new_position = rot_mat * (radial_vector * distance_from_scene_center).expand(S::one());
        let new_position = new_position.contract();
        
        self.position = new_position;

        &self.position
    }
}

pub struct CubeLight<S> {
    light: PointLight<S>,
    kinematics: OrbitalKinematics<S>,
}

impl<S> CubeLight<S> where S: ScalarFloat {
    pub fn new(light: PointLight<S>, kinematics: OrbitalKinematics<S>) -> Self {
        Self {
            light: light,
            kinematics: kinematics,
        }
    }

    #[inline]
    pub fn model(&self) -> &PointLightModel<S> {
        self.light.model()
    }

    #[inline]
    pub fn position(&self) -> Vector3<S> {
        self.light.position()
    }

    #[inline]
    pub fn update(&mut self, elapsed: S) {
        self.kinematics.update(elapsed);
        self.light.update_position_world(&self.kinematics.position);
    }

    #[inline]
    pub fn model_matrix(&self) -> Matrix4<S> {
        self.light.model_matrix()
    }
}

pub struct FlashLightKinematics<S> {
    position: Vector3<S>,
    forward: Vector3<S>,
    up: Vector3<S>,
    right: Vector3<S>,
    axis: Vector3<S>,
}

impl<S> FlashLightKinematics<S> where S: ScalarFloat {
    pub fn new(camera: &PerspectiveFovCamera<S>) -> Self {
        Self {
            position: camera.position(),
            forward: camera.forward_axis(),
            up: camera.up_axis(),
            right: camera.right_axis(),
            axis: camera.rotation_axis(),
        }
    }

    pub fn update(&mut self, camera: &PerspectiveFovCamera<S>, _elapsed: S) {
        self.position = camera.position();
        self.forward = camera.forward_axis();
        self.up = camera.up_axis();
        self.right = camera.right_axis();
        self.axis = camera.rotation_axis();
    }
}

pub struct FlashLight<S> {
    light: SpotLight<S>,
    kinematics: FlashLightKinematics<S>,
}

impl<S> FlashLight<S> where S: ScalarFloat {
    pub fn new(light: SpotLight<S>, kinematics: FlashLightKinematics<S>) -> Self {
        Self {
            light: light,
            kinematics: kinematics,
        }
    }

    #[inline]
    pub fn model(&self) -> &SpotLightModel<S> {
        self.light.model()
    }

    #[inline]
    pub fn direction(&self) -> Vector3<S> {
        self.kinematics.forward
    }

    #[inline]
    pub fn position(&self) -> Vector3<S> {
        self.light.position()
    }

    #[inline]
    pub fn update(&mut self, camera: &PerspectiveFovCamera<S>, _elapsed: S) {
        self.kinematics.update(camera, _elapsed);
        self.light.update_position_world(&self.kinematics.position);
    }

    #[inline]
    pub fn model_matrix(&self) -> Matrix4<S> {
        self.light.model_matrix()
    }
}

