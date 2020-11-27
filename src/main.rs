extern crate rand;

use std::fs::File;
use std::io::prelude::*;
use rand::Rng;


#[derive(Clone, Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 {x, y, z}
    }

    pub fn add(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )
    }

    pub fn addf(&self, value: f32) -> Vec3 {
        Vec3::new(
            self.x + value,
            self.y + value,
            self.z + value,
        )
    }

    pub fn multiply(&self, value: f32) -> Vec3 {
        Vec3::new(
            self.x * value,
            self.y * value,
            self.z * value,
        )
    }

    pub fn substract(&self, other: &Vec3) -> Vec3 {
        self.add(&other.multiply(-1.0))
    }

    pub fn length(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }

    pub fn normalized(&self) -> Vec3 {
        let length = self.length();
        Vec3::new(
            self.x / length,
            self.y / length,
            self.z / length,
        )
    }

    pub fn rotate_z(&self, angle: f32) -> Vec3 {
        Vec3::new(
            self.x * angle.cos() - self.y * angle.sin(),
            self.x * angle.sin() + self.y * angle.cos(),
            self.z
        )
    }

    pub fn rotate_y(&self, angle: f32) -> Vec3 {
        Vec3::new(
            self.x * angle.cos() + self.z * angle.sin(),
            self.y,
            - self.x * angle.sin() + self.z * angle.cos(),
        )
    }

    pub fn dot_product(&self, other: &Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn get_normal(&self) -> Vec3 {
        Vec3::new(
            -self.z,
            // self.x,
            self.y,
            self.x,
        )
    }

    pub fn angle_between(&self, other: &Vec3) -> f32 {
        (
            self.normalized().dot_product(&other.normalized())
        ).acos()
    }

    pub fn cross_product(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
}

struct Sphere {
    position: Vec3,
    color: Vec3,
    radius: f32,
    reflective: f32
}

impl Sphere {
    pub fn new(position: Vec3, color: Vec3, radius: f32, reflective: f32) -> Sphere {
        Sphere {position, color, radius, reflective}
    }

    pub fn intersects(&self, ray: &Ray) -> Option<(f32, Vec3, Vec3)> {
        // https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        let common_part = ray.direction.dot_product(
            &ray.origin.substract(&self.position)
        );
        let delta = common_part.powi(2) - (
            ray.origin
            .substract(&self.position)
            .length().powi(2) - self.radius.powi(2)
        );

        if delta < 0.0 {
            return None
        }
        let delta_sqrt = delta.sqrt();

        let d1 = (- common_part) - delta_sqrt;
        let d2 = (- common_part) + delta_sqrt;
        let distance = d1.min(d2) - 0.001;
        let ddd = distance;
        let point = ray.origin.add(&ray.direction.multiply(distance));
        let normal_vector = point.substract(&self.position).normalized();
        let nnn = normal_vector;

        if distance < 0.0 {
            return None
        }
        return Some((ddd, point, nnn));
        //
        // let initial_distance = ray.origin.add(
        //     &ray.direction
        // ).substract(&self.position).length() - self.radius;
        //
        // let mut step_size = initial_distance / 2.0;
        // let mut cur_dist = step_size;
        //
        // for _ in 0..100 {
        //     let point_to_check = ray.origin.add(&ray.direction.multiply(cur_dist - 0.001));
        //     let distance = point_to_check.substract(&self.position).length() - self.radius;
        //
        //     if distance < 0.0 {
        //         let normal_vector = point_to_check.substract(&self.position).normalized();
        //         return Some((0.0, point_to_check, normal_vector))
        //     } else if distance < 0.0001 {
        //         let normal_vector = point_to_check.substract(&self.position).normalized();
        //         return Some((cur_dist, point_to_check, normal_vector))
        //     } else if distance > 1000.0 {
        //         return None
        //     }
        //
        //     step_size = distance * 0.99;
        //     cur_dist += step_size;
        // }
        //
        // // return Some((ddd, point, nnn));
        // None
    }
}

struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Ray {
        Ray{
            origin,
            direction
        }
    }
}

struct World {
    spheres: Vec<Sphere>,
    light_point: Vec3,
}

impl World {
    pub fn new() -> World {
        let spheres = vec![
            Sphere::new(
                Vec3::new(10.0, 0.5, 0.5),
                Vec3::new(1.0, 0.0, 0.0),
                1.0,
                0.5
            ),
            Sphere::new(
                Vec3::new(10.0, 1.0, 0.5),
                Vec3::new(0.0, 0.5, 0.5),
                1.0,
                0.2
            ),
            Sphere::new(
                Vec3::new(11.5, -1.1, 0.5),
                Vec3::new(1.0, 1.0, 0.0),
                0.7,
                0.2
            ),
            Sphere::new(
                Vec3::new(10.0, 0.0, -200.0),
                Vec3::new(1.0, 1.0, 1.0),
                199.0,
                0.2
            ),
            Sphere::new(
                Vec3::new(15.0, 0.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
                2.5,
                0.8
            ),
            Sphere::new(
                Vec3::new(10.0, -3.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
                1.5,
                0.8
            ),
            Sphere::new(
                Vec3::new(8.0, -2.0, -1.0),
                Vec3::new(0.0, 0.0, 1.0),
                1.0,
                0.0
            ),
            Sphere::new(
                Vec3::new(8.0, 0.0, -1.0),
                Vec3::new(0.0, 0.0, 1.0),
                1.0,
                0.0
            ),
        ];
        World {
            spheres,
            light_point: Vec3::new(5.0, 10.0, 20.0),
        }
    }

    pub fn point_sees_light(&self, point: &Vec3, sphere: &Sphere) -> f32 {
        let mut rng = rand::thread_rng();

        let point_to_sphere = sphere.position.substract(point).normalized();
        let normal1 = point_to_sphere.cross_product(&Vec3::new(1.0, 0.0, 0.0));
        let normal2 = point_to_sphere.cross_product(&normal1);

        let mut sum = 0;
        let mut total = 0;
        for (ring_size, count_in_ring) in [
            (0.0, 1),
            (0.333, 7),
            (0.666, 15),
            (1.0, 30)
        ].iter() {
            for i in 0..*count_in_ring {
                total += 1;
                let multiplier = (i as f32) / (*count_in_ring as f32) * 3.14159 * 2.0;
                let light_point = sphere.position
                    .add(&normal1.multiply((multiplier).cos() * sphere.radius * ring_size))
                    .add(&normal2.multiply((multiplier).sin() * sphere.radius * ring_size));

                let ray_direction = light_point.substract(point).normalized();

                let mut works = true;
                for sphere in self.spheres.iter() {
                    match sphere.intersects(&Ray::new(
                        point.clone(),
                        ray_direction.clone(),
                    )) {
                        Some(_) => {
                            works = false;
                            break;
                        },
                        _ => {
                        }
                    }
                }
                if works {
                    sum += 1;
                }
            }

        }
        sum as f32 / total as f32
    }

    pub fn ray_sees_light(&self, ray: &Ray) -> bool {
        for sphere in self.spheres.iter() {
            match sphere.intersects(ray) {
                Some(_) => {
                    return false
                },
                _ => {}
            }
        }
        return true
    }

    pub fn calc_ray(&self, ray: &Ray, depth: i32) -> Vec3{
        // Find what ray intersects
        let mut closest_sphere : Option<(f32, Vec3, Vec3, &Sphere)> = None;
        for sphere in self.spheres.iter() {
            match sphere.intersects(ray) {
                Some((distance, intersection_point, normal_vec)) => {
                    match &closest_sphere {
                        Some((current_distance, _, _, _)) => {
                            if *current_distance > distance {
                                closest_sphere = Some((distance, intersection_point, normal_vec, sphere))
                            }
                        },
                        _ => {
                            closest_sphere = Some((distance, intersection_point, normal_vec, sphere))
                        }
                    }
                },
                _ => {},
            }
        }

        // Get the color based on match
        match closest_sphere {
            Some((_, new_ray_position, normal_vec, sphere)) => {
                let new_ray_direction = ray.direction.substract(&normal_vec.multiply(
                    ray.direction.dot_product(&normal_vec) * 2.0)
                ).normalized();
                let new_ray = Ray::new(new_ray_position.clone(), new_ray_direction);

                let point_to_light = Ray::new(
                    new_ray_position.clone(),
                    self.light_point.substract(&new_ray_position).normalized()
                );

                let to_light_angle =
                    point_to_light.direction
                    .angle_between(&new_ray.direction);

                let light_multiplier = self.point_sees_light(
                    &new_ray_position,
                    &Sphere::new(
                        self.light_point.clone(),
                        Vec3::new(1.0, 1.0, 1.0),
                        3.0,
                        1.0
                    )
                );

                let diffuse_color = sphere
                    .color.clone()
                    .multiply(1.0 - to_light_angle / 3.14159)
                    .multiply(light_multiplier * 0.8 + 0.2);

                // let diffuse_color = if self.ray_sees_light(&point_to_light) {
                // let diffuse_color = if self.ray_sees_light(&point_to_light) {
                //     sphere.color.clone().multiply(1.0 - to_light_angle / 3.14159)
                // } else {
                //     sphere.color.clone().multiply(0.1)
                // };

                if sphere.reflective > 0.0 && depth > 0 {
                    return self
                        .calc_ray(&new_ray, depth - 1)
                        .multiply(sphere.reflective)
                        .add(&diffuse_color.multiply(1.0 - sphere.reflective))
                }

                return diffuse_color
            },
            _ => {
                return Vec3::new(0.0, 0.0, 0.0)
            }
        }
    }
}

struct Camera {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Camera {
    pub fn new(origin: Vec3, direction: Vec3) -> Camera {
        Camera{
            origin,
            direction
        }
    }

    pub fn see(&self, world: &World) {
        // let x_res = 250;
        // let y_res = 250;
        let x_res = 100 * 20;
        let y_res = 75 * 20;
        // let x_res = 198;
        // let y_res = 198;
        // let x_res = 200;
        // let y_res = 200;

        // println!("P3");
        // println!("{} {}", x_res, y_res);
        // println!("255");

        let data_size = x_res * y_res * 3;
        let file_size = data_size + 54;
        let mut binary_data: Vec<u8> = vec![0; x_res * y_res * 3];

        for y in 0..y_res {
            for x in 0..x_res {
                let x_angle: f32 = -(x as f32 - (x_res as f32 - 1.0) / 2.0) / (x_res as f32 - 1.0);
                let mut z_angle: f32 = (y as f32 - (y_res as f32 - 1.0) / 2.0) / (y_res as f32  - 1.0);
                z_angle *= y_res as f32 / x_res as f32;

                // Angles from -0.5 to 0.5

                let zoom: f32 = 0.5;

                let x_perpendicular = self.direction
                    .cross_product(&Vec3::new(0.0, 0.0, 1.0));
                let z_perpendicular = self.direction
                    .cross_product(&x_perpendicular);

                // println!("{:?}", self.direction);
                // println!("{:?}", x_perpendicular);
                // println!("{:?}", z_perpendicular);
                // println!("{:?}", self.direction.angle_between(&x_perpendicular));
                // println!("{:?}", z_perpendicular.angle_between(&x_perpendicular));

                let new_direction =
                    self.direction
                    .add(&x_perpendicular.multiply(x_angle * zoom))
                    .add(&z_perpendicular.multiply(z_angle * zoom));

                    // self.direction
                    // .rotate_z(x_angle * multiplier)
                    // .rotate_y(y_angle * multiplier);

                let ray = Ray::new(self.origin.clone(), new_direction.normalized());
                let color = world.calc_ray(&ray, 4);

                let i = x;
                let j = y_res - y - 1;

                binary_data[(i + j * x_res) * 3 + 0] = (color.z * 255.0) as u8;
                binary_data[(i + j * x_res) * 3 + 1] = (color.y * 255.0) as u8;
                binary_data[(i + j * x_res) * 3 + 2] = (color.x * 255.0) as u8;

                // println!(
                //     "{} {} {}",
                //     (color.x * 255.0) as u8,
                //     (color.y * 255.0) as u8,
                //     (color.z * 255.0) as u8,
                // )
            }
        }

        match File::create("kuva.bmp") {
            Ok(mut file) => {
                file.write_all(&[
                    b'B', b'M',
                    file_size as u8,
                    (file_size >> 8) as u8,
                    (file_size >> 16) as u8,
                    (file_size >> 24) as u8,
                    0, 0, 0, 0,
                    54, 0, 0, 0,
                ]);
                file.write_all(&[
                    40, 0, 0, 0,
                    x_res as u8,
                    (x_res >> 8) as u8,
                    (x_res >> 16) as u8,
                    (x_res >> 24) as u8,
                    y_res as u8,
                    (y_res >> 8) as u8,
                    (y_res >> 16) as u8,
                    (y_res >> 24) as u8,
                    // 0, 0, 0, 0,
                    // 0, 0, 0, 0,
                    1, 0, 24, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                ]);
                file.write_all(binary_data.as_slice());
            },
            Err(E) => {}
        }
    }
}

fn main() {
    let camera = Camera::new(
        Vec3::new(-10.0, 8.0, 12.0),
        Vec3::new(1.0, -0.4, -0.5).normalized(),
    );

    let world = World::new();

    camera.see(&world)
    // sphere.intersects(ray);
}
