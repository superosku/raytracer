extern crate rand;

use std::fs::File;
use std::io::prelude::*;
use rayon::prelude::*;
use rand::prelude::*;


#[derive(Clone, Debug)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 {x, y, z}
    }

    pub fn add(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )
    }

    // pub fn addf(&self, value: f64) -> Vec3 {
    //     Vec3::new(
    //         self.x + value,
    //         self.y + value,
    //         self.z + value,
    //     )
    // }

    pub fn multiply(&self, value: f64) -> Vec3 {
        Vec3::new(
            self.x * value,
            self.y * value,
            self.z * value,
        )
    }

    pub fn multiplyv(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
        )
    }

    pub fn substract(&self, other: &Vec3) -> Vec3 {
        self.add(&other.multiply(-1.0))
    }

    pub fn length(&self) -> f64 {
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

    // pub fn rotate_z(&self, angle: f64) -> Vec3 {
    //     Vec3::new(
    //         self.x * angle.cos() - self.y * angle.sin(),
    //         self.x * angle.sin() + self.y * angle.cos(),
    //         self.z
    //     )
    // }

    // pub fn rotate_y(&self, angle: f64) -> Vec3 {
    //     Vec3::new(
    //         self.x * angle.cos() + self.z * angle.sin(),
    //         self.y,
    //         - self.x * angle.sin() + self.z * angle.cos(),
    //     )
    // }

    pub fn dot_product(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    // pub fn get_normal(&self) -> Vec3 {
    //     Vec3::new(
    //         -self.z,
    //         // self.x,
    //         self.y,
    //         self.x,
    //     )
    // }

    pub fn angle_between(&self, other: &Vec3) -> f64 {
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
    radius: f64,
    reflective: f64,
    opaque: bool,
    refractive_ratio: f64,
    emitter: bool,
}

impl Sphere {
    pub fn new(position: Vec3, color: Vec3, radius: f64, reflective: f64) -> Sphere {
        Sphere {
            position,
            color,
            radius,
            reflective,
            opaque: false,
            refractive_ratio: 1.0,
            emitter: false,
        }
    }

    pub fn new_emitter(position: Vec3, color: Vec3, radius: f64) -> Sphere {
        Sphere {
            position,
            color,
            radius,
            reflective: 0.0,
            opaque: false,
            refractive_ratio: 1.0,
            emitter: true,
        }
    }

    pub fn new_opaque(position: Vec3, radius: f64, refractive_ratio: f64, reflective: f64) -> Sphere {
        Sphere {
            position,
            color: Vec3::new(1.0, 1.0, 1.0),
            radius,
            // reflective: 0.0,
            reflective,
            opaque: true,
            refractive_ratio,
            emitter: false,
        }
    }

    pub fn intersects(&self, ray: &Ray) -> Option<(f64, Vec3, Vec3, bool)> {
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
        let distance = d1.min(d2);// - 0.0001;
        let point = ray.origin.add(&ray.direction.multiply(distance));
        let normal_vector = point.substract(&self.position).normalized();

        let is_inside = ray.origin.substract(&self.position).length() < self.radius;

        if distance < 0.0 {
            return None
        }
        return Some((distance, point, normal_vector, is_inside));
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
    // light_point: Vec3,
}

impl World {
    pub fn new() -> World {
        let spheres = vec![
            // Box
            Sphere::new(
                Vec3::new(0.0, 10010.0, 0.0),
                Vec3::new(1.0, 0.5, 0.0),
                10000.0,
                0.0,
            ),
            Sphere::new(
                Vec3::new(0.0, -10000.0, 0.0),
                Vec3::new(0.5, 0.5, 1.0),
                10000.0,
                0.0,
            ),
            Sphere::new(
                Vec3::new(0.0, 0.0, 10010.0),
                Vec3::new(1.0, 1.0, 1.0),
                10000.0,
                0.0,
            ),
            Sphere::new(
                Vec3::new(0.0, 0.0, -10000.0),
                Vec3::new(1.0, 1.0, 1.0),
                10000.0,
                0.0,
            ),
            Sphere::new(
                Vec3::new(10010.0, 0.0, 0.0),
                Vec3::new(0.5, 0.5, 0.5),
                10000.0,
                0.7,
            ),
            Sphere::new(
                Vec3::new(-10000.0, 0.0, 0.0),
                Vec3::new(0.1, 0.1, 0.1),
                10000.0,
                0.0,
            ),

            // Balls
            Sphere::new(
                Vec3::new(5.0, 3.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
                1.0,
                0.9,
            ),
            Sphere::new_opaque(
                Vec3::new(5.0, 7.0, 1.0),
                1.0,
                1.5,
                0.1,
            ),

            // Emitter
            Sphere::new_emitter(
                Vec3::new(5.0, 5.0, 13.0),
                Vec3::new(1.0, 1.0, 1.0),
                4.0,
            )
        ];
        World {
            spheres,
            // light_point: Vec3::new(5.0, 5.0, 7.0),
        }
    }

    // pub fn point_sees_light(&self, point: &Vec3, sphere: &Sphere) -> f64 {
    //     let point_to_sphere = sphere.position.substract(point).normalized();
    //     let point_to_sphere_distance = sphere.position.substract(point).length();
    //     let normal1 = point_to_sphere.cross_product(&Vec3::new(1.0, 0.0, 0.0));
    //     let normal2 = point_to_sphere.cross_product(&normal1);
    //
    //     let mut sum = 0;
    //     let mut total = 0;
    //     for (ring_size, count_in_ring) in [
    //         (0.0, 1),
    //         (0.333, 7),
    //         (0.666, 15),
    //         (1.0, 30)
    //     ].iter() {
    //         for i in 0..*count_in_ring {
    //             total += 1;
    //             let multiplier = (i as f64) / (*count_in_ring as f64) * 3.14159 * 2.0;
    //             let light_point = sphere.position
    //                 .add(&normal1.multiply((multiplier).cos() * sphere.radius * ring_size))
    //                 .add(&normal2.multiply((multiplier).sin() * sphere.radius * ring_size));
    //
    //             let ray_direction = light_point.substract(point).normalized();
    //
    //             let mut works = true;
    //             for sphere in self.spheres.iter() {
    //                 match sphere.intersects(&Ray::new(
    //                     point.clone(),
    //                     ray_direction.clone(),
    //                 )) {
    //                     Some((distance, _, _, _)) => {
    //                         if !sphere.opaque && distance < point_to_sphere_distance {
    //                             works = false;
    //                             break;
    //                         }
    //                     },
    //                     _ => {
    //                     }
    //                 }
    //             }
    //             if works {
    //                 sum += 1;
    //             }
    //         }
    //
    //     }
    //     sum as f64 / total as f64
    // }

    // pub fn ray_sees_light(&self, ray: &Ray) -> bool {
    //     for sphere in self.spheres.iter() {
    //         match sphere.intersects(ray) {
    //             Some(_) => {
    //                 return false
    //             },
    //             _ => {}
    //         }
    //     }
    //     return true
    // }

    pub fn calc_ray(&self, ray: &Ray, depth: i32) -> Vec3{
        if depth == 0 {
            return Vec3::new(0.0, 0.0, 0.0);
        }

        // Find what ray intersects
        let mut closest_sphere : Option<(f64, Vec3, Vec3, &Sphere, bool)> = None;
        for sphere in self.spheres.iter() {
            match sphere.intersects(ray) {
                Some((distance, intersection_point, normal_vec, is_inside)) => {
                    match &closest_sphere {
                        Some((current_distance, _, _, _, _)) => {
                            if *current_distance > distance {
                                closest_sphere = Some((
                                    distance,
                                    intersection_point,
                                    normal_vec,
                                    sphere,
                                    is_inside
                                ))
                            }
                        },
                        _ => {
                            closest_sphere = Some((
                                distance,
                                intersection_point,
                                normal_vec,
                                sphere,
                                is_inside
                            ))
                        }
                    }
                },
                _ => {},
            }
        }

        let mut rng = rand::thread_rng();

        // Get the color based on match
        match closest_sphere {
            Some((_, new_ray_exact_position, circle_normal_vec, sphere, is_inside)) => {
                if sphere.emitter {
                    return sphere.color.clone()
                }

                let normal_vec =
                    if is_inside {circle_normal_vec.multiply(-1.0)}
                    else {circle_normal_vec};
                let new_ray_position = new_ray_exact_position
                    .add(&normal_vec.multiply(0.001));

                if sphere.reflective >= 0.0 && rng.gen::<f64>() < sphere.reflective {
                    let new_ray_direction = ray.direction.substract(&normal_vec.multiply(
                        ray.direction.dot_product(&normal_vec) * 2.0)
                    ).normalized();
                    let new_ray = Ray::new(new_ray_position.clone(), new_ray_direction);

                    return self
                        .calc_ray(&new_ray, depth - 1)
                        // .multiply(sphere.reflective)
                        // .add(&diffuse_color.multiply(1.0 - sphere.reflective))
                }

                if sphere.opaque {
                    let new_ray_position = new_ray_exact_position
                        .add(&normal_vec.multiply(-0.001)); // Must be negative so we can hop through the sphere
                    let refractive_ratio =
                        if is_inside {sphere.refractive_ratio}
                        else {1.0/sphere.refractive_ratio};
                    let c = ray.direction.dot_product(&normal_vec.multiply(-1.0));
                    let new_ray_direction =
                        ray.direction.multiply(refractive_ratio)
                        .add(
                            &normal_vec.multiply(
                                refractive_ratio * c
                                -(
                                    1.0 -
                                    refractive_ratio.powi(2) *
                                    (1.0 - c.powi(2))
                                ).sqrt()
                            )
                        );
                    let new_ray = Ray::new(new_ray_position.clone(), new_ray_direction);

                    return self.calc_ray(&new_ray, depth - 1);
                }

                let mut new_hemisphere_vector = Vec3::new(
                    rng.gen::<f64>() - 0.5,
                    rng.gen::<f64>() - 0.5,
                    rng.gen::<f64>() - 0.5,
                );
                while new_hemisphere_vector.length() > 1.0 {
                    new_hemisphere_vector.x = rng.gen::<f64>();
                    new_hemisphere_vector.y = rng.gen::<f64>();
                    new_hemisphere_vector.z = rng.gen::<f64>();
                }

                if new_hemisphere_vector.angle_between(&normal_vec) > 3.14159 / 2.0 {
                    new_hemisphere_vector.x = -new_hemisphere_vector.x;
                    new_hemisphere_vector.y = -new_hemisphere_vector.y;
                    new_hemisphere_vector.z = -new_hemisphere_vector.z;
                }

                new_hemisphere_vector = new_hemisphere_vector.normalized();

                let new_ray = Ray::new(
                    new_ray_position.clone(),
                    new_hemisphere_vector
                );

                let rec_color = self.calc_ray(&new_ray, depth - 1);

                return sphere.color.multiplyv(&rec_color);

                return Vec3::new(0.0, 0.0, 0.0);

                //
                //
                //
                //
                //
                // let point_to_light = Ray::new(
                //     new_ray_position.clone(),
                //     self.light_point.substract(&new_ray_position).normalized()
                // );
                //
                // let to_light_angle =
                //     point_to_light.direction
                //     .angle_between(&new_ray.direction);
                //
                // let light_multiplier = self.point_sees_light(
                //     &new_ray_position,
                //     &Sphere::new(
                //         self.light_point.clone(),
                //         Vec3::new(1.0, 1.0, 1.0),
                //         1.0,
                //         1.0
                //     )
                // );
                //
                // let diffuse_color = sphere
                //     .color.clone()
                //     .multiply(1.0 - to_light_angle / 3.14159)
                //     .multiply(light_multiplier * 0.8 + 0.2);
                //
                // if sphere.reflective > 0.0 && depth > 0 {
                //     return self
                //         .calc_ray(&new_ray, depth - 1)
                //         .multiply(sphere.reflective)
                //         .add(&diffuse_color.multiply(1.0 - sphere.reflective))
                // }
                //
                // return diffuse_color
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
        let x_res = 100 * 20;
        let y_res = 75 * 20;

        let data_size = x_res * y_res * 3;
        let file_size = data_size + 54;
        let mut collected_binary_data: Vec<u8> = vec![0; x_res * y_res * 3];

        let mut xy_pairs : Vec<(usize, usize)> = Vec::new();
        for y in 0..y_res {
            for x in 0..x_res {
                xy_pairs.push((x, y))
            }
        }

        // 0m10.433s

        const THREAD_COUNT: i64 = 16;
        const PER_PIXEL_STEPS: i64 = 50;
        let zoom: f64 = 2.5;

        let vec: Vec<i64> = (0..THREAD_COUNT).collect();
        let all_binary_datas: Vec<Vec<u8>> = vec.par_iter().map(|thread_index| {
            let mut binary_data: Vec<u8> = vec![0; x_res * y_res * 3];

            let thread_len = xy_pairs.len() / THREAD_COUNT as usize;

            for pixel_index in 0..thread_len {
                let (x, y) = xy_pairs[pixel_index * THREAD_COUNT as usize + *thread_index as usize];

                let x_angle: f64 = -(x as f64 - (x_res as f64 - 1.0) / 2.0) / (x_res as f64 - 1.0);
                let mut z_angle: f64 = (y as f64 - (y_res as f64 - 1.0) / 2.0) / (y_res as f64  - 1.0);
                z_angle *= y_res as f64 / x_res as f64;
                // x_angle goes from -0.5 to 0.5

                let x_perpendicular = self.direction
                    .cross_product(&Vec3::new(0.0, 0.0, 1.0));
                let z_perpendicular = self.direction
                    .cross_product(&x_perpendicular);

                let new_direction =
                    self.direction
                    .add(&x_perpendicular.multiply(x_angle * zoom))
                    .add(&z_perpendicular.multiply(z_angle * zoom));

                let ray = Ray::new(self.origin.clone(), new_direction.normalized());

                let mut color = Vec3::new(0.0, 0.0, 0.0);
                for _ in 0..PER_PIXEL_STEPS {
                    let step_color = world.calc_ray(&ray, 20);
                    color = color.add(&step_color);
                }
                color = color.multiply(7.5 * 1.0 / PER_PIXEL_STEPS as f64);

                let i = x;
                let j = y_res - y - 1;

                binary_data[(i + j * x_res) * 3 + 0] = (color.z * 255.0) as u8;
                binary_data[(i + j * x_res) * 3 + 1] = (color.y * 255.0) as u8;
                binary_data[(i + j * x_res) * 3 + 2] = (color.x * 255.0) as u8;
            }
            return binary_data
        }).collect();

        for (x, y) in xy_pairs.iter() {
            for binary_data in all_binary_datas.iter() {
                let index = (x + y * x_res) * 3;
                collected_binary_data[index + 0] = binary_data[index + 0].max(collected_binary_data[index + 0]);
                collected_binary_data[index + 1] = binary_data[index + 1].max(collected_binary_data[index + 1]);
                collected_binary_data[index + 2] = binary_data[index + 2].max(collected_binary_data[index + 2]);
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
                ]).unwrap();
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
                    1, 0, 24, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                ]).unwrap();
                file.write_all(collected_binary_data.as_slice()).unwrap();
            },
            Err(_) => {}
        }
    }
}

fn main() {
    let camera = Camera::new(
        Vec3::new(1.0, 5.0, 1.5),
        Vec3::new(1.0, 0.0, -0.0),
        // Vec3::new(1.0, 9.0, 9.0),
        // Vec3::new(1.0, -1.0, -1.0),
        // Vec3::new(1.0, -0.33, -0.3).normalized(),
    );

    let world = World::new();

    camera.see(&world)
    // sphere.intersects(ray);
}
