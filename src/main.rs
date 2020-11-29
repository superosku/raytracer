extern crate rand;

use std::fs::File;
use std::io::prelude::*;
use rayon::prelude::*;
use rand::prelude::*;


const THREAD_COUNT: i64 = 16;
const PER_PIXEL_STEPS: i64 = 1000;


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

    pub fn dot_product(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

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
        }
    }

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

    pub fn see(&self, world: &World, x_res: usize, y_res: usize) -> Vec<f64> {
        let mut collected_colors: Vec<f64> = vec![0.0; x_res * y_res * 3];

        let mut xy_pairs : Vec<(usize, usize)> = Vec::new();
        for y in 0..y_res {
            for x in 0..x_res {
                xy_pairs.push((x, y))
            }
        }

        let zoom: f64 = 2.5;

        let vec: Vec<i64> = (0..THREAD_COUNT).collect();
        let all_color_datas: Vec<Vec<f64>> = vec.par_iter().map(|thread_index| {
            let mut color_data: Vec<f64> = vec![0.0; x_res * y_res * 3];

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

                color_data[(i + j * x_res) * 3 + 0] = color.z;
                color_data[(i + j * x_res) * 3 + 1] = color.y;
                color_data[(i + j * x_res) * 3 + 2] = color.x;
            }
            return color_data;
        }).collect();

        for (x, y) in xy_pairs.iter() {
            for color_data in all_color_datas.iter() {
                let index = (x + y * x_res) * 3;
                collected_colors[index + 0] = color_data[index + 0].max(collected_colors[index + 0]);
                collected_colors[index + 1] = color_data[index + 1].max(collected_colors[index + 1]);
                collected_colors[index + 2] = color_data[index + 2].max(collected_colors[index + 2]);
            }
        }

        collected_colors

    }
}

fn main() {
    let camera = Camera::new(
        Vec3::new(1.0, 5.0, 1.5),
        Vec3::new(1.0, 0.0, -0.0),
    );

    let world = World::new();

    let x_res = 100 * 20;
    let y_res = 75 * 20;

    let mut all_colors: Vec<Vec<f64>> = Vec::new();
    let mut loop_index = 0;

    while true {
        loop_index += 1;

        let colors =  camera.see(&world, x_res, y_res);
        all_colors.push(colors);

        let x_res: usize = 100 * 20;
        let y_res: usize = 75 * 20;

        let data_size = x_res * y_res * 3;
        let file_size = data_size + 54;

        let mut binary_data: Vec<u8> = vec![0; data_size];
        for i in 0..data_size {
            let mut pixel_colo_sum = 0.0;
            for colors in all_colors.iter() {
                pixel_colo_sum += colors[i]
            }
            binary_data[i] = (pixel_colo_sum * 255.0 / all_colors.len() as f64) as u8;
        }

        let file_name = format!("pic-{}.bmp", loop_index * PER_PIXEL_STEPS);
        match File::create(file_name) {
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
                file.write_all(binary_data.as_slice()).unwrap();
            },
            Err(_) => {}
        }
    }
}
