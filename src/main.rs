extern crate rand;

use std::fs::File;
use std::io::prelude::*;
use rayon::prelude::*;
use rand::prelude::*;


const THREAD_COUNT: i64 = 16;
const PER_PIXEL_STEPS: i64 = 100;


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

    pub fn new_random() -> Vec3 {
        let mut rng = rand::thread_rng();

        let mut new = Vec3::new(
            (rng.gen::<f64>() - 0.5) * 2.0,
            (rng.gen::<f64>() - 0.5) * 2.0,
            (rng.gen::<f64>() - 0.5) * 2.0,
        );
        while new.length() > 1.0 {
            new.x = (rng.gen::<f64>() - 0.5) * 2.0;
            new.y = (rng.gen::<f64>() - 0.5) * 2.0;
            new.z = (rng.gen::<f64>() - 0.5) * 2.0;
        }

        new
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
    emitter: Sphere,
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
                0.0,
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
                Vec3::new(5.0, 5.0, 9.0),
                Vec3::new(1.0, 1.0, 1.0),
                1.0,
            )
        ];
        World {
            spheres,
            emitter: Sphere::new_emitter(
                Vec3::new(5.0, 5.0, 8.0),
                Vec3::new(1.0, 1.0, 1.0),
                1.0,
            )
        }
    }

    pub fn fid_intersection(&self, ray: &Ray, ignore_emitters: bool) -> Option<(f64, Vec3, Vec3, &Sphere, bool)> {
        let mut closest_sphere : Option<(f64, Vec3, Vec3, &Sphere, bool)> = None;
        for sphere in self.spheres.iter() {
            if ignore_emitters && sphere.emitter {
                continue
            }
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

        return closest_sphere
    }

    pub fn sees(&self, point1: &Vec3, point2: &Vec3) -> bool {
        let dist_between = point1.substract(&point2).length();
        match self.fid_intersection(&Ray::new(
            point1.clone(),
            point2.substract(&point1).normalized()
        ), true) {
            Some((distance, _, _, sphere, _)) => {
                // println!("FOUND INTERSECTION {} {}", distance, dist_between);
                // println!("FOUND INTERSECTION {:?} {}", sphere.position, sphere.radius);
                // println!("fff {} {}", distance, dist_between);
                if distance > dist_between {
                    return true
                }
                if sphere.emitter {
                    return true
                }
                return false
            },
            _ => {
                return true
            }
        }
    }

    pub fn calc_ray(&self, ray: &Ray, depth: i32, ignore_emitters: bool) -> Option<(Vec3, Vec3, bool)> {
        if depth == 0 {
            return Some((
                ray.origin.clone(),
                Vec3::new(1.0, 1.0, 1.0),
                false
            ))
            // return Vec3::new(0.0, 0.0, 0.0);
        }

        // Find what ray intersects
        let mut closest_sphere = self.fid_intersection(ray, ignore_emitters);

        let mut rng = rand::thread_rng();

        // Get the color based on match
        match closest_sphere {
            Some((_, new_ray_exact_position, circle_normal_vec, sphere, is_inside)) => {

                let normal_vec =
                    if is_inside {circle_normal_vec.multiply(-1.0)}
                    else {circle_normal_vec};
                let new_ray_position = new_ray_exact_position
                    .add(&normal_vec.multiply(0.001));

                if sphere.emitter {
                    return Some((
                        new_ray_position,
                        // ray.origin.clone(),
                        sphere.color.clone(),
                        true
                    ))
                }

                if sphere.reflective >= 0.0 && rng.gen::<f64>() < sphere.reflective {
                    let new_ray_direction = ray.direction.substract(&normal_vec.multiply(
                        ray.direction.dot_product(&normal_vec) * 2.0)
                    ).normalized();
                    let new_ray = Ray::new(new_ray_position.clone(), new_ray_direction);

                    return self.calc_ray(&new_ray, depth, ignore_emitters)
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

                    return self.calc_ray(&new_ray, depth, ignore_emitters);
                }

                let mut new_hemisphere_vector = Vec3::new_random().normalized();

                if new_hemisphere_vector.angle_between(&normal_vec) > 3.14159 / 2.0 {
                    new_hemisphere_vector.x = -new_hemisphere_vector.x;
                    new_hemisphere_vector.y = -new_hemisphere_vector.y;
                    new_hemisphere_vector.z = -new_hemisphere_vector.z;
                }

                let new_ray = Ray::new(
                    new_ray_position.clone(),
                    new_hemisphere_vector
                );

                match self.calc_ray(&new_ray, depth - 1, ignore_emitters) {
                    Some((position, color, hit_light)) => {
                        return Some((
                            position,
                            // sphere.color.clone()
                            sphere.color.multiplyv(&color),
                            hit_light
                        ));
                    },
                    _ => {}
                }
                return None
                // let rec_color = self.calc_ray(&new_ray, depth - 1);
                //
                // return sphere.color.multiplyv(&rec_color);
                //
                // return Vec3::new(0.0, 0.0, 0.0);
            },
            _ => {
                return None;
                // return Vec3::new(0.0, 0.0, 0.0)
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

    pub fn see(&self, world: &World, x_res: usize, y_res: usize, a: i32, b: i32) -> Vec<f64> {
        let mut collected_colors: Vec<f64> = vec![0.0; x_res * y_res * 3];

        let mut xy_pairs : Vec<(usize, usize)> = Vec::new();
        for y in 0..y_res {
            for x in 0..x_res {
                xy_pairs.push((x, y))
            }
        }

        let zoom: f64 = 2.7;

        let vec: Vec<i64> = (0..THREAD_COUNT).collect();

        let all_color_datas: Vec<Vec<f64>> = vec.par_iter().map(|thread_index| {
            // let mut total_counter = 0;
            // let mut hit_counter = 0;

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
                    // let choices = vec![
                    //     (1, 0),
                    //     (1, 1),
                    //     (1, 2),
                    //     (1, 3),
                    //     (2, 0),
                    //     (2, 1),
                    //     (2, 2),
                    //     (2, 3),
                    //     (3, 0),
                    //     (3, 1),
                    //     (3, 2),
                    //     (3, 3),
                    // ];
                    //
                    // for (a, b) in choices.iter() {
                        let mut step_color = Vec3::new(0.0, 0.0, 0.0);
                        match world.calc_ray(&ray, a, false) {
                            Some((ray_position, color1, hit_light)) => {
                                if hit_light {
                                    step_color = color1;
                                } else {
                                    let random_emitter_pos = Vec3::new_random()
                                        .multiply(world.emitter.radius)
                                        .add(&world.emitter.position);
                                    let random_emitter_dir = Vec3::new_random().normalized();

                                    let emitter_ray = Ray::new(
                                        random_emitter_pos.clone(),
                                        random_emitter_dir,
                                    );

                                    // step_color = color1;

                                    match world.calc_ray(&emitter_ray, b, true) {
                                        Some((light_position, color2, _)) => {
                                            let sees = world.sees(&ray_position, &light_position);
                                            // if sees {
                                            //     println!("asdf {} {:?} {:?}", sees, ray_position, light_position);
                                            // }

                                            // total_counter += 1;
                                            if sees {
                                                // hit_counter += 1;
                                                step_color = color1.multiplyv(&color2);
                                            } else {
                                                step_color = Vec3::new(0.0, 0.0, 0.0);
                                            }
                                        },
                                        _ => {
                                            step_color = Vec3::new(0.0, 0.0, 0.0);
                                        }
                                    }
                                }
                            },
                            _ => {
                                // step_color = color1;
                                // println!("BUU");
                                step_color = Vec3::new(1.0, 0.0, 1.0);
                            }
                        }
                        color = color.add(&step_color);
                    }
                // }

                color = color.multiply(0.125 * 7.5 * 1.0 / PER_PIXEL_STEPS as f64);

                let i = x;
                let j = y_res - y - 1;

                color_data[(i + j * x_res) * 3 + 0] = color.z;
                color_data[(i + j * x_res) * 3 + 1] = color.y;
                color_data[(i + j * x_res) * 3 + 2] = color.x;
            }

            // println!("counters {} {} {}", hit_counter, total_counter, hit_counter as f64 / total_counter as f64);

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
        Vec3::new(0.1, 5.0, 1.5),
        Vec3::new(1.0, 0.0, 0.2),
    );

    let world = World::new();

    // let temp_sees = world.sees(
    //     &Vec3::new(1.0, 1.0, 1.0),
    //     &Vec3::new(9.0, 9.0, 9.0),
    // );
    //
    // println!("TEST {}", temp_sees);
    //
    // return;

    // let x_res = 100 * 4;
    // let y_res = 75 * 4;
    // let x_res = 100 * 5;
    // let y_res = 75 * 5;
    let x_res = 100 * 20;
    let y_res = 75 * 20;
    let data_size = x_res * y_res * 3;
    let file_size = data_size + 54;

    println!("data_size {}", data_size);
    println!("file_size {}", file_size);


    let choices = vec![
        // (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        // (3, 3),
        (4, 0),
        (4, 1),
        // (4, 2),
        // (4, 3),
        // (5, 0),
        // (5, 1),
        // (5, 2),
        // (5, 3),
    ];

    let mut all_all_colors: Vec<Vec<f64>> = Vec::new();
    for (a, b) in choices.iter() {
        let mut all_colors: Vec<Vec<f64>> = Vec::new();
        let mut loop_index = 0;

        while true {
            loop_index += 1;

            let colors = camera.see(&world, x_res, y_res, *a, *b);
            all_colors.push(colors.clone());
            all_all_colors.push(colors);

            let mut binary_data: Vec<u8> = vec![0; data_size];
            for i in 0..data_size {
                let mut pixel_colo_sum = 0.0;
                for colors in all_colors.iter() {
                    pixel_colo_sum += colors[i]
                }
                binary_data[i] = (pixel_colo_sum * 255.0 / all_colors.len() as f64) as u8;
            }

            let file_name = format!("outputs/pict-{}-{}-{}.bmp", a, b, loop_index * PER_PIXEL_STEPS);

            println!("Writing {}", file_name);
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

            break;
        }
    }

    let mut binary_data: Vec<u8> = vec![0; data_size];
    for i in 0..data_size {
        let mut pixel_colo_sum = 0.0;
        for colors in all_all_colors.iter() {
            pixel_colo_sum += colors[i]
        }
        binary_data[i] = (2.0 * pixel_colo_sum * 255.0 / all_all_colors.len() as f64) as u8;
    }

    let file_name = format!("outputs/pict-final-{}.bmp", PER_PIXEL_STEPS);

    println!("Writing {}", file_name);
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
