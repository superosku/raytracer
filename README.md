
# Ray Tracer

This is my raytracer written in rust

# How To Run

Unoptimized
```
cargo run
```

Optimized version:
```
cargo build --release && time ./target/release/raytracer
```

Output is a bmp file

# Example Render

![alt text](kuva.png)

Now with soft shadows!

![alt text](kuva2.png)

Now with opaque spheres!

![alt text](kuva3.png)
