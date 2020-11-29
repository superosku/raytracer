
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

# Example Render (5000 rays per pixel)

![alt text](pic-5000.png)

# Old versions (single ray per pixel)

First render was at command line

![alt text](old2.png)

First color render had just the shape

![alt text](old3.png)

Added some shadows

![alt text](old1.png)

The perspective was wrong at the beginning

![alt text](old4.png)

First well working version

![alt text](kuva.png)

Added soft shadows

![alt text](kuva2.png)

And opaque spheres

![alt text](kuva3.png)
