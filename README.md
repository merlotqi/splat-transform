
# Splat Transform

A C++ library for handling and transforming splat data formats, with support for PLY and SOG formats.

## Features

- **Format Support**: Handles both PLY and SOG splat formats
- **GPU Acceleration**: CUDA-based processing for performance
- **Parallel Processing**: Built-in support for parallel operations
- **Mathematical Operations**: Includes k-means, k-d tree, and b-tree implementations
- **Data Compression**: WebP codec support for efficient storage

## Dependencies

- CUDA (with compute capability 7.5, 8.0, or 8.9)
- Eigen3
- WebP
- nlohmann_json

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Project Structure

```
splat_transform/
├── app/           # Main application entry point
├── include/       # Header files
├── src/          # Source implementations
├── docs/         # Documentation
└── examples/     # Example scripts
```

## Usage Example

Download sample splat files:

```bash
cd examples
./download.sh
```

This will download several sample splat files in parallel to the `splat_downloads` directory.

## Documentation

- [PLY Format](docs/ply.md)
- [SOG Format](docs/sog.md)

## License

This project uses C++17 and CUDA 17 standards. For Windows builds, `_USE_MATH_DEFINES` is automatically defined.

