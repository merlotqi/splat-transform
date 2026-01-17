
# SplatLib

A C++ library for reading, writing, and processing 3D Gaussian Splatting (splat) files used in real-time neural rendering.

SplatLib provides comprehensive support for various Gaussian splat formats, enabling efficient conversion, manipulation, and optimization of 3D scene data for applications like web-based real-time rendering and neural graphics.

## Features

- **Multiple Format Support**:
  - **PLY**: Industry-standard uncompressed format for training, editing, and archival storage
  - **SOG**: Compressed format optimized for web delivery (15-20× smaller than PLY)
  - **KSPLAT, SPZ, LCC**: Additional specialized formats for different use cases

- **GPU Acceleration**: CUDA-based processing for high-performance operations including SOG compression and spatial algorithms

- **Spatial Data Structures**:
  - Octree for hierarchical spatial partitioning
  - K-D tree for nearest neighbor searches
  - B-tree for efficient data organization
  - Morton order encoding for spatial coherence

- **Mathematical Operations**:
  - K-means clustering (CPU and GPU implementations)
  - Spherical harmonic (SH) rotation and manipulation
  - Coordinate transformations and data processing

- **Level of Detail (LOD)**: Support for multi-resolution Gaussian representations with chunk-based organization

- **Data Compression**: WebP codec integration for efficient texture compression

- **Parallel Processing**: Built-in thread pools and parallel algorithms for multi-core performance

- **Python Bindings**: pybind11-based Python interface for integration with Python workflows

- **Command-line Tool**: `SplatTransform` utility for format conversion, filtering, and processing

## Dependencies

### Required
- **CUDA** (compute capability 7.5, 8.0, or 8.9) - for GPU acceleration
- **Eigen3** - linear algebra library
- **WebP** - image compression library
- **nlohmann_json** - JSON parsing
- **Abseil** (absl) - C++ utilities
- **ZLIB** - compression library

### Optional
- **Doxygen** - for generating API documentation
- **pybind11** - for Python bindings (if enabled)

## Building

```bash
# Clone the repository
git clone https://github.com/merlotqi/SplitLib.git
cd SplitLib

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DBUILD_SPLAT_TRANSFORM_TOOL=ON -DBUILD_PYTHON_BINDINGS=OFF

# Build
make -j$(nproc)
```

### CMake Options
- `BUILD_SPLAT_TRANSFORM_TOOL`: Build the command-line transform utility (default: OFF)
- `BUILD_PYTHON_BINDINGS`: Build Python bindings (default: OFF)
- `ENABLE_CLANG_TIDY`: Enable clang-tidy static analysis (default: OFF)

### Platform-specific Notes
- **Linux**: Uses pkg-config for dependency detection
- **Windows**: Uses find_package for CUDA and other dependencies
- C++17 and CUDA 17 standards are required
- On Windows, `_USE_MATH_DEFINES` is automatically defined

## Project Structure

```
SplitLib/
├── include/splat/          # Public header files
│   ├── splat.h            # Main library header
│   ├── io/                # File I/O modules
│   ├── maths/             # Mathematical utilities
│   ├── models/            # Data models (PLY, SOG, etc.)
│   ├── op/                # Operations (combine, transform)
│   ├── spatial/           # Spatial data structures
│   └── utils/             # Utilities (compression, logging)
├── src/                   # Source implementations
├── python/                # Python bindings
├── transform/             # Command-line tool
├── docs/                  # Documentation
├── examples/              # Example scripts and data
├── thirdparty/            # External dependencies
└── cmake/                 # CMake utilities
```

## Usage

### Command-line Tool

The `SplatTransform` tool provides command-line interface for format conversion and processing:

```bash
# Convert PLY to SOG with GPU acceleration
SplatTransform input.ply output.sog --gpu 0 --iterations 10

# Process multiple files with LOD
SplatTransform input1.ply input2.ply output.sog --lod 0.5

# List available GPUs
SplatTransform --list-gpus
```

### Library Usage (C++)

```cpp
#include <splat/splat.h>

// Read a PLY file
auto data = splat::readPly("scene.ply");

// Process the data (e.g., apply transformations)
auto transformed = splat::transform(data, transformation_matrix);

// Write to compressed SOG format
splat::writeSog("scene.sog", transformed);
```

### Python Bindings

```python
import splat_transform_cpp

# Example usage (bindings under development)
result = splat_transform_cpp.add(1, 2)
print(result)  # 3
```

## Examples

Download sample Gaussian splat datasets:

```bash
cd examples
./download.sh
```

This downloads several sample splat files in parallel for testing and development.

## Documentation

- [PLY Format Specification](docs/ply.md) - Industry standard for Gaussian splats
- [SOG Format Specification](docs/sog.md) - Compressed format for web delivery
- [API Documentation](docs/index.md) - Library overview and format comparison

Generate Doxygen documentation:

```bash
cd build
make doc
# Open docs/html/index.html
```

## Version

Current version: **1.2.0**

See [ChangeLog](ChangeLog) for detailed version history and updates.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

See [LICENSE](LICENSE) and [COPYRIGHT](COPYRIGHT) files for details.

## Contributing

Contributions are welcome! Please ensure code follows the established patterns and includes appropriate tests.

## Related Projects

- [PlayCanvas](https://playcanvas.com/) - Web-based real-time rendering engine that uses SOG format
- [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Original research implementation
