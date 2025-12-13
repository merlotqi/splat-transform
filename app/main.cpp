#include <gflags/gflags.h>
#include <splat/splat_version.h>

#include <iostream>


DEFINE_string(translate, "", "Translate splats by (x, y, z). Format: x,y,z");
DEFINE_string(rotate, "", "Rotate splats by Euler angles (x, y, z) in degrees. Format: x,y,z");
DEFINE_double(scale, 1.0, "Uniformly scale splats by factor.");
DEFINE_int32(filter_harmonics, -1, "Remove spherical harmonic bands > n (0|1|2|3).");
DEFINE_bool(filter_nan, false, "Remove Gaussians with NaN or Inf values.");
DEFINE_string(filter_box, "", "Remove Gaussians outside box (min, max corners). Format: x,y,z,X,Y,Z");
DEFINE_int32(lod, -1, "Specify the level of detail, n >= 0.");

DEFINE_bool(quiet, false, "Suppress non-error output.");
DEFINE_bool(overwrite, false, "Overwrite output file if it exists.");
DEFINE_bool(cpu, false, "Use CPU for SOG spherical harmonic compression.");
DEFINE_int32(iterations, 10, "Iterations for SOG SH compression (more=better). Default: 10");
DEFINE_string(viewer_settings, "", "HTML viewer settings JSON file.");
DEFINE_string(lod_select, "", "Comma-separated LOD levels to read from LCC input.");
DEFINE_int32(lod_chunk_count, 512, "Approximate number of Gaussians per LOD chunk in K. Default: 512");
DEFINE_double(lod_chunk_extent, 16.0, "Approximate size of an LOD chunk in world units (m). Default: 16");

int main(int argc, char** argv) {
  gflags::SetVersionString("0.1.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_quiet) {
    std::cout << "Running in quiet mode..." << std::endl;
  }

  return 0;
}
