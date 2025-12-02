#pragma once

#include <string>
#include <vector>

struct Options {
  bool overwrite;
  bool help;
  bool version;
  bool quiet;
  int iterations;
  bool listGpus;
  int device;

  std::vector<double> lodSelect;

  std::string viewerSettingPath;
  bool unbundled;

  int lodChunkCount;
  int lodChunkExtent;

  Options()
      : overwrite(false),
        help(false),
        version(false),
        quiet(false),
        iterations(1),
        listGpus(false),
        device(-1),
        lodChunkCount(1),
        lodChunkExtent(1) {}
};

using Param = std::pair<std::string, std::string>;
