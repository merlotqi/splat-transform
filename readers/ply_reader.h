#pragma once

#include <string>
#include <vector>

#include "data_table.h"

struct PlyProperty {
  std::string name;  // 'x', 'f_dc_0', etc
  std::string type;  // 'float' 'char', etc
};

struct PlyElement {
  std::string name;  // 'vertex', 'face', etc
  size_t count;      // number of items in this element
  std::vector<PlyProperty> properties;
};

struct PlyHeader {
  std::vector<std::string> comments;
  std::vector<PlyElement> elements;
};

struct PlyData {
  std::vector<std::string> comments;
  std::vector<std::pair<std::string, DataTable>> elements;
};
