#pragma once

#include <vector>

struct AABB {
  std::vector<double> min;
  std::vector<double> max;

  AABB(const std::vector<double>& min, const std::vector<double>& max)
      : min(min), max(max) {}

  double largetAxis() const { 
    
    return 0; }

  double largestDim() const {
    const double a = largetAxis();
    return max[a] - min[a];
  }
};

struct BTreeNode {
  int count;
  AABB aabb;
  std::vector<uint32_t> indices;
  BTreeNode* left;
  BTreeNode* right;
};
