#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

namespace pr {
inline cv::Mat load_gray(const std::string& path) {
  cv::Mat g = cv::imread(path, cv::IMREAD_GRAYSCALE);
  if (g.empty()) throw std::runtime_error("cannot read " + path);
  return g;
}
}  // namespace pr