
#include "place_rec/euroc_sequence.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "place_rec/image_io.hpp"

namespace pr {

namespace {

struct Entry {
  double stamp;
  std::string file;
};

std::vector<Entry> read_list(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("EurocSequence: cannot open " + path);
  std::vector<Entry> out;
  std::string line;
  while (std::getline(f, line)) {
    // removing wdws line ending
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    Entry e{};
    std::string s_stamp;
    std::getline(ss, s_stamp, ',');
    if (!std::getline(ss, e.file) || e.file.empty())
      throw std::runtime_error("EurocSequence: corrupted line in " + path + ": " + line);
    try {
      std::size_t pos = 0;
      const std::int64_t ns = std::stoll(s_stamp, &pos);
      if (pos != s_stamp.size()) throw std::invalid_argument("trailing characters");
      e.stamp = static_cast<double>(ns) / 1e9;
    } catch (const std::exception&) {
      throw std::runtime_error("EurocSequence: bad timestamp in " + path + ": " + line);
    }

    out.push_back(e);
  }
  return out;
}
}  // namespace

EurocSequence::EurocSequence(const std::string& root) {
  const std::string csv = root + "/mav0/cam0/data.csv";
  const auto list = read_list(csv);
  if (list.empty()) throw std::runtime_error("EurocSequence: no frames in " + csv);
  if (!std::is_sorted(list.begin(), list.end(),
                      [](const Entry& e1, const Entry& e2) { return e1.stamp < e2.stamp; }))
    throw std::runtime_error("EurocSequence: timestamps not ascending in " + csv);

  frames_.reserve(list.size());

  for (const auto& entry : list) {
    frames_.push_back({entry.stamp, root + "/mav0/cam0/data/" + entry.file});
  }
}

std::vector<EurocFrame> EurocSequence::keyframes(double period) const {
  std::vector<EurocFrame> out;
  for (const auto& f : frames_)
    // floating-point rounding at 20 Hz
    if (out.empty() || f.stamp - out.back().stamp >= period - 1e-6) out.push_back(f);
  return out;
}

cv::Mat EurocSequence::load_gray(const std::string& path) { return pr::load_gray(path); }

}  // namespace pr
