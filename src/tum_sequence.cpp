#include "place_rec/tum_sequence.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

namespace pr{

namespace {

struct Entry {
    double stamp;
    std::string file;

};

std::vector<Entry> read_list(const std::string& path){
    std::ifstream f(path);
    if (!f) throw std::runtime_error("TumSequence: cannot open " + path);
    std::vector<Entry> out;
    std::string line;
    while (std::getline(f , line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Entry e;
        if (ss >> e.stamp >> e.file) out.push_back(e);
    }
    return out;

    }
} // namespace

TumSequence::TumSequence(const std::string& root, double max_dt) {
    const auto rgb = read_list(root + "/rgb.txt");
    const auto depth = read_list(root + "/depth.txt");

    for (const auto& r : rgb) {
        auto it = std::lower_bound(depth.begin(), depth.end(), r.stamp,
                    [](const Entry& e, double t) {return e.stamp < t;});

        const Entry* best = nullptr;
        if (it != depth.end()) best = &*it;
        if (it != depth.begin() && 
            (!best || std::abs((it-1) -> stamp - r.stamp) < std::abs(best->stamp - r.stamp)))  best = &*(it -1);
        if (best && std::abs(best -> stamp - r.stamp) <= max_dt) frames_.push_back({r.stamp, root + "/" + r.file, root + "/" + best->file});
    }
}


std::vector<TumFrame> TumSequence::keyframes(double period) const {
    std::vector<TumFrame> out;
    for (const auto& f : frames_)
        if (out.empty() || f.stamp - out.back().stamp >= period) out.push_back(f);
    return out;
}

cv::Mat TumSequence::load_gray(const std::string& path){
    cv::Mat g = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (g.empty()) throw std::runtime_error("cannot read "+ path);
    return g;
}

cv::Mat TumSequence::load_depth_m(const std::string& path){
    cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (raw.empty() || raw.type() != CV_16U) throw std::runtime_error("bad depth "+ path);
    cv::Mat m;
    raw.convertTo(m, CV_32F, 1.0 / 5000.0); // from 5000 units = 1 meter
    return m;
}


} //namespace pr



