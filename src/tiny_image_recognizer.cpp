#include "place_rec/tiny_image_recognizer.hpp"
#include "place_rec/types.hpp"
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <vector>

namespace  pr {

cv::Mat TinyImageRecognizer::describe(const cv::Mat& gray){
    cv::Mat small, f;
    cv::resize(gray, small, cv::Size(32,34), 0, 0, cv::INTER_AREA);
    small.convertTo(f, CV_32F);
    f = f.reshape(1,1);
    f-=cv::mean(f)[0];
    const double n = cv::norm(f);
    if (n > 1e-6) f/=n;
    return f;
}

std::vector<Candidate> TinyImageRecognizer::query(const Keyframe& kf, double max_stamp, int max_results) const{
    const cv::Mat q = describe(kf.gray);
    std::vector<Candidate> out;
    for (size_t i = 0; i < descs_.size(); i++)
        if (stamps_[i] <=max_stamp) out.push_back({static_cast<int>(i), q.dot(descs_[i])});
    const size_t n = std::min(out.size(), static_cast<size_t>(max_results));
    std::partial_sort(out.begin(), out.begin() + n, out.end(), 
                    [](const Candidate& a, const Candidate&b) {return a.score > b.score; });
    out.resize(n);
    return out;
}

void TinyImageRecognizer::add(const Keyframe& kf){
    if (kf.id != static_cast<int>(descs_.size()))
        throw std::logic_error("TinyImageRecognizer::add: ids must be sequential");
    descs_.push_back(describe(kf.gray));
    stamps_.push_back(kf.stamp);
}

} //namespace pr