
#include "place_rec/bow_recognizer.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "place_rec/binary_descriptor.hpp"
#include "place_rec/types.hpp"
#include "place_rec/vocabulary_tree.hpp"
namespace pr {

void BowRecognizer::add(const Keyframe& kf) {
  if (kf.id != static_cast<int>(stamps_.size()))
    throw std::logic_error("BowRecognizer::add: ids must be sequential");

  const auto bow = bow_of(kf);
  const int id = index_.add(bow);
  if (id != kf.id) throw std::logic_error("kf.id must be equal to index_.add(bow)");
  stamps_.push_back(kf.stamp);
}

VocabularyTree::BowVector BowRecognizer::bow_of(const Keyframe& kf) const {
  auto packed = pack(kf.descriptors);
  std::vector<const Desc256*> ptrs;
  ptrs.reserve(packed.size());
  for (const auto& d : packed) ptrs.push_back(&d);
  return voc_->bow(ptrs);
}

std::vector<Candidate> BowRecognizer::query(const Keyframe& kf, double max_stamp,
                                            int max_results) const {
  std::vector<Candidate> candidates;
  if (max_results <= 0 || stamps_.empty()) return candidates;
  auto bow = bow_of(kf);
  if (bow.empty()) return candidates;
  std::vector<float> sim_scores;
  // sim_scores.size() always equals stamps_.size(), because every add appends to both
  sim_scores = index_.scores(bow);

  candidates.reserve(sim_scores.size());

  for (std::size_t i = 0; i < stamps_.size(); i++) {
    if (stamps_[i] <= max_stamp && sim_scores[i] > 0.0f) {
      candidates.push_back({static_cast<int>(i), sim_scores[i]});
    }
  }

  const size_t n = std::min(candidates.size(), static_cast<size_t>(max_results));
  std::partial_sort(candidates.begin(), candidates.begin() + n, candidates.end(),
                    [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
  candidates.resize(n);

  return candidates;
}

}  // namespace pr
