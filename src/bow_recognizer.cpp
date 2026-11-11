
#include "place_rec/bow_recognizer.hpp"

#include <stdexcept>
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

}  // namespace pr