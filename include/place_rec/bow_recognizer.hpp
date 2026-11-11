
#pragma once
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "place_rec/place_recognizer.hpp"
#include "place_rec/types.hpp"
#include "place_rec/vocabulary_tree.hpp"
namespace pr {

using shr_cst_voc = std::shared_ptr<const VocabularyTree>;
class BowRecognizer : public PlaceRecognizer {
 public:
  explicit BowRecognizer(shr_cst_voc voc)
      : voc_(check_voc(std::move(voc))), index_(voc_->num_words()) {}

  void add(const Keyframe& kf) override;

  std::vector<Candidate> query(const Keyframe& Kf, double max_stamp,
                               int max_results) const override;

 private:
  static shr_cst_voc check_voc(shr_cst_voc voc) {
    if (!voc) throw std::invalid_argument("BowRecognizer: vocabulary is null");
    return voc;
  }

  VocabularyTree::BowVector bow_of(const Keyframe& kf) const;
  shr_cst_voc voc_;
  // voc_ must be declared before since it needs voc->num_words()
  InvertedIndex index_;
  std::vector<double> stamps_;
};
}  // namespace pr