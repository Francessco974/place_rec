#pragma once
#include <memory>
#include <string>

#include "place_rec/place_recognizer.hpp"
#include "place_rec/vocabulary_tree.hpp"
namespace pr {
enum class Backend { Tiny, Bow };
struct RecognizerOptions {
  std::shared_ptr<const VocabularyTree> vocabulary;
};
Backend parse_backend(const std::string& name);
std::unique_ptr<PlaceRecognizer> make_recognizer(Backend backend, const RecognizerOptions& options);

}  // namespace pr