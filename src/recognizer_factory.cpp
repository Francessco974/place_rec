#include "place_rec/recognizer_factory.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#include "place_rec/bow_recognizer.hpp"
#include "place_rec/place_recognizer.hpp"
#include "place_rec/tiny_image_recognizer.hpp"

namespace pr {
Backend parse_backend(const std::string& name) {
  if (name == "tiny") {
    return Backend::Tiny;
  }
  if (name == "bow") {
    return Backend::Bow;
  }

  // Throw an exception listing the valid names if no match is found
  throw std::invalid_argument("Invalid backend: '" + name +
                              "'. Valid names are: \"tiny\", \"bow\".");
}

std::unique_ptr<PlaceRecognizer> make_recognizer(Backend backend,
                                                 const RecognizerOptions& options) {
  switch (backend) {
    case Backend::Tiny:
      return std::make_unique<TinyImageRecognizer>();

    case Backend::Bow:
      if (!options.vocabulary)
        throw std::invalid_argument("Bow backend needs a non-null vocabulary");
      return std::make_unique<BowRecognizer>(options.vocabulary);
  }
  throw std::invalid_argument("Unknown backend");
}

}  // namespace pr