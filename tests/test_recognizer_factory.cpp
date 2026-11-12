#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>

#include "place_rec/recognizer_factory.hpp"
#include "place_rec/vocabulary_tree.hpp"
#include "test_helpers.hpp"

using namespace pr::test;

namespace {
std::shared_ptr<const pr::VocabularyTree> small_vocab() {
  const auto a1 = zeros(), a2 = with_low8(zeros(), true);
  const auto b1 = ones(), b2 = with_low8(ones(), false);
  const std::vector<pr::Desc256> data = {a1, a1, a2, a2, a1, b1, b2, b2};
  auto t = std::make_shared<pr::VocabularyTree>();
  t->train(ptrs(data), {2, 2, 25, 42}, {0, 0, 0, 0, 1, 1, 2, 2});
  return t;
}

pr::Keyframe make_kf(int id, double stamp, const std::vector<pr::Desc256>& d) {
  pr::Keyframe kf;
  kf.id = id;
  kf.stamp = stamp;
  kf.descriptors = cv::Mat(static_cast<int>(d.size()), 32, CV_8U);
  for (size_t i = 0; i < d.size(); ++i)
    std::memcpy(kf.descriptors.ptr<uchar>(static_cast<int>(i)), d[i].data(), 32);
  return kf;
}
}  // namespace

TEST(RecognizerFactory, ParseBackend) {
  EXPECT_EQ(pr::parse_backend("tiny"), pr::Backend::Tiny);
  EXPECT_EQ(pr::parse_backend("bow"), pr::Backend::Bow);
}

TEST(RecognizerFactory, UnknownNameThrowsAndListsValidNames) {
  try {
    pr::parse_backend("netvlad");
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("tiny"), std::string::npos);
    EXPECT_NE(msg.find("bow"), std::string::npos);
  }
}

TEST(RecognizerFactory, BowWithoutVocabularyThrows) {
  EXPECT_THROW(pr::make_recognizer(pr::Backend::Bow, {}), std::invalid_argument);
}

TEST(RecognizerFactory, TinyIgnoresOptions) {
  EXPECT_NE(pr::make_recognizer(pr::Backend::Tiny, {}), nullptr);
}

TEST(RecognizerFactory, BowWorksThroughBasePointer) {
  pr::RecognizerOptions opt;
  opt.vocabulary = small_vocab();
  std::unique_ptr<pr::PlaceRecognizer> rec = pr::make_recognizer(pr::Backend::Bow, opt);
  ASSERT_NE(rec, nullptr);

  const std::vector<pr::Desc256> d = {zeros(), with_low8(zeros(), true), ones()};
  rec->add(make_kf(0, 1.0, d));
  const auto r = rec->query(make_kf(1, 5.0, d), 2.0, 3);
  ASSERT_EQ(r.size(), 1u);
  EXPECT_EQ(r[0].id, 0);
  EXPECT_NEAR(r[0].score, 1.0, 1e-5);
}