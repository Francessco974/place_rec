#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "place_rec/euroc_sequence.hpp"

namespace fs = std::filesystem;
using pr::EurocSequence;

namespace {
std::string make_root(const char* name, const std::string& csv) {
  const fs::path root = fs::temp_directory_path() / name;
  fs::remove_all(root);
  fs::create_directories(root / "mav0" / "cam0");
  std::ofstream(root / "mav0" / "cam0" / "data.csv", std::ios::binary) << csv;
  return root.string();
}
}  // namespace

TEST(EurocSequence, ParsesRowsWithCrAndBlankLine) {
  const auto root = make_root("euroc_ok",
                              "#timestamp [ns],filename\r\n"
                              "1403636579763555584,1403636579763555584.png\r\n"
                              "\n"
                              "1403636579813555584,1403636579813555584.png\n");
  EurocSequence s(root);
  ASSERT_EQ(s.size(), 2u);
  const auto k = s.keyframes(0.0);
  ASSERT_EQ(k.size(), 2u);
  EXPECT_NEAR(k[0].stamp, 1403636579.763555584, 1e-6);  // catches stoi overflow
  EXPECT_NEAR(k[1].stamp - k[0].stamp, 0.05, 1e-6);
  EXPECT_EQ(k[0].gray_path, root + "/mav0/cam0/data/1403636579763555584.png");
}

TEST(EurocSequence, MissingFolderThrows) {
  EXPECT_THROW(EurocSequence("/nonexistent_euroc_dir"), std::runtime_error);
}

TEST(EurocSequence, HeaderOnlyThrows) {
  EXPECT_THROW(EurocSequence(make_root("euroc_empty", "#timestamp [ns],filename\n")),
               std::runtime_error);
}

TEST(EurocSequence, UnsortedThrows) {
  EXPECT_THROW(EurocSequence(make_root("euroc_unsorted", "2000000000,b.png\n1000000000,a.png\n")),
               std::runtime_error);
}

TEST(EurocSequence, MalformedLinesThrow) {
  EXPECT_THROW(EurocSequence(make_root("euroc_nocomma", "1000000000\n")), std::runtime_error);
  EXPECT_THROW(EurocSequence(make_root("euroc_badstamp", "abc,a.png\n")), std::runtime_error);
  EXPECT_THROW(EurocSequence(make_root("euroc_junk", "123abc,a.png\n")), std::runtime_error);
}

TEST(EurocSequence, KeyframesBySpacing) {
  // 0.25 s steps (exactly representable): period 0.5 keeps frames 0, 2, 4
  const auto root = make_root(
      "euroc_kf", "0,a.png\n250000000,b.png\n500000000,c.png\n750000000,d.png\n1000000000,e.png\n");
  const auto k = EurocSequence(root).keyframes(0.5);
  ASSERT_EQ(k.size(), 3u);
  EXPECT_NEAR(k[1].stamp, 0.5, 1e-9);
  EXPECT_NEAR(k[2].stamp, 1.0, 1e-9);
}