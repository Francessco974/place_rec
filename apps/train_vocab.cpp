// train_vocab: build a BoW vocabulary tree from EuRoC cam0 images.
// usage: train_vocab --seq <root1,root2,...> --out <vocab.bin>
//                    [--stride-seconds 0.5] [--k 10] [--L 5]
// Writes <vocab.bin> and a JSON sidecar <vocab.json> next to it.
#include <chrono>
#include <cstddef>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "place_rec/binary_descriptor.hpp"
#include "place_rec/euroc_sequence.hpp"
#include "place_rec/feature_extractor.hpp"
#include "place_rec/types.hpp"
#include "place_rec/vocabulary_tree.hpp"

#ifndef PLACE_REC_GIT_HASH
#define PLACE_REC_GIT_HASH "unknown"
#endif

namespace {
using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::istringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep))
    if (!item.empty()) out.push_back(item);
  return out;
}

// Escape the two characters that break a JSON string: backslash and double quote.
std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '\\' || c == '"') out += '\\';
    out += c;
  }
  return out;
}

std::string now_iso8601() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  std::ostringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
  return ss.str();
}

struct TrainStats {
  std::vector<std::string> sequences;
  double stride_seconds = 0.0;
  std::size_t frames_total = 0;
  std::size_t images_used = 0;
  std::size_t images_skipped = 0;
  std::size_t descriptors = 0;
  double extract_seconds = 0.0;
  double train_seconds = 0.0;
};

void write_sidecar(const std::string& json_path, const std::string& vocab_path,
                   const pr::VocabularyTree::Params& p, const pr::OrbParams& orb,
                   const TrainStats& s, const pr::VocabularyTree& tree) {
  std::ofstream f(json_path);
  if (!f) throw std::runtime_error("train_vocab: cannot write " + json_path);
  f << std::setprecision(6);
  f << "{\n";
  f << "  \"vocab_file\": \"" << json_escape(std::filesystem::path(vocab_path).filename().string())
    << "\",\n";
  f << "  \"created\": \"" << now_iso8601() << "\",\n";
  f << "  \"git_hash\": \"" << json_escape(PLACE_REC_GIT_HASH) << "\",\n";
  f << "  \"format_version\": 1,\n";
  f << "  \"k\": " << p.k << ", \"L\": " << p.L << ", \"max_iters\": " << p.max_iters
    << ", \"seed\": " << p.seed << ",\n";
  f << "  \"orb\": {\"n_features\": " << orb.n_features
    << ", \"scale_factor\": " << orb.scale_factor << ", \"n_levels\": " << orb.n_levels
    << ", \"fast_threshold\": " << orb.fast_threshold << "},\n";
  f << "  \"training\": {\n";
  f << "    \"dataset\": \"EuRoC cam0\",\n";
  f << "    \"sequences\": [";
  for (std::size_t i = 0; i < s.sequences.size(); ++i)
    f << (i ? ", " : "") << '"' << json_escape(s.sequences[i]) << '"';
  f << "],\n";
  f << "    \"stride_seconds\": " << s.stride_seconds << ",\n";
  f << "    \"frames_total\": " << s.frames_total << ",\n";
  f << "    \"images_used\": " << s.images_used << ",\n";
  f << "    \"images_skipped_no_descriptors\": " << s.images_skipped << ",\n";
  f << "    \"descriptors\": " << s.descriptors << "\n";
  f << "  },\n";
  f << "  \"result\": {\"num_nodes\": " << tree.num_nodes()
    << ", \"num_words\": " << tree.num_words() << ", \"extract_seconds\": " << s.extract_seconds
    << ", \"train_seconds\": " << s.train_seconds << "}\n";
  f << "}\n";
  if (!f) throw std::runtime_error("train_vocab: write failed for " + json_path);
}

void usage() {
  std::cerr << "usage: train_vocab --seq <root1,root2,...> --out <vocab.bin>\n"
               "                   [--stride-seconds 0.5] [--k 10] [--L 5]\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const std::set<std::string> known{"--seq", "--out", "--stride-seconds", "--k", "--L"};
    std::map<std::string, std::string> arg{
        {"--stride-seconds", "0.5"}, {"--k", "10"}, {"--L", "5"}};
    for (int i = 1; i < argc; i += 2) {
      const std::string key = argv[i];
      if (!known.count(key)) throw std::invalid_argument("unknown option " + key);
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
      arg[key] = argv[i + 1];
    }
    if (!arg.count("--seq") || !arg.count("--out")) {
      usage();
      return 1;
    }

    const std::vector<std::string> sequences = split(arg["--seq"], ',');
    if (sequences.empty()) {
      std::cerr << "train_vocab: no sequence given in --seq\n";
      usage();
      return 1;
    }

    TrainStats stats;
    stats.sequences = sequences;
    stats.stride_seconds = std::stod(arg["--stride-seconds"]);

    pr::VocabularyTree::Params p;  // defaults, then overrides
    p.k = std::stoi(arg["--k"]);
    p.L = std::stoi(arg["--L"]);

    const std::string out = arg["--out"];
    const std::string json_path = std::filesystem::path(out).replace_extension(".json").string();

    const pr::OrbParams orb{};
    const pr::FeatureExtractor extractor{orb};

    std::vector<pr::Desc256> descs;
    std::vector<int> imgs_ids;
    int img_id = 0;  // continues across sequences

    const auto te0 = Clock::now();
    for (const auto& root : sequences) {
      const pr::EurocSequence seq(root);
      const auto frames = seq.keyframes(stats.stride_seconds);
      const std::size_t descs_before = descs.size();
      std::size_t skipped_here = 0;

      for (std::size_t i = 0; i < frames.size(); ++i) {
        pr::Keyframe kf{};
        kf.id = img_id;
        kf.stamp = frames[i].stamp;
        kf.gray = pr::EurocSequence::load_gray(frames[i].gray_path);
        extractor.extract(kf);

        const auto d = pr::pack(kf.descriptors);
        if (d.empty()) {
          ++skipped_here;
          continue;
        }
        descs.insert(descs.end(), d.begin(), d.end());
        imgs_ids.insert(imgs_ids.end(), d.size(), img_id);
        ++img_id;

        if ((i + 1) % 100 == 0)
          std::cout << "  " << (i + 1) << " / " << frames.size() << " frames\n";
      }

      stats.frames_total += frames.size();
      stats.images_skipped += skipped_here;
      std::cout << root << ": " << frames.size() << " frames, " << skipped_here << " skipped, "
                << (descs.size() - descs_before) << " descriptors\n";
    }
    stats.extract_seconds = seconds(te0, Clock::now());
    stats.images_used = static_cast<std::size_t>(img_id);
    stats.descriptors = descs.size();

    if (descs.empty()) {
      std::cerr << "train_vocab: no descriptors extracted, nothing to train\n";
      return 1;
    }

    std::cout << "images used: " << stats.images_used << " (skipped " << stats.images_skipped
              << " of " << stats.frames_total << ")\n"
              << "descriptors: " << stats.descriptors << "\n"
              << "extraction: " << stats.extract_seconds << " s\n"
              << "training k=" << p.k << " L=" << p.L << " ...\n";

    // Pointers only after descs is complete: earlier pointers would dangle on reallocation.
    std::vector<const pr::Desc256*> x;
    x.reserve(descs.size());
    for (const auto& d : descs) x.push_back(&d);

    pr::VocabularyTree tree;
    const auto tt0 = Clock::now();
    tree.train(x, p, imgs_ids);
    stats.train_seconds = seconds(tt0, Clock::now());

    std::cout << "training: " << stats.train_seconds << " s\n"
              << "words: " << tree.num_words() << ", nodes: " << tree.num_nodes() << "\n";

    tree.save(out);
    write_sidecar(json_path, out, p, orb, stats, tree);
    std::cout << "saved " << out << " and " << json_path << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "train_vocab: " << e.what() << "\n";
    return 1;
  }
}