// run_eval (v2): runs one place-recognition backend on a TUM sequence.
// usage: run_eval --seq <tum_dir> --out <root> [--backend tiny|bow] [--vocab <file>]
//                 [--period 1.0] [--exclude 20] [--topk 10]
// Creates <root>/<backend>_<timestamp>/ with keyframes.csv, candidates.csv, loops.csv (header
// only), timing.csv and run.json. The last line printed is the run folder.
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "app_utils.hpp"
#include "place_rec/feature_extractor.hpp"
#include "place_rec/recognizer_factory.hpp"
#include "place_rec/tum_sequence.hpp"
#include "place_rec/vocabulary_tree.hpp"

#ifndef PLACE_REC_GIT_HASH
#define PLACE_REC_GIT_HASH "unknown"
#endif
#ifndef PLACE_REC_BUILD_TYPE
#define PLACE_REC_BUILD_TYPE "unknown"
#endif

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Rewrites the whole file each time: called at the start (finished=false) and at the end (true).
void write_run_json(const std::filesystem::path& path, bool finished, const std::string& started,
                    const std::string& backend, const std::map<std::string, std::string>& arg,
                    const pr::OrbParams& orb) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path.string());
  f << "{\n";
  f << "  \"backend\": \"" << apps::json_escape(backend) << "\",\n";
  f << "  \"started\": \"" << started << "\",\n";
  f << "  \"finished\": " << (finished ? "true" : "false") << ",\n";
  f << "  \"git_hash\": \"" << apps::json_escape(PLACE_REC_GIT_HASH) << "\",\n";
  f << "  \"build_type\": \"" << apps::json_escape(PLACE_REC_BUILD_TYPE) << "\",\n";
  f << "  \"orb\": {\"n_features\": " << orb.n_features
    << ", \"scale_factor\": " << orb.scale_factor << ", \"n_levels\": " << orb.n_levels
    << ", \"fast_threshold\": " << orb.fast_threshold << "},\n";
  f << "  \"args\": {";
  bool first = true;
  for (const auto& [k, v] : arg) {
    f << (first ? "" : ", ") << '"' << apps::json_escape(k) << "\": \"" << apps::json_escape(v)
      << '"';
    first = false;
  }
  f << "}\n}\n";
  if (!f) throw std::runtime_error("write failed for " + path.string());
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const std::set<std::string> known{"--period", "--exclude", "--topk", "--backend",
                                      "--vocab",  "--seq",     "--out"};
    std::map<std::string, std::string> arg{
        {"--period", "1.0"}, {"--exclude", "20"}, {"--topk", "10"}, {"--backend", "tiny"}};

    for (int i = 1; i < argc; i += 2) {
      const std::string key = argv[i];
      if (!known.count(key)) throw std::invalid_argument("unknown option " + key);
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
      arg[argv[i]] = argv[i + 1];
    }
    if (!arg.count("--seq") || !arg.count("--out")) {
      std::cerr << "usage: run_eval --seq <tum_dir> --out <root> [options]\n";
      return 1;
    }

    pr::Backend backend = pr::parse_backend(arg["--backend"]);

    if (backend == pr::Backend::Bow && !arg.count("--vocab")) {
      std::cerr << "usage: run_eval --backend bow needs --vocab <file> " << '\n';
      return 1;
    }
    const double period = std::stod(arg["--period"]);
    const double exclude = std::stod(arg["--exclude"]);
    const int topk = std::stoi(arg["--topk"]);

    const pr::OrbParams orb{};
    const std::string started = apps::now_iso8601();
    const std::filesystem::path run_dir =
        std::filesystem::path(arg["--out"]) / (arg["--backend"] + "_" + apps::now_filename());
    std::filesystem::create_directories(run_dir);
    write_run_json(run_dir / "run.json", false, started, arg["--backend"], arg, orb);

    pr::RecognizerOptions rec_opt{};
    if (arg.count("--vocab")) {
      auto tree = pr::VocabularyTree::load(arg["--vocab"]);
      rec_opt.vocabulary = std::make_shared<const pr::VocabularyTree>(std::move(tree));
    }

    std::unique_ptr<pr::PlaceRecognizer> rec = pr::make_recognizer(backend, rec_opt);
    const std::string out = run_dir.string();
    std::ofstream f_kf(out + "/keyframes.csv"), f_cand(out + "/candidates.csv"),
        f_loop(out + "/loops.csv"), f_time(out + "/timing.csv");
    if (!f_kf || !f_cand || !f_loop || !f_time) {
      std::cerr << "cannot write into " << out << ' ' << '\n';
      return 1;
    }
    f_kf << "id,stamp\n";
    f_cand << "query,rank,cand,score\n";
    f_loop << "query,cand,score,matches,inliers,tx,ty,tz,qx,qy,qz,qw\n";  // rows from v3
    f_time << "query,extract_ms,query_ms,verify_ms\n";
    f_kf.precision(6);
    f_kf << std::fixed;  // stamps like 1305031910.765238 need 6 decimals

    pr::TumSequence seq(arg["--seq"]);
    const auto frames = seq.keyframes(period);
    std::cout << "keyframes: " << frames.size() << "\n";

    pr::FeatureExtractor extractor{orb};

    for (size_t i = 0; i < frames.size(); ++i) {
      const auto t0 = Clock::now();
      pr::Keyframe kf;
      kf.id = static_cast<int>(i);
      kf.stamp = frames[i].stamp;
      kf.gray = pr::TumSequence::load_gray(frames[i].rgb_path);

      extractor.extract(kf);
      const auto t1 = Clock::now();

      // Query BEFORE add: a keyframe must never match itself.
      const auto cands = rec->query(kf, kf.stamp - exclude, topk);
      const auto t2 = Clock::now();
      for (size_t r = 0; r < cands.size(); ++r)
        f_cand << i << ',' << r << ',' << cands[r].id << ',' << cands[r].score << '\n';

      rec->add(kf);
      f_kf << i << ',' << kf.stamp << '\n';
      f_time << i << ',' << ms(t0, t1) << ',' << ms(t1, t2) << ",0\n";
    }
    write_run_json(run_dir / "run.json", true, started, arg["--backend"], arg, orb);
    std::cout << "processed " << frames.size() << " keyframes\n";
    std::cout << out << "\n";  // last line: the run folder, for scripts
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "run_eval: " << e.what() << "\n";
    return 1;
  }
}