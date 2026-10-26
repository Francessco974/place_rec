// run_eval (v1, tiny backend only)
// usage: run_eval --seq <tum_dir> --out <dir> [--period 1.0] [--exclude 20] [--topk 10]
// Writes keyframes.csv, candidates.csv, loops.csv (header only), timing.csv into <dir> (must exist).
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "place_rec/tiny_image_recognizer.hpp"
#include "place_rec/tum_sequence.hpp"

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> arg{
      {"--period", "1.0"}, {"--exclude", "20"}, {"--topk", "10"}};
  for (int i = 1; i + 1 < argc; i += 2) arg[argv[i]] = argv[i + 1];
  if (!arg.count("--seq") || !arg.count("--out")) {
    std::cerr << "usage: run_eval --seq <tum_dir> --out <dir> [options]\n";
    return 1;
  }
  const double period = std::stod(arg["--period"]);
  const double exclude = std::stod(arg["--exclude"]);
  const int topk = std::stoi(arg["--topk"]);

  const std::string out = arg["--out"];
  std::ofstream f_kf(out + "/keyframes.csv"), f_cand(out + "/candidates.csv"),
      f_loop(out + "/loops.csv"), f_time(out + "/timing.csv");
  if (!f_kf || !f_cand || !f_loop || !f_time) {
    std::cerr << "cannot write into " << out << " (does the folder exist?)\n";
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

  std::unique_ptr<pr::PlaceRecognizer> rec = std::make_unique<pr::TinyImageRecognizer>();

  for (size_t i = 0; i < frames.size(); ++i) {
    const auto t0 = Clock::now();
    pr::Keyframe kf;
    kf.id = static_cast<int>(i);
    kf.stamp = frames[i].stamp;
    kf.gray = pr::TumSequence::load_gray(frames[i].rgb_path);
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
  std::cout << "processed " << frames.size() << " keyframes -> " << out << "\n";
  return 0;
}