#include "place_rec/vocabulary_tree.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "place_rec/binary_descriptor.hpp"

namespace pr {
namespace {
constexpr std::int32_t kMaxNodes = 20'000'000;
constexpr std::int32_t kMaxK = 4096;  // sanity cap on branching factor read from a file
static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(Desc256) == 32,
              "file format assumes 4-byte int/float and 32-byte descriptors");
}  // namespace

auto VocabularyTree::kmeans(const std::vector<const Desc256*>& x, int k, int max_iters,
                            std::mt19937& rng) -> KmeansOutput {
  KmeansOutput results;
  if (k <= 0) throw std::invalid_argument("k must be greater than 0");

  int point_num = static_cast<int>(x.size());
  if (!point_num) return results;

  if (point_num <= k) {
    results.centroids.reserve(point_num);
    results.assign.reserve(point_num);

    for (int i = 0; i < point_num; ++i) {
      results.centroids.push_back(*x[i]);
      results.assign.push_back(i);
    }

    return results;
  }

  // k-means++ seeding
  std::vector<int> dist(point_num);
  std::vector<double> weights(point_num);

  int idx = std::uniform_int_distribution<int>(0, point_num - 1)(rng);
  results.centroids.push_back(*x[idx]);  // store the first cluster

  for (int i = 0; i < point_num; i++) {
    dist[i] = hamming(*x[i], *x[idx]);
  }

  while (results.centroids.size() < static_cast<std::size_t>(k)) {
    double sum = 0;
    for (int i = 0; i < point_num; i++) {
      weights[i] = static_cast<double>(dist[i]) * static_cast<double>(dist[i]);
      sum += weights[i];
    }
    if (sum <= 1e-6) break;

    idx = std::discrete_distribution<int>(weights.begin(), weights.end())(rng);
    results.centroids.push_back(*x[idx]);

    for (int i = 0; i < point_num; i++) {
      dist[i] = std::min(dist[i], hamming(*x[i], results.centroids.back()));
    }
  }

  // Lloyd loop
  results.assign.assign(point_num, -1);
  for (int iter = 0; iter < max_iters; iter++) {
    const int c = static_cast<int>(results.centroids.size());  // may be < k
    bool changed = false;

    // assign: nearest centroid, ties go to the lower index
    for (int i = 0; i < point_num; i++) {
      int best = -1, best_dist = 257;  // 257 > any Hamming distance
      for (int j = 0; j < c; j++) {
        const int d = hamming(*x[i], results.centroids[j]);
        if (d < best_dist) {
          best_dist = d;
          best = j;
        }
      }
      if (best != results.assign[i]) {
        results.assign[i] = best;
        changed = true;
      }
    }
    if (!changed) break;

    // update: group in one pass, then majority vote
    std::vector<std::vector<const Desc256*>> members(c);
    for (int i = 0; i < point_num; i++) members[results.assign[i]].push_back(x[i]);
    for (int j = 0; j < c; j++)
      if (!members[j].empty()) results.centroids[j] = majority(members[j]);

    // compact: remap table, in-place copy, then rewrite assign
    std::vector<int> remap(c, -1);
    int next = 0;
    for (int j = 0; j < c; j++) {
      if (members[j].empty()) continue;
      remap[j] = next;
      results.centroids[next++] = results.centroids[j];
    }
    results.centroids.resize(next);
    for (int& a : results.assign) a = remap[a];
  }
  return results;
}

void VocabularyTree::build(int node, const std::vector<const Desc256*>& x, int depth,
                           std::mt19937& rng) {
  auto make_leaf = [&] {
    tree_[node].word = static_cast<int>(word_node_.size());
    word_node_.push_back(node);
  };

  // Base cases: max depth reached, or too few points to split meaningfully.
  if (depth == params_.L || x.size() <= 1) {
    make_leaf();
    return;
  }

  auto out = kmeans(x, params_.k, params_.max_iters, rng);

  // All points identical (or one cluster survived): a split adds nothing.
  if (out.centroids.size() <= 1) {
    make_leaf();
    return;
  }

  std::vector<std::vector<const Desc256*>> subsets(out.centroids.size());
  for (size_t i = 0; i < x.size(); ++i) subsets[out.assign[i]].push_back(x[i]);

  for (size_t i = 0; i < out.centroids.size(); i++) {
    const int id = static_cast<int>(tree_.size());
    // Create child node
    Node child{};
    child.centroid = out.centroids[i];
    // add to tree
    tree_.push_back(child);
    tree_[node].children.push_back(id);
    build(id, subsets[i], depth + 1, rng);
  }
}

void VocabularyTree::train(const std::vector<const Desc256*>& x, const Params& params,
                           const std::vector<int>& imgs_ids) {
  /*imgs_ids[i] is the image of x[i]; ids are >= 0; descriptors of one image are contiguous*/
  // Initialize
  tree_.clear();
  word_node_.clear();
  idf_.clear();

  // Set params
  params_ = params;
  // Create root
  Node root;
  tree_.push_back(root);
  if (!(x.size() == imgs_ids.size()))
    throw std::invalid_argument("x and imgs_ids size must be equal");
  if (x.empty()) return;

  if (*std::min_element(imgs_ids.begin(), imgs_ids.end()) < 0)
    throw std::invalid_argument("imgs_ids must be >= 0");

  if (!std::is_sorted(imgs_ids.begin(), imgs_ids.end()))
    throw std::invalid_argument("imgs_ids must be non-decreasing (contiguous per image)");

  int num_images = 1;
  int curr_img = imgs_ids[0];
  // Start building
  std::mt19937 rng(params_.seed);
  build(0, x, 0, rng);

  std::vector<int> doc_freq(word_node_.size(), 0);
  std::vector<int> last_img(num_words(), -1);
  idf_.assign(word_node_.size(), 0.0);
  for (std::size_t i = 0; i < x.size(); i++) {
    if (curr_img != imgs_ids[i]) {
      curr_img = imgs_ids[i];
      num_images++;
    }
    auto word = transform(*x[i]);
    if (word == -1) continue;
    if (imgs_ids[i] == last_img[word]) continue;  // already seen in the image

    doc_freq[word] += 1;
    last_img[word] = imgs_ids[i];
  }
  for (std::size_t c = 0; c < word_node_.size(); c++) {
    if (doc_freq[c] > 0)
      idf_[c] = std::log(static_cast<float>(num_images) / static_cast<float>(doc_freq[c]));
  }
}

int VocabularyTree::transform(const Desc256& descriptor) const {
  if (tree_.empty()) return -1;
  int curr = 0;
  while (!tree_[curr].children.empty()) {
    int best = tree_[curr].children[0];  // initial guess
    int d = hamming(descriptor, tree_[best].centroid);

    for (int child : tree_[curr].children) {
      int d_cmp = hamming(descriptor, tree_[child].centroid);
      if (d > d_cmp) {
        d = d_cmp;
        best = child;  // Node id
      }
    }
    curr = best;
  }
  return tree_[curr].word;
}

VocabularyTree::BowVector VocabularyTree::bow(const std::vector<const Desc256*>& descs) const {
  BowVector out{};
  if (descs.empty()) return out;
  if (tree_.empty()) return out;
  if (idf_.empty()) return out;

  std::map<int, float> acc;
  for (const auto* desc : descs) {
    int word = transform(*desc);
    if (word >= 0) {
      acc[word] += 1.0f;
    }
  }
  if (acc.empty()) return out;
  out.reserve(acc.size());
  float sum{0.0f};
  for (const auto& [word, freq] : acc) {
    float weight = idf_[word] * freq;
    if (weight <= 0.0f) continue;
    sum += weight;
    out.emplace_back(word, weight);
  }
  if (out.empty()) return out;

  for (auto& [word, weight] : out) {
    weight /= sum;
  }
  return out;
}

namespace {

template <typename T>
void write_pod(std::ostream& out, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>, "write_pod needs a trivially copyable type");
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
void write_vct(std::ostream& out, const std::vector<T>& v) {
  static_assert(std::is_trivially_copyable_v<T>, "write_vct needs a trivially copyable type");
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(T)));
}

}  // namespace

void VocabularyTree::save(const std::string& path) const {
  //   field	type
  // magic	8 bytes VOCTREE1
  // version	uint32
  // k, L, max_iters	int32
  // seed	uint32
  // num_nodes	int32
  // per node: centroid (32 B), word int32, n_children uint32, children int32[n]
  // num_words	int32
  // word_node_	int32[num_words]
  // idf_	float[num_words]
  if (tree_.empty()) throw std::logic_error("VocabularyTree::save: tree is not trained");
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("VocabularyTree::save: cannot open " + path);
  out.write("VOCTREE1", 8);
  const std::uint32_t version = 1;
  // header
  write_pod(out, (version));
  write_pod(out, static_cast<std::int32_t>(params_.k));
  write_pod(out, static_cast<std::int32_t>(params_.L));
  write_pod(out, static_cast<std::int32_t>(params_.max_iters));
  write_pod(out, static_cast<std::uint32_t>(params_.seed));
  // tree
  // nodes
  write_pod(out, static_cast<std::int32_t>(num_nodes()));
  for (const auto& n : tree_) {
    write_pod(out, n.centroid);
    write_pod(out, static_cast<std::int32_t>(n.word));
    write_pod(out, static_cast<std::uint32_t>(n.children.size()));
    write_vct(out, n.children);
  }
  // words
  write_pod(out, static_cast<std::int32_t>(num_words()));
  write_vct(out, word_node_);
  write_vct(out, idf_);

  if (!out) throw std::runtime_error("VocabularyTree::save has some errors " + path);
}

namespace {

template <typename T>
void read_pod(std::istream& in, T& value) {
  static_assert(std::is_trivially_copyable_v<T>);
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) throw std::runtime_error("VocabularyTree::load: unexpected end of file");
}

template <typename T>
void read_vct(std::istream& in, std::vector<T>& v) {
  static_assert(std::is_trivially_copyable_v<T>, "read_vct needs a trivially copyable type");
  in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
  if (!in) throw std::runtime_error("VocabularyTree::load: unexpected end of file");
}

}  // namespace

VocabularyTree VocabularyTree::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);

  if (!in) throw std::runtime_error("VocabularyTree::load: cannot open " + path);

  // header
  char magic[8];
  in.read(magic, 8);
  if (in.gcount() != 8 || std::string_view(magic, 8) != "VOCTREE1") {
    throw std::runtime_error("VocabularyTree::load: invalid file format (magic) in " + path);
  }

  std::uint32_t version = 0;
  read_pod(in, version);

  if (version != 1) {
    throw std::runtime_error("VocabularyTree::load: unsupported version " +
                             std::to_string(version) + " (expected 1)");
  }

  // Start loading the tree
  VocabularyTree t;

  std::int32_t k = 0, L = 0, max_iters = 0;
  std::uint32_t seed = 0;

  read_pod(in, k);
  read_pod(in, L);
  read_pod(in, max_iters);
  read_pod(in, seed);

  if (k < 1 || k > kMaxK) throw std::runtime_error("VocabularyTree::load: bad k");
  if (L < 0) throw std::runtime_error("VocabularyTree::load: bad L");
  t.params_ = {k, L, max_iters, seed};

  // nodes
  std::int32_t num_nodes = 0;
  read_pod(in, num_nodes);
  if (num_nodes < 1 || num_nodes > kMaxNodes) throw std::runtime_error("bad node count");

  t.tree_.resize(num_nodes);
  for (auto& n : t.tree_) {
    read_pod(in, n.centroid);
    read_pod(in, n.word);

    std::uint32_t n_children = 0;
    read_pod(in, n_children);
    if (n_children > static_cast<std::uint32_t>(k))
      throw std::runtime_error("VocabularyTree::load: bad child count");

    n.children.resize(n_children);
    read_vct(in, n.children);
  }

  // words
  std::int32_t num_words = 0;
  read_pod(in, num_words);
  if (num_words < 0 || num_words > num_nodes)
    throw std::runtime_error("VocabularyTree::load: bad word count");

  t.word_node_.resize(num_words);
  t.idf_.resize(num_words);
  read_vct(in, t.word_node_);
  read_vct(in, t.idf_);

  // validation: corrupted file?
  for (int p = 0; p < num_nodes; ++p) {
    const Node& n = t.tree_[p];
    for (int c : n.children)
      if (c <= p || c >= num_nodes)  // c > p: indices strictly grow, so transform terminates
        throw std::runtime_error("VocabularyTree::load: bad child index");
    if (n.word < -1 || n.word >= num_words)
      throw std::runtime_error("VocabularyTree::load: bad word id");
    if (n.word >= 0 && (!n.children.empty() || t.word_node_[n.word] != p))
      throw std::runtime_error("VocabularyTree::load: inconsistent leaf");
    if (n.children.empty() && n.word < 0 && num_nodes > 1)
      throw std::runtime_error("VocabularyTree::load: leaf without word");
  }
  for (int w = 0; w < num_words; ++w) {
    const int node = t.word_node_[w];
    if (node < 0 || node >= num_nodes || t.tree_[node].word != w)
      throw std::runtime_error("VocabularyTree::load: bad word_node_ entry");
    if (!std::isfinite(t.idf_[w]) || t.idf_[w] < 0.0f)
      throw std::runtime_error("VocabularyTree::load: bad idf value");
  }

  return t;
}

}  // namespace pr
