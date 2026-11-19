#pragma once
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
namespace apps {

inline std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::istringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep))
    if (!item.empty()) out.push_back(item);
  return out;
}

// Escape the two characters that break a JSON string: backslash and double quote.
inline std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '\\' || c == '"') out += '\\';
    out += c;
  }
  return out;
}

inline std::string now_iso8601() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  std::ostringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
  return ss.str();
}

inline std::string now_filename() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  std::ostringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
  return ss.str();
}

}  // namespace apps