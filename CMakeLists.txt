cmake_minimum_required(VERSION 3.16)
project(place_rec LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs features2d calib3d)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ROS-free core library: only OpenCV + Eigen.
add_library(place_rec
  src/binary_descriptor.cpp
  src/vocabulary_tree.cpp
  src/bow_database.cpp
  src/bow_recognizer.cpp
  src/tiny_image_recognizer.cpp
  src/temporal_filter.cpp
  src/feature_extractor.cpp
  src/geometric_verifier.cpp
  src/tum_sequence.cpp)
target_include_directories(place_rec PUBLIC include)
target_link_libraries(place_rec PUBLIC ${OpenCV_LIBS} Eigen3::Eigen)
target_compile_options(place_rec PRIVATE -Wall -Wextra)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_compile_options(place_rec PRIVATE -mpopcnt)  # 1 instruction per 64-bit Hamming word
endif()  # aarch64 (Orin): popcount maps to NEON CNT without extra flags

add_executable(train_vocab apps/train_vocab.cpp)
target_link_libraries(train_vocab PRIVATE place_rec)
add_executable(run_eval apps/run_eval.cpp)
target_link_libraries(run_eval PRIVATE place_rec)

enable_testing()
find_package(GTest REQUIRED)
add_executable(unit_tests tests/test_bow.cpp tests/test_verifier.cpp)
target_link_libraries(unit_tests PRIVATE place_rec GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(unit_tests)