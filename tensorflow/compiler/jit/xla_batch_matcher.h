/* Copyright 2026 The TensorFlow Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_JIT_XLA_BATCH_MATCHER_H_
#define TENSORFLOW_COMPILER_JIT_XLA_BATCH_MATCHER_H_

#include <cstdint>
#include <string>
#include <vector>

namespace tensorflow {

// Define the maximum allowed batch size: 2147483648 >> 1 = 1073741824
// Exceeding this value will make the next power of two exceed the safe range
constexpr int kMaxBatch = 2147483648ULL >> 1;

class XlaBatchMatcher {
 public:
  XlaBatchMatcher();
  virtual ~XlaBatchMatcher() = default;
  int64_t get_xla_compile_batch(int64_t real_batch);
  std::vector<int64_t> get_all_batches() { return all_batches_; }

 private:
  void parse_env_config();
  void print_all_batches();
  std::vector<int64_t> parse_single_item(const std::string& item);
  int64_t find_min_larger_batch(int64_t real_batch);

  std::vector<int64_t> all_batches_;
  std::string env_str_;
  int64_t last_batch_ = -1;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_BATCH_MATCHER_H_
