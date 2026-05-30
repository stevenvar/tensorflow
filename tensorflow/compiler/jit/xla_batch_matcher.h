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

#include "absl/container/flat_hash_map.h"

namespace tensorflow {

// Define the maximum allowed batch size: 2147483648 >> 1 = 1073741824
// Exceeding this value will make the next power of two exceed the safe range
constexpr int kMaxBatch = 2147483648ULL >> 1;

class XlaBatchMatcher {
 public:
  XlaBatchMatcher();
  virtual ~XlaBatchMatcher() = default;

  // Per-cluster-key API (recommended): isolates padding candidates per cluster.
  int64_t get_xla_compile_batch(const std::string& cluster_key,
                                int64_t real_batch);

  // Backward-compatible API: uses a default (global) key.
  int64_t get_xla_compile_batch(int64_t real_batch);

  // For debugging.
  std::vector<int64_t> get_all_batches(const std::string& cluster_key);

 private:
  struct ClusterState {
    std::vector<int64_t> all_batches;
    bool updated = false;
  };

  void parse_env_config();
  void print_all_batches(const std::string& cluster_key,
                         const std::vector<int64_t>& batches);
  std::vector<int64_t> parse_single_item(const std::string& item);

  int64_t find_min_larger_batch(ClusterState* state, int64_t real_batch);

  absl::flat_hash_map<std::string, ClusterState> clusters_;
  std::string env_str_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_BATCH_MATCHER_H_
