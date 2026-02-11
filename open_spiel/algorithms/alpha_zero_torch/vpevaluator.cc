// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"

#include <cstdint>
#include <iostream>
#include <memory>

#include "open_spiel/abseil-cpp/absl/hash/hash.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/utils/stats.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {

VPNetEvaluator::VPNetEvaluator(DeviceManager *device_manager, int batch_size,
                               int threads, int cache_size, int cache_shards)
    : device_manager_(*device_manager), // Take the address and then Dereference
      batch_size_(batch_size), queue_(batch_size * threads * 4),
      batch_size_hist_(batch_size + 1) {
  cache_shards = std::max(1, cache_shards);
  cache_.reserve(cache_shards);
  for (int i = 0; i < cache_shards; ++i) {
    cache_.push_back(
        std::make_unique<LRUCache<uint64_t, VPNetModel::InferenceOutputs>>(
            cache_size / cache_shards));
  }
  if (batch_size_ <= 1) {
    threads = 0;
  }
  inference_threads_.reserve(threads);
  for (int i = 0; i < threads; ++i) {
    inference_threads_.emplace_back([this]() { this->Runner(); });
  }
}

VPNetEvaluator::~VPNetEvaluator() {
  stop_.Stop();
  queue_.BlockNewValues();
  queue_.Clear();
  for (auto &t : inference_threads_) {
    t.join();
  }
}

void VPNetEvaluator::ClearCache() {
  for (auto &c : cache_) {
    c->Clear();
  }
}

LRUCacheInfo VPNetEvaluator::CacheInfo() {
  LRUCacheInfo info;
  for (auto &c : cache_) {
    info += c->Info();
  }
  return info;
}

std::vector<double> VPNetEvaluator::Evaluate(const State &state) {
  // TODO(author5): currently assumes zero-sum.
  double p0value = Inference(state).value;
  return {p0value, -p0value};
}

open_spiel::ActionsAndProbs VPNetEvaluator::Prior(const State &state) {
  if (state.IsChanceNode()) {
    return state.ChanceOutcomes();
  } else {
    return Inference(state).policy;
  }
}

VPNetModel::InferenceOutputs VPNetEvaluator::Inference(const State &state) {
  VPNetModel::InferenceInputs inputs = {state.LegalActions(),
                                        state.ObservationTensor()};

  uint64_t key;
  int cache_shard;
  if (!cache_.empty()) {
    // Create cache key
    key = absl::Hash<VPNetModel::InferenceInputs>{}(inputs);
    cache_shard = key % cache_.size();
    absl::optional<const VPNetModel::InferenceOutputs> opt_outputs =
        cache_[cache_shard]->Get(key);
    if (opt_outputs) {
      return *opt_outputs;
    }
  }

  VPNetModel::InferenceOutputs outputs;
  if (batch_size_ <= 1) {
    outputs = device_manager_.Get(1)->Inference(std::vector{inputs})[0];
  } else {
    // Pushing state to inference queue

    std::promise<VPNetModel::InferenceOutputs> prom;
    std::future<VPNetModel::InferenceOutputs> fut = prom.get_future();
    // std::cout << "DEBUG: Pushing to queue" << std::endl;
    queue_.Push(QueueItem{inputs, &prom});
    // std::cout << "DEBUG: Waiting for future" << std::endl;
    outputs = fut.get();
    // std::cout << "DEBUG: Future returned" << std::endl;
  }
  if (!cache_.empty()) {
    cache_[cache_shard]->Set(key, outputs);
  }
  return outputs;
}

void VPNetEvaluator::Runner() {
  std::vector<VPNetModel::InferenceInputs> inputs;
  std::vector<std::promise<VPNetModel::InferenceOutputs> *> promises;
  inputs.reserve(batch_size_);
  promises.reserve(batch_size_);
  // std::cout << "DEBUG: Runner started" << std::endl;

  absl::Time last_log_time = absl::Now();
  int items_since_last_log = 0;
  int batches_since_last_log = 0;
  double total_inference_time_ms = 0;

  while (!stop_.StopRequested()) {
    {
      // Only one thread at a time should be listening to the queue to maximize
      // batch size and minimize latency.
      absl::MutexLock lock(inference_queue_m_);
      absl::Time deadline = absl::InfiniteFuture();
      for (int i = 0; i < batch_size_; ++i) {
        absl::optional<QueueItem> item = queue_.Pop(deadline);
        if (!item) { // Hit the deadline.
          break;
        }
        if (inputs.empty()) {
          deadline = absl::Now() + absl::Milliseconds(1);
        }
        // std::cout << "DEBUG: Runner popped item" << std::endl;
        inputs.push_back(item->inputs);
        promises.push_back(item->prom);
      }
    }

    if (inputs.empty()) { // Almost certainly StopRequested.
      continue;
    }

    size_t real_batch_size = inputs.size();
    if (real_batch_size < batch_size_) {
      // Pad with the first item to ensure valid data and fixed batch size
      // This prevents cuDNN thrashing/recompilation for variable batch sizes
      VPNetModel::InferenceInputs dummy = inputs[0];
      for (size_t i = real_batch_size; i < batch_size_; ++i) {
        inputs.push_back(dummy);
      }
    }

    {
      absl::MutexLock lock(stats_m_);
      batch_size_stats_.Add(real_batch_size);
      batch_size_hist_.Add(real_batch_size);
    }

    absl::Time start = absl::Now();
    std::vector<VPNetModel::InferenceOutputs> outputs =
        device_manager_.Get(inputs.size())->Inference(inputs);
    double latency = absl::ToDoubleMilliseconds(absl::Now() - start);

    items_since_last_log += real_batch_size;
    batches_since_last_log++;
    total_inference_time_ms += latency;

    if (absl::ToDoubleSeconds(absl::Now() - last_log_time) >= 5.0) {
      double interval = absl::ToDoubleSeconds(absl::Now() - last_log_time);
      std::cout << "Inference Speed: " << (items_since_last_log / interval)
                << " items/s | Avg Batch: "
                << (double)items_since_last_log / (double)batches_since_last_log
                << " | Avg Latency: "
                << (total_inference_time_ms / batches_since_last_log) << " ms"
                << std::endl;
      last_log_time = absl::Now();
      items_since_last_log = 0;
      batches_since_last_log = 0;
      total_inference_time_ms = 0;
    }
    // std::cout << "DEBUG: Runner finished Inference With " << "Size "
    //           << inputs.size() << std::endl;
    for (int i = 0; i < promises.size(); ++i) {
      promises[i]->set_value(outputs[i]);
    }
    inputs.clear();
    promises.clear();
  }
}

void VPNetEvaluator::ResetBatchSizeStats() {
  absl::MutexLock lock(stats_m_);
  batch_size_stats_.Reset();
  batch_size_hist_.Reset();
}

open_spiel::BasicStats VPNetEvaluator::BatchSizeStats() {
  absl::MutexLock lock(stats_m_);
  return batch_size_stats_;
}

open_spiel::HistogramNumbered VPNetEvaluator::BatchSizeHistogram() {
  absl::MutexLock lock(stats_m_);
  return batch_size_hist_;
}

} // namespace torch_az
} // namespace algorithms
} // namespace open_spiel
