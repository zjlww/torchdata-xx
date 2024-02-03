#include "sampler.h"

#include <ATen/ops/from_blob.h>
#include <ATen/ops/pad_sequence.h>
#include <c10/core/ScalarType.h>
#include <tbb/enumerable_thread_specific.h>
#include <torch/torch.h>

#include <boost/thread/sync_bounded_queue.hpp>
#include <memory>
#include <variant>

#include "dataset.h"
#include "types.h"

namespace data {
struct SegmentedSampler final : Sampler {
    SamplerHandle base{};
    size_t segmentSize{};
    int64_t dim{0};
    std::string bufferKey;
    tbb::enumerable_thread_specific<TensorBuffer> buffers;
    SegmentedSampler(SamplerHandle s, std::string_view bufferKey,
                     size_t segmentSize, int64_t dim)
        : base{std::move(s)},
          bufferKey{bufferKey},
          segmentSize{segmentSize},
          dim{dim} {}

    Item sample() override {
        auto& buffer = buffers.local();
        buffer.dim = dim;
        while (buffer.size() < segmentSize) {
            auto item = base->sample();
            Tensor A = std::get<Tensor>(item[bufferKey]);
            buffer.push(A);
        }
        Tensor A = buffer.pop(segmentSize);
        Item it;
        it.emplace(bufferKey, A);
        return it;
    }
};
SamplerHandle segmentSampler(SamplerHandle s, std::string_view bufferKey,
                             size_t segmentSize, int64_t dim) {
    return std::make_shared<SegmentedSampler>(s, bufferKey, segmentSize, dim);
}

/*
Given a Sampler sampling items { buffer_key: arr<DType> }. This function
transforms samples by generating slices from each sampled item. For each
input item, it generates a list of slices. Notice that the length of
buffer must be no less than segment_size!
*/
struct SliceSegmentedSampler final : BatchSampler {
    SamplerHandle base;
    size_t segmentSize;
    std::string bufferKey;
    int64_t dim;
    tbb::enumerable_thread_specific<std::mt19937> rng;
    SliceSegmentedSampler(SamplerHandle s, std::string_view bufferKey,
                          size_t segmentSize, int64_t dim)
        : base{std::move(s)},
          bufferKey{bufferKey},
          segmentSize{segmentSize},
          dim{dim} {}

    ItemList sample() override {
        // Initialize the thread local RNG:
        bool rng_exists;
        auto& _rng = rng.local(rng_exists);
        if (not rng_exists) {
            _rng.seed(std::random_device()());
        }

        // Sample an item:
        auto it = base->sample();
        Tensor A = std::get<Tensor>(it[bufferKey]);
        int64_t N = A.size(dim);
        if (N < segmentSize) {
            throw std::runtime_error(
                "SliceSegmentedSampler received too short sequence.");
        }
        // Rotate the columns of array A randomly.
        auto dist = std::uniform_int_distribution<int64_t>(0, N - 1);
        auto dice = dist(_rng);
        A = torch::roll(A, {dice}, {dim});
        // Cut A in dim, each slice has segmentSize elements.
        ItemList lst;
        for (size_t i = 0; i <= N - segmentSize; i += segmentSize) {
            auto S = A.slice(dim, i, i + segmentSize);
            lst.push_back(Item{{bufferKey, S}});
        }
        return lst;
    }
};

BatchSamplerHandle segmentSamplerSlicing(SamplerHandle s,
                                         std::string_view bufferKey,
                                         size_t segmentSize, int64_t dim) {
    return std::make_shared<SliceSegmentedSampler>(s, bufferKey, segmentSize,
                                                   dim);
}

/*
Given a Sampler of { buffer_key : Tensor, class_key : int64_t }. This
function works just like segment_Sampler. However it ensures that all
segments are from the same class_key. All storage are thread local for
this class.
*/
struct ClasswiseSegmentedSampler final : Sampler {
    SamplerHandle base;
    size_t segmentSize;
    std::string bufferKey;
    std::string classKey;
    int64_t dim;

    tbb::enumerable_thread_specific<std::map<int64_t, TensorBuffer>> buffers;
    tbb::enumerable_thread_specific<TensorBuffer*> currentBuffer;
    tbb::enumerable_thread_specific<int64_t> currentCls;

    ClasswiseSegmentedSampler(SamplerHandle s, std::string_view bufferKey,
                              std::string_view classKey, size_t segmentSize,
                              int64_t dim)
        : base{std::move(s)},
          bufferKey{bufferKey},
          classKey{classKey},
          segmentSize{segmentSize},
          dim{dim} {}

    Item popCurrentBuffer() {
        auto& _currentBuffer = currentBuffer.local();
        Item it;
        auto A = _currentBuffer->pop(segmentSize);
        it.emplace(bufferKey, A);
        it.emplace(classKey, currentCls.local());
        return it;
    }

    Item sample() override {
        auto& _currentCls = currentCls.local();
        auto& _currentBuffer = currentBuffer.local();
        auto& _buffers = buffers.local();
        if (_currentBuffer != nullptr and
            _currentBuffer->size() >= segmentSize) {
            return popCurrentBuffer();
        }
        while (true) {
            auto it = base->sample();
            _currentCls = std::get<int64_t>(it[classKey]);
            _currentBuffer = &_buffers[_currentCls];
            _currentBuffer->dim = dim;
            Tensor A = std::get<Tensor>(it[bufferKey]);
            _currentBuffer->push(A);
            if (_currentBuffer->size() >= segmentSize) {
                return popCurrentBuffer();
            }
        }
    }
};

SamplerHandle segmentSamplerClasswise(SamplerHandle s,
                                      std::string_view bufferKey,
                                      std::string_view classKey,
                                      size_t segmentSize, int64_t dim) {
    return std::make_shared<ClasswiseSegmentedSampler>(s, bufferKey, classKey,
                                                       segmentSize, dim);
}

struct SampledDataset final : Sampler {
    DatasetHandle base;
    tbb::enumerable_thread_specific<std::mt19937> rng;
    explicit SampledDataset(DatasetHandle base) : base{base} {}
    Item sample() override {
        bool rng_exists;
        auto& _rng = rng.local(rng_exists);
        if (not rng_exists) {
            _rng.seed(std::random_device()());
        }
        auto dist = std::uniform_int_distribution<size_t>(0, base->size() - 1);
        auto dice = dist(_rng);

        auto key = base->getKey(dice);
        auto it = (*base)[key];

        it.insert_or_assign("key", key.data());
        return it;
    }
};

SamplerHandle sampleDataset(DatasetHandle d) {
    return std::make_shared<SampledDataset>(d);
}

struct SampledSamplers final : Sampler {
    SamplerList bases;
    StringList samplerIDs;
    std::vector<double> weights;
    tbb::enumerable_thread_specific<std::mt19937> rng;
    SampledSamplers(SamplerList samplers, StringList samplerIDs,
                    DoubleList weights)
        : bases{std::move(samplers)},
          samplerIDs{samplerIDs},
          weights{weights} {}

    Item sample() override {
        bool rng_exist;
        auto& _rng = rng.local(rng_exist);
        if (not rng_exist) {
            _rng.seed(std::random_device()());
        }
        auto dist =
            std::discrete_distribution<size_t>(weights.begin(), weights.end());
        auto dice = dist(_rng);
        auto base = bases[dice];
        return base->sample();
    }
};

SamplerHandle sampleSamplers(SamplerList samplers, StringList samplerIDs,
                             DoubleList weights) {
    assert(samplers.size() == samplerIDs.size() &&
           samplerIDs.size() == weights.size());
    return std::make_shared<SampledSamplers>(
        std::move(samplers), std::move(samplerIDs), std::move(weights));
}

struct MappedSampler final : Sampler {
    SamplerHandle base;
    ItemTransform func;
    MappedSampler(SamplerHandle base, ItemTransform func)
        : base{base}, func{func} {}
    Item sample() override { return func(base->sample()); }
};

SamplerHandle mapSampler(SamplerHandle s, ItemTransform func) {
    return std::make_shared<MappedSampler>(std::move(s), std::move(func));
}

struct FilteredSampler final : Sampler {
    SamplerHandle base;
    ItemPredicate pred;
    FilteredSampler(SamplerHandle base, ItemPredicate pred)
        : base{base}, pred{std::move(pred)} {}
    Item sample() override {
        auto item = (*base).sample();
        while (not pred(item)) {
            item = (*base).sample();
        }
        return item;
    }
};

SamplerHandle filterSampler(SamplerHandle s, ItemPredicate pred) {
    return std::make_shared<FilteredSampler>(std::move(s), std::move(pred));
}

using Queue = boost::concurrent::sync_bounded_queue<Item>;
inline static void push_queue_forever(std::stop_token st, SamplerHandle sampler,
                                      Queue& queue) {
    while (not st.stop_requested()) {
        try {
            queue.push(sampler->sample());
        } catch (...) {
        }
    }
}

struct QueuedSampler final : Sampler {
    SamplerHandle base;
    std::vector<std::jthread> workers;
    Queue q;
    Item sample() override { return q.pull(); }
    QueuedSampler(SamplerHandle base, size_t nThreads, size_t queueSize)
        : base{std::move(base)}, q(queueSize) {
        for (size_t i = 0; i < nThreads; ++i) {
            workers.emplace_back([this](std::stop_token st) {
                push_queue_forever(st, this->base, q);
            });
        }
    }
    virtual ~QueuedSampler() {
        for (auto& worker : workers) {
            worker.request_stop();
        }
        while (!q.empty()) {
            q.pull();
        }
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        q.close();
    }
};

SamplerHandle queueSampler(SamplerHandle sampler, size_t nThreads,
                           size_t queueSize) {
    return std::make_shared<QueuedSampler>(std::move(sampler), nThreads,
                                           queueSize);
}

struct BucketizedSampler final : BatchSampler {
    using BucketGrid = std::vector<std::vector<Item>>;
    SamplerHandle base;
    std::string sortKey;
    Partition p;
    tbb::enumerable_thread_specific<BucketGrid> buckets;

    BucketizedSampler(SamplerHandle s, std::string_view sortKey, Partition p)
        : base{std::move(s)}, sortKey{sortKey}, p{std::move(p)} {}

    ItemList sample() override {
        bool exists;
        auto& _buckets = buckets.local(exists);
        if (not exists) {
            _buckets = BucketGrid(p.size());
        }
        while (true) {
            auto it = base->sample();
            auto len = std::get<int64_t>(it[sortKey]);
            int bin_idx = -1;
            for (int i = 0; i < p.size(); ++i) {
                auto [a, b, c] = p[i];
                if (a <= len and len < b) {
                    _buckets[i].push_back(std::move(it));
                    bin_idx = i;
                    break;
                }
            }
            if (bin_idx == -1) {
                continue;  // Drop this item
            }
            auto [a, b, c] = p[bin_idx];
            if (_buckets[bin_idx].size() == c) {
                ItemList items = std::move(_buckets[bin_idx]);
                _buckets[bin_idx].clear();
                // Sort the items in descending order.
                std::sort(items.begin(), items.end(),
                          [this](auto&& u, auto&& v) {
                              auto u_val = std::get<int64_t>(u[sortKey]);
                              auto v_val = std::get<int64_t>(v[sortKey]);
                              return v_val < u_val;
                          });
                return items;
            }
        }
    }
};

BatchSamplerHandle bucketSampler(SamplerHandle s, std::string_view sortKey,
                                 Partition p) {
    return std::make_shared<BucketizedSampler>(s, sortKey, p);
}

struct FixedSizeBatchedSampler final : BatchSampler {
    SamplerHandle base;
    size_t batchSize;
    FixedSizeBatchedSampler(SamplerHandle base, size_t batchSize)
        : base{std::move(base)}, batchSize(batchSize) {}

    ItemList sample() override {
        ItemList items;
        for (int i = 0; i < batchSize; ++i) {
            items.push_back(base->sample());
        }
        return items;
    }
};

BatchSamplerHandle sampleFixedBatch(SamplerHandle s, size_t batchSize) {
    return std::make_shared<FixedSizeBatchedSampler>(s, batchSize);
}

template <typename T>
std::vector<T> gather_values(ItemList const& items, std::string_view key) {
    std::vector<T> result;
    for (const Item& item : items) {
        result.push_back(std::get<T>(item.at(key.data())));
    }
    return result;
}

template <typename S, torch::Dtype T> Tensor to_tensor(std::vector<S>& v) {
    auto opts = torch::TensorOptions().dtype(T);
    return torch::from_blob(v.data(), int64_t(v.size()), opts);
}

// This function transforms a list of items into a single item.
// It accepts [T, ...] tensors, and pad into [B, T, ...] tensors.
// And call torch::pad_sequence on the elements.
// For double / int64_t values, it stacks them into a tensor.
// For all other values, it dumps them.
Item stack_items(ItemList const& items) {
    Item result;
    auto N = items.size();
    if (N == 0) return {};
    Item const& first = *(items.begin());
    for (auto const& p : first) {
        auto const& [k, v] = p;
        if (std::holds_alternative<int64_t>(v)) {
            std::vector<int64_t> vs = gather_values<int64_t>(items, k);
            result[k] = to_tensor<int64_t, torch::kInt64>(vs);
        } else if (std::holds_alternative<double>(v)) {
            std::vector<double> vs = gather_values<double>(items, k);
            result[k] = to_tensor<double, torch::kDouble>(vs);
        } else if (std::holds_alternative<Tensor>(v)) {
            // Construct the padded tensor:
            std::vector<Tensor> vs = gather_values<Tensor>(items, k);
            Tensor V = torch::pad_sequence(vs, true, 0.0);
            result[k] = V;

            // Construct the lens tensor:
            std::vector<int64_t> lens(N);
            for (size_t i = 0; i < N; ++i) lens[i] = vs[i].size(0);
            result[k + "_lens"] = to_tensor<int64_t, torch::kInt64>(lens);
        }
    }
    return result;
}

struct StackedBatchSampler final : Sampler {
    BatchSamplerHandle base;
    StackedBatchSampler(BatchSamplerHandle base) : base{std::move(base)} {}
    Item sample() override {
        auto items = base->sample();
        return stack_items(items);
    }
};

SamplerHandle stackBatch(BatchSamplerHandle s) {
    return std::make_shared<StackedBatchSampler>(std::move(s));
}

struct FlattenedBatchSampler final : Sampler {
    BatchSamplerHandle base;
    tbb::enumerable_thread_specific<ItemList> lists;
    FlattenedBatchSampler(BatchSamplerHandle base) : base{std::move(base)} {}

    Item sample() override {
        ItemList& lst = lists.local();
        while (lst.empty()) {
            lst = base->sample();
        }
        Item it = std::move(lst.back());
        lst.pop_back();
        return it;
    }
};

SamplerHandle flattenBatch(BatchSamplerHandle s) {
    return std::make_shared<FlattenedBatchSampler>(s);
}

// The key of the item must be stored in item[key].
SamplerHandle zipSamplerDataset(SamplerHandle s, DatasetHandle d) {
    return mapSampler(s, [d](Item it) {
        std::string const& key = std::get<std::string>(it["key"]);
        auto dit = (*d)[key];
        it.merge(std::move(dit));
        return it;
    });
}

}  // namespace data