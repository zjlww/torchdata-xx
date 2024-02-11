#pragma once
#include <functional>
#include <stdexcept>
#include <string_view>

#include "tensor_utils.h"
#include "types.h"

namespace data {

SamplerHandle mapSampler(SamplerHandle, ItemTransformHandle);
SamplerHandle filterSampler(SamplerHandle, ItemPredicateHandle);

SamplerHandle sampleDataset(DatasetHandle d);
SamplerHandle permuteSampleDataset(DatasetHandle d);

SamplerHandle sampleSamplers(SamplerList samplers, StringList samplerIDs,
                             DoubleList weights);

SamplerHandle queueSampler(SamplerHandle s, size_t nThreads, size_t queueSize);

SamplerHandle segmentSampler(SamplerHandle s, std::string_view bufferKey,
                             size_t segmentSize, int64_t dim);
BatchSamplerHandle segmentSamplerSlicing(SamplerHandle s,
                                         std::string_view bufferKey,
                                         size_t segmentSize, int64_t dim);
SamplerHandle segmentSamplerClasswise(SamplerHandle s,
                                      std::string_view bufferKey,
                                      std::string_view classKey,
                                      size_t segmentSize, int64_t dim);

SamplerHandle zipSamplerDataset(SamplerHandle s, DatasetHandle d,
                                std::string keyKey);

SamplerHandle sampleShard(SamplerHandle s, std::string shardPathKey,
                          std::string shardIDKey, size_t samplesPerShard);

BatchSamplerHandle sampleFixedBatch(SamplerHandle s, size_t batchSize);
BatchSamplerHandle bucketSampler(SamplerHandle s, std::string_view sortKey,
                                 Partition p);

struct Sampler : public std::enable_shared_from_this<Sampler> {
    virtual Item sample() = 0;

    // Apply a transform to all the samples.
    // The transform is lazy, only applied when sample() is called.
    SamplerHandle map(ItemTransformHandle func) {
        return mapSampler(shared_from_this(), func);
    }
    // Drop samples that do not pass the test.
    SamplerHandle filter(ItemPredicateHandle pred) {
        return filterSampler(shared_from_this(), pred);
    }
    // Create n_threads workers that samples from this sampler and store the
    // samples into a shared queue.
    SamplerHandle queue(size_t nThreads, size_t queueSize) {
        return queueSampler(shared_from_this(), nThreads, queueSize);
    }
    BatchSamplerHandle batch(size_t batchSize) {
        return sampleFixedBatch(shared_from_this(), batchSize);
    }
    SamplerHandle zipDataset(DatasetHandle d, std::string keyKey) {
        return zipSamplerDataset(shared_from_this(), d, keyKey);
    }
    SamplerHandle segment(std::string_view bufferKey, size_t segmentSize,
                          int64_t dim) {
        return segmentSampler(shared_from_this(), bufferKey, segmentSize, dim);
    }
    BatchSamplerHandle segmentSlicing(std::string_view bufferKey,
                                      size_t segmentSize, int64_t dim) {
        return segmentSamplerSlicing(shared_from_this(), bufferKey, segmentSize,
                                     dim);
    }
    SamplerHandle segmentClasswise(std::string_view bufferKey,
                                   std::string_view classKey,
                                   size_t segmentSize, int64_t dim) {
        return segmentSamplerClasswise(shared_from_this(), bufferKey, classKey,
                                       segmentSize, dim);
    }

    BatchSamplerHandle bucket(std::string_view sortKey, Partition p) {
        return bucketSampler(shared_from_this(), sortKey, p);
    }

    SamplerHandle sampleShard(std::string shardPathKey, std::string shardIDKey,
                              size_t samplesPerShard) {
        return data::sampleShard(shared_from_this(), shardPathKey, shardIDKey,
                                 samplesPerShard);
    }

    virtual ~Sampler() = default;

   protected:
    Sampler() = default;
};

SamplerHandle stackBatch(BatchSamplerHandle s);
SamplerHandle flattenBatch(BatchSamplerHandle s);

struct BatchSampler : public std::enable_shared_from_this<BatchSampler> {
    virtual ItemList sample() = 0;
    // Stacking returned batch
    SamplerHandle stack() { return stackBatch(shared_from_this()); }
    // Converts back to a sampler.
    SamplerHandle flatten() { return flattenBatch(shared_from_this()); }
    virtual ~BatchSampler() = default;

   protected:
    BatchSampler() = default;
};

}  // namespace data