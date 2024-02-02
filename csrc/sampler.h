#pragma once
#include <tbb/enumerable_thread_specific.h>

#include <functional>
#include <stdexcept>
#include <string_view>

#include "types.h"

namespace data {

SamplerHandle mapSampler(SamplerHandle, ItemTransform);

SamplerHandle filterSampler(SamplerHandle, ItemPredicate);
SamplerHandle sampleDataset(DatasetHandle);
SamplerHandle sampleSamplers(std::vector<SamplerHandle>,
                             std::vector<std::string_view>,
                             std::vector<double>);
SamplerHandle queueSampler(SamplerHandle sampler, size_t n_threads,
                           size_t queue_size);
BatchSamplerHandle sampleFixedBatch(SamplerHandle s, size_t batch_size);
BatchSamplerHandle bucketSampler(SamplerHandle s, std::string_view sort_key,
                                 Partition p);

struct Sampler : public std::enable_shared_from_this<Sampler> {
    virtual Item sample() = 0;

    // Apply a transform to all the samples.
    // The transform is lazy, only applied when sample() is called.
    SamplerHandle map(ItemTransform func);

    // Drop samples that do not pass the test.
    SamplerHandle filter(ItemPredicate pred);

    // Create n_threads workers that samples from this sampler and store the
    // samples into a shared queue.
    SamplerHandle queue(size_t n_threads, size_t queue_size);

    // See segment_sampler for details.
    // template <typename DType>
    // SamplerHandle segment(std::string_view buffer_key, size_t segment_size) {
    //     return segment_sampler<DType>(shared_from_this(), buffer_key,
    //                                   segment_size);
    // }

    // See segment_sampler for details.
    // template <typename DType>
    // BatchSamplerHandle segmentSlices(std::string_view buffer_key,
    //                                  size_t segment_size) {
    //     return segment_sampler_slice<DType>(shared_from_this(), buffer_key,
    //                                         segment_size);
    // }

    // See segment_sampler_classwise for details.
    // template <typename DType, typename CType>
    // SamplerHandle segmentWithCls(std::string_view buffer_key,
    //                                std::string_view class_key,
    //                                size_t segment_size) {
    //     return segment_sampler_classwise<DType, CType>(
    //         shared_from_this(), buffer_key, class_key, segment_size);
    // }

    // For each sampled Item, seek for additional information in a dataset, then
    // merge the located Item into the current Item.
    // SamplerHandle zip(DatasetHandle d);

    // Collect batch_size items into a list and return a batch in each sample.
    // BatchSamplerHandle batch(size_t batch_size);

    // Given a list of partition of the form [(1, 2, 10), (2, 4, 5), ...];
    // Where each tuple means (begin, end, n_samples);
    // This sampler buckets multiples samples according to their length (stored
    // in sort_key) and returns the regrouped sample.
    // BatchSamplerHandle bucket(std::string_view sort_key, partition p);

    virtual ~Sampler() = default;

   protected:
    Sampler() = default;
};

SamplerHandle stack_batch(BatchSamplerHandle s, key_list arr_keys,
                          key_list int_keys);

struct batch_sampler : public std::enable_shared_from_this<batch_sampler> {
    virtual item_list sample() = 0;

    SamplerHandle stack(key_list int_keys, key_list int_arr_keys);
    SamplerHandle stack_auto();

    // Converts back to a sampler.
    SamplerHandle flatten();

    virtual ~batch_sampler() = default;

   protected:
    batch_sampler() = default;
};

/*
Given a sampler sampling items { buffer_key : arr<DType> }. This function
transforms samples by dumping the arrays to a buffer. It returns a fixed
length sample by concatenating multiple samples from the input sampler.
See eigen_buffer<> for more details.
*/
template <typename DType>
inline SamplerHandle segment_sampler(SamplerHandle s,
                                     std::string_view buffer_key,
                                     size_t segment_size) {
    struct segmented_sampler final : sampler {
        SamplerHandle base;
        size_t segment_size;
        std::string buffer_key;
        eigen_buffer<DType> buffer;
        segmented_sampler(SamplerHandle s, std::string_view buffer_key,
                          size_t segment_size)
            : base{std::move(s)},
              buffer_key(buffer_key),
              segment_size(segment_size) {}

        Item sample() override {
            while (buffer.size() < segment_size) {
                auto Item = base->sample();
                auto&& A = std::get<arr<DType>>(Item[buffer_key]);
                buffer.push(std::move(A));
            }
            auto A = buffer.pop(segment_size);
            Item it;
            it.emplace(buffer_key, std::move(A));
            return it;
        }
    };
    return std::make_shared<segmented_sampler>(s, buffer_key, segment_size);
}

/*
Given a sampler sampling items { buffer_key: arr<DType> }. This function
transforms samples by generating slices from each sampled Item. For each input
Item, it generates a list of slices.
Notice that the length of buffer must be no less than segment_size!
*/
template <typename DType>
inline BatchSamplerHandle segment_sampler_slice(SamplerHandle s,
                                                std::string_view buffer_key,
                                                size_t segment_size) {
    struct slice_segmented_sampler final : batch_sampler {
        SamplerHandle base;
        size_t segment_size;
        std::string buffer_key;
        tbb::enumerable_thread_specific<std::mt19937> rng;
        slice_segmented_sampler(SamplerHandle s, std::string_view buffer_key,
                                size_t segment_size)
            : base{std::move(s)},
              buffer_key(buffer_key),
              segment_size(segment_size) {}

        item_list sample() override {
            // Initialize the thread local RNG:
            bool rng_exists;
            auto& _rng = rng.local(rng_exists);
            if (not rng_exists) {
                _rng.seed(std::random_device()());
            }

            // Sample an Item:
            auto it = base->sample();
            auto&& A = std::get<arr<DType>>(it[buffer_key]);
            auto rows = A.rows();
            auto cols = A.cols();
            if (cols < segment_size) {
                throw std::runtime_error(
                    fmt::format("The number of columns in the array ({}) is "
                                "less than the segment size ({}).",
                                cols, segment_size));
            }

            // Rotate the columns of array A randomly.
            auto dist = std::uniform_int_distribution<size_t>(0, A.cols() - 1);
            auto dice = dist(_rng);
            arr<DType> B(rows, cols);
            B.rightCols(cols - dice) = A.leftCols(cols - dice);
            B.leftCols(dice) = A.rightCols(dice);

            // Cut B by columns, each slice has segment_size columns.
            item_list lst;
            for (size_t i = 0; i <= cols - segment_size; i += segment_size) {
                arr<DType> slice = B.middleCols(i, segment_size);
                lst.push_back(Item{{buffer_key, std::move(slice)}});
            }
            return lst;
        }
    };
    return std::make_shared<slice_segmented_sampler>(s, buffer_key,
                                                     segment_size);
}

/*
Given a sampler of { buffer_key : arr<DType>, class_key : KType }. This function
works just like segment_sampler. However it ensures that all segments are from
the same class_key.
All storage are thread local for this class.
TODO: Write tests for this class.
*/
template <typename DType, typename CType>
inline SamplerHandle segment_sampler_classwise(SamplerHandle s,
                                               std::string_view buffer_key,
                                               std::string_view class_key,
                                               size_t segment_size) {
    struct classwise_segmented_sampler final : sampler {
        SamplerHandle base;
        size_t segment_size;
        std::string buffer_key;
        std::string class_key;

        tbb::enumerable_thread_specific<std::map<CType, eigen_buffer<DType>>>
            buffers;
        tbb::enumerable_thread_specific<eigen_buffer<DType>*> current_buffer;
        tbb::enumerable_thread_specific<CType> current_cls;

        classwise_segmented_sampler(SamplerHandle s,
                                    std::string_view buffer_key,
                                    std::string_view class_key,
                                    size_t segment_size)
            : base{std::move(s)},
              buffer_key{buffer_key},
              class_key{class_key},
              segment_size{segment_size} {}

        Item pop_current_buffer() {
            auto& _current_buffer = current_buffer.local();
            Item it;
            auto A = _current_buffer->pop(segment_size);
            it.emplace(buffer_key, std::move(A));
            it.emplace(class_key, current_cls.local());
            return it;
        }

        Item sample() override {
            auto& _current_cls = current_cls.local();
            auto& _current_buffer = current_buffer.local();
            auto& _buffers = buffers.local();
            if (_current_buffer != nullptr and
                _current_buffer->size() >= segment_size) {
                return pop_current_buffer();
            }
            while (true) {
                auto it = base->sample();
                _current_cls = std::get<CType>(it[class_key]);
                _current_buffer = &_buffers[_current_cls];
                auto&& A = std::get<arr<DType>>(it[buffer_key]);
                _current_buffer->push(std::move(A));
                if (_current_buffer->size() >= segment_size) {
                    return pop_current_buffer();
                }
            }
        }
    };
    return std::make_shared<classwise_segmented_sampler>(
        s, buffer_key, class_key, segment_size);
}

}  // namespace data