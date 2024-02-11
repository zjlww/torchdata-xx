#pragma once
#include "sampler.h"
#include "types.h"

namespace data {

// Helper functions:
DatasetHandle immediateDataset(ItemDict items);
DatasetHandle loadShard(std::string_view path);

DatasetHandle mapDataset(DatasetHandle d, ItemTransformHandle func);
DatasetHandle filterDataset(DatasetHandle d, KeyPredicateHandle pred);
DatasetHandle zipDatasets(DatasetList const& datasets);
DatasetHandle unionDatasets(DatasetList const& datasets);
DatasetHandle prefixDataset(DatasetHandle d, std::string_view prefix);

// Dataset interface:
struct Dataset : public std::enable_shared_from_this<Dataset> {
   public:
    // The list of keys must be sorted in a Dataset.
    KeyList keys{};

    virtual size_t size() { return keys.size(); }

    virtual bool contains(std::string_view key) {
        return std::binary_search(keys.begin(), keys.end(), key);
    }

    virtual Item operator[](std::string_view key) = 0;
    virtual Item getItem(size_t idx) { return (*this)[keys[idx]]; }
    virtual std::string_view getKey(size_t idx) { return keys[idx]; }

    // Apply a transform to all the items in the Dataset.
    // The transform is lazy, only applied when operator[] is called.
    DatasetHandle map(ItemTransformHandle func) {
        return mapDataset(shared_from_this(), std::move(func));
    }

    // Filter the dataset by keys.
    DatasetHandle filter(KeyPredicateHandle pred) {
        return filterDataset(shared_from_this(), std::move(pred));
    }

    // Zip two dataset, for each key, the returned item from two datasets are
    // merged.
    DatasetHandle zip(DatasetHandle other) {
        return zipDatasets({shared_from_this(), std::move(other)});
    }

    // Merge two dataset into a large one. The user must ensure the keys are not
    // overlapping.
    DatasetHandle merge(DatasetHandle other) {
        return unionDatasets({shared_from_this(), std::move(other)});
    }

    // Prefix all the keys in a dataset.
    DatasetHandle prefix(std::string_view prefix) {
        return prefixDataset(shared_from_this(), prefix);
    }

    // Convert a dataset into a sampler.
    SamplerHandle sample() { return sampleDataset(shared_from_this()); }
    SamplerHandle permuteSample() {
        return permuteSampleDataset(shared_from_this());
    }

    // Save all items in a dataset to a map. This can be slow.
    ItemDict toMap() {
        ItemDict key_items;
        for (auto&& key : keys) {
            key_items.emplace_hint(key_items.end(), key, (*this)[key]);
        }
        return key_items;
    }
    virtual ~Dataset() = default;

   protected:
    Dataset() = default;
    explicit Dataset(KeyList const& keys) : keys{keys} {};
    explicit Dataset(KeyList&& keys) : keys{std::move(keys)} {};
};

}  // namespace data