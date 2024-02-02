#include "dataset.h"

#include <torch/script.h>

#include <algorithm>
#include <memory>
#include <ranges>
#include <stdexcept>

namespace data {

struct ZippedDataset final : Dataset {
    DatasetList p_bases;
    Item operator[](std::string_view key) override {
        Item item;
        for (auto& p_base : std::ranges::reverse_view(p_bases)) {
            auto& t = p_base;
            auto&& part_item = (*p_base)[key];
            item.merge(part_item);
        }
        return item;
    }

    ZippedDataset(DatasetList const& datasets) : p_bases{std::move(datasets)} {
        // Compute the common keys:
        KeyList common_keys = datasets[0]->keys;
        for (int i = 1; i < datasets.size(); ++i) {
            KeyList buffer;
            std::set_intersection(common_keys.begin(), common_keys.end(),
                                  datasets[i]->keys.begin(),
                                  datasets[i]->keys.end(),
                                  std::back_inserter(buffer));
            std::swap(buffer, common_keys);
            buffer.clear();
        }
        keys = std::move(common_keys);
    }
};

DatasetHandle zipDatasets(DatasetList const& datasets) {
    assert(datasets.size() > 1);
    return std::make_shared<ZippedDataset>(datasets);
}

struct UnionedDataset final : Dataset {
    DatasetList p_bases;
    using KeyIDList = std::vector<std::pair<std::string, int>>;
    KeyIDList key_ids;

    UnionedDataset(DatasetList const& datasets) : p_bases{datasets} {
        assert(datasets.size() > 1);

        // First compute the union of keys.
        for (int i = 0; i < datasets.size(); ++i) {
            for (const auto& key : datasets[i]->keys) {
                key_ids.emplace_back(key, i);
                keys.emplace_back(key);
            }
        }

        std::sort(key_ids.begin(), key_ids.end());
        std::sort(keys.begin(), keys.end());

        auto it = std::adjacent_find(keys.begin(), keys.end());
        if (it != keys.end()) {
            throw std::runtime_error(
                "Duplicated keys found in union_datasets.");
        }
    }

    Item operator[](std::string_view key) override {
        auto it = std::lower_bound(
            key_ids.begin(), key_ids.end(), key,
            [](const auto& pair, const auto& k) { return pair.first < k; });
        if (it != key_ids.end() && it->first == key) {
            return (*p_bases[it->second])[key];
        } else {
            throw std::runtime_error("Key not found in unioned_datasets.");
        }
    }
};

// The user is responsible to ensure that the keys do not overlap.
DatasetHandle unionDatasets(DatasetList const& datasets) {
    return std::make_shared<UnionedDataset>(datasets);
}

struct PrefixedDataset final : Dataset {
    DatasetHandle base;
    size_t prefix_length;
    PrefixedDataset(DatasetHandle base, std::string_view prefix)
        : Dataset{base->keys},
          base{std::move(base)},
          prefix_length(prefix.length()) {
        for (auto& key : keys) {
            key = prefix.data() + key;
        }
    }

    Item operator[](std::string_view key) override {
        std::string_view stripped_key = key.substr(prefix_length);
        return (*base)[stripped_key];
    }
};

DatasetHandle prefixDataset(DatasetHandle base, std::string_view prefix) {
    return std::make_shared<PrefixedDataset>(base, prefix);
}

struct MappedDataset final : Dataset {
    DatasetHandle base;
    ItemTransform func;
    MappedDataset(DatasetHandle base, ItemTransform func)
        : Dataset{base->keys}, base{std::move(base)}, func{std::move(func)} {}
    Item operator[](std::string_view key) override {
        auto&& b = *base;
        return func(b[key]);
    }
};

DatasetHandle mapDataset(DatasetHandle base, ItemTransform func) {
    return std::make_shared<MappedDataset>(std::move(base), std::move(func));
}

struct FilteredDataset final : Dataset {
    DatasetHandle base;
    KeyPredicate pred;
    FilteredDataset(DatasetHandle base, KeyPredicate pred)
        : Dataset{base->keys}, base{std::move(base)}, pred{std::move(pred)} {
        keys.erase(std::remove_if(keys.begin(), keys.end(), std::not_fn(pred)),
                   keys.end());
    }
    Item operator[](std::string_view key) override {
        if (pred(key)) {
            return (*base)[key];
        } else {
            throw std::runtime_error("Key not found in FilteredDataset");
        }
    }
};

DatasetHandle filterDataset(DatasetHandle base, KeyPredicate pred) {
    return std::make_shared<FilteredDataset>(std::move(base), std::move(pred));
}

struct LoadedShard final : Dataset {
    Path path;
    torch::jit::Module m;
    LoadedShard(Path path) : path{path} {
        m = torch::jit::load(path.string());
        auto item_lst = m.named_modules();
        auto it = item_lst.begin();
        for (int i = 0; i < item_lst.size(); ++i) {
            if (i < 1)
                ++it;
            else {
                keys.push_back((*it).name);
            }
        }
    }

    Item operator[](std::string_view key) override {
        auto const& item_module = m.attr(key.data()).toModule();
        auto const& lst = item_module.named_attributes(false);

        Item item{};
        auto it = lst.begin();
        for (int i = 0; i < lst.size(); ++i) {
            if (i < 2)
                ++it;  // Skip the first two elements.
            else {
                auto name = (*it).name;
                auto const& value = (*it).value;
                if (value.isInt()) {
                    item[name] = value.toInt();
                } else if (value.isDouble()) {
                    item[name] = value.toDouble();
                } else if (value.isString()) {
                    item[name] = value.toStringRef();
                } else if (value.isTensor()) {
                    item[name] = value.toTensor();
                } else {
                    throw std::runtime_error(
                        "Found unsupported value type in shard item.");
                }
            }
        }
        return item;
    }
};

DatasetHandle loadShard(Path path) {
    return std::make_shared<LoadedShard>(path);
}

}  // namespace data