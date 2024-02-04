#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <any>
#include <array>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <variant>

namespace data {

using namespace std::literals;

// Pipeline Components:
struct Dataset;
using DatasetHandle = std::shared_ptr<Dataset>;
struct Sampler;
using SamplerHandle = std::shared_ptr<Sampler>;
struct BatchSampler;
using BatchSamplerHandle = std::shared_ptr<BatchSampler>;

// Data Types
using Tensor = torch::Tensor;
using IValue = torch::IValue;
using ValueType = std::variant<bool, int64_t, double, std::string, Tensor,
                               DatasetHandle, SamplerHandle>;
using Item = std::map<std::string, ValueType>;
using Partition = std::vector<std::tuple<int, int, int>>;

// Functional Types
struct ItemTransform : public std::enable_shared_from_this<ItemTransform> {
    virtual Item operator()(Item item) = 0;
};
struct ItemPredicate : public std::enable_shared_from_this<ItemPredicate> {
    virtual bool operator()(Item const& item) = 0;
};
struct KeyPredicate : public std::enable_shared_from_this<KeyPredicate> {
    virtual bool operator()(std::string_view sv) = 0;
};
using ItemTransformHandle = std::shared_ptr<ItemTransform>;
using ItemPredicateHandle = std::shared_ptr<ItemPredicate>;
using KeyPredicateHandle = std::shared_ptr<KeyPredicate>;

// List Types
using StringList = std::vector<std::string>;
using DoubleList = std::vector<double>;
using ItemList = std::vector<Item>;
using KeyList = std::vector<std::string>;
using DatasetList = std::vector<DatasetHandle>;
using SamplerList = std::vector<SamplerHandle>;

using ItemDict = std::map<std::string, Item>;

// Utility Templates
// Concatente multiple std::vector<ElementType>.
template <typename ElementType, typename... Args>
std::vector<ElementType> concatenate_vectors(std::vector<ElementType> v,
                                             Args... args) {
    (v.insert(v.end(), std::make_move_iterator(args.begin()),
              std::make_move_iterator(args.end())),
     ...);
    return v;
}

// Concatenate multiple std::array<ElementType>.
template <typename ElementType, std::size_t... Sizes>
constexpr auto concatenate_array(
    const std::array<ElementType, Sizes>&... arrays) {
    std::array<ElementType, (Sizes + ...)> result;
    std::size_t index{};

    ((std::copy_n(arrays.begin(), Sizes, result.begin() + index),
      index += Sizes),
     ...);

    return result;
}

// Gather all the keys of the given type in an item, and return as a vector.
template <typename Type> inline KeyList gather_keys_of_type(Item const& it) {
    KeyList keys;
    for (auto& [key, value] : it) {
        if (std::holds_alternative<Type>(value)) {
            keys.push_back(key);
        }
    }
    return keys;
}

}  // namespace data