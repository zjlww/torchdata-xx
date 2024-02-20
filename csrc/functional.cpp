#include "functional.h"

#include <fstream>
#include <memory>

#include "types.h"

namespace data {
struct Roll final : ItemTransform {
    std::string key;
    int dim{};
    int shift{};
    Roll(std::string key, int dim, int shift)
        : key{key}, dim{dim}, shift{shift} {}
    Item operator()(Item item) override {
        Tensor t = std::get<Tensor>(item[key]);
        item[key] = torch::roll(t, {shift}, {dim});
        return item;
    }
};

ItemTransformHandle roll(std::string key, int dim, int shift) {
    return std::make_shared<Roll>(key, dim, shift);
}

struct RandomRoll final : ItemTransform {
    std::string key;
    int dim;
    int shiftMin;
    int shiftMax;
    RandomRoll(std::string key, int dim, int shiftMin, int shiftMax)
        : key{key}, dim{dim}, shiftMin{shiftMin}, shiftMax{shiftMax} {}
    Item operator()(Item item) override {
        static thread_local auto rng = std::mt19937(std::random_device()());

        Tensor t = std::get<Tensor>(item[key]);
        auto dist = std::uniform_int_distribution<int>(shiftMin, shiftMax);
        auto shift = dist(rng);
        item[key] = torch::roll(t, {shift}, {dim});
        return item;
    }
};

ItemTransformHandle randomRoll(std::string key, int dim, int shiftMin,
                               int shiftMax) {
    return std::make_shared<RandomRoll>(key, dim, shiftMin, shiftMax);
}

struct RightPadSequenceFrame final : ItemTransform {
    std::string key;
    std::string frameKey;
    int dim;
    int frameSize;

    RightPadSequenceFrame(std::string key, std::string frameKey, int dim,
                          int frameSize)
        : key{key}, frameKey{frameKey}, dim{dim}, frameSize{frameSize} {}

    Item operator()(Item item) override {
        Tensor t = std::get<Tensor>(item[key]);
        int n = t.size(dim);
        int m = (n + frameSize - 1) / frameSize;
        int n_pad = m * frameSize - n;
        item[frameKey] = m;
        if (n_pad == 0) {
            return item;
        }
        auto sz = t.sizes();
        auto nsz = std::vector<int64_t>(sz.begin(), sz.end());
        nsz[dim] = n_pad;
        auto p =
            t.new_zeros(torch::IntArrayRef(nsz.data(), nsz.data() + sz.size()));
        item[key] = torch::cat({t, p}, dim);
        return item;
    }
};

ItemTransformHandle rightPadSequenceFrame(std::string key, std::string frameKey,
                                          int dim, int frameSize) {
    return std::make_shared<RightPadSequenceFrame>(key, frameKey, dim,
                                                   frameSize);
}

struct RightTruncateSequenceFrame final : ItemTransform {
    std::string key;
    std::string frameKey;
    int dim;
    int frameSize;
    RightTruncateSequenceFrame(std::string key, std::string frameKey, int dim,
                               int frameSize)
        : key{key}, frameKey{frameKey}, dim{dim}, frameSize{frameSize} {}
    Item operator()(Item item) override {
        Tensor t = std::get<Tensor>(item[key]);
        int n = t.size(dim);
        int m = n / frameSize;
        int nTrunc = m * frameSize;
        item[key] = t.slice(dim, 0, nTrunc);
        item[frameKey] = m;
        return item;
    }
};

ItemTransformHandle rightTruncateSequenceFrame(std::string key,
                                               std::string frameKey, int dim,
                                               int frameSize) {
    return std::make_shared<RightTruncateSequenceFrame>(key, frameKey, dim,
                                                        frameSize);
}

struct AddInt64 final : ItemTransform {
    std::string keyA;
    std::string keyB;
    std::string keyC;
    int64_t bias{0};
    AddInt64(std::string keyA, std::string keyB, std::string key_c,
             int64_t bias)
        : keyA(keyA), keyB(keyB), keyC(key_c), bias(bias) {}
    Item operator()(Item item) override {
        auto a = std::get<int64_t>(item[keyA]);
        auto b = std::get<int64_t>(item[keyB]);
        item[keyC] = a + b + bias;
        return item;
    }
};

ItemTransformHandle addInt64(std::string keyA, std::string keyB,
                             std::string keyC, int64_t bias) {
    return std::make_shared<AddInt64>(keyA, keyB, keyC, bias);
}

struct ReadFile final : ItemTransform {
    std::string pathKey;
    std::string textKey;
    ReadFile(std::string pathKey, std::string textKey)
        : pathKey{pathKey}, textKey{textKey} {}
    Item operator()(Item item) override {
        auto p = std::get<std::string>(item.at(pathKey));
        std::ifstream file(p);
        if (!file) {
            throw std::runtime_error("Could not open file: " + p);
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        item.emplace(textKey, std::move(content));
        return item;
    }
};

ItemTransformHandle readFile(std::string pathKey, std::string textKey) {
    return std::make_shared<ReadFile>(pathKey, textKey);
}

struct TotalLength final : ItemTransform {
    Item operator()(Item item) override {
        auto n_phone = std::get<int64_t>(item["n_phone"]);
        auto n_frame = std::get<int64_t>(item["n_frame"]);
        item["n_total"] = n_phone + 2 * n_frame;
        return item;
    }
};

ItemTransformHandle addTotalLength() { return std::make_shared<TotalLength>(); }

struct TotalLengthWithRef final : ItemTransform {
    Item operator()(Item item) override {
        auto n_phone = std::get<int64_t>(item["n_phone"]);
        auto n_frame = std::get<int64_t>(item["n_frame"]);
        auto n_frame_ref = std::get<int64_t>(item["n_frame_ref"]);
        item["n_total"] = n_phone + 2 * n_frame + n_frame_ref;
        return item;
    }
};

ItemTransformHandle addTotalLengthWithRef() {
    return std::make_shared<TotalLengthWithRef>();
};
}  // namespace data