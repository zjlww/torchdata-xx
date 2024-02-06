#pragma once
#include <ATen/core/ATen_fwd.h>
#include <torch/torch.h>

#include <cstdint>
#include <stdexcept>

#include "types.h"

namespace data {
struct TensorBuffer {
    int64_t dim{0};
    IValue buffer{};

    TensorBuffer(){};
    TensorBuffer(int64_t dim) : dim{dim} {};

    void push(Tensor t) {
        if (buffer.isNone()) {
            buffer = t;
        } else {
            buffer = torch::cat({buffer.toTensor(), t}, dim);
        }
    }

    int size() {
        if (buffer.isNone()) {
            return 0;
        } else {
            return buffer.toTensor().size(dim);
        }
    }

    Tensor pop(int64_t n) {
        auto b = buffer.toTensor();
        if (n > b.size(dim)) {
            throw std::out_of_range("TensorBuffer pop out of range");
        }
        auto a = b.slice(dim, 0, n);
        auto c = b.slice(dim, n);
        buffer = c;
        return a;
    }
};

inline Tensor padTensor(Tensor t, int64_t dim, int64_t len) {
    int cur_len = t.size(dim);
    int pad_len = len - cur_len;
    if (pad_len == 0) return t;
    auto size = t.sizes();
    auto p_size = std::vector<int64_t>(size.begin(), size.end());
    p_size[dim] = pad_len;
    Tensor p = t.new_zeros(torch::IntArrayRef(p_size));
    return torch::cat({t, p}, dim);
}

inline TensorList padTensorList(TensorList ts, int64_t dim, int64_t len) {
    for (int i = 0; i < ts.size(); ++i) ts[i] = padTensor(ts[i], dim, len);
    return ts;
}

inline Tensor pad_sequence(TensorList ts, int64_t dim, int64_t len) {
    TensorList nts = padTensorList(ts, dim, len);
    return torch::stack(nts, 0);
}

}  // namespace data