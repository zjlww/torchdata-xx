#include "text/utils.h"

#include <cstdint>
#include <regex>
#include <string>
#include <string_view>

#include "text/en_data.h"
#include "types.h"

namespace data {
std::string trim(std::string_view sv) {
    static auto regex_trim = std::regex("^\\s+|\\s+$");
    return std::regex_replace(sv.data(), regex_trim, "");
}

// Encode IPA string int a 1D array of symbol ID, and a 2D array of symbol
// property flags.
std::pair<Tensor, Tensor> encodeIPA(std::string const& IPA) {
    static auto token_regex =
        std::regex(_puncspace_regex() + "|" + _phone_regex());

    std::vector<std::string> unit_list{}, extra_list{};

    auto iter = std::sregex_token_iterator(IPA.begin(), IPA.end(), token_regex,
                                           {0, 1, 2, 3, 4});
    auto end = std::sregex_token_iterator();
    while (iter != end) {
        auto piece = iter->str();
        ++iter;
        auto sep = iter->str();
        ++iter;
        auto prefix = iter->str();
        ++iter;
        auto phone = iter->str();
        ++iter;
        auto suffix = iter->str();
        ++iter;

        if (sep.size() > 0) {
            unit_list.emplace_back(" ");
            extra_list.push_back(std::move(trim(piece)));
        } else if (phone.size() > 0) {
            unit_list.push_back(std::move(phone));
            extra_list.push_back(prefix + suffix);
        }
    }

    const int L = static_cast<int>(unit_list.size());

    // Encode the units into a vector:
    std::vector<int> _A(L, 0);
    for (int idx = 0; idx < L; ++idx) {
        auto const& unit = unit_list[idx];
        auto unit_code = symbol_to_id.at(unit);
        _A[idx] = unit_code;
    }
    Tensor phoneID = to_tensor<int, torch::kInt32>(_A);

    // Encode extra IPA features into a matrix:
    static auto extra_regex = std::regex(or_pattern(extras));
    std::vector<int8_t> _P(L * N_EXTRA, 0);

    for (int idx = 0; idx < L; ++idx) {
        auto const& s = extra_list[idx];
        auto iter =
            std::sregex_token_iterator(s.begin(), s.end(), extra_regex, 0);
        auto end = std::sregex_token_iterator();
        while (iter != end) {
            auto extra = iter->str();
            auto extra_code = extra_to_id.at(extra);
            _P[idx * N_EXTRA + extra_code] += 1;
            ++iter;
        }
    }
    Tensor extra = to_tensor<int8_t, torch::kInt8>(_P);
    extra = extra.reshape({L, N_EXTRA});
    return std::pair(std::move(phoneID), std::move(extra));
}

struct EncodeIPATransform final : ItemTransform {
    std::string IPAKey;
    std::string phoneIDKey;
    std::string extraKey;
    std::string nPhoneKey;
    EncodeIPATransform(std::string IPAKey, std::string phoneIDKey,
                       std::string extraKey, std::string nPhoneKey)
        : IPAKey{IPAKey},
          phoneIDKey{phoneIDKey},
          extraKey{extraKey},
          nPhoneKey{nPhoneKey} {}
    Item operator()(Item item) override {
        const auto& IPA = std::get<std::string>(item[IPAKey]);
        auto [phoneID, extra] = encodeIPA(IPA);
        item[nPhoneKey] = phoneID.numel();
        item[phoneIDKey] = phoneID;
        item[extraKey] = extra;
        return item;
    }
};

ItemTransformHandle encodeIPATransform(std::string IPAKey,
                                       std::string phoneIDKey,
                                       std::string extraKey,
                                       std::string nPhoneKey) {
    return std::make_shared<EncodeIPATransform>(IPAKey, phoneIDKey, extraKey,
                                                nPhoneKey);
}

}  // namespace data