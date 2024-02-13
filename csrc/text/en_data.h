#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

#include "types.h"

namespace data {

using namespace std::literals;

using symbol_tab = std::map<std::string_view, size_t>;

// Assign integer IDs to symbols in symbol_array, and store the
// symbol to ID map in a table (symbol_tab).
inline auto init_symbol_tab(auto const& symbol_array) {
    symbol_tab tab;
    for (int i = 0; i < symbol_array.size(); ++i) {
        tab.insert(std::make_pair(symbol_array[i], i));
    }
    return tab;
}

// Given an array of symbols, concatenate them into a regex subpattern,
// of form "(a|b|c)". All symbols are escaped.
template <typename T, std::size_t N>
std::string or_pattern(std::array<T, N> const& arr,
                       std::string_view escape = "\\"sv) {
    std::string p = "(";
    for (auto r : arr) {
        p += std::string(escape) + r + R"(|)"s;
    }
    p.pop_back();
    p += ")";
    return p;
}

constexpr inline auto upper_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"sv;
constexpr inline auto lower_letters = "abcdefghijklmnopqrstuvwxyz"sv;

// HACK: < is used for BOS and > is used for EOS.
constexpr inline std::array punctuations{"-",  "!", "?", ";", ":",
                                         "\"", ",", ".", "<", ">"};

constexpr inline std::array vowels{
    "a", "ɶ", "ɑ", "ɒ", "æ", "ɐ", "ɛ", "œ", "ɜ", "ɞ", "ʌ", "ɔ", "ə", "e", "ø",
    "ɘ", "ɵ", "ɤ", "o", "ɪ", "ʏ", "ʊ", "i", "y", "ɨ", "ʉ", "ɯ", "u", "ɚ", "ɝ"};

constexpr inline std::array consonants_pulmonic{
    "p", "b", "t", "d", "ʈ", "ɖ", "c", "ɟ", "k", "ɡ", "q", "ɢ", "ʔ", "m", "ɱ",
    "n", "ɳ", "ɲ", "ŋ", "ɴ", "ʙ", "r", "ʀ", "ⱱ", "ɾ", "ɽ", "ɸ", "β", "f", "v",
    "θ", "ð", "s", "z", "ʃ", "ʒ", "ʂ", "ʐ", "ç", "ʝ", "x", "ɣ", "χ", "ʁ", "ħ",
    "ʕ", "h", "ɦ", "ɬ", "ɮ", "ʋ", "ɹ", "ɻ", "j", "ɰ", "l", "ɭ", "ʎ", "ʟ",
};

constexpr inline std::array consonants_non_pulmonic{
    "ʘ", "ǀ", "ǃ", "ǂ", "ǁ", "ɓ", "ɗ", "ʄ", "ɠ", "ʛ",
};

constexpr inline std::array other_ipa{
    "ʍ", "ɕ", "ʑ", "w", "ɺ", "ɥ", "ɧ", "ʜ", "ʡ", "ʢ", "ɫ", "ᵻ",
};

constexpr inline std::array diphthongs{"ɔɪ", "eɪ", "aʊ", "oʊ",
                                       "eʊ", "oɪ", "əʊ", "aɪ"};

constexpr inline std::array affricates{"tʃ", "ts", "dʒ"};

constexpr inline auto phones =
    concatenate_array(diphthongs, affricates, vowels, consonants_pulmonic,
                      consonants_non_pulmonic, other_ipa);

constexpr inline auto symbols = concatenate_array(std::array{"_", " "}, phones);

constexpr size_t N_SYMBOLS = symbols.size();

inline auto symbol_to_id = init_symbol_tab(symbols);

// [0] Combining vertical line below (IPA Syllabic consonant).
// [1] Combining tilde (IPA Nasalization).
// [2] Modifier letter small j (Palatalized).
constexpr inline std::array diacritics{"\U00000329", "\U00000303", "ʲ"};
constexpr inline std::array durations{"ː"};
constexpr inline std::array stresses{"ˈ", "ˌ"};
constexpr inline auto prefixes = stresses;
constexpr inline auto suffixes = concatenate_array(diacritics, durations);
constexpr inline auto extras =
    concatenate_array(prefixes, suffixes, punctuations);
constexpr size_t N_EXTRA = extras.size();

inline auto extra_to_id = init_symbol_tab(extras);

// This function generates an regex for matching IPA phonemes.
// A phoneme is prefix{0,1} + phone + suffix{0,1}.
// TODO: Currently only supports a single suffix character. You'll need to
// extend the code if you need to handle more general phonemes.
inline auto _phone_regex() {
    static auto prefix_pattern = or_pattern(prefixes, "") + "?";
    static auto phone_pattern = or_pattern(phones, "");
    static auto suffix_pattern = or_pattern(suffixes, "") + "?";
    return prefix_pattern + phone_pattern + suffix_pattern;
}

// Generates an regex for matching consecutive punctuations / spaces.
inline auto _puncspace_regex() {
    static auto p =
        or_pattern(concatenate_array(std::array{" "}, punctuations));
    return p + "+";
}

}  // namespace data