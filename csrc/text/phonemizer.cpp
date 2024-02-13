#include "text/phonemizer.h"

#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>

#include "speak_lib.h"
#include "text/en_data.h"
#include "text/utils.h"

namespace data {

espeak_phonemizer::espeak_phonemizer() {
    auto rate = espeak_Initialize(
        espeak_AUDIO_OUTPUT::AUDIO_OUTPUT_SYNCH_PLAYBACK, 0, nullptr, 0);
    auto ret = espeak_SetVoiceByName("en-us");
    if (ret != EE_OK) {
        throw std::runtime_error("failed to initialize phonemizer");
    }
}

// Phonemize UTF-8 input string, returns IPA in UTF-8.
// This is a wrapper around the ESPEAK-NG API.
std::string espeak_phonemizer::phonemize_segment(std::string_view sv) {
    auto text = sv.data();
    std::string result;
    char const** pt = &text;
    char const* end = *pt + sv.length();
    do {
        char const* out = espeak_TextToPhonemes(
            reinterpret_cast<void const**>(pt), espeakCHARS_UTF8, 2);
        result += out;
    } while (*pt != nullptr);
    return result;
}

std::string _segment_pattern() {
    std::string pattern = "(";
    for (std::string punc : punctuations) {
        if (punc != "-"s) {
            pattern += "\\" + punc + "|";
        }
    }
    pattern += R"(\s+\-|\-\s+|\s+\-\s+))";
    return pattern;
}

// Phonemize complete strings.
std::string espeak_phonemizer::phonemize(std::string_view sv) {
    auto s = trim(sv);
    s = " " + s + " ";

    // Preprocessing:
    static auto regex_dash = std::regex("—");
    s = std::regex_replace(s, regex_dash, " -- ");

    // Hack: Hyphen requires special processing, we view foo-bar as a single
    // word, and foo- bar, foo - bar, foo -bar as separated words. This
    // complicates the regex pattern.

    static auto segment_regex = std::regex(_segment_pattern());

    // Split the sequence s at punctuations that has surrounding spaces.
    // And phonemize each segment.
    // Finally paste each results as a long sequence.
    std::string result;
    std::sregex_token_iterator iter(s.begin(), s.end(), segment_regex, {-1, 0});
    std::sregex_token_iterator end;
    while (iter != end) {
        result += phonemize_segment(iter->str());
        ++iter;
        if (iter == end) break;
        result += iter->str();
        ++iter;
    }
    return trim(result);
}

}  // namespace data