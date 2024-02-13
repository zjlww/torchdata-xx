#pragma once

#include <cstdint>
#include <regex>
#include <string>
#include <string_view>

#include "text/en_data.h"
#include "types.h"

namespace data {
// Trim initial and terminal spaces.
std::string trim(std::string_view sv);

// Encode UTF-8 IPA sequence into two tensors.
// First IntTensor [N_phone] contains the phoneme IDs.
// Second IntTensor [N_phone, N_extra] encodes the prefix / suffix /
// punctuations.
std::pair<Tensor, Tensor> encodeIPA(std::string const& IPA);

ItemTransformHandle encodeIPATransform(std::string IPAKey,
                                       std::string phoneIDKey,
                                       std::string extraKey,
                                       std::string nPhoneKey);

}  // namespace data
