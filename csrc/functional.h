#pragma once
#include "types.h"

/*
This file contains some example transformations of Items. The user can implement
any custom function that binds to ItemTransform, i.e. std::function<Item(Item)>.
*/

namespace data {

ItemTransformHandle roll(std::string key, int dim, int shift);
ItemTransformHandle randomRoll(std::string key, int dim, int shiftMin,
                               int shiftMax);
ItemTransformHandle rightPadSequenceFrame(std::string key, std::string frameKey,
                                          int dim, int frameSize);
ItemTransformHandle rightTruncateSequenceFrame(std::string key,
                                               std::string frameKey, int dim,
                                               int frameSize);
ItemTransformHandle addInt64(std::string keyA, std::string keyB,
                             std::string keyC, int64_t bias);
ItemTransformHandle readFile(std::string pathKey, std::string textKey);
}  // namespace data