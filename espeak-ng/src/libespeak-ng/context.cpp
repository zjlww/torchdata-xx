/*
 * Copyright (C) 2005 to 2013 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2013-2017 Reece H. Dunn
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see: <http://www.gnu.org/licenses/>.
 */

#include "config.h"

#include "context.hpp"
#include <cstdlib>

espeak::context_t& espeak::context_t::global() {
    static thread_local espeak::context_t ctx {};
    return ctx;
}

espeak_ng_ERROR_CONTEXT espeak::context_t::GetError() {
    auto ctx = error_context;
    error_context = nullptr;
    return ctx;
}

espeak::context_t::~context_t() {
    free(voices);
}

#pragma GCC visibility push(default)

ESPEAK_NG_API espeak_ng_CONTEXT* espeak_ng_ctx_New(void) {
    return new espeak_ng_CONTEXT {};
}

ESPEAK_NG_API void espeak_ng_ctx_Free(espeak_ng_CONTEXT*ctx) {
    delete ctx;
}

ESPEAK_NG_API espeak_ng_ERROR_CONTEXT espeak_ng_ctx_GetError(espeak_ng_CONTEXT *ctx) {
    return ctx->GetError();
}

ESPEAK_NG_API const espeak_VOICE **espeak_ng_ctx_ListVoices(espeak_ng_CONTEXT *ctx, espeak_VOICE *voice_spec) {
    return ctx->ListVoices(voice_spec);
}

ESPEAK_NG_API void espeak_ng_ctx_InitializePath(espeak_ng_CONTEXT *ctx, const char *path) {
    return ctx->InitializePath(path);
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_Initialize(espeak_ng_CONTEXT *ctx) {
    return ctx->Initialize();
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_InitializeOutput(
  espeak_ng_CONTEXT *ctx,
  espeak_ng_OUTPUT_MODE output_mode,
  int buffer_length,
  const char *device
) {
    return ctx->InitializeOutput(output_mode, buffer_length, device);
}
ESPEAK_NG_API int espeak_ng_ctx_GetSampleRate(espeak_ng_CONTEXT *ctx) {
    return ctx->GetSampleRate();
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetRandSeed(espeak_ng_CONTEXT *ctx, long seed) {
    return ctx->SetRandSeed(seed);
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetParameter(espeak_ng_CONTEXT *ctx, espeak_PARAMETER parameter, int value, int relative) {
    return ctx->SetParameter(parameter, value, relative);
}
ESPEAK_NG_API int espeak_ng_ctx_GetParameter(espeak_ng_CONTEXT *ctx, espeak_PARAMETER parameter, int current) {
    return ctx->GetParameter(parameter, current);
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetPhonemeEvents(espeak_ng_CONTEXT *ctx, int enable, int ipa) {
    return ctx->SetPhonemeEvents(enable, ipa);
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetPunctuationList(espeak_ng_CONTEXT *ctx, const wchar_t *punctlist) {
    return ctx->SetPunctuationList(punctlist);
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetVoiceByName(espeak_ng_CONTEXT *ctx, const char *name) {
    return ctx->SetVoiceByName(name);
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetVoiceByFile(espeak_ng_CONTEXT *ctx, const char *filename) {
    return ctx->SetVoiceByFile(filename);
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetVoiceByProperties(espeak_ng_CONTEXT *ctx, espeak_VOICE *voice_selector) {
    return ctx->SetVoiceByProperties(voice_selector);
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SpeakKeyName(espeak_ng_CONTEXT *ctx, const char *key_name) {
    return ctx->SpeakKeyName(key_name);
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SpeakCharacter(espeak_ng_CONTEXT *ctx, wchar_t character) {
    return ctx->SpeakCharacter(character);
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_Cancel(espeak_ng_CONTEXT *ctx) {
    return ctx->Cancel();
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_Synchronize(espeak_ng_CONTEXT *ctx) {
    return ctx->Synchronize();
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_Terminate(espeak_ng_CONTEXT *ctx) {
    return ctx->Terminate();
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetOutputHooks(espeak_ng_CONTEXT *ctx, espeak_ng_OUTPUT_HOOKS* hooks) {
    return ctx->SetOutputHooks(hooks);
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetConstF0(espeak_ng_CONTEXT *ctx, int f0) {
    return ctx->SetConstF0(f0);
}
ESPEAK_NG_API const char* espeak_ng_ctx_TextToPhonemes(espeak_ng_CONTEXT *ctx, const void **textptr, int textmode, int phonememode) {
    return ctx->TextToPhonemes(textptr, textmode, phonememode);
}

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetSynthCallback(espeak_ng_CONTEXT *ctx, t_espeak_callback* SynthCallback) {
    return ctx->SetSynthCallback(SynthCallback);
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetUriCallback(espeak_ng_CONTEXT *ctx, int (*UriCallback)(int, const char*, const char*)) {
    return ctx->SetUriCallback(UriCallback);
}
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetPhonemeCallback(espeak_ng_CONTEXT *ctx, int (*PhonemeCallback)(const char *)) {
    return ctx->SetPhonemeCallback(PhonemeCallback);
}

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_Synthesize(
  espeak_ng_CONTEXT *ctx,
  const void *text,
  size_t size,
  unsigned int position,
  espeak_POSITION_TYPE position_type,
  unsigned int end_position,
  unsigned int flags,
  unsigned int *unique_identifier,
  void *user_data
) {
    return ctx->Synthesize(text, size, position, position_type, end_position, flags, unique_identifier, user_data);
}

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SynthesizeMark(
  espeak_ng_CONTEXT *ctx,
  const void *text,
  size_t size,
  const char *index_mark,
  unsigned int end_position,
  unsigned int flags,
  unsigned int *unique_identifier,
  void *user_data
) {
    return ctx->SynthesizeMark(text, size, index_mark, end_position, flags, unique_identifier, user_data);
}

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_CompileDictionary(
  espeak_ng_CONTEXT *ctx,
  const char *dsource,
  const char *dict_name,
  FILE *log,
  int flags
) {
    return ctx->CompileDictionary(dsource, dict_name, log, flags);
}

#if USE_MBROLA
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_CompileMbrolaVoice(
  espeak_ng_CONTEXT *ctx,
  const char *path,
  FILE *log
) {
    return ctx->CompileMbrolaVoice(path, log);
}
#endif

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_CompilePhonemeDataPath(
  espeak_ng_CONTEXT *ctx,
  long rate,
  const char *source_path,
  const char *destination_path,
  FILE *log
) {
    return ctx->CompilePhonemeDataPath(rate, source_path, destination_path, log);
}

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_CompileIntonationPath(
  espeak_ng_CONTEXT *ctx,
  const char *source_path,
  const char *destination_path,
  FILE *log
) {
    return ctx->CompileIntonationPath(source_path, destination_path, log);
}

#pragma GCC visibility pop
