
/* eSpeak NG API.
 *
 * Copyright (C) 2022 Yury Popov <git@phoenix.dj>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef ESPEAK_NG_CTX_H
#define ESPEAK_NG_CTX_H

#include "espeak_ng.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Contextual API */

typedef struct espeak_ng_CONTEXT espeak_ng_CONTEXT;

ESPEAK_NG_API espeak_ng_CONTEXT* espeak_ng_ctx_New(void);
ESPEAK_NG_API void espeak_ng_ctx_Free(espeak_ng_CONTEXT* ctx);

ESPEAK_NG_API espeak_ng_ERROR_CONTEXT
espeak_ng_ctx_GetError(espeak_ng_CONTEXT* ctx);

ESPEAK_NG_API const espeak_VOICE** espeak_ng_ctx_ListVoices(
    espeak_ng_CONTEXT* ctx, espeak_VOICE* voice_spec);
ESPEAK_NG_API void espeak_ng_ctx_InitializePath(espeak_ng_CONTEXT* ctx,
                                                const char* path);
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_Initialize(espeak_ng_CONTEXT* ctx);
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_InitializeOutput(
    espeak_ng_CONTEXT* ctx, espeak_ng_OUTPUT_MODE output_mode,
    int buffer_length, const char* device);
ESPEAK_NG_API int espeak_ng_ctx_GetSampleRate(espeak_ng_CONTEXT* ctx);
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetRandSeed(espeak_ng_CONTEXT* ctx,
                                                         long seed);
ESPEAK_NG_API espeak_ng_STATUS
espeak_ng_ctx_SetParameter(espeak_ng_CONTEXT* ctx, espeak_PARAMETER parameter,
                           int value, int relative);
ESPEAK_NG_API int espeak_ng_ctx_GetParameter(espeak_ng_CONTEXT* ctx,
                                             espeak_PARAMETER parameter,
                                             int current);
ESPEAK_NG_API espeak_ng_STATUS
espeak_ng_ctx_SetPhonemeEvents(espeak_ng_CONTEXT* ctx, int enable, int ipa);
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetPhonemeTrace(
    espeak_ng_CONTEXT* ctx, int phonememode, FILE* stream);
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetPunctuationList(
    espeak_ng_CONTEXT* ctx, const wchar_t* punctlist);
ESPEAK_NG_API espeak_ng_STATUS
espeak_ng_ctx_SetVoiceByName(espeak_ng_CONTEXT* ctx, const char* name);
ESPEAK_NG_API espeak_ng_STATUS
espeak_ng_ctx_SetVoiceByFile(espeak_ng_CONTEXT* ctx, const char* filename);
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetVoiceByProperties(
    espeak_ng_CONTEXT* ctx, espeak_VOICE* voice_selector);
ESPEAK_NG_API espeak_ng_STATUS
espeak_ng_ctx_SpeakKeyName(espeak_ng_CONTEXT* ctx, const char* key_name);
ESPEAK_NG_API espeak_ng_STATUS
espeak_ng_ctx_SpeakCharacter(espeak_ng_CONTEXT* ctx, wchar_t character);
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_Cancel(espeak_ng_CONTEXT* ctx);
ESPEAK_NG_API espeak_ng_STATUS
espeak_ng_ctx_Synchronize(espeak_ng_CONTEXT* ctx);
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_Terminate(espeak_ng_CONTEXT* ctx);
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetOutputHooks(
    espeak_ng_CONTEXT* ctx, espeak_ng_OUTPUT_HOOKS* hooks);
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetConstF0(espeak_ng_CONTEXT* ctx,
                                                        int f0);
ESPEAK_NG_API const char* espeak_ng_ctx_TextToPhonemes(espeak_ng_CONTEXT* ctx,
                                                       const void** textptr,
                                                       int textmode,
                                                       int phonememode);

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetSynthCallback(
    espeak_ng_CONTEXT* ctx, t_espeak_callback* SynthCallback);
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetUriCallback(
    espeak_ng_CONTEXT* ctx, int (*UriCallback)(int, const char*, const char*));
ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SetPhonemeCallback(
    espeak_ng_CONTEXT* ctx, int (*PhonemeCallback)(const char*));

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_Synthesize(
    espeak_ng_CONTEXT* ctx, const void* text, size_t size,
    unsigned int position, espeak_POSITION_TYPE position_type,
    unsigned int end_position, unsigned int flags,
    unsigned int* unique_identifier, void* user_data);

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_SynthesizeMark(
    espeak_ng_CONTEXT* ctx, const void* text, size_t size,
    const char* index_mark, unsigned int end_position, unsigned int flags,
    unsigned int* unique_identifier, void* user_data);

ESPEAK_NG_API espeak_ng_STATUS
espeak_ng_ctx_CompileDictionary(espeak_ng_CONTEXT* ctx, const char* dsource,
                                const char* dict_name, FILE* log, int flags);

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_CompileMbrolaVoice(
    espeak_ng_CONTEXT* ctx, const char* path, FILE* log);

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_CompilePhonemeDataPath(
    espeak_ng_CONTEXT* ctx, long rate, const char* source_path,
    const char* destination_path, FILE* log);

ESPEAK_NG_API espeak_ng_STATUS espeak_ng_ctx_CompileIntonationPath(
    espeak_ng_CONTEXT* ctx, const char* source_path,
    const char* destination_path, FILE* log);

#ifdef __cplusplus
}
#endif

#endif
