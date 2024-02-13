#include <getopt.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "espeak_ng_ctx.h"

#ifndef PROGRAM_NAME
#    define PROGRAM_NAME "espeak-ng"
#endif

namespace espeak {
extern ESPEAK_NG_API void strncpy0(char* to, const char* from, int size);
extern ESPEAK_NG_API int utf8_in(int* c, const char* buf);
}  // namespace espeak

using namespace espeak;

int main(int argc, char** argv) {
    static const struct option long_options[] = {
        {"compile", optional_argument, 0, 0x102},
        {"path", required_argument, 0, 0x107},
        {"compile-intonations", no_argument, 0, 0x10f},
        {"compile-phonemes", optional_argument, 0, 0x110},
        {0, 0, 0, 0}};

    espeak_VOICE voice_select;
    char* data_path = NULL;  // use default path for espeak-ng-data
    int flag_compile = 0;
    int option_index = 0;
    int c;
    char* optarg2;

    char voicename[40];

    while (true) {
        c = getopt_long(argc, argv, "a:b:Dd:f:g:hk:l:mp:P:qs:v:w:xXz",
                        long_options, &option_index);

        // Detect the end of the options.
        if (c == -1) break;
        optarg2 = optarg;

        switch (c) {
        case 0x102:  // --compile
            if (optarg2 != NULL && *optarg2) {
                strncpy0(voicename, optarg2, sizeof(voicename));
                flag_compile = c;
                break;
            } else {
                fprintf(stderr, "Voice name to '%s' not specified.\n",
                        c == 0x101 ? "--compile-debug" : "--compile");
                exit(EXIT_FAILURE);
            }
        case 0x107:  // --path
            data_path = optarg2;
            break;
        case 0x10f:  // --compile-intonations
        {
            espeak_ng_InitializePath(data_path);
            espeak_ng_ERROR_CONTEXT context = NULL;
            espeak_ng_STATUS result =
                espeak_ng_CompileIntonation(stdout, &context);
            if (result != ENS_OK) {
                espeak_ng_PrintStatusCodeMessage(result, stderr, context);
                espeak_ng_ClearErrorContext(&context);
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }
        case 0x110:  // --compile-phonemes
        {
            espeak_ng_InitializePath(data_path);
            espeak_ng_ERROR_CONTEXT context = NULL;
            espeak_ng_STATUS result;
            if (optarg2) {
                result = espeak_ng_CompilePhonemeDataPath(22050, optarg2, NULL,
                                                          stdout, &context);
            } else {
                result = espeak_ng_CompilePhonemeData(22050, stdout, &context);
            }
            if (result != ENS_OK) {
                espeak_ng_PrintStatusCodeMessage(result, stderr, context);
                espeak_ng_ClearErrorContext(&context);
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }
        default:
            exit(0);
        }
    }

    espeak_ng_InitializePath(data_path);
    espeak_ng_ERROR_CONTEXT context = NULL;
    espeak_ng_STATUS result = espeak_ng_Initialize(&context);
    if (result != ENS_OK) {
        espeak_ng_PrintStatusCodeMessage(result, stderr, context);
        espeak_ng_ClearErrorContext(&context);
        exit(1);
    }

    if (voicename[0] == 0) strcpy(voicename, ESPEAKNG_DEFAULT_VOICE);

    result = espeak_ng_SetVoiceByName(voicename);
    if (result != ENS_OK) {
        memset(&voice_select, 0, sizeof(voice_select));
        voice_select.languages = voicename;
        result = espeak_ng_SetVoiceByProperties(&voice_select);
        if (result != ENS_OK) {
            espeak_ng_PrintStatusCodeMessage(result, stderr, NULL);
            exit(EXIT_FAILURE);
        }
    }

    if (flag_compile) {
        // This must be done after the voice is set
        espeak_ng_ERROR_CONTEXT context = NULL;
        espeak_ng_STATUS result = espeak_ng_CompileDictionary(
            "", NULL, stderr, flag_compile & 0x1, &context);
        if (result != ENS_OK) {
            espeak_ng_PrintStatusCodeMessage(result, stderr, context);
            espeak_ng_ClearErrorContext(&context);
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    espeak_ng_Terminate();
    return 0;
}
