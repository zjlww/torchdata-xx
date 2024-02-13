/*
 * Copyright (C) 2005 to 2007 by Jonathan Duddington
 * email: jonsd@users.sourceforge.net
 * Copyright (C) 2013-2015 Reece H. Dunn
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

#pragma once

#include <espeak-ng/espeak_ng_ctx.h>
#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>

#include "config.h"
#include "speech.hpp"
#include "translate.hpp"
#include "phoneme.hpp"
#include "synthesize.hpp"
#include "readclause.hpp"
#include "ssml.hpp"
#include "soundicon.hpp"
#include "wavegen.hpp"

#if USE_LIBSONIC
#include "sonic.h"
#endif

#if USE_KLATT
#include "klatt.hpp"
#endif

#if USE_SPEECHPLAYER
#include "sPlayer.hpp"
#endif

struct audio_object;

namespace espeak {
    struct CompileContext;
    struct Translator;
    struct voice_t;
    struct SSML_STACK;
    struct ALPHABET;
    struct PHONEME_TAB;
    struct WORD_TAB;
    struct FMT_PARAMS;
    struct PHONEME_LIST;
    struct MatchRecord;
    struct t_espeak_command;
    struct frame_t;
    struct SYLLABLE;
    struct wavegen_peaks_t;
    struct SpectFrame;
    struct MBROLA_TAB;

    struct context_t {
    private:
        espeak_ng_ERROR_CONTEXT error_context;

        uint32_t espeak_rand_state = 0;

        int samplerate = 0; // this is set by Wavegeninit()
        int out_samplerate = 0;
        int voice_samplerate = 22050;

        long count_samples;
        #if USE_LIBPCAUDIO
        audio_object *my_audio = NULL;
        #endif

        unsigned int my_unique_identifier = 0;
        void *my_user_data = NULL;
        espeak_ng_OUTPUT_MODE my_mode = ENOUTPUT_MODE_SYNCHRONOUS;
        const int min_buffer_length = 60; // minimum buffer length in ms
        espeak_ng_STATUS async_err = ENS_OK;

        char path_home[N_PATH_HOME]; // this is the espeak-ng-data directory
        char word_phonemes[N_WORD_PHONEMES]; // a word translated into phoneme codes

        // Translator *translator = NULL; // the main translator
        Translator *translator2 = NULL; // secondary translator for certain words
        Translator *translator3 = NULL; // tertiary translator for certain words
        char translator2_language[20] = { 0 };
        char translator3_language[20] = { 0 };

        int ignore_next_n = 0;
        char voice_change_name[40];

        int count_sayas_digits;
        int count_words;
        bool new_sentence;
        int word_emphasis = 0; // set if emphasis level 3 or 4
        int embedded_flag = 0; // there are embedded commands to be applied to the next phoneme, used in TranslateWord2()

        int max_clause_pause = 0;
        bool any_stressed_words;
        ALPHABET *current_alphabet;

        int embedded_ix;
        int embedded_read;

        // the source text of a single clause (UTF8 bytes)
        char source[N_TR_SOURCE+40]; // extra space for embedded command & voice change info at end

        // Several phoneme tables may be loaded into memory. phoneme_tab points to
        // one for the current voice
        // copy the current phoneme table into here
        int n_phoneme_tab;
        PHONEME_TAB *phoneme_tab[N_PHONEME_TAB];

        int n_phoneme_tables;
        PHONEME_TAB_LIST phoneme_tab_list[N_PHONEME_TABS];
        int phoneme_tab_number = 0;
        int current_phoneme_table;

        unsigned short *phoneme_index = NULL;
        char *phondata_ptr = NULL;
        unsigned char *phoneme_tab_data = NULL;

        int n_replace_phonemes;
        REPLACE_PHONEMES replace_phonemes[N_REPLACE_PHONEMES];

        int n_ph_list2;
        PHONEME_LIST2 ph_list2[N_PHONEME_LIST]; // first stage of text->phonemes

        int n_param_stack;
        PARAM_STACK param_stack[N_PARAM_STACK];

        int speech_parameters[N_SPEECH_PARAM]; // Parameters saved on synthesis start
        int saved_parameters[N_SPEECH_PARAM]; // Parameters saved on synthesis start

        int embedded_value[N_EMBEDDED_VALUES];

        int n_digit_lookup;
        char *digit_lookup;
        int speak_missing_thousands;
        int number_control;
        char ph_ordinal2[12];
        char ph_ordinal2x[12];

        // list of phonemes in a clause
        int n_phoneme_list;
        PHONEME_LIST phoneme_list[N_PHONEME_LIST+1];

        unsigned int embedded_list[N_EMBEDDED_LIST];

        int n_soundicon_tab;
        SOUND_ICON soundicon_tab[N_SOUNDICON_TAB];

        int n_tunes;
        TUNE *tunes;

        voice_t voicedata;
        voice_t *voice = &voicedata;

        #define N_VOICES_LIST  350
        int n_voices_list = 0;
        espeak_VOICE *voices_list[N_VOICES_LIST];
        espeak_VOICE current_voice_selected;

        espeak_VOICE **voices = NULL;
        char voice_id[50];
        char voice_identifier[40]; // file name for  current_voice_selected
        char voice_name[40];       // voice name for current_voice_selected
        char ssml_voice_name[40];
        char voice_languages[100]; // list of languages and priorities for current_voice_selected
        char variant_name[40];
        espeak_VOICE voice_variants[N_VOICE_VARIANTS];
        voice_t *new_voice = NULL;

        espeak_VOICE base_voice;
        char base_voice_variant_name[40] = { 0 };
        char current_voice_id[40] = { 0 };

        frameref_t frames_buf[N_SEQ_FRAMES];

        int frame_pool_ix = 0;
        frame_t frame_pool[N_FRAME_POOL];

        int tone_points[12] = { 600, 170, 1200, 135, 2000, 110, 3000, 110, -1, 0 };

        unsigned char *out_ptr;
        unsigned char *out_end;
        unsigned char *outbuf = NULL;
        unsigned char *out_start;
        int outbuf_size = 0;

        int seq_len_adjust; // temporary fix to advance the start point for playing the wav sample

        unsigned char *wavefile_data = NULL;

        int echo_head;
        int echo_tail;
        int echo_amp;
        short echo_buf[N_ECHO_BUF];
        int echo_length = 0; // period (in sample\) to ensure completion of echo at the end of speech, set in WavegenSetEcho()

        int formant_rate[9]; // max rate of change of each formant, values adjusted for actual sample rate
        SPEED_FACTORS speed;

        espeak_ng_OUTPUT_HOOKS* output_hooks = NULL;

        // waveform shape table for HF peaks, formants 6,7,8
        int wavemult_offset = 0;
        int wavemult_max = 0;

        // the presets are for 22050 Hz sample rate.
        // A different rate will need to recalculate the presets in WavegenInit()
        unsigned char wavemult[N_WAVEMULT] = {
            0,   0,   0,   2,   3,   5,   8,  11,  14,  18,  22,  27,  32,  37,  43,  49,
            55,  62,  69,  76,  83,  90,  98, 105, 113, 121, 128, 136, 144, 152, 159, 166,
            174, 181, 188, 194, 201, 207, 213, 218, 224, 228, 233, 237, 240, 244, 246, 249,
            251, 252, 253, 253, 253, 253, 252, 251, 249, 246, 244, 240, 237, 233, 228, 224,
            218, 213, 207, 201, 194, 188, 181, 174, 166, 159, 152, 144, 136, 128, 121, 113,
            105,  98,  90,  83,  76,  69,  62,  55,  49,  43,  37,  32,  27,  22,  18,  14,
            11,   8,   5,   3,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
        };

        const unsigned char *pk_shape;

        // the queue of operations passed to wavegen from sythesize
        intptr_t wcmdq[N_WCMDQ][4];
        int wcmdq_head = 0;
        int wcmdq_tail = 0;

        char skip_marker[N_MARKER_LENGTH];
        wchar_t option_punctlist[N_PUNCTLIST] = { 0 }; // which punctuation characters to announce

        char dictionary_name[40];
        int dictionary_skipwords;

        FILE *f_trans = NULL; // phoneme output text
        int option_tone_flags = 0; // bit 8=emphasize allcaps, bit 9=emphasize penultimate stress
        int option_phonemes = 0;
        int option_phoneme_events = 0;
        int option_endpause = 0; // suppress pause after end of text
        int option_capitals = 0;
        int option_punctuation = 0;
        int option_sayas = 0;
        int option_sayas2 = 0; // used in translate_clause()
        int option_emphasis = 0; // 0=normal, 1=normal, 2=weak, 3=moderate, 4=strong
        int option_ssml = 0;
        int option_phoneme_input = 0; // allow [[phonemes]] in input
        int option_wordgap = 0;
        int option_linelength = 0; // treat lines shorter than this as end-of-clause

        // int count_characters = 0;
        int count_sentences;
        int skip_characters;
        int skip_words;
        int skip_sentences;
        bool skipping_text; // waiting until word count, sentence count, or named marker is reached
        int end_character_position;
        int clause_start_char;
        int clause_start_word;
        
        int namedata_ix = 0;
        int n_namedata = 0;
        char *namedata = NULL;

        int pre_pause;

        int event_list_ix = 0;
        int n_event_list;

        int len_speeds[3] = { 130, 121, 118 };
        int more_syllables = 0;

        const char *xmlbase = ""; // base URL from <speak>

        int n_ssml_stack;
        SSML_STACK ssml_stack[N_SSML_STACK];

        bool ignore_text = false; // set during <sub> ... </sub>  to ignore text which has been replaced by an alias
        bool audio_text = false; // set during <audio> ... </audio>
        bool clear_skipping_text = false; // next clause should clear the skipping_text flag
        int sayas_mode;
        int sayas_start;

        int ungot_char2 = 0;
        int ungot_char;

        char ungot_string[N_XML_BUF2+4];
        int ungot_string_ix = -1;

        int (*uri_callback)(int, const char *, const char *) = NULL;
        t_espeak_callback *synth_callback = NULL;
        int (*phoneme_callback)(const char *) = NULL;

        char word_replacement[N_WORD_BYTES];
        char *phon_out_buf = NULL;   // passes the result of GetTranslatedPhonemeString()
        unsigned int phon_out_size = 0;

        int tone_pitch_env; // used to return pitch envelope
        int number_pre;
        int number_tail;
        int last_primary;
        int tone_posn;
        int tone_posn2;
        int no_tonic;

        int last_pitch_cmd;
        int last_amp_cmd;
        frame_t  *last_frame;
        int last_wcmdq;
        int pitch_length;
        int amp_length;
        int modn_flags;
        int fmt_amplitude = 0;

        int syllable_start;
        int syllable_end;
        int syllable_centre;
        int wave_flag = 0;

        int gen_ix;
        int embedded_ix_syn;
        int word_count_syn;
        int sourceix = 0;
        WORD_PH_DATA worddata;

        voice_t *wvoice = NULL;

        int option_harmonic1 = 10;
        int flutter_amp = 64;

        int general_amplitude = 60;
        int consonant_amp = 26;

        int PHASE_INC_FACTOR;

        wavegen_peaks_t peaks[N_PEAKS];
        int peak_harmonic[N_PEAKS];
        int peak_height[N_PEAKS];

        int voicing;
        RESONATOR rbreath[N_PEAKS];

        int harm_inc[N_LOWHARM]; // only for these harmonics do we interpolate amplitude between steps
        int *harmspect;
        int hswitch = 0;
        int hspect[2][MAX_HARMONIC]; // 2 copies, we interpolate between then

        int nsamples = 0; // number to do
        int modulation_type = 0;
        int glottal_flag = 0;
        int glottal_reduce = 0;

        WGEN_DATA wdata;

        int amp_ix;
        int amp_inc;
        unsigned char *amplitude_env = NULL;

        int samplecount = 0; // number done
        int samplecount_start = 0; // count at start of this segment
        int end_wave = 0; // continue to end of wave cycle
        int wavephase;
        int phaseinc;
        int cycle_samples; // number of samples in a cycle at current pitch
        int cbytes;
        int hf_factor;

        double minus_pi_t;
        double two_pi_t;

        int const_f0 = 0;

        int Flutter_ix = 0;
        int agc = 256;
        int h_switch_sign = 0;
        int cycle_count = 0;
        int amplitude2 = 0; // adjusted for pitch
        int maxh, maxh2;
        int n_samples;
        int PlayWave_n_samples;
        int PlayWave_ix = 0;
        voice_t wavegen_voice2;
        bool wavegen_resume = false;
        int echo_complete = 0;
        int Flutter_inc;

        void LoadConfig(void);
        espeak_ng_STATUS ReadPhFile(void **ptr, const char *fname, int *size, espeak_ng_ERROR_CONTEXT *context);
        int compile_dictlist_file(CompileContext *ctx, const char *path, const char *filename);
        espeak_ng_STATUS compile_dictrules(CompileContext *ctx, FILE *f_in, FILE *f_out);
        char *compile_rule(CompileContext *ctx, char *input);
        int LoadDictionary(Translator *tr, const char *name, int no_error);
        espeak_ng_STATUS LoadPhData(int *srate, espeak_ng_ERROR_CONTEXT *context);
        voice_t *LoadVoice(const char *vname, int control);
        int check_data_path(const char *path, int allow_directory);
        espeak_ng_STATUS LoadSoundFile(const char *fname, int index, espeak_ng_ERROR_CONTEXT *context);
        int LoadSoundFile2(const char *fname);
        int LookupSoundicon(int c);
        espeak_ng_STATUS ReadPhondataManifest(CompileContext *ctx, espeak_ng_ERROR_CONTEXT *context);
        int SetVoiceScores(espeak_VOICE *voice_select, espeak_VOICE **voices, int control);
        char const *SelectVoice(espeak_VOICE *voice_select, int *found);
        void VoiceReset(int tone_only);
        const char *VoiceFromStack(SSML_STACK *ssml_stack, int n_ssml_stack, espeak_VOICE *base_voice, char base_voice_variant_name[40]);
        int GetVoiceAttributes(wchar_t *pw, int tag_type, SSML_STACK *ssml_sp, SSML_STACK *ssml_stack, int n_ssml_stack, char current_voice_id[40], espeak_VOICE *base_voice, char *base_voice_variant_name);
        int ProcessSsmlTag(wchar_t *xml_buf, char *outbuf, int *outix, int n_outbuf, const char *xmlbase, bool *audio_text, char *current_voice_id, espeak_VOICE *base_voice, char *base_voice_variant_name, bool *ignore_text, bool *clear_skipping_text, int *sayas_mode, int *sayas_start, SSML_STACK *ssml_stack, int *n_ssml_stack, int *n_param_stack, int *speech_parameters);
        voice_t *LoadVoiceVariant(const char *vname, int variant_num);
        int SpeakNextClause(int control);
        int AnnouncePunctuation(Translator *tr, int c1, int *c2_ptr, char *output, int *bufix, int end_clause);
        int SetAlternateTranslator(const char *new_language, Translator **translator, char translator_language[20]);
        int SetTranslator2(const char *new_language);
        int SetTranslator3(const char *new_language);
        void LookupLetter(Translator *tr, unsigned int letter, int next_byte, char *ph_buf1, int control);
        int TranslateLetter(Translator *tr, char *word, char *phonemes, int control, const ALPHABET *current_alphabet);
        const char *LookupCharName(char buf[80], Translator *tr, int c, bool only);
        int TranslateWord2(Translator *tr, char *word, WORD_TAB *wtab, int pre_pause);
        void TranslateClause(Translator *tr, int *tone_out, char **voice_change);
        int TranslateWord3(Translator *tr, char *word_start, WORD_TAB *wtab, char *word_out, bool *any_stressed_words, ALPHABET *current_alphabet, char word_phonemes[], size_t size_word_phonemes);
        char *SpeakIndividualLetters(Translator *tr, char *word, char *phonemes, int spell_word, const ALPHABET *current_alphabet, char word_phonemes[]);
        int TranslateRules(Translator *tr, char *p_start, char *phonemes, int ph_size, char *end_phonemes, int word_flags, unsigned int *dict_flags);
        int TranslateWord(Translator *tr, char *word_start, WORD_TAB *wtab, char *word_out);
        int compile_line(CompileContext *ctx, char *linebuf, char *dict_line, int n_dict_line, int *hash);
        int Lookup(Translator *tr, const char *word, char *ph_out);
        int CheckDotOrdinal(Translator *tr, char *word, char *word_end, WORD_TAB *wtab, int roman);
        void CombineFlag(Translator *tr, WORD_TAB *wtab, char *word, int *flags, unsigned char *p, char *word_phonemes);
        int TranslateRoman(Translator *tr, char *word, char *ph_out, WORD_TAB *wtab);
        int TranslateNumber_1(Translator *tr, char *word, char *ph_out, unsigned int *flags, WORD_TAB *wtab, int control);
        int TranslateNumber(Translator *tr, char *word1, char *ph_out, unsigned int *flags, WORD_TAB *wtab, int control);
        int LookupLetter2(Translator *tr, unsigned int letter, char *ph_buf);
        void LookupAccentedLetter(Translator *tr, unsigned int letter, char *ph_buf);
        void addPluralSuffixes(int flags, Translator *tr, char last_char, char *word_phonemes);
        int Unpronouncable2(Translator *tr, char *word);
        int Unpronouncable(Translator *tr, char *word, int posn);
        int LookupDictList(Translator *tr, char **wordptr, char *ph_out, unsigned int *flags, int end_flags, WORD_TAB *wtab);
        const char *LookupSpecial(Translator *tr, const char *string, char *text_out, size_t out_sz);
        int LookupFlags(Translator *tr, const char *word, unsigned int flags_out[2]);
        void DollarRule(char *word[], char *word_start, int consumed, int group_length, char *word_buf, Translator *tr, int command, int *failed, int *add_points);
        void MatchRule(Translator *tr, char *word[], char *word_start, int group_length, char *rule, MatchRecord *match_out, int word_flags, int dict_flags);
        int LookupThousands(Translator *tr, int value, int thousandplex, int thousands_exact, char *ph_out);
        int LookupNum2(Translator *tr, int value, int thousandplex, const int control, char *ph_out);
        int LookupNum3(Translator *tr, int value, char *ph_out, bool suppress_null, int thousandplex, int control);
        const char * M_Variant(int value);
        const char *LookupDict2(Translator *tr, const char *word, const char *word2, char *phonetic, unsigned int *flags, int end_flags, WORD_TAB *wtab);
        int DoSpect2(PHONEME_TAB *this_ph, int which, FMT_PARAMS *fmt_params,  PHONEME_LIST *plist, int modulation);
        int Generate(PHONEME_LIST *phoneme_list, int *n_ph, bool resume);
        int Wavegen(int length, int modulation, bool resume, frame_t *fr1, frame_t *fr2, voice_t *wvoice);
        int PlaySilence(int length, bool resume);
        int PlayWave(int length, bool resume, unsigned char *data, int scale, int amp);
        void WavegenSetVoice(voice_t *v);
        int WavegenFill2(void);
        int WavegenFill(void);
        bool InterpretCondition(Translator *tr, int control, PHONEME_LIST *plist, unsigned short *p_prog, WORD_PH_DATA *worddata);
        void InterpretPhoneme(Translator *tr, int control, PHONEME_LIST *plist, PHONEME_DATA *phdata, WORD_PH_DATA *worddata);
        void InterpretPhoneme2(int phcode, PHONEME_DATA *phdata);
        void ReInterpretPhoneme(PHONEME_TAB *ph, PHONEME_TAB *ph2, PHONEME_LIST *plist3, Translator *tr, PHONEME_DATA *phdata, WORD_PH_DATA *worddata);
        void MarkerEvent(int type, unsigned int char_position, int value, int value2, unsigned char *out_ptr);
        void MakePhonemeList(Translator *tr, int post_pause, bool start_sentence);
        char *WritePhMnemonic(char *phon_out, PHONEME_TAB *ph, PHONEME_LIST *plist, int use_ipa, int *flags);
        void CalcLengths(Translator *tr);
        const char *GetTranslatedPhonemeString(int phoneme_mode);
        int PhonemeCode(unsigned int mnem);
        void SetUpPhonemeTable(int number);
        void SelectPhonemeTable(int number);
        const char *EncodePhonemes(const char *p, char *outptr, int *bad_phoneme);
        void SetWordStress(Translator *tr, char *output, unsigned int *dictionary_flags, int tonic, int control);
        void AppendPhonemes(Translator *tr, char *string, int size, const char *ph);
        void CalcPitches_Tone(Translator *tr);
        int SubstitutePhonemes(PHONEME_LIST *plist_out);
        void SetRegressiveVoicing(int regression, PHONEME_LIST2 *plist2, PHONEME_TAB *ph, Translator *tr);
        void CalcPitches(Translator *tr, int clause_type);
        frameref_t *LookupSpect(PHONEME_TAB *this_ph, int which, FMT_PARAMS *fmt_params,  int *n_frames, PHONEME_LIST *plist);
        bool StressCondition(Translator *tr, PHONEME_LIST *plist, int condition, int control);
        int GetVowelStress(Translator *tr, unsigned char *phonemes, signed char *vowel_stress, int *vowel_count, int *stressed_syllable, int control);
        void DecodePhonemes(const char *inptr, char *outptr);
        int CountSyllables(unsigned char *phonemes);
        void ChangeWordStress(Translator *tr, char *word, int new_stress);
        int SelectPhonemeTableName(const char *name);
        void SwitchLanguage(char *word, char *word_phonemes);
        int LookupPhonemeString(const char *string);
        void ApplySpecialAttribute2(Translator *tr, char *phonemes, int dict_flags);
        void PhonemeReplacement(char *p);
        void DecodeWithPhonemeMode(char *buf, size_t buf_sz, char *phonemes, Translator *tr, Translator *tr2, unsigned int flags[]);
        int LookupPhonemeTable(const char *name);
        void InitText2(void);
        void InitText(int control);
        void DoEmbedded(int *embix, int sourceix);
        void FreePhData(void);
        int calc_pitches2(SYLLABLE *syllable_tab, int start, int end,  int tune_number);
        int LookupTune(const char *name);
        int calc_pitches(SYLLABLE *syllable_tab, int control, int start, int end,  int tune_number);
        void LoadLanguageOptions(Translator *translator, int key, char *keyValue );
        int DoSample3(PHONEME_DATA *phdata, int length_mod, int amp);
        int FormantTransition2(frameref_t *seq, int *n_frames, unsigned int data1, unsigned int data2, PHONEME_TAB *other_ph, int which);
        void SetSpeed(int control);
        void set_frame_rms(frame_t *fr, int new_rms);
        void VoiceFormant(char *p);
        void formants_reduce_hf(frame_t *fr, int level);
        void AdjustFormants(frame_t *fr, int target, int min, int max, int f1_adj, int f3_adj, int hf_reduce, int flags);
        void DoEmbedded2(int *embix);
        int LoadWavefile(CompileContext *ctx, FILE *f, const char *fname);
        espeak_ng_STATUS LoadDataFile(CompileContext *ctx, const char *path, int control, int *addr);
        void CompileToneSpec(CompileContext *ctx);
        void CompileSound(CompileContext *ctx, int keyword, int isvowel);
        int CompilePhoneme(CompileContext *ctx, int compile_phoneme);
        void CompilePhonemeFiles(CompileContext *ctx);
        void DoPause(int length, int control);
        int DoSample2(int index, int which, int std_length, int control, int length_mod, int amp);
        void WavegenInit(int rate, int wavemult_fact);
        void WavegenSetEcho(void);
        int PeaksToHarmspect(wavegen_peaks_t *peaks, int pitch, int *htab, int control);
        void InitBreath(void);
        void SetEmbedded(int control, int value);
        double GetFrameRms(SpectFrame *frame, int seq_amplitude);
        espeak_ng_STATUS LoadSpect(CompileContext *ctx, const char *path, int control, int *addr);
        int PauseLength(int pause, int control);
        void SmoothSpect(void);
        void EndPitch(int voice_break);
        void DoPitch(const unsigned char *env, int pitch1, int pitch2);
        int GetAmplitude(void);
        void SetPitchFormants(void);
        void SetPitch2(voice_t *voice, int pitch1, int pitch2, int *pitch_base, int *pitch_range);
        void SetPitch(int length, unsigned char *env, int pitch1, int pitch2);
        void Word_EmbeddedCmd(void);
        int EmbeddedCommand(unsigned int *source_index_out);
        void EndAmplitude(void);
        void DoAmplitude(int amp, const unsigned char *amp_env);
        void DoPhonemeAlignment(char* pho, int type);
        void SetSynth(int length, int modn, frame_t *fr1, frame_t *fr2, voice_t *v);
        espeak_ng_STATUS DoVoiceChange(voice_t *v);
        void DoPhonemeMarker(int type, int char_posn, int length, char *name);
        void DoMarker(int type, int char_posn, int length, int value);
        void StartSyllable(void);
        void InitGroups(Translator *tr);
        Translator *NewTranslator(void);
        Translator *SelectTranslator(const char *name);
        int CheckDottedAbbrev(char *word1);
        int RemoveEnding(Translator *tr, char *word, int end_type, char *word_copy);
        int SubstituteChar(Translator *tr, unsigned int c, unsigned int next_in, const char *next, int *insert, int *wordflags);
        int TranslateChar(Translator *tr, char *ptr, int prev_in, unsigned int c, unsigned int next_in, int *insert, int *wordflags);
        void ProcessParamStack(char *outbuf, int *outix, int n_param_stack, PARAM_STACK *param_stack, int *speech_parameters);
        void PopParamStack(int tag_type, char *outbuf, int *outix, int *n_param_stack, PARAM_STACK *param_stack, int *speech_parameters);
        void InitNamedata(void);
        int AddNameData(const char *name, int wide);
        int create_events(short *outbuf, int length, espeak_EVENT *event_list);
        int dispatch_audio(short *outbuf, int length, espeak_EVENT *event);
        espeak_VOICE *SelectVoiceByName(espeak_VOICE **voices, const char *name2);
        void GetVoices(const char *path, int len_path_voices, int is_language_file);
        int AddToVoicesList(const char *fname, int len_path_voices, int is_language_file);
        void FreeVoiceList(void);
        char *ExtractVoiceVariantName(char *vname, int variant_num, int add_dir);
        const unsigned char *GetEnvelope(int index);
        frame_t *AllocFrame(void);
        frame_t *CopyFrame(frame_t *frame1, int copy);
        frame_t *DuplicateLastFrame(frameref_t *seq, int n_frames, int length);
        void WavegenFini(void);
        void SetVoiceStack(espeak_VOICE *v, const char *variant_name);
        int ApplyBreath(void);
        void count_pitch_vowels(SYLLABLE *syllable_tab, int start, int end, int clause_end);
        void SetPlist2(PHONEME_LIST2 *p, unsigned char phcode);
        void cancel_audio(void);
        void SynthesizeInit(void);
        void TerminateBufWithSpaceAndZero(char *buf, int index, int *ungetc);
        void AdvanceParameters(void);
        void setresonator(RESONATOR *rp, int freq, int bwidth, int init);
        void SetAmplitude(int length, unsigned char *amp_env, int value);
        void SetBreath(void);

        long espeak_rand(long min, long max);
        void espeak_srand(long seed);

        int Eof(void);
        int GetC(void);
        void UngetC(int c);

        void WcmdqStop(void);
        void WcmdqInc(void);
        int WcmdqFree(void);
        void WcmdqIncHead(void);
        int WcmdqUsed(void);

        espeak_ng_STATUS _SetParameter(int parameter, int value, int relative);

        espeak_ng_STATUS Synthesize(unsigned int unique_identifier, const void *text, int flags);

        espeak_ng_STATUS sync_espeak_Synth(unsigned int unique_identifier, const void *text,
                                        unsigned int position, espeak_POSITION_TYPE position_type,
                                        unsigned int end_position, unsigned int flags, void *user_data);
        espeak_ng_STATUS sync_espeak_Synth_Mark(unsigned int unique_identifier, const void *text,
                                                const char *index_mark, unsigned int end_position,
                                                unsigned int flags, void *user_data);
        espeak_ng_STATUS sync_espeak_Key(const char *key);
        espeak_ng_STATUS sync_espeak_Char(wchar_t character);
        void sync_espeak_SetPunctuationList(const wchar_t *punctlist);
        int sync_espeak_terminated_msg(uint32_t unique_identifier, void *user_data);
        t_espeak_command *create_espeak_voice_spec(espeak_VOICE *voice);
        int delete_espeak_command(t_espeak_command *the_command);
        espeak_ng_STATUS close_stream(void);

        #if USE_ASYNC
        void process_espeak_command(t_espeak_command *the_command);
        void say_thread();
        void fifo_init(void);
        void fifo_terminate(void);
        void _fifo_init(int process_parameters);
        #endif

        #if USE_MBROLA
        int mbrola_delay;
        char mbrola_name[20];

        MBROLA_TAB *mbrola_tab = NULL;
        int mbrola_control = 0;
        int mbr_name_prefix = 0;

        char output_mbr[50];
        int phix_mbr;
        int embedded_ix_mbr;
        int word_count_mbr;
        int n_samples_mbr;

        espeak_ng_STATUS LoadMbrolaTable(const char *mbrola_voice, const char *phtrans, int *srate);
        int MbrolaTranslate(PHONEME_LIST *plist, int n_phonemes, bool resume, FILE *f_mbrola);
        int MbrolaGenerate(PHONEME_LIST *phoneme_list, int *n_ph, bool resume);
        int MbrolaFill(int length, bool resume, int amplitude);
        int GetMbrName(PHONEME_LIST *plist, PHONEME_TAB *ph, PHONEME_TAB *ph_prev, PHONEME_TAB *ph_next, int *name2, int *split, int *control);
        char *WritePitch(int env, int pitch1, int pitch2, int split, int final);
        #endif

        #if USE_KLATT
        double parwave_noise;
        double parwave_voice;
        double parwave_vlast;
        double parwave_glotlast;
        double parwave_sourc;

        double impulsive_source_vwave;
        double natural_source_vwave;

        klatt_frame_t kt_frame;
        klatt_global_t kt_globals;
        int klattp[N_KLATTP];
        double klattp1[N_KLATTP];
        double klattp_inc[N_KLATTP];

        int kt_nsamples;
        int kt_sample_count;
        int kt_time_count;
        long kt_skew;

        double kt_nlast;
        frame_t kt_prev_fr;

        klatt_peaks_t kt_peaks[N_PEAKS];
        int kt_end_wave;

        double impulsive_source(void);
        double natural_source(void);
        double sampled_source(int source_num);
        void frame_init(klatt_frame_ptr frame);
        void pitch_synch_par_reset(klatt_frame_ptr frame);
        void setabc(long int f, long int bw, resonator_ptr rp);
        void setzeroabc(long int f, long int bw, resonator_ptr rp);
        void flutter(klatt_frame_ptr frame);
        double gen_noise(double noise);
        int parwave(klatt_frame_ptr frame, WGEN_DATA *wdata);
        void KlattInit(void);
        void KlattFini(void);
        void KlattReset(int control);
        int Wavegen_Klatt(int length, int resume, frame_t *fr1, frame_t *fr2, WGEN_DATA *wdata, voice_t *wvoice);
        void SetSynth_Klatt(int length, frame_t *fr1, frame_t *fr2, voice_t *wvoice, int control);
        #endif

        #if USE_SPEECHPLAYER
        speechPlayer_handle_t speechPlayerHandle = NULL;
        void KlattInitSP(void);
        void KlattFiniSP(void);
        void KlattResetSP(void);
        int Wavegen_KlattSP(WGEN_DATA *wdata, voice_t *wvoice, int length, int resume, frame_t *fr1, frame_t *fr2);
        bool isKlattFrameFollowing(void);
        #endif

        #if USE_LIBSONIC
        sonicStream sonicSpeedupStream = NULL;
        double sonicSpeed = 1.0;
        int SpeedUp(short *outbuf, int length_in, int length_out, int end_of_text);
        void DoSonicSpeed(int value);
        #endif

        // expose path_home
        friend const char* ::espeak_Info(const char **ptr);
        // expose dictionary_name
        friend void ::espeak_CompileDictionary(const char *path, FILE *log, int flags);
        // expose current_voice_selected
        friend espeak_VOICE *::espeak_GetCurrentVoice(void);

    public: // TESTS
        espeak_ng_TEXT_DECODER* p_decoder = NULL;
        espeak_EVENT *event_list = NULL;
        Translator *translator = NULL; // the main translator
        int ReadClause(Translator *tr, char *buf, short *charix, int *charix_top, int n_buf, int *tone_type, char *voice_change);
        int count_characters = 0;

    public:
        static context_t& global(void);

        ~context_t();

        espeak_ng_ERROR_CONTEXT GetError();
        const espeak_VOICE **ListVoices(espeak_VOICE *voice_spec);
        void InitializePath(const char *path);
        espeak_ng_STATUS Initialize();
        espeak_ng_STATUS InitializeOutput(
            espeak_ng_OUTPUT_MODE output_mode,
            int buffer_length,
            const char *device
        );
        int GetSampleRate() const;

        espeak_ng_STATUS SetRandSeed(long seed);

        espeak_ng_STATUS SetParameter(espeak_PARAMETER parameter, int value, int relative);
        int              GetParameter(espeak_PARAMETER parameter, int current);

        espeak_ng_STATUS SetPhonemeEvents(int enable, int ipa);
        espeak_ng_STATUS SetPunctuationList(const wchar_t *punctlist);
        espeak_ng_STATUS SetVoiceByName(const char *name);
        espeak_ng_STATUS SetVoiceByFile(const char *filename);

        espeak_ng_STATUS SetVoiceByProperties(espeak_VOICE *voice_selector);
        espeak_ng_STATUS SpeakKeyName(const char *key_name);
        espeak_ng_STATUS SpeakCharacter(wchar_t character);
        espeak_ng_STATUS Cancel();
        espeak_ng_STATUS Synchronize();
        espeak_ng_STATUS Terminate();
        espeak_ng_STATUS SetOutputHooks(espeak_ng_OUTPUT_HOOKS* hooks);
        espeak_ng_STATUS SetConstF0(int f0);
        const char *     TextToPhonemes(const void **textptr, int textmode, int phonememode);

        espeak_ng_STATUS SetSynthCallback(t_espeak_callback* SynthCallback);
        espeak_ng_STATUS SetUriCallback(int (*UriCallback)(int, const char*, const char*));
        espeak_ng_STATUS SetPhonemeCallback(int (*PhonemeCallback)(const char *));
        espeak_ng_STATUS SetPhonemeTrace(int phonememode, FILE *stream);

        espeak_ng_STATUS Synthesize(
            const void *text,
            size_t size,
            unsigned int position,
            espeak_POSITION_TYPE position_type,
            unsigned int end_position,
            unsigned int flags,
            unsigned int *unique_identifier,
            void *user_data
        );

        espeak_ng_STATUS SynthesizeMark(
            const void *text,
            size_t size,
            const char *index_mark,
            unsigned int end_position,
            unsigned int flags,
            unsigned int *unique_identifier,
            void *user_data
        );

        espeak_ng_STATUS CompileDictionary(
            const char *dsource,
            const char *dict_name,
            FILE *log,
            int flags
        );

#if USE_MBROLA
        espeak_ng_STATUS CompileMbrolaVoice(
            const char *path,
            FILE *log
        );
#endif

        espeak_ng_STATUS CompilePhonemeDataPath(
            long rate,
            const char *source_path,
            const char *destination_path,
            FILE *log
        );

        espeak_ng_STATUS CompileIntonationPath(
            const char *source_path,
            const char *destination_path,
            FILE *log
        );
    };
}

struct espeak_ng_CONTEXT : public espeak::context_t {};
