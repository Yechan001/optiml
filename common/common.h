/**
 * @file common.h
 * @brief Common definitions and utilities for the project
 * 
 * This header contains shared structures, constants and utility functions
 * used throughout the project.
 */

#pragma once

#ifndef OPTIML_COMMON_H
#define OPTIML_COMMON_H

#include "llama.h"

#include "sampling.h"

// Logging configuration
#define LOG_NO_FILE_LINE_FUNCTION
#include "log.h"

// Platform detection
#if defined(_WIN32)
#define PLATFORM_WINDOWS 1
#elif defined(__linux__)
#define PLATFORM_LINUX 1
#elif defined(__APPLE__)
#define PLATFORM_APPLE 1
#endif

#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <thread>
#include <unordered_map>
#include <tuple>

#ifdef _WIN32
#define DIRECTORY_SEPARATOR '\\'
#else
#define DIRECTORY_SEPARATOR '/'
#endif // _WIN32

#endif // OPTIML_COMMON_H

#define die(msg)          do { fputs("error: " msg "\n", stderr);                exit(1); } while (0)
#define die_fmt(fmt, ...) do { fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); exit(1); } while (0)

#define print_build_info() do {                                                                     \
    fprintf(stderr, "%s: build = %d (%s)\n", __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);           \
    fprintf(stderr, "%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);    \
} while(0)

// build info
extern int LLAMA_BUILD_NUMBER;
extern char const *LLAMA_COMMIT;
extern char const *LLAMA_COMPILER;
extern char const *LLAMA_BUILD_TARGET;

//
// CLI argument parsing
//
int32_t get_num_physical_cores();

/**
 * @brief Parameters for GPT model inference
 * 
 * Contains all configurable parameters for running model inference,
 * including sampling parameters, context size, and generation settings.
 */
struct gpt_params {
    /// @name Basic Configuration
    /// @{
    uint32_t seed                           = -1;    ///< RNG seed (-1 for random)
    /// @}

    /// @name Performance Parameters
    /// @{
    int32_t n_threads                       = get_num_physical_cores(); ///< Number of threads for generation
    int32_t n_threads_batch                 = -1;    ///< Threads for batch processing (-1 = use n_threads)
    int32_t n_predict                       = -1;    ///< New tokens to predict (-1 = infinite)
    int32_t n_ctx                           = 512;   ///< Context window size
    int32_t n_batch                         = 512;   ///< Batch size for prompt processing
    int32_t n_keep                          = 0;     ///< Tokens to keep from initial prompt
    int32_t n_draft                         = 16;    ///< Tokens to draft during speculative decoding
    int32_t n_chunks                        = -1;    ///< Max chunks to process (-1 = unlimited)
    int32_t n_parallel                      = 1;     ///< Parallel sequences to decode
    int32_t n_sequences                     = 1;     ///< Sequences to decode
    float   p_accept                        = 0.5f;  ///< Speculative decoding accept probability
    float   p_split                         = 0.1f;  ///< Speculative decoding split probability
    /// @}
    /// @name GPU Configuration
    /// @{
    int32_t n_gpu_layers                    = -1;    ///< Layers to store in VRAM (-1 = default)
    int32_t n_gpu_layers_draft              = -1;    ///< Layers for draft model (-1 = default)
    int32_t main_gpu                        = 0;     ///< Main GPU index
    float   tensor_split[LLAMA_MAX_DEVICES] = {0};   ///< Tensor split across GPUs
    int32_t n_beams                         = 0;     ///< Beam search width (0 = disabled)
    /// @}

    /// @name RoPE/YaRN Parameters
    /// @{
    float   rope_freq_base                  = 0.0f;  ///< RoPE base frequency
    float   rope_freq_scale                 = 0.0f;  ///< RoPE frequency scaling
    float   vram_budget_gb                  = -1.0f; ///< VRAM budget in GB (-1 = auto)
    float   yarn_ext_factor                 = -1.0f; ///< YaRN extrapolation mix
    float   yarn_attn_factor                = 1.0f;  ///< YaRN magnitude scaling
    float   yarn_beta_fast                  = 32.0f; ///< YaRN low correction dim
    float   yarn_beta_slow                  = 1.0f;  ///< YaRN high correction dim
    int32_t yarn_orig_ctx                   = 0;     ///< YaRN original context length
    int8_t  rope_scaling_type               = LLAMA_ROPE_SCALING_UNSPECIFIED; ///< RoPE scaling type
    /// @}

    /// @name Sampling Parameters
    /// @{
    struct llama_sampling_params sparams; ///< Sampling parameters
    /// @}

    /// @name Model & Prompt Configuration
    /// @{
    std::string model             = "models/7B/ggml-model-f16.gguf"; ///< Model file path
    std::string model_draft       = "";                              ///< Draft model path
    std::string model_alias       = "unknown"; ///< Model display name
    std::string prompt            = ""; ///< Initial prompt
    std::string prompt_file       = ""; ///< External prompt file
    std::string path_prompt_cache = ""; ///< Prompt cache path
    std::string input_prefix      = ""; ///< Prefix for user inputs
    std::string input_suffix      = ""; ///< Suffix for user inputs
    std::vector<std::string> antiprompt; ///< Strings that trigger new input
    std::string logdir            = ""; ///< Directory for YAML logs
    /// @}

    /// @name LoRA Configuration
    /// @{
    std::vector<std::tuple<std::string, float>> lora_adapter; ///< LoRA adapters with scales
    std::string lora_base  = "";                              ///< Base model for LoRA
    /// @}

    /// @name GPU Management
    /// @{
    bool reset_gpu_index   = false; ///< Refresh GPU index file
    bool disale_gpu_index  = false; ///< Disable GPU index loading
    /// @}

    /// @name Evaluation Parameters
    /// @{
    int  ppl_stride        = 0;     ///< Perplexity calculation stride
    int  ppl_output_type   = 0;     ///< Perplexity output format
    bool hellaswag         = false; ///< Enable HellaSwag evaluation
    size_t hellaswag_tasks = 400;   ///< Number of HellaSwag tasks
    /// @}

    /// @name Memory & Performance Flags
    /// @{
    bool mul_mat_q         = true;  ///< Use custom matmul kernels
    bool memory_f16        = true;  ///< Use FP16 for KV cache
    bool use_mmap          = true;  ///< Use memory mapping
    bool use_mlock         = false; ///< Lock model in memory
    bool numa              = false; ///< NUMA optimizations
    /// @}

    /// @name Interaction Flags
    /// @{
    bool random_prompt     = false; ///< Randomize empty prompts
    bool use_color         = false; ///< Colorized output
    bool interactive       = false; ///< Interactive mode
    bool prompt_cache_all  = false; ///< Cache all prompts
    bool prompt_cache_ro   = false; ///< Read-only prompt cache
    bool embedding         = false; ///< Embedding-only mode
    bool escape            = false; ///< Process escape sequences
    bool interactive_first = false; ///< Immediate interactive
    bool multiline_input   = false; ///< Multiline input mode
    bool simple_io         = false; ///< Simplified I/O
    bool cont_batching     = false; ///< Continuous batching
    bool input_prefix_bos  = false; ///< Prefix BOS token
    bool ignore_eos        = false; ///< Ignore EOS tokens
    bool instruct          = false; ///< Instruction mode
    bool logits_all        = false; ///< Return all logits
    bool verbose_prompt    = false; ///< Verbose prompt
    bool infill            = false; ///< Infill mode
    /// @}

    /// @name Multimodal Parameters
    /// @{
    std::string mmproj = ""; ///< Multimodal projector path
    std::string image = "";  ///< Image file path
    /// @}
};

bool gpt_params_parse_ex(int argc, char ** argv, gpt_params & params);

bool gpt_params_parse(int argc, char ** argv, gpt_params & params);

void gpt_print_usage(int argc, char ** argv, const gpt_params & params);

std::string get_system_info(const gpt_params & params);

std::string gpt_random_prompt(std::mt19937 & rng);

void process_escapes(std::string& input);

//
// Model utils
//

// TODO: avoid tuplue, use struct
std::tuple<struct llama_model *, struct llama_context *> llama_init_from_gpt_params(gpt_params & params);

struct llama_model_params   llama_model_params_from_gpt_params  (const gpt_params & params);
struct llama_context_params llama_context_params_from_gpt_params(const gpt_params & params);

// Batch utils

void llama_batch_clear(struct llama_batch & batch);

void llama_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits);

//
// Vocab utils
//

// tokenizes a string into a vector of tokens
// should work similar to Python's `tokenizer.encode`
std::vector<llama_token> llama_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_bos,
                        bool   special = false);

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const std::string & text,
                        bool   add_bos,
                        bool   special = false);

// tokenizes a token into a piece
// should work similar to Python's `tokenizer.id_to_piece`
std::string llama_token_to_piece(
        const struct llama_context * ctx,
                       llama_token   token);

// TODO: these should be moved in llama.h C-style API under single `llama_detokenize` function
//       that takes into account the tokenizer type and decides how to handle the leading space
//
// detokenizes a vector of tokens into a string
// should work similar to Python's `tokenizer.decode`
// removes the leading space from the first non-BOS token
std::string llama_detokenize_spm(
                         llama_context * ctx,
        const std::vector<llama_token> & tokens);

// detokenizes a vector of tokens into a string
// should work similar to Python's `tokenizer.decode`
std::string llama_detokenize_bpe(
                         llama_context * ctx,
        const std::vector<llama_token> & tokens);

//
// YAML utils
//

bool create_directory_with_parents(const std::string & path);
void dump_vector_float_yaml(FILE * stream, const char * prop_name, const std::vector<float> & data);
void dump_vector_int_yaml(FILE * stream, const char * prop_name, const std::vector<int> & data);
void dump_string_yaml_multiline(FILE * stream, const char * prop_name, const char * data);
std::string get_sortable_timestamp();

void dump_non_result_info_yaml(
    FILE * stream, const gpt_params & params, const llama_context * lctx,
    const std::string & timestamp, const std::vector<int> & prompt_tokens, const char * model_desc);
