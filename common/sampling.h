/**
 * @file sampling.h
 * @brief Text generation sampling algorithms and parameters
 * 
 * Contains structures and functions for controlling text generation sampling.
 */

#pragma once

#ifndef OPTIML_SAMPLING_H
#define OPTIML_SAMPLING_H

#include "llama.h"

#include "grammar-parser.h"

#include <string>
#include <vector>
#include <unordered_map>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Parameters for controlling text generation sampling
 * 
 * These parameters affect how tokens are selected during text generation,
 * allowing control over creativity, diversity and quality of output.
 */
typedef struct llama_sampling_params {
    /// @name Context Memory
    /// @{
    int32_t n_prev            = 64;    ///< Number of previous tokens to remember
    int32_t n_probs           = 0;     ///< Output top n_probs probabilities (>0)
    /// @}

    /// @name Core Sampling Parameters
    /// @{
    int32_t top_k             = 40;    ///< Top-k sampling (<=0 = vocab size)
    float   top_p             = 0.95f; ///< Nucleus sampling (1.0 = disabled)
    float   min_p             = 0.05f; ///< Minimum probability cutoff (0.0 = disabled)
    float   tfs_z             = 1.00f; ///< Tail free sampling (1.0 = disabled)
    float   typical_p         = 1.00f; ///< Locally typical sampling (1.0 = disabled)
    float   temp              = 0.80f; ///< Temperature (1.0 = disabled)
    /// @}

    /// @name Repetition Penalty
    /// @{
    int32_t penalty_last_n    = 64;    ///< Last n tokens to penalize (0 = disable, -1 = ctx size)
    float   penalty_repeat    = 1.10f; ///< Repeat penalty (1.0 = disabled)
    float   penalty_freq      = 0.00f; ///< Frequency penalty (0.0 = disabled)
    float   penalty_present   = 0.00f; ///< Presence penalty (0.0 = disabled)
    bool    penalize_nl       = true;  ///< Penalize newlines
    /// @}

    /// @name Mirostat Parameters  
    /// @{
    int32_t mirostat          = 0;     ///< 0=disabled, 1=mirostat, 2=mirostat 2.0
    float   mirostat_tau      = 5.00f; ///< Target entropy
    float   mirostat_eta      = 0.10f; ///< Learning rate
    /// @}

    /// @name Grammar Constraints
    /// @{
    std::string grammar;  ///< BNF-like grammar to constrain sampling
    /// @}

    /// @name Classifier-Free Guidance
    /// @{
    std::string cfg_negative_prompt; ///< Negative prompt for guidance
    float       cfg_scale     = 1.f; ///< Guidance strength
    /// @}

    /// @name Logit Bias
    /// @{
    std::unordered_map<llama_token, float> logit_bias; ///< Logit bias for specific tokens
    /// @}
} llama_sampling_params;

// general sampler context
// TODO: move to llama.h
struct llama_sampling_context {
    // parameters that will be used for sampling
    llama_sampling_params params;

    // mirostat sampler state
    float mirostat_mu;

    llama_grammar * grammar;

    // internal
    grammar_parser::parse_state parsed_grammar;

    // TODO: replace with ring-buffer
    std::vector<llama_token>      prev;
    std::vector<llama_token_data> cur;
};

#include "common.h"

/**
 * @brief Initialize a new sampling context
 * 
 * Creates and initializes a new sampling context with the given parameters.
 * 
 * @param params Sampling parameters to use
 * @return Pointer to newly created sampling context
 */
struct llama_sampling_context * llama_sampling_init(const struct llama_sampling_params & params);

#ifdef __cplusplus
}
#endif

#endif // OPTIML_SAMPLING_H

/**
 * @brief Reset sampling context state
 * 
 * Clears previous tokens and resets grammar state while keeping parameters.
 * 
 * @param ctx Sampling context to reset
 */
void llama_sampling_reset(llama_sampling_context * ctx);

/**
 * @brief Copy sampling context
 * 
 * Copies all state and parameters from source to destination context.
 * 
 * @param src Source sampling context
 * @param dst Destination sampling context
 */
void llama_sampling_cp(llama_sampling_context * src, llama_sampling_context * dst);

/**
 * @brief Get last sampled token
 * 
 * @param ctx Sampling context
 * @return Last sampled token
 */
llama_token llama_sampling_last(llama_sampling_context * ctx);

/**
 * @brief Get string representation of recent tokens
 * 
 * @param ctx_sampling Sampling context
 * @param ctx_main Main llama context
 * @param n Number of recent tokens to include
 * @return String representation of tokens
 */
std::string llama_sampling_prev_str(llama_sampling_context * ctx_sampling, llama_context * ctx_main, int n);

/**
 * @brief Format sampling parameters as string
 * 
 * @param params Sampling parameters
 * @return Formatted string representation
 */
std::string llama_sampling_print(const llama_sampling_params & params);

/**
 * @brief Sample next token
 * 
 * Main sampling function that selects next token based on current state.
 * 
 * @param ctx_sampling Sampling context with parameters and state
 * @param ctx_main Main llama context with model and logits
 * @param ctx_cfg Optional context for classifier-free guidance
 * @param idx Index of logits to sample from (default 0)
 * @return Sampled token
 * 
 * @note When using multiple sequences, caller must reset sampling context
 *       when a sequence ends.
 */
llama_token llama_sampling_sample(
        struct llama_sampling_context * ctx_sampling,
        struct llama_context * ctx_main,
        struct llama_context * ctx_cfg,
        int idx = 0);

/**
 * @brief Accept sampled token
 * 
 * Updates sampling context with accepted token and applies grammar rules.
 * 
 * @param ctx_sampling Sampling context
 * @param ctx_main Main llama context
 * @param id Accepted token ID
 * @param apply_grammar Whether to apply grammar rules
 */
void llama_sampling_accept(
        struct llama_sampling_context * ctx_sampling,
        struct llama_context * ctx_main,
        llama_token id,
        bool apply_grammar);
