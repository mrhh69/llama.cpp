// Various helper functions and utilities

#pragma once

#include "llama.h"

#include <string>
#include <vector>
#include <random>
#include <thread>

//
// CLI argument parsing
//

struct gpt_params {
    int32_t seed          = -1;   // RNG seed
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_predict     = 128;  // new tokens to predict
    int32_t repeat_last_n = 64;   // last n tokens to penalize
    int32_t n_parts       = -1;   // amount of model parts (-1 = determine from model dimensions)
    int32_t n_ctx         = 512;  // context size

    // sampling parameters
    int32_t top_k = 40;
    float   top_p = 0.95f;
    float   temp  = 0.80f;
    float   repeat_penalty  = 1.10f;

    std::string model  = "models/lamma-7B/ggml-model.bin"; // model path
    std::string input_prefix = ""; // string to prefix user inputs with

    bool memory_f16        = true;  // use f16 instead of f32 for memory kv

    bool embedding         = false; // get only sentence embedding

    bool use_mlock         = false; // use mlock to keep model in memory

    // for server mode:
    bool always_reload     = false; // load ctx within the request handler, rather than at server start
};

bool gpt_params_parse(int argc, char ** argv, gpt_params & params);

void gpt_print_usage(int argc, char ** argv, const gpt_params & params);

//
// Vocab utils
//

std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos);
