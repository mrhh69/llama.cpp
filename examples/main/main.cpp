// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common.h"
#include "llama.h"
#include "build-info.h"

#include "server.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

static llama_context * ctx;
static gpt_params params;
static std::vector<llama_token> inp_pfx; // prefix & suffix for instruct mode
static std::vector<llama_token> inp_sfx;


static void model_load () {
    // load the model and apply lora adapter, if any
    ctx = llama_init_from_gpt_params(params);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        exit(1);
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

	inp_pfx = ::llama_tokenize(ctx, " Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n", true);
	inp_sfx = ::llama_tokenize(ctx, "\n\n### Response:\n", false);
}


void process (
	char *s
) {
    bool is_interacting;
	std::string prompt = std::string(s);

	/* if (params.always_reload) {
		model_load();
	} */

    auto embd_inp = ::llama_tokenize(ctx, prompt, true);

    const int n_ctx = llama_n_ctx(ctx);

    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        exit(1);
    }


    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct) {
        params.n_keep = (int)embd_inp.size();
    }

    // prefix & suffix for instruct mode
    /* const auto inp_pfx = ::llama_tokenize(ctx, "\n\n### Instruction:\n\n", true);
    const auto inp_sfx = ::llama_tokenize(ctx, "\n\n### Response:\n\n", false); */

    // enable interactive mode if reverse prompt or interactive start is specified
    if (params.antiprompt.size() != 0 || params.interactive_first) {
        params.interactive = true;
    }

    // determine newline token
    auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    if (params.interactive) {

        fprintf(stderr, "%s: interactive mode on.\n", __func__);

        if (params.antiprompt.size()) {
            for (auto antiprompt : params.antiprompt) {
                fprintf(stderr, "Reverse prompt: '%s'\n", antiprompt.c_str());
            }
        }
    }

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    if (params.interactive) {
        is_interacting = params.interactive_first;
    }

    bool is_antiprompt        = false;
    // bool input_echo           = true;

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;

    // the first thing we will do is to output the prompt, so set color accordingly
    std::vector<llama_token> embd;

    while (n_remain != 0 || params.interactive) {
        // predict
        if (embd.size() > 0) {
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int) embd.size() > n_ctx) {
                const int n_left = n_past - params.n_keep;

                // always keep the first token - BOS
                n_past = std::max(1, params.n_keep);

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());
            }

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }
                if (llama_eval(ctx, &embd[i], n_eval, n_past, params.n_threads)) {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    exit(1);
                }
                n_past += n_eval;
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // out of user input, sample next token
            const float   temp            = params.temp;
            const int32_t top_k           = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
            const float   top_p           = params.top_p;
            const float   tfs_z           = params.tfs_z;
            const float   typical_p       = params.typical_p;
            const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
            const float   repeat_penalty  = params.repeat_penalty;
            const float   alpha_presence  = params.presence_penalty;
            const float   alpha_frequency = params.frequency_penalty;
            const int     mirostat        = params.mirostat;
            const float   mirostat_tau    = params.mirostat_tau;
            const float   mirostat_eta    = params.mirostat_eta;
            const bool    penalize_nl     = params.penalize_nl;

            llama_token id = 0;

            {
                auto logits  = llama_get_logits(ctx);
                auto n_vocab = llama_n_vocab(ctx);

                // Apply params.logit_bias map
                for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                    logits[it->first] += it->second;
                }

                std::vector<llama_token_data> candidates;
                candidates.reserve(n_vocab);
                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                }

                llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                // Apply penalties
                float nl_logit = logits[llama_token_nl()];
                auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
                llama_sample_repetition_penalty(ctx, &candidates_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, repeat_penalty);
                llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, alpha_frequency, alpha_presence);
                if (!penalize_nl) {
                    logits[llama_token_nl()] = nl_logit;
                }

                if (temp <= 0) {
                    // Greedy sampling
                    id = llama_sample_token_greedy(ctx, &candidates_p);
                } else {
                    if (mirostat == 1) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        const int mirostat_m = 100;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                    } else if (mirostat == 2) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                    } else {
                        // Temperature sampling
                        llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                        llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                        llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                        llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token(ctx, &candidates_p);
                    }
                }

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // replace end of text token with newline token when in interactive mode
            if (id == llama_token_eos() && params.interactive && !params.instruct) {
                id = llama_token_newline.front();
                if (params.antiprompt.size() != 0) {
                    // tokenize and inject first reverse prompt
                    const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false);
                    embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                }
            }

            // add it to the context
            embd.push_back(id);

            // echo this to console
            // input_echo = true;

            // decrement remaining sampling budget
            --n_remain;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        // if (input_echo) {
            for (auto id : embd) {
                printf("%s", llama_token_to_str(ctx, id));
                new_token_str(llama_token_to_str(ctx, id));
            }
        // }

        // in interactive mode, and not currently processing queued inputs;
        // check if we should prompt the user for more
        if (params.interactive && (int) embd_inp.size() <= n_consumed) {

            // check for reverse prompt
            if (params.antiprompt.size()) {
                std::string last_output;
                for (auto id : last_n_tokens) {
                    last_output += llama_token_to_str(ctx, id);
                }

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                for (std::string & antiprompt : params.antiprompt) {
                    if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
                        is_interacting = true;
                        is_antiprompt = true;
                        break;
                    }
                }
            }

            if (n_past > 0 && is_interacting) {
                std::string buffer;
                if (!params.input_prefix.empty()) {
                    buffer += params.input_prefix;
                    printf("%s", buffer.c_str());
                }

                std::string line;
                // bool another_line = true;
                do {
                    // TODO: add this to server.c
                    line = std::string(sock_readline());
                    buffer += line;
                } while (false);

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {
                    // append input suffix if any
                    if (!params.input_suffix.empty()) {
                        buffer += params.input_suffix;
                        // printf("%s", params.input_suffix.c_str());
                        new_token_str(params.input_suffix.c_str());
                    }

                    /* // instruct mode: insert instruction prefix
                    if (params.instruct && !is_antiprompt) {
                        n_consumed = embd_inp.size();
                        embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
                    } */

                    auto line_inp = ::llama_tokenize(ctx, buffer, false);
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

                    /* // instruct mode: insert response suffix
                    if (params.instruct) {
                        embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                    } */

                    n_remain -= line_inp.size();
                }

                // input_echo = false; // do not echo this again
            }

            if (n_past > 0) {
                is_interacting = false;
            }
        }

        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos()) {
            if (params.instruct) {
                is_interacting = true;
            } else {
                fprintf(stderr, " [end of text]\n");
                break;
            }
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        if (params.interactive && n_remain <= 0 && params.n_predict != -1) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }
}



int main(int argc, char ** argv) {
	params.model = "models/llama-7B/ggml-model.bin";

    fprintf(stderr, "%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

	if (gpt_params_parse(argc, argv, params) == false) {
		return 1;
	}

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    }
	if (params.seed <= 0) {
		params.seed = time(NULL);
	}

// load the model
	// if (!params.always_reload) {
		model_load();
	// }


	/* infinite server loop */
	server_loop();
}
