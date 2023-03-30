#include "common.h"
#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>



void process (
	llama_context * ctx,
	gpt_params params,
	std::vector<llama_token> inp_pfx,
	std::vector<llama_token> inp_sfx,
	std::string prompt
) {

	// tokenize the prompt
	auto embd_inp = ::llama_tokenize(ctx, prompt, false);
	// insert prefix/suffix
	embd_inp.insert(embd_inp.begin(), inp_pfx.begin(), inp_pfx.end());
	embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());




	const int n_ctx = llama_n_ctx(ctx);

	if ((int) embd_inp.size() > n_ctx - 4) {
		fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
		exit(1);
	}

	// TODO: replace with ring-buffer
	std::vector<llama_token> last_n_tokens(n_ctx);
	std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

	bool input_noecho = true;

	int n_past	 = 0;
	int n_remain   = params.n_predict;
	int n_consumed = 0;


	std::vector<llama_token> embd;

	while (n_remain != 0) {
		// predict
		if (embd.size() > 0) {
			// infinite text generation via context swapping
			// if we run out of context:
			// - take the n_keep first tokens from the original prompt (via n_past)
			// - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
			if (n_past + (int) embd.size() > n_ctx) {
				const int n_left = n_past;
				n_past = 0;

				// insert n_left/2 tokens at the start of embd from last_n_tokens
				embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());
			}

			if (llama_eval(ctx, embd.data(), embd.size(), n_past, params.n_threads)) {
				fprintf(stderr, "%s : failed to eval\n", __func__);
				exit(1);
			}
		}

		n_past += embd.size();
		embd.clear();


		if ((int) embd_inp.size() <= n_consumed) {
			// out of user input, sample next token
			const float top_k		  = params.top_k;
			const float top_p		  = params.top_p;
			const float temp		   = params.temp;
			const float repeat_penalty = params.repeat_penalty;

			llama_token id = 0;

			{
				id = llama_sample_top_p_top_k(ctx,
						last_n_tokens.data() + n_ctx - params.repeat_last_n,
						params.repeat_last_n, top_k, top_p, temp, repeat_penalty);

				last_n_tokens.erase(last_n_tokens.begin());
				last_n_tokens.push_back(id);
			}

			// add it to the context
			embd.push_back(id);

			// echo this to console
			input_noecho = false;

			// decrement remaining sampling budget
			--n_remain;
		} else {
			// some user input remains from prompt or interaction, forward it to processing
			while ((int) embd_inp.size() > n_consumed) {
				embd.push_back(embd_inp[n_consumed]);
				last_n_tokens.erase(last_n_tokens.begin());
				last_n_tokens.push_back(embd_inp[n_consumed]);
				++n_consumed;
			}
		}

    /* THIS is the part that will be sent over the server */
		// display text
		if (!input_noecho) {
			for (auto id : embd) {
				printf("%s", llama_token_to_str(ctx, id));
			}
			fflush(stdout);
		}

		// end of text token
		if (embd.back() == llama_token_eos()) {
      /* die here */
			fprintf(stderr, " [end of text]\n");
			break;
		}
  }
}



int main(int argc, char ** argv) {
	gpt_params params;
	params.model = "models/llama-7B/ggml-model.bin";

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


	llama_context * ctx;

	// load the model
	{
		auto lparams = llama_context_default_params();

		lparams.n_ctx	  = params.n_ctx;
		lparams.n_parts	= params.n_parts;
		lparams.seed	   = params.seed;
		lparams.f16_kv	 = params.memory_f16;
		lparams.use_mlock  = params.use_mlock;

		ctx = llama_init_from_file(params.model.c_str(), lparams);

		if (ctx == NULL) {
			fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
			return 1;
		}
	}

	// print system information
	{
		fprintf(stderr, "\n");
		fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
				params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
	}



	// prefix & suffix for instruct mode
	const auto inp_pfx = ::llama_tokenize(ctx, " Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n", true);
	const auto inp_sfx = ::llama_tokenize(ctx, "\n\n### Response:\n", false);


  /* HERE: I am going to add the server part, and then each request has a new prompt */

  std::string prompt = "tell me about alpacas";
  std::string prompt2 = "do I have rizz?";

	// Add a space in front of the first character to match OG llama tokenizer behavior
	//prompt.insert(0, 1, ' ');
  //prompt2.insert(0, 1, ' ');


  process(ctx, params, inp_pfx, inp_sfx, prompt);

  printf("\n\nnext\n\n");

  process(ctx, params, inp_pfx, inp_sfx, prompt2);



	llama_free(ctx);

	return 0;
}
