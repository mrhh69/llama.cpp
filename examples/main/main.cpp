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
#include <signal.h>

#include <sys/select.h>

#include "ws.h"


ws_cli_conn_t *cur_client;
int cur_pid;
int cur_pipe[2];


llama_context * ctx;
gpt_params params;
std::vector<llama_token> inp_pfx;
std::vector<llama_token> inp_sfx;
// prefix & suffix for instruct mode

void new_token_str(const char *s) {
	//printf("%s", s);
	//fflush(stdout);
	char c = 1;
	write(cur_pipe[1], &c, 1);
	write(cur_pipe[1], s, strlen(s) + 1);
	//if (cur_client == NULL) {printf("no client???"); return;}
	//ws_sendframe_txt(cur_client, s);
}


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
				new_token_str(llama_token_to_str(ctx, id));
				//printf("%s", llama_token_to_str(ctx, id));
			}
			fflush(stdout);
		}

		// end of text token
		if (embd.back() == llama_token_eos()) {
      /* die here */
			//fprintf(stderr, " [end of text]\n");
			break;
		}
  }
	/* tell that done */
	char c = 2;
	write(cur_pipe[1], &c, 1);
	close(cur_pipe[1]);
}




void onopen(ws_cli_conn_t *client) {
	if (cur_client) {
		ws_sendframe_txt(client, "busy");
		ws_close_client(client);
	}
	else {
		printf("opened!\n");
		cur_client = client;
		ws_sendframe_txt(client, "ok");
	}
}
void onclose(ws_cli_conn_t *client) {
	printf("closed...\n");
	if (cur_pipe[0]) close(cur_pipe[0]);
}
void onmessage(ws_cli_conn_t *client, const unsigned char *msg, uint64_t size, int type) {
	//printf("message!! (%i)\n", type);
	if (cur_client == client) {
		if (!cur_pid) {
			printf("prompt: '%s'\n", msg);
			std::string prompt = std::string((char *)msg);
			pipe(cur_pipe);
			if ((cur_pid = fork()) == 0) {
				close(cur_pipe[0]);
				process(ctx, params, inp_pfx, inp_sfx, prompt);
				exit(0);
			}
			close(cur_pipe[1]);
			cur_pipe[1] = 0;
		}
		else {
			//printf("ping (kinda)!\n");
			if (cur_client) {
				/* try read from pipe here */
				fd_set set;
				struct timeval tv = {0, 0};
				for (;;) {
					FD_ZERO(&set);
					FD_SET(cur_pipe[0], &set);
					if (select(FD_SETSIZE, &set, NULL, NULL, &tv) == -1) {printf("select\n"); exit(1);}
					if ((FD_ISSET(cur_pipe[0], &set))) {
						/* fd's ready */
						char c, buf[256];
						read(cur_pipe[0], &c, 1);
						//printf("cmd:%u\n", c);
						if (c == 1) {
							int i = 0;
							do {
								read(cur_pipe[0], &buf[i], 1);
							} while (buf[i++]);
							//printf("%s\n", buf);
							ws_sendframe_txt(cur_client, &buf[0]);
						}
						else if (c == 2) {
							printf("done!\n");
							ws_close_client(cur_client);
							cur_client = 0;
							close(cur_pipe[0]);
							cur_pipe[0] = 0;
							kill(cur_pid, SIGKILL);
							cur_pid = 0;
							return;
						}
					}
					else break;
				}
			}
		}
	}
}


int main(int argc, char ** argv) {
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

	inp_pfx = ::llama_tokenize(ctx, " Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n", true);
	inp_sfx = ::llama_tokenize(ctx, "\n\n### Response:\n", false);


	struct ws_events evs;
  evs.onopen    = &onopen;
  evs.onclose   = &onclose;
  evs.onmessage = &onmessage;

	/* infinite server loop */
	ws_socket(&evs, 3002, 0, -1);


	llama_free(ctx);

	return 0;
}
