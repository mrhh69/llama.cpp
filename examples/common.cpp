#include "common.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>


#define NEXT_ARG() if (++i >= argc) {invalid_param = true;break;}

bool gpt_params_parse(int argc, char ** argv, gpt_params & params) {
    // determine sensible default number of threads.
    // std::thread::hardware_concurrency may not be equal to the number of cores, or may return 0.
    if (params.n_threads == 0) {
        params.n_threads = std::max(1, (int32_t) std::thread::hardware_concurrency());
    }

    bool invalid_param = false;
    std::string arg;
    gpt_params default_params;

    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            NEXT_ARG();
            params.seed = std::stoi(argv[i]);
        } else if (arg == "-t" || arg == "--threads") {
            NEXT_ARG();
            params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-n" || arg == "--n_predict") {
            NEXT_ARG();
            params.n_predict = std::stoi(argv[i]);
        } else if (arg == "--top_k") {
            NEXT_ARG();
            params.top_k = std::stoi(argv[i]);
        } else if (arg == "-c" || arg == "--ctx_size") {
            NEXT_ARG();
            params.n_ctx = std::stoi(argv[i]);
        } else if (arg == "--memory_f32") {
            params.memory_f16 = false;
        } else if (arg == "--top_p") {
            NEXT_ARG();
            params.top_p = std::stof(argv[i]);
        } else if (arg == "--temp") {
            NEXT_ARG();
            params.temp = std::stof(argv[i]);
        } else if (arg == "--repeat_last_n") {
            NEXT_ARG();
            params.repeat_last_n = std::stoi(argv[i]);
        } else if (arg == "--repeat_penalty") {
            NEXT_ARG();
            params.repeat_penalty = std::stof(argv[i]);
        } else if (arg == "-m" || arg == "--model") {
            NEXT_ARG();
            params.model = argv[i];
        } else if (arg == "--embedding") {
            params.embedding = true;
        } else if (arg == "--mlock") {
            params.use_mlock = true;
        } else if (arg == "--no-mmap") {
            params.use_mmap = false;
        } else if (arg == "--n_parts") {
            NEXT_ARG();
            params.n_parts = std::stoi(argv[i]);
        } else if (arg == "-h" || arg == "--help") {
            gpt_print_usage(argc, argv, default_params);
            exit(0);
        } else if (arg == "--in-prefix") {
            NEXT_ARG();
            params.input_prefix = argv[i];
        } else if (arg == "--always_reload") {
          params.always_reload = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            gpt_print_usage(argc, argv, default_params);
            exit(1);
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        gpt_print_usage(argc, argv, default_params);
        exit(1);
    }

    return true;
}

void gpt_print_usage(int /*argc*/, char ** argv, const gpt_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for <= 0)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  --in-prefix STRING    string to prefix user inputs with (default: empty)\n");
    fprintf(stderr, "  -n N, --n_predict N   number of tokens to predict (default: %d, -1 - infinity)\n", params.n_predict);
    fprintf(stderr, "  --top_k N             top-k sampling (default: %d)\n", params.top_k);
    fprintf(stderr, "  --top_p N             top-p sampling (default: %.1f)\n", params.top_p);
    fprintf(stderr, "  --repeat_last_n N     last n tokens to consider for penalize (default: %d)\n", params.repeat_last_n);
    fprintf(stderr, "  --repeat_penalty N    penalize repeat sequence of tokens (default: %.1f)\n", params.repeat_penalty);
    fprintf(stderr, "  -c N, --ctx_size N    size of the prompt context (default: %d)\n", params.n_ctx);
    fprintf(stderr, "  --ignore-eos          ignore end of stream token and continue generating\n");
    fprintf(stderr, "  --memory_f32          use f32 instead of f16 for memory key+value\n");
    fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", params.temp);
    fprintf(stderr, "  --n_parts N           number of model parts (default: -1 = determine from dimensions)\n");
    if (llama_mlock_supported()) {
        fprintf(stderr, "  --mlock               force system to keep model in RAM rather than swapping or compressing\n");
    }
    if (llama_mmap_supported()) {
        fprintf(stderr, "  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)\n");
    }
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  --always_restart      server reloads model each time\n");
    fprintf(stderr, "\n");
}

// TODO: not great allocating this every time
std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
    // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
    std::vector<llama_token> res(text.size() + (int)add_bos);
    int n = llama_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
    assert(n >= 0);
    res.resize(n);

    return res;
}
