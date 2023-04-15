#!/bin/bash

#
# Temporary script - will be removed in the future
#

cd ~/Downloads/llama/

<<<<<<< HEAD
llama.cpp/bin/main --color --instruct --threads 4 --mlock \
       --model ./models/ggml-gpt4all-q4.bin \
       --file ./llama.cpp/prompts/alpaca.txt \
       --batch_size 1024 --ctx_size 2048 \
=======
./main --color --instruct --threads 4 \
       --model ./models/gpt4all-7B/gpt4all-lora-quantized.bin \
       --file ./prompts/alpaca.txt \
       --batch_size 8 --ctx_size 2048 -n -1 \
>>>>>>> upstream/master
       --repeat_last_n 64 --repeat_penalty 1.3 \
       --n_predict 2048 --temp 0.3 --top_k 40 --top_p 0.95
