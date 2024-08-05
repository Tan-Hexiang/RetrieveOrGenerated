This script is being modified.
### Requirement
- Environment: `requirements.txt`
- To access llama, you need to download [llama-2-7b/13b-chat](https://huggingface.co/meta-llama) and put them in `llama_model/13b-chat` or `llama_model/7b-chat`.             
- To access gpt-3.5 and gpt-4, you need to add your openai key in `src/key.py: used_keys(list)`. We recommend using multiple keys from independent openai accounts to improve speed. Our program will automatically split the data and make parallel requests based on the number of keys.

### 1. Answer with only Retrieved Contexts

```
python read.py \
        --dataset nq \
        --subset_name nq \
        --pid 1 \
        --retrieved_name contriever \
        --ctxs_file backgrounds-retrieval/nq/retrieval_result.jsonl \
        --ctxs_key contriever \
        --ctxs_num 1 \
        --generated_name none \
        --generated_num 0 \
        --generated_file none \
        --generated_key none \
        --merge_order random \
        --engine gpt-3.5-turbo-0613 \
        --subdir prompt_similar_length
```

### 2. Answer with only Generated Contexts

```
python read.py \
        --dataset nq \
        --subset_name nq \
        --pid 1 \
        --retrieved_name none \
        --ctxs_file none \
        --ctxs_key none \
        --ctxs_num 0 \
        --generated_name gpt-3.5-turbo-0613 \
        --generated_num 1 \
        --generated_file Generated-context-greedy-gpt-3.5-turbo-0613/nq/gen_with_similar_length.jsonl \
        --generated_key response \
        --merge_order random \
        --engine gpt-3.5-turbo-0613 \
        --subdir prompt_similar_length
```

### 3. Answer with only Hybrid Contexts

```
python read.py \
        --dataset nq \
        --subset_name nq \
        --pid 1 \
        --retrieved_name contriever \
        --ctxs_file backgrounds-retrieval/nq/retrieval_result.jsonl \
        --ctxs_key contriever \
        --ctxs_num 1 \
        --generated_name gpt-3.5-turbo-0613 \
        --generated_num 1 \
        --generated_file Generated-context-greedy-gpt-3.5-turbo-0613/nq/gen_with_similar_length.jsonl \
        --generated_key response \
        --merge_order random \
        --engine gpt-3.5-turbo-0613 \
        --subdir prompt_similar_length
```