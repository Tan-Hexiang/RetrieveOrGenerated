# RetrieveOrGenerated
This is the code repository for the ACL2024 paper "Blinded by Generated Contexts: How Language Models Merge Generated and Retrieved Contexts for Open-Domain QA?"

We are currently organizing the code and data to enhance its readability and conciseness, and will upload it soon.

## Prepare Datasets

### Requirement

- Environment: `requirements.txt`
- To access llama, you need to download [llama-2-7b/13b-chat](https://huggingface.co/meta-llama) and put them in `llama_model/13b-chat` or `llama_model/7b-chat`.
- To access gpt-3.5 and gpt-4, you need to add your openai key in `src/key.py: used_keys(list)`. We recommend using multiple keys from independent openai accounts to improve speed. Our program will automatically split the data and make parallel requests based on the number of keys.

### Download QA Dataset

- Download QA datasets from the official website: [NQ/TQA](https://github.com/facebookresearch/DPR)
- We use a uniform format version of the dataset (NQ and TQA) from [GenRead](https://github.com/wyu97/GenRead?tab=readme-ov-file): [Link](https://drive.google.com/drive/folders/1lFFTklW_0HuR53hLpFdLClgfSAhXn_2f)
- Before the experiment, we merge `nq-test.jsonl` and `nq-dev.jsonl` to a single `nq.jsonl`; merge `tqa-test.jsonl` and `tqa-dev.jsonl` to a single `tqa.jsonl`

### 1. Prepare Retrieved Contexts

- We employ the official code of [contriever](https://github.com/facebookresearch/contriever) to get the top-1 passages. 
  <!-- We then process the retrieval result to fit our format. -->
- Due to the request for anonymity, we will share the link to our processed retrieval results data after the review is completed.

### 2. Prepare Generated Contexts

Generate contexts with different length constraints in the prompt.

```
for id in 3 4 5 6 7
do
    for dataset in nq tqa
    do
        python generate.py \
        --dataset $dataset \
        --type Gen \
        --split none \
        --engine gpt-3.5-turbo-0613 \
        --decoding greedy \
        --pid $id
    done
done
```

Find the contexts closest to the retrieved contexts in length.

```
python find_gen_ctx_length_similar_to_ret_cxt.py --all_gen_dir Generated-context-greedy-gpt-3.5-turbo-0613/nq --ret_file backgrounds-retrieval/nq/retrieval_result.jsonl
```

### 3. Answer with only Retrieved Contexts

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

### 4. Answer with only Generated Contexts

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

### 5. Answer with only Hybrid Contexts

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

### 6. Construct Context-Conflicting Datasets and Compute DiffGR

```
python compute_preference.py \
--generated_only_path Answer-with-gpt-3.5-turbo-0613/nq/prompt_similar_length/Retrieved-none-0-Generated-gpt-3.5-turbo-0613-p1-trunclen-0.jsonl \
--retrieved_only_path Answer-with-gpt-3.5-turbo-0613/nq/prompt_similar_length/Retrieved-contriever-1-Generated-none-p1-trunclen-0.jsonl \
--hybrid_path Answer-with-gpt-3.5-turbo-0613/nq/prompt_similar_length/Retrieved-contriever-1-Generated-gpt-3.5-turbo-0613-p1-trunclen-0.jsonl \
```

We will provide all the data and intermediate results after review.