This script is being modified.
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