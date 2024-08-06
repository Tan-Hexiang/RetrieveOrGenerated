# #  GPT 3.5 read without context
# python read.py \
#         --dataset tqa \
#         --pid 3 \
#         --retrieved_name none \
#         --ctxs_file backgrounds-retrieval/tqa/retrieval_result.jsonl \
#         --ctxs_key none \
#         --ctxs_num 1 \
#         --generated_name none \
#         --generated_num 0 \
#         --generated_file none \
#         --generated_key none \
#         --merge_order random \
#         --engine gpt-3.5-turbo-0613 \
#         --subdir prompt_similar_length

# #  GPT 4 read without context
# python read.py \
#         --dataset tqa \
#         --pid 3 \
#         --retrieved_name none \
#         --ctxs_file backgrounds-retrieval/tqa/retrieval_result.jsonl \
#         --ctxs_key none \
#         --ctxs_num 1 \
#         --generated_name none \
#         --generated_num 0 \
#         --generated_file none \
#         --generated_key none \
#         --merge_order random \
#         --engine gpt-4-0613 \
#         --subdir prompt_similar_length


# llama2-13b read without context
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 25676 read.py \
        --dataset tqa \
        --pid 3 \
        --retrieved_name none \
        --ctxs_file backgrounds-retrieval/tqa/retrieval_result.jsonl \
        --ctxs_key none \
        --ctxs_num 1 \
        --generated_name none \
        --generated_num 0 \
        --generated_file none \
        --generated_key none \
        --merge_order random \
        --engine llama_model/13b-chat \
        --batch_size 2 \
        --device_map balanced \
        --subdir prompt_similar_length

# llama2-7b read without context
    CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node 1 --master_port 25677 read.py \
        --dataset tqa \
        --pid 3 \
        --retrieved_name none \
        --ctxs_file backgrounds-retrieval/tqa/retrieval_result.jsonl \
        --ctxs_key none \
        --ctxs_num 1 \
        --generated_name none \
        --generated_num 0 \
        --generated_file none \
        --generated_key none \
        --merge_order random \
        --engine llama_model/7b-chat \
        --batch_size 2 \
        --device_map balanced \
        --subdir prompt_similar_length