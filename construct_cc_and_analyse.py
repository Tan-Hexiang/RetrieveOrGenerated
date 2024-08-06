from src.combine import context_conflicting_dataset
# 7b-chat 13b-chat gpt-3.5-turbo-0613
reader = "13b-chat"
generator = "13b-chat"
# nq tqa
dataset = "nq"

# -----------------------------------------------------------------------------------------------
def full_name(name):
    if name in ['7b-chat', '13b-chat']:
        return f"llama_model/{name}"
    else:
        return name

path = {
    'com_path':'Answer-with-{}/{}/prompt_similar_length/Retrieved-contriever-1-Generated-{}-p1-trunclen-0.jsonl'.format(full_name(reader), dataset, generator),
    'gen_path':'Answer-with-{}/{}/prompt_similar_length/Retrieved-none-0-Generated-{}-p1-trunclen-0.jsonl'.format(full_name(reader), dataset, generator),
    'ir_path':'Answer-with-{}/{}/prompt_similar_length/Retrieved-contriever-1-Generated-none-p1-trunclen-0.jsonl'.format(full_name(reader), dataset),
    'llm_path': f'Answer-with-{full_name(reader)}/{dataset}/prompt_similar_length/Retrieved-none-1-Generated-none-p3-trunclen-0.jsonl'
}

    
data = context_conflicting_dataset(**path)
print("\n\nDiffGR : \n", data.get_diffgr())

# save cc dataset if need
# data.save_all_to_dict("your dir to save the cc datasets")
