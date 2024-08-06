# RetrieveOrGenerated
This is the code repository for the ACL2024 paper "Blinded by Generated Contexts: How Language Models Merge Generated and Retrieved Contexts When Knowledge Conflicts?"

## Quickly get the **C**ontext-**C**onflicting datasets (**CC**) and analysis result.
- Environment:
`pip install -r requirements.txt`
- Download prepared context and intermediate results to the project directory.
[link](https://drive.google.com/drive/folders/1Kisw8LXoa12kLm9XyT_1u4TJ64DgJfBR?usp=sharing)
- Build the CC dataset and calculate DiffGR.
`python construct_cc_and_analyse.py`

The resulting DiffGR matches Table 5, depicting the bias after excluding the effects of parametric knowledge.
By adjusting the parameters in `construct_cc_and_analyse.py`, you can obtain datasets with different combinations of readers and generators (**7b-chat, 13b-chat, gpt-4-0613 or gpt-3.5-turbo-0613**).
| Reader | Generator | Dataset | DiffGR | 
| ------- | ------- | ------- | ------- |
| 13b-chat   | 13b-chat   | NQ-AIR   | 0.5785 |
| 13b-chat   | 13b-chat   | NQ-AIG   | 0.9012 |
| 13b-chat   | 13b-chat   | TQA-AIR   | 0.6069 |
| 13b-chat   | 13b-chat   | TQA-AIG   | 0.8968 |


## Building the CC Dataset from Scratch
1. First, we need to prepare the retrieved context and generated context and ensure their lengths match, following `scripts/prepare_contexts.md`.

2. Next, LLM answers questions based on different contexts, following `scripts/answer_with_different_contexts.md`.
   - Without context: To determine LLM's parametric knowledge.
   - Using only one type of context (generated/retrieved): To determine what each type of context provides.
   - Using both generated and retrieved context simultaneously: To analyze which context LLM relies on.

3. Construct the CC dataset and analyze LLM preference:
`python construct_cc_and_analyse.py`
