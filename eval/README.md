# `al-qasida/eval`

This includes the second (2nd) set of instructions for running AL-QASIDA: 
running the main evaluation (for monolingual, crosslingual, and MT tasks).

Please complete the steps in [`../data_processing/README.md`](../data_processing/README.md) before 
continuing here.

## Eval process

Run `python3 evaluator.py --data-dir ../data_processing/data` with different additional options to run AL-QASIDA evaluation

Different tasks:
- The *monolingual evaluation task* runs by default
- For the *crosslingual evaluation task*, set `--task crosslingual` 
- For the *MT task*, set `--task mt`

Set each of these tags to use different LLMs:
- AceGPT: `--llm acegpt`
- Llama-3: `--llm llama`
- Llama-3 (base): `--llm llama-base`
- JAIS: `--llm jais`
- SILMA: `--llm silma`

### Example commands

Examples:
- For Llama-3 with the monolingual task: `python3 evaluator.py --data-dir ../data_processing/data --llm llama`
- For Llama-3 (base) with the crosslingual task: `python3 evaluator.py --data-dir ../data_processing/data --llm llama-base --task crosslingual`
- For SILMA with the MT task: `python3 evaluator.py --data-dir ../data_processing/data --llm silma --task mt`

***Note***: Before running any of these commands, be sure to read the [HuggingFace tokens](#huggingface-tokens) section below. 

### Results

Resulting scores should appear in `../llm_outputs/<MODEL_NAME>_<TASK_NAME>/*metrics.csv`.

## File contents

- `correct_scores.py`: script to redo a subset of evaluations (in the case of mistakes)
- `data_organizers.py`: contains classes for data organization, imported in `./evaluator.py` 
- `evaluator.py`: primary script to run AL-QASIDA evaluation
- `jais.py`: contains function to run JAIS model (since this is significantly different from running the other HF models)
- `maps.py`: contains constants imported in `./evaluator.py`
- `tokens.py`: to store a personal HuggingFace token

## HuggingFace tokens

Some LLMs require a HuggingFace account token to run. To run `evaluator.py` with any of these models you must:

1. Obtain a token by following instructions on HuggingFace
2. Create a file in this directory named `tokens.py` (or edit the existing one)
3. In the file insert the text `HF_TOKEN="<insert your token here>"` (template already exists for your convenience)

After completing the steps in this README, please proceed to [`../humevals/README.md`](../humevals/README.md) 
