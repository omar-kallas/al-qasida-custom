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

## LLM-as-judge with OpenRouter

You can score generated sample outputs with an OpenRouter judge model:

```
export OPENROUTER_API_KEY="<your key>"
python3 openrouter_judge.py ../llm_outputs/allam_steered_beirut_monolingual_l14_c3.0
```

This writes per-sample scores to
`../llm_outputs/<OUTPUT_DIR>/llm_judge_samples.jsonl`, per-directory aggregate
metrics to `../llm_outputs/<OUTPUT_DIR>/llm_judge_metrics.jsonl`, and a global
one-line-per-directory file to `../llm_outputs/llm_judge_directory_metrics.jsonl`.

Each sample receives integer 1-5 scores for `dialect_authenticity`,
`coherence`, `arabic_fluency`, and `msa_formality`. The judge returns scores
only, without rationales. Runs are resumable; existing `(file, row_index)` rows
are skipped unless `--force` is passed.

For steered output directories, `dialect_authenticity` is judged against the
steering dialect parsed from the directory name: `beirut` -> Lebanese Arabic,
`cairo` -> Egyptian Arabic, `rabat` -> Moroccan Arabic, and `riyadh` -> Saudi
Arabic. The sample filename dialect is still recorded as `file_dialect`.

If a matching prompt CSV is missing, the default `--missing-prompt-policy warn`
still judges the output and records `prompt_available: false`; use
`--missing-prompt-policy skip` or `--missing-prompt-policy error` for stricter
runs.

Useful smoke-test command:

```
python3 openrouter_judge.py --dry-run --sample-limit 20 ../llm_outputs/allam_steered_beirut_monolingual_l14_c3.0
```

Use `--max-samples-per-file` when you want a balanced cap from each sample CSV,
or `--sample-limit` / `--limit-samples` when you want a total per-directory cap.
Normal judging runs show a `tqdm` progress bar; pass `--no-progress` to disable
it.

## File contents

- `correct_scores.py`: script to redo a subset of evaluations (in the case of mistakes)
- `data_organizers.py`: contains classes for data organization, imported in `./evaluator.py` 
- `evaluator.py`: primary script to run AL-QASIDA evaluation
- `openrouter_judge.py`: LLM-as-judge scoring over `*_samples.csv` outputs using OpenRouter
- `jais.py`: contains function to run JAIS model (since this is significantly different from running the other HF models)
- `maps.py`: contains constants imported in `./evaluator.py`
- `tokens.py`: to store a personal HuggingFace token

## HuggingFace tokens

Some LLMs require a HuggingFace account token to run. To run `evaluator.py` with any of these models you must:

1. Obtain a token by following instructions on HuggingFace
2. Create a file in this directory named `tokens.py` (or edit the existing one)
3. In the file insert the text `HF_TOKEN="<insert your token here>"` (template already exists for your convenience)

After completing the steps in this README, please proceed to [`../humevals/README.md`](../humevals/README.md) 
