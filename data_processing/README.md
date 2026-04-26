# `al-qasida/data_processing`

This contains is the first (1st) set of instructions for running AL-QASIDA: 
getting data ready. 

## Instructions for getting data ready 

### Download of FLORES

⚠️ Note this repo has been updated to be used for *dev* by default, rather than testing (for the [AMIYA](https://sites.google.com/view/vardial-2026/shared-tasks) shared task)

To download FLORES-200 data needed for this evaluation automatically, you can simply run `python download_flores.py`. 
After downloading, you should be able to run `wc -l bitexts/flores200_dataset/dev/*` 
and verify that each of the 7 files downloaded has length `1012`.

You can also download and situate FLORES-200 (or its living counterpart, FLORES+) manually, by visiting 
this [FLORES GitHub repo](https://github.com/facebookresearch/flores/blob/main/flores200/README.md) and follow instructions for download. 

Once you have downloaded the datasets, please organize them in the `./bitexts` directory so the files look 
like `./bitexts/flores200_dataset/dev/<XXX>.dev`, where `<XXX>` represents the FLORES code of each
Arabic language variety in the set (plus English). Those we included in our published study are:

- `arb_Arab`: Modern Standard Arabic
- `ary_Arab`: Moroccan Arabic 
- `arz_Arab`: Egyptian Arabic 
- `ajp_Arab`: South Levantine (Palestinian) Arabic (**NOTE**: Seemlingly not included in FLORES+, only in the original FLORES-200.)
- `apc_Arab`: North Levantine (Syrian) Arabic
- `ars_Arab`: Najdi (Saudi) Arabic 
- `eng_Latn`: English

For `create_dataset.py` to run without errors later, you should also have the `dev` sets downloaded (in 
addition to `dev`). This will happen automatically if you run `download_flores.py`. 

### Download of MADAR-26

To download MADAR-26, please visit the [download page](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/) 
and follow download instructions. 

Then organize them in `./bitexts` such that each file looks like `./bitexts/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/MADAR_Corpus/MADAR.corpus.<XXX>.tsv`, where `<XXX>` is the MADAR dialect city name. 
Those we used in our experiments were: 

- `MADAR.corpus.Riyadh.tsv` (Saudi)
- `MADAR.corpus.Damascus.tsv` (Syrian)
- `MADAR.corpus.Jerusalem.tsv` (Palestinian)
- `MADAR.corpus.Khartoum.tsv` (Sudanese)
- `MADAR.corpus.Cairo.tsv` (Egyptian)
- `MADAR.corpus.Algiers.tsv` (Algerian)
- `MADAR.corpus.Fes.tsv` (Moroccan)

Once this is accomplished, copy or move the provided file `bitexts/BTEC_for_MADAR/MADAR.corpus.English.tsv` 
to the directory `./bitexts/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/MADAR_Corpus` (to sit 
alongside the files it is aligned with). 

### Download of NADI-2023-TWT
NO NEED

#### MADAR-TWITTER

NO NEED

### HABIBI data

We provide the HABIBI data on this repo, since it was released with a CC-BY-NC license. 
Please do not restribute or use without crediting its original creators, and do not use for commercial 
purposes (per CC-BY-NC restrictions). Refer to [`./monotexts/HABIBI/README.md`](./monotexts/HABIBI/README.md).

## Processing data

Once all data is downloaded, you should be able to run the primary script in this directory, 
`create_dataset.py`. This should organize and store the data for your next step: 
running the AL-QASIDA evaluation in the [`../eval`](../eval) directory. 

Example command: 

```
python create_dataset.py
```

## Directory contents

- `analysis`: This directory contains on old preliminary analysis script of ours (safe to ignore).
- `bitexts`: This is where to place downloaded files from FLORES-200 and MADAR-26.
- `create_dataset.py`: the primary script in this directory, this organizes data and writes files to `./data` ready to be run through the AL-QASIDA evaluation.
- `data`: This is where outputs of `./create_dataset.py` are written (after pertinent datasets are downloaded).
- `format_xtext.py`: This contains necessary functions and classes to be imported in `./create_dataset.py`
- `monotexts`: This is where we host the HABIBI data and where the MADAR-Twitter data should be placed.
- `prompt_templates`: This directory contains prompt templates for monolingual commands (and the scripts and files to recreate them).
- `src2info.py`: This contains constants to be imported in `./create_dataset.py`.

We have already run `./prompt_templates/json/make_template_jsons.py`
to prepare prompt templates for monolingual commands; outputs are already provided. 
We just included the file in case people want to customize. 
If customizing, this script would need to be run after `create_dataset.py` before running 
the AL-QASIDA evaluation.

After completing the steps in this README, please proceed to [../eval/README.md](../eval/README.md)
