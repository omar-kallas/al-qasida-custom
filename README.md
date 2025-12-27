# al-qasida

Welcome! This is the official repo for AL-QASIDA (Robinson et al., 2025), 
the first comprehensive evaluation for LLM dialectal Arabic proficiency, presented in 
this paper: 
[AL-QASIDA: **A**nalyzing **L**LM **Q**uality and **A**ccuracy **S**ystematically **i**n **D**ialectal **A**rabic](https://arxiv.org/abs/2412.04193)

Please cite:

```
@inproceedings{robinson-etal-2025-al,
    title = "{AL}-{QASIDA}: Analyzing {LLM} Quality and Accuracy Systematically in Dialectal {A}rabic",
    author = "Robinson, Nathaniel Romney  and
      Abdelmoneim, Shahd  and
      Marchisio, Kelly  and
      Ruder, Sebastian",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1137/",
    doi = "10.18653/v1/2025.findings-acl.1137",
    pages = "22048--22065",
    ISBN = "979-8-89176-256-5",
    abstract = "Dialectal Arabic (DA) varieties are under-served by language technologies, particularly large language models (LLMs). This trend threatens to exacerbate existing social inequalities and limits LLM applications, yet the research community lacks operationalized performance measurements in DA. We present a framework that comprehensively assesses LLMs' DA modeling capabilities across four dimensions: fidelity, understanding, quality, and diglossia. We evaluate nine LLMs in eight DA varieties and provide practical recommendations. Our evaluation suggests that LLMs do not produce DA as well as they understand it, not because their DA fluency is poor, but because they are reluctant to generate DA. Further analysis suggests that current post-training can contribute to bias against DA, that few-shot examples can overcome this deficiency, and that otherwise no measurable features of input text correlate well with LLM DA performance."
}
```

⚠️ Recent updates:

- 3 Dec 2025 - Switch to using templative prompts by default for models labeled "Instruct"

- 11 Nov 2025 - Note this repo has been updated to be used for *dev* by default, rather than testing (for the [AMIYA](https://sites.google.com/view/vardial-2026/shared-tasks) shared task)

## Current to-dos 

To update soon:
- 🚧 I noticed an issue that the `pre_humeval` outputs for MT tasks are not getting written properly without use of `eval/correct_scores.py`, due to absense of `llm_outputs/**/*samples*` files. Need to address this ASAP.
- 🚧 Need to rename hehe --> cohere, TVD --> ADI2
- 🚧 Make tutorial video in Arabic

## Tutorial video

See our video tutorial for running AL-QASIDA [here](https://youtu.be/_BVEitNmtCI). Or feel free to go through the instructions at your own pace. 

## Instructions

To set up the environment, run:

```
bash setup.sh
```

For instructions on running AL-QASIDA, refer to 

1. [`data_processing/README.md`](data_processing/README.md)
2. [`eval/README.md`](eval/README.md)
3. [`humevals/README.md`](humevals/README.md)

in that order. 

You could also skip to step #3 if you don't want to run AL-QASIDA on your own model. 
The results for the models we evaluated in our paper are already in `llm_outputs`.

## Contact 

If you have any questions or difficulty running our evaluation, please feel free to raise issues or 
pull requests, or to contact the authors. 
(Primary contact: [nrobin38@jhu.edu](mailto:nrobin38@jhu.edu))
