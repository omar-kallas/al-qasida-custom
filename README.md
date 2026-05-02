This repo is forked from the official Al-Qasida repo https://github.com/JHU-CLSP/al-qasida, please follow instructions below (original repo README) for usage, including data_processing and evaluation.
The modifications in this repo are meant to be used to evaluate steered LLMs in this repo https://github.com/KareemElozeiri/SteeringArabicDialects
To apply them, this repo must be cloned inside the `SteeringArabicDialects` repo and run there.

The below content is all from the original alqasida repo

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
