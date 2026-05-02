#!/bin/bash

declare -A city_args

city_args["beirut"]="syr pse"
city_args["cairo"]="egy sdn"
city_args["rabat"]="dza mar"
city_args["riyadh"]="sau kwt"

# for city in beirut cairo rabat riyadh; do
#     python plot_summary_deltas.py \
#     "allam_steered_${city}_monolingual_l14_c3.0/summary.csv" \
#     "allam_baseline_monolingual/summary.csv"
# done

# for city in beirut cairo rabat riyadh; do
#     python plot_summary_deltas.py \
#     "allam_steered_${city}_crosslingual_l14_c3.0/summary.csv" \
#     "allam_baseline_crosslingual/summary.csv"
# done

for city in beirut cairo rabat riyadh; do
    python plot_summary_deltas.py \
    "allam_steered_${city}_mt_l14_c3.0/summary.csv" \
    "allam_baseline_mt/summary.csv" \
    --dialects ${city_args[$city]}

    python plot_summary_deltas.py \
    "allam_steered_${city}_mt_l14_c3.0/summary.csv" \
    "allam_baseline_mt/summary.csv" \
    --metric SpBLEU_corpus_score \
    --dialects ${city_args[$city]}

    python plot_summary_deltas.py \
    "allam_steered_${city}_mt_l14_c3.0/summary.csv" \
    "allam_baseline_mt/summary.csv" \
    --metric ChrF_corpus_score \
    --dialects ${city_args[$city]}
done
