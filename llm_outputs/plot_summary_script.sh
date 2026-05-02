#!/bin/bash

# for city in beirut cairo rabat riyadh; do
#     python plot_summary_deltas.py \
#     "allam_steered_${city}_monolingual_l14_c3.0/summary.csv" \
#     "allam_baseline_monolingual/summary.csv"
# done

for city in beirut cairo rabat riyadh; do
    python plot_summary_deltas.py \
    "allam_steered_${city}_crosslingual_l14_c3.0/summary.csv" \
    "allam_baseline_crosslingual/summary.csv"
done

# for city in beirut cairo rabat riyadh; do
#     python plot_summary_deltas.py \
#     "allam_steered_${city}_mt_l14_c3.0/summary.csv" \
#     "allam_baseline_mt/summary.csv"

#     python plot_summary_deltas.py \
#     "allam_steered_${city}_mt_l14_c3.0/summary.csv" \
#     "allam_baseline_mt/summary.csv" \
#     --metric SpBLEU_corpus_score

#     python plot_summary_deltas.py \
#     "allam_steered_${city}_mt_l14_c3.0/summary.csv" \
#     "allam_baseline_mt/summary.csv" \
#     --metric ChrF_corpus_score
# done
