#!/bin/bash

for i in $(seq 14 20); do
    python plot_directory_summary_deltas.py \
        "allam_steered_cairo_monolingual_l${i}_c1.0" \
        "allam_steered_cairo_monolingual_l${i}_c2.0" \
        "allam_steered_cairo_monolingual_l${i}_c3.0" \
        "allam_steered_cairo_monolingual_l${i}_c4.0" \
        "allam_steered_cairo_monolingual_l${i}_c5.0" \
        --reference allam_monolingual \
        --output "plots/mono_summary_l$i.png" \
        --dialects egy
done

for i in $(seq 14 20); do
    python plot_directory_summary_deltas.py \
        "allam_steered_cairo_mt_l${i}_c1.0" \
        "allam_steered_cairo_mt_l${i}_c2.0" \
        "allam_steered_cairo_mt_l${i}_c3.0" \
        "allam_steered_cairo_mt_l${i}_c4.0" \
        "allam_steered_cairo_mt_l${i}_c5.0" \
        --reference allam_baseline_mt \
        --output "plots/mt_summary_l$i.png" \
        --dialects eng-egy msa-egy
done
