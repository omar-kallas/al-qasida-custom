#!/bin/bash

for i in $(seq 14 20); do
    for j in $(seq 1 5); do
        python aggregate_results.py "allam_steered_cairo_monolingual_l${i}_c${j}.0"
    done
done


for i in $(seq 14 20); do
    for j in $(seq 1 5); do
        python aggregate_results.py "allam_steered_cairo_mt_l${i}_c${j}.0"
    done
done