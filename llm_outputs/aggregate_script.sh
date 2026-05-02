#!/bin/bash

for task in monolingual crosslingual mt; do
    for city in beirut cairo rabat riyadh; do
        python aggregate_results.py "allam_steered_${city}_${task}_l14_c3.0"
    done
done
