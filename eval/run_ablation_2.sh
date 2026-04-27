python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_cairo" --use_config "configs/allam_steered_cairo_l16_c1_config.json" --task "mt" --dialects egy-eng egy-msa eng-egy msa-egy --out-dir ./llm_with_samples
python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_cairo" --use_config "configs/allam_steered_cairo_l16_c2_config.json" --task "mt" --dialects egy-eng egy-msa eng-egy msa-egy --out-dir ./llm_with_samples
python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_cairo" --use_config "configs/allam_steered_cairo_l16_c3_config.json" --task "mt" --dialects egy-eng egy-msa eng-egy msa-egy --out-dir ./llm_with_samples
python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_cairo" --use_config "configs/allam_steered_cairo_l16_c4_config.json" --task "mt" --dialects egy-eng egy-msa eng-egy msa-egy --out-dir ./llm_with_samples
python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_cairo" --use_config "configs/allam_steered_cairo_l16_c5_config.json" --task "mt" --dialects egy-eng egy-msa eng-egy msa-egy --out-dir ./llm_with_samples
runpodctl stop pod jhpu5fcoi5z3vw
