python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_beirut" --use_config "allam_steered_beirut_config.json" --task "monolingual"
python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_rabat" --use_config "allam_steered_rabat_config.json" --task "monolingual"

python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_beirut" --use_config "allam_steered_beirut_config.json" --task "crosslingual"
python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_rabat" --use_config "allam_steered_rabat_config.json" --task "crosslingual"

python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_beirut" --use_config "allam_steered_beirut_config.json" --task "mt"
python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_rabat" --use_config "allam_steered_rabat_config.json" --task "mt"