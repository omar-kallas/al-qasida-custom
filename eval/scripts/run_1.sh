python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_cairo" --use_config "allam_steered_cairo_config.json" --task "monolingual"
python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_riyadh" --use_config "allam_steered_riyadh_config.json" --task "monolingual"

python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_cairo" --use_config "allam_steered_cairo_config.json" --task "crosslingual"
python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_riyadh" --use_config "allam_steered_riyadh_config.json" --task "crosslingual"

python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_cairo" --use_config "allam_steered_cairo_config.json" --task "mt"
python evaluator.py --data-dir "../data_processing/data" --llm "allam_steered_riyadh" --use_config "allam_steered_riyadh_config.json" --task "mt"