CUDA_VISIBLE_DEVICES=0 python3 main.py --OUTPUT_DIR "./model/output_model_icews14_1024" --DATA_PATH "./data/processed/train/icews14/icews14_1024.json"

CUDA_VISIBLE_DEVICES=1 python3 main.py --OUTPUT_DIR "./model/output_model_icews18_1024" --DATA_PATH "./data/processed/train/icews18/icews18_1024.json"

CUDA_VISIBLE_DEVICES=2 python3 main.py --OUTPUT_DIR "./model/output_model_GDELT_1024" --DATA_PATH "./data/processed/train/GDELT/GDELT_1024.json"

CUDA_VISIBLE_DEVICES=3 python3 main.py --OUTPUT_DIR "./model/output_model_YAGO_1024" --DATA_PATH "./data/processed/train/YAGO/YAGO_1024.json"
