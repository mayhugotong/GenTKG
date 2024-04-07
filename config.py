import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Config of Llama2-lora")

    parser.add_argument("--MICRO_BATCH_SIZE", type=int, default=2, help="Per device train batch size") #ori: 4
    parser.add_argument("--BATCH_SIZE", type=int, default=128, help="batch size") #ori: 128
    parser.add_argument('--EPOCHS', type=int, default=50, help='Training epochs') #ori: 100
    parser.add_argument('--WARMUP_STEPS', type=int, default=100, help='Warmup steps')
    parser.add_argument('--LEARNING_RATE', type=float, default= 3e-4 , help='Training learning rate') #ori: 2e-5
    parser.add_argument('--CONTEXT_LEN', type=int, default=256, help='Truncation length of context (in json)') #ori: 256
    parser.add_argument('--TARGET_LEN', type=int, default=36, help='Truncation length of target (in json)') #ori: 256
    parser.add_argument('--TEXT_LEN', type=int, default=256, help='Truncation length of text (in txt)')
    parser.add_argument('--LORA_R', type=int, default=8, help='Lora low rank')
    parser.add_argument('--LORA_ALPHA', type=int, default=16, help='Lora Alpha')
    parser.add_argument('--LORA_DROPOUT', type=float, default=0.05, help='Lora dropout')
    parser.add_argument('--MODEL_NAME', type=str, default="TheBloke/Llama-2-7B-fp16", help='Model name') #ori: THUDM/chatglm2-6b
    parser.add_argument('--LOGGING_STEPS', type=int, default=10, help='Logging steps in training') #ori 128
    parser.add_argument('--LOAD_BEST_MODEL_AT_END', type=int, default=0, help='set 1 to save the best checkpoint')
    parser.add_argument('--OUTPUT_DIR', type=str, default="./output_model", help='Output dir')
    parser.add_argument('--DATA_PATH', type=str, default="./train_data.json", help='Input dir of trainset')
    parser.add_argument('--DATA_TYPE', type=str, choices= ["json" , "txt"], default="json", help='Input trainsetfile type')
    
    parser.add_argument('--EVAL_STRATEGY', type=str, default="no", help='eval by the history fact file')
    parser.add_argument('--EVAL_BY_HF', type=int, default=1, help='set 1 to eval by the history fact file')
    parser.add_argument('--EVAL_PATH', type=str, default=None, help='Input dir of evalset')
    parser.add_argument('--EVAL_TYPE', type=str, choices= ["json" , "txt"], default="txt", help='Input evalset file type')
    parser.add_argument('--EVAL_STEPS', type=int, default=10, help='Eval the model according to steps')
    
    parser.add_argument('--SAVE_STEPS', type=int, default=20, help='Save the model according to steps')
    parser.add_argument('--SAVE_TOTAL_LIMIT', type=int, default=None, help='The number of the checkpoint you will save (Excluding the final one)')
    parser.add_argument('--BIT_8', default=False, action="store_true", help='Use 8-bit')
    parser.add_argument('--BIT_4', default=False, action="store_true", help='Use 4-bit')

    parser.add_argument('--REPORT_TO', type=str, default=None, help='logging to e.g. wandb')
    parser.add_argument('--PROJ_NAME', type=str, default=None, help='Project name for e.g. wandb')
    parser.add_argument('--RUN_NAME', type=str, default=None, help='Run name for e.g. wandb')

    parser.add_argument('--W_RESUME', type=int, default=0, help='set 1 to enable WANDB_RESUME')
    parser.add_argument('--W_ID', type=str, default=0, help='set 1 to enable WANDB_RESUME')

    return parser.parse_args()