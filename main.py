from config import parse_args
from utils import get_lora_config, get_model_and_tokenizer, process_data, get_trainer, get_dataset
import wandb
import re
from datetime import datetime
import os

def set_wandb_para(proj_name):
    os.environ["WANDB_PROJECT"] = proj_name # name your W&B project 
    os.environ["WANDB_LOG_MODEL"] = "checkpoint" # log all model checkpoints
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB__SERVICE_WAIT"] = "300" #prevent issue https://github.com/wandb/wandb/issues/3911

def prepair_data(args, tokenizer):
    if args.EVAL_PATH is not None and args.EVAL_STRATEGY != "no": #using separate train and eval sets
        train_dataset = get_dataset(args.DATA_TYPE, args.DATA_PATH)
        if args.EVAL_BY_HF: 
            with open(args.EVAL_PATH, 'r', encoding='utf-8') as file:
                content = file.read() # Read the entire document content, all tests together
             # Separate content with blank lines as parts to get each test individually
            eval_dataset = content.split('\n\n')
            eval_dataset = get_dataset(args.EVAL_TYPE, eval_dataset)
        else:
            eval_dataset = get_dataset(args.EVAL_TYPE, args.EVAL_PATH)
        train_data = process_data(args, tokenizer, args.DATA_TYPE, train_dataset)
        eval_data = process_data(args, tokenizer, args.EVAL_TYPE, eval_dataset)
        data = {"train": train_data, "test": eval_data}
    else: 
        dataset = get_dataset(args.DATA_TYPE, args.DATA_PATH)
        data = process_data(args, tokenizer, args.DATA_TYPE, dataset)
    return data

def wandb_resume(id, outdir):
    run = wandb.init(entity="tkg_forecaster", project=os.environ["WANDB_PROJECT"], id=id, resume="must")
    trainer = get_trainer(args, model, data, tokenizer) 
    run_name = run.name
    ckpt_name = f"checkpoint-{run_name}:latest"
    ckpt_artifact = run.use_artifact(ckpt_name)
    ckpt_dir = ckpt_artifact.download() #get ckpt from server wandb
    trainer.train(resume_from_checkpoint=ckpt_dir)
    print("Resumed wandb run: ", str(run.id))
    trainer.save_model(outdir + "/model_final")

def dumm_wandb_resume(trainer, id, outdir):
    wandb.init(project=os.environ["WANDB_PROJECT"], id=id, resume="must")

def generate_run_name(outdir, time=None):
    match = re.search(r'([^\/]+)$', outdir)
    filename = match.group(1) if match else ""
    run_name = filename + "_"
    if time is None:
        run_name = run_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        run_name = run_name + time
    return run_name

if __name__ == "__main__":
    args = parse_args()
    if args.REPORT_TO == "wandb":
        #init to wandb
        wandb.login()

        if args.RUN_NAME is None:
                args.RUN_NAME = generate_run_name(args.OUTPUT_DIR)
                print("gen_name", args.RUN_NAME)
        #args to wandb
        set_wandb_para(args.PROJ_NAME)

    print("load_best_model_at_end is set as", args.LOAD_BEST_MODEL_AT_END)

    lora_config = get_lora_config(args)
    model, tokenizer = get_model_and_tokenizer(args, lora_config)

    data = prepair_data(args, tokenizer)
    #wandb is combined with transformers.Trainer, by using TrainingArguments, as shown in the official site
    if args.W_RESUME == 1:
        #trainer.train(resume_from_checkpoint="/mnt/fast-local/Llama2-LoRA-Trainer_RT/output_models_llama2/output_model_1024_bs_8bit_4096_test/checkpoint-3")
        wandb_resume(args.W_ID, args.OUTPUT_DIR)
    else:
        trainer = get_trainer(args, model, data, tokenizer) 
        trainer.train(resume_from_checkpoint=False)
        trainer.save_model(args.OUTPUT_DIR + "/model_final")
