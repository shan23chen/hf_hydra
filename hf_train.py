from __future__ import print_function
from cProfile import run
import pandas as pd
import numpy as np

from pyparsing import Optional # to set the python random seed
import torch

# Ignore excessive warnings
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

# WandB – Import the wandb and hydra
import wandb
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

#raytune
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining

#huggingface
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#import helper functions
from src.n_proc import *
from src.n_metrics import *
from src.n_trainer_classes import *
from src.n_yaml_config import TOXConfig

# specifiy and clear device   
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 386

# instanlize hydra config class
cs = ConfigStore.instance()
cs.store(name="tox_config", node=TOXConfig)

@hydra.main(config_path="config", config_name="config")
def main(cfg: TOXConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    MODEL = cfg.run.hf_model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    # log into wandb
    wandb.login()
    wandb.init(project=cfg.paths.project_name, entity="bitterman")

    # TODO: write a ulti to only include the info thats not NONE

    run_setup = [str(val) for _, val in cfg.run.items()]
    run_setup = [i for i in run_setup if i != 'None']
    wandb.run.name = '_'.join(run_setup)

    # Importing Data
    dataset = load_dataset("csv", data_files={  'train': cfg.paths.data+'train.csv',
                                                "valid": cfg.paths.data+'dev.csv',
                                                'test': cfg.paths.data+'test.csv'})
    
    ###marking labels:
    dataset = dataset.map(lambda x: grade_preproc(x['Free Text Grade']))
    if cfg.run.num_classes == 2:
        # grade 0 or grade 1,2,3
        dataset = dataset.map(lambda x: binary_preproc(x['labels']))

    elif cfg.run.num_classes == 22:
        # grade 0&1 or grade 2&3
        dataset = dataset.map(lambda x: bibi_preproc(x['labels'])) 
    
    elif cfg.run.num_classes == 3:
        # grade 0, 1, 2&3
        dataset = dataset.map(lambda x: trinary_preproc(x['labels']))    
    
    elif cfg.run.num_classes == 6:
        #joint training
        dataset = dataset.map(lambda x: grade6_preproc(x['labels']))
    
    ###text pre_select part: 
    dataset = dataset.map(lambda x: project_text(x))

    if cfg.run.ih:
        dataset = dataset.map(lambda x: text_ih(x))

    if cfg.run.ap:
        dataset = dataset.map(lambda x: text_ap(x))
    
    if cfg.run.exam:
        dataset = dataset.map(lambda x: text_exam(x))

    if cfg.run.ros:
        dataset = dataset.map(lambda x: text_ros(x))

    if cfg.run.rot:
        dataset = dataset.map(lambda x: text_rot(x))

    if cfg.run.sec:
        dataset = dataset.map(lambda x: text_sec(x))

    if cfg.run.rot is None and cfg.run.ros is None and cfg.run.exam is None and cfg.run.ap is None and cfg.run.ih is None and cfg.run.sec is None:
        print('no changes addded to original text')
    else:
        dataset = dataset.map(lambda x: text_full_text(x))

    if cfg.run.struc:
        dataset = dataset.map(lambda x: struc_text_preproc1(x))

    if cfg.run.clean == 'clean':
        print('### loading and cleaning text ###')
        dataset = dataset.map(lambda x: text_preproc(x['Full Text']))
        print('!!! Done loading and cleaning text !!!')

    ###tokenizations:
    if cfg.run.clean == 'clean':
        dataset = dataset.map(lambda x: tokenizer(x['clean_text'], padding='max_length', truncation=True, max_length=MAX_LEN), batched=True)
    else:
        dataset = dataset.map(lambda x: tokenizer(x['Full Text'], padding='max_length', truncation=True, max_length=MAX_LEN), batched=True)
    
    ### label class count
    label_list = dataset['train'].unique('labels')
    num_labels = len(label_list)
    print(f'Total label numbers are {num_labels}, and the label list is {label_list}')

 
    # config = cfg.params
    T_args= TrainingArguments(
        report_to = 'wandb',                    
        output_dir = cfg.params.output_dir,      
        num_train_epochs = cfg.params.epochs,
        overwrite_output_dir = True,
        evaluation_strategy = 'steps',         
        learning_rate = cfg.params.learning_rate,  
        max_steps = cfg.params.max_steps,          # will overwrite num_train_epochs
        warmup_steps= cfg.params.warmup_steps,                
        weight_decay= cfg.params.weight_decay,      
        logging_steps = cfg.params.logging_steps,                
        eval_steps = cfg.params.eval_steps,                     
        save_steps = int(cfg.params.logging_steps)*4,
        load_best_model_at_end = cfg.params.load_best_model_at_end,
        per_device_train_batch_size=cfg.params.batch_size, 
        gradient_accumulation_steps=cfg.params.gradient_accumulation_steps, 
        per_device_eval_batch_size=cfg.params.per_device_eval_batch_size,
        tf32=cfg.params.tf32,
        metric_for_best_model = cfg.params.metric_for_best_model
    )

    print(f'The back bone model name is {MODEL}')
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)
    # model.resize_token_embeddings(len(tokenizer))

    if cfg.run.num_classes == 3:
        trainer = CELTrainer(
            args=T_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset=dataset['valid'],
            # test_dataset=dataset['test'],
            model=model,
            compute_metrics=compute_metrics3,
        )
    
    elif cfg.run.num_classes == 2:
        trainer = CELTrainer(
            args=T_args,
            # tokenizer=tokenizer,
            train_dataset=dataset['train'],
            # eval_dataset=dataset['valid'],
            # train_dataset=dataset['valid'],
            eval_dataset=dataset['test'],
            model=model,
            compute_metrics=compute_metrics2,
        )
 
    elif cfg.run.num_classes == 22:
        trainer = CELTrainer(
            args=T_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            # eval_dataset=dataset['valid'],
            # train_dataset=dataset['valid'],
            eval_dataset=dataset['test'],
            model=model,
            compute_metrics=compute_metrics22,
        )
    
    elif cfg.run.num_classes == 6:
        trainer = BCETrainer(
            args=T_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset=dataset['valid'],
            # test_dataset=dataset['test'],
            model=model,
            compute_metrics=compute_metrics6,
        ) 

    # use raytune to params search
    if cfg.run.raytune:
        tune_config = {
            "per_device_train_batch_size": 16,  
            "per_device_eval_batch_size": 128,
            # "num_train_epochs": tune.choice([2, 3, 4, 5]),
            # "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
        }

        scheduler = PopulationBasedTraining(
                        time_attr="training_iteration",
                        metric="eval_accuracy",
                        mode="max",
                        # perturbation_interval=1,
                        hyperparam_mutations={
                        "weight_decay": tune.uniform(0.0, 0.3),
                        "learning_rate": tune.uniform(1e-5, 5e-5),
                        "per_device_train_batch_size": [8, 16, 18, 20, 32],
            },
        )

        reporter = CLIReporter(
            parameter_columns={
                "weight_decay": "w_decay",
                "learning_rate": "lr",
                "per_device_train_batch_size": "train_bs/gpu",
                "num_train_epochs": "num_epochs",
            },
            metric_columns=["eval_acc", "eval_accuracy", "epoch", "training_iteration"],
        )

        if cfg.run.num_classes == 22:
            def model_init():
                return AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
            trainer = CELTrainer(
                args=T_args,
                tokenizer=tokenizer,
                train_dataset=dataset['train'],
                eval_dataset=dataset['valid'],
                # eval_dataset=dataset['test'],
                model_init=model_init,
                compute_metrics=compute_metrics22,
            )

        elif cfg.run.num_classes == 3:
            def model_init():
                return AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=cfg.run.num_classes)
            trainer = CELTrainer(
                args=T_args,
                tokenizer=tokenizer,
                train_dataset=dataset['train'],
                eval_dataset=dataset['valid'],
                # eval_dataset=dataset['test'],
                model_init=model_init,
                compute_metrics=compute_metrics3,
            )

        trainer.hyperparameter_search(
            hp_space=lambda _: tune_config,
            backend="ray",
            n_trials=10,
            # resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
            scheduler=scheduler,
            keep_checkpoints_num=1,
            checkpoint_score_attr="training_iteration",
            # stop={"training_iteration": 1} if smoke_test else None,
            progress_reporter=reporter,
            # local_dir="~/ray_results/",
            # name="tune_transformer_pbt",
            log_to_file=True,
        )

    else:
        # trainer.train(resume_from_checkpoint = True)
        trainer.train()
        # print(trainer.predict(dataset['test']).metrics)
        print(trainer.predict(dataset['valid']).metrics)
        with open('out.txt', 'a') as f:
            print('Runname:', wandb.run.name, trainer.predict(dataset['test']).metrics, file=f)

if __name__ == "__main__":
    main()