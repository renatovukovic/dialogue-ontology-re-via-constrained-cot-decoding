# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Renato Vukovic (renato.vukovic@hhu.de)
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

####################################################################################################################
#Train autoregressive model on causal language modelling on the concatenated input, output pairs
####################################################################################################################

###ignore warning regarding too large sequence length
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainingArguments
#transformer reinforcement learning library for supervised fine-tuning trainer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import sys
from datasets import load_from_disk
import argparse
import torch
import numpy as np

from training.finetuning_config_list import *
sys.path.append("../LLM_experiments")
from handle_logging_config import *



def train(config_name="gemma_2b_multiwoz21_training"):

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_name", help="Name of the config to use", type=str, default=None, required=False)

    args = parser.parse_args()

    

    #load the training data
    #setup logging
    #if run from command line with config name
    if args.config_name:
        config_name = args.config_name

    logger = setup_logging("training_" + config_name)

    logger.info(f"Running with config: {config_name}")

    #load the config
    config = get_config(config_name)
    config_param_dict = config.to_dict()
    logger.info(f"Loaded config: {config_param_dict}")

    #if there is a seed, set it
    if config.seed:
        torch.manual_seed(config.seed)

    #load the model

    logger.info(f"Load the model {config.model_name}")

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    #put model in train
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else  "cpu")

    #set device to m1 GPU if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    #set the maximum sequence length
    maximum_sequence_length = min(config.model_max_length, tokenizer.model_max_length) #if the output for model_max_length is some random number, which is the for LLaMa, OPT

    logger.info(f"Using device: {device}")


    model.to(device)

    logger.info("Loading the model done.")

    logger.info(f"Load the dataset {config.dataset_name}.")
    sft_dataset = load_from_disk(f"../data/{config.dataset_name}_ontology_relation_sft_dataset")
    logger.info("Successfully loaded the dataset.")


    #setup training
    batch_size = config.batch_size
    model_name = config.model_name.replace("/", "-") + "_trained_on_" + config.dataset_name  
    if config.seed:
        model_name += "_seed_" + str(config.seed)
    model_dir = f"models/{model_name}"
    args = Seq2SeqTrainingArguments(
        model_dir,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        logging_strategy="steps",
        logging_steps=config.eval_steps / 10,
        save_strategy="steps",
        save_steps=config.eval_steps * 2,
        learning_rate=4e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=config.num_train_epochs,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        generation_max_length=maximum_sequence_length,
        warmup_ratio=0.1,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
    )



    ### based on transformer reinforcement learning library SFT tutorial: https://huggingface.co/docs/trl/sft_trainer ###
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            if "gemma" in config.model_name: #also add the end of turn template
                text = f"{config.instruction_prefix} {example['instruction'][i]}\n{config.answer_prefix} {example['output'][i]}<end_of_turn>"
            else:
                text = f"{config.instruction_prefix} {example['instruction'][i]}\n{config.answer_prefix} {example['output'][i]}"
            output_texts.append(text)
        return output_texts

    response_template_with_context = "\n" + config.answer_prefix
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`

    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    
    #lora config
    peft_config = LoraConfig(
        #r=16,
        #lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #set tokenizer truncation side to left
    tokenizer.truncation_side = "left"

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=sft_dataset["train"],
        eval_dataset=sft_dataset["validation"],
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
        peft_config=peft_config,
        max_seq_length=maximum_sequence_length,
        tokenizer=tokenizer,
    )


    logger.info("Start training the model.")
    trainer.train()
    logger.info("Training done.")

    logger.info(f"Training done. Save model to {model_dir}")
    trainer.save_model()
    logger.info("Saving successful.")

    logger.info("Make predictions on the validation dataset.")
    val_evaluation = trainer.evaluate()
    logger.info(f"Results on validation set after training: \n {val_evaluation}")


    logger.info("Make predictions on the test dataset.")
    test_evaluation = trainer.predict(sft_dataset["test"])
    logger.info(f"Results on test set after training: \n {test_evaluation}")
    logger.info("DONE")

if __name__=="__main__":
    train()