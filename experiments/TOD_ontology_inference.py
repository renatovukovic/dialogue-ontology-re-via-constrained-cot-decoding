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

#take as input the model name, and the data-set, either multiwoz or sgd and also with different splits as input
#for each dialogue there is a list of terms from the dialogue as input between which the ontology hierarchy relations should be predicted
import os
import sys
import json
from tqdm import tqdm
from pathlib import Path
import transformers
import torch
#from convlab.util import load_dataset, load_ontology, load_database
import argparse
import logging
import random

from handle_logging_config import setup_logging, get_git_info
from configs import *
from LLM_predictor import LLM_predictor
from evaluation_functions import extract_list_from_string, build_hierarchical_memory, get_one_hop_neighbour_relations_for_termlist
from prompt_generator_instances import get_prompt_generator



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default="gpt_multiwoz_validation")
    parser.add_argument('--seed', type=int, default=None, help='Seed for reproducibility')


    args = parser.parse_args()

    #setup logging
    logger = setup_logging("inference_" + args.config_name)

    logger.info(f"Running with config: {args.config_name}")

    #load the config
    config = get_config(args.config_name)
    config_param_dict = config.to_dict()
    logger.info(f"Loaded config: {config_param_dict}")

    prompt_generator = get_prompt_generator(config)


    logger.info("Loading dataset")
    with Path("../data/" + config.dataset + "_dialogue_term_dict.json").open("r") as file:
        dialogue_term_dict = json.load(file)
    
    logger.info(f"Loaded dataset with splits: {config.splits}")

    
    prompt_dict_path = prompt_generator.get_prompt_dict_path()
    logger.info(f"Loading prompt dict from: {prompt_dict_path}")
    with Path(prompt_dict_path).open("r") as promptfile:
        prompt = json.load(promptfile)

    prompt_generator.set_prompt_dict(prompt)

    if args.seed and "seed" not in args.config_name:
        logger.info(f"Setting seed to {args.seed}")
        torch.manual_seed(args.seed)

    
    result_filename = "results/"
    result_filename += "config_"
    result_filename += args.config_name
    if args.seed and "seed" not in args.config_name:
        result_filename += "_seed_" + str(args.seed)
    checkpoint_filename = result_filename + "_LLM_TOD_ontology_inference_results_checkpoint.json"
    result_filename += "_LLM_TOD_ontology_inference_results.json"
    config_filename = "results/" + args.config_name + "_config.json"

    if config.predict_for_cot_decoding or config.analyse_top_k_tokens:
        result_filename = result_filename.replace(".json", ".pt")
        checkpoint_filename = checkpoint_filename.replace(".json", ".pt")
    
    if Path(checkpoint_filename).is_file():
        logger.info(f"Loading checkpoint from {checkpoint_filename} and continue from there")
        if config.predict_for_cot_decoding or config.analyse_top_k_tokens:
            response_per_dialogue = torch.load(checkpoint_filename)
        else:
            with Path(checkpoint_filename).open("r") as file:
                response_per_dialogue = json.load(file)
    else:
        #initialise the dialogue id, InstructGPT response dictionary
        logger.info("Initialising response per dialogue dictionary")
        response_per_dialogue = {}
        for split in config.splits:
            response_per_dialogue[split] = {}

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #initialise the LLM predictor
    logger.info(f"Initialising LLM predictor with model {config.model_name}")
    LLM = LLM_predictor(config, device)
    logger.info("LLM predictor initialised")


    counter = 0
    relations_so_far = set()

    for split in config.splits:
        #only consider a subset of the data for faster inference and thus more experiments
        if config.subset_size is not None:
            logger.info(f"Using a subset of {config.subset_size} dialogues for the {split} split")
            #get the first subset_size dialogues
            dialogue_texts = dict(list(dialogue_term_dict[split].items())[:config.subset_size])
        else:
            dialogue_texts = dialogue_term_dict[split]
        logger.info(f"Run inference on {split} split with {len(dialogue_texts)} dialogues with model {config.model_name}")
        #response_per_dialogue[split] = {}
        for dial_id, content_triplet in tqdm(dialogue_texts.items()):
            text = content_triplet["text"]
            terms = content_triplet["terms"]
            relations = content_triplet["relational triplets"]

            if dial_id not in response_per_dialogue[split]: #if there arise problems with connection or model only do the missing dialogues
                
                counter += 1
                current_responses = []
                output_string = ""
                
                for i in range(config.steps):
                    if config.finetuned_model:
                        LLM_input = prompt_generator.generate_prompt(step=i, dialogue=text, term_list=terms, relations_so_far=relations_so_far, additional_input=current_responses, instruction_prefix=config.instruction_prefix, answer_prefix=config.answer_prefix)
                    else:
                        LLM_input = prompt_generator.generate_prompt(step=i, dialogue=text, term_list=terms, relations_so_far=relations_so_far, additional_input=current_responses)

                    try: #if it fails save the current dict and then log the error and continue

                        relationlist=["has slot", "has value", "has domain", "refers to same concept as"]
                        if config.only_hasslot:
                            relationlist = ["has slot"]
                        elif config.only_hasvalue:
                            relationlist = ["has value"]
                        elif config.only_hasdomain:
                            relationlist = ["has domain"]
                        elif config.only_equivalence:
                            relationlist = ["refers to same concept as"]

                        if config.predict_for_cot_decoding and config.constrain_generation:
                            branch_strings, branch_tokens, branch_logits, entropy = LLM.predict(LLM_input, constrain_generation=config.constrain_generation, predict_for_cot_decoding=config.predict_for_cot_decoding, entropy_branching_threshold=config.entropy_branching_threshold, term_list=terms, relation_list=relationlist)
                            branch_length = 0
                        elif config.predict_for_cot_decoding:
                            branch_strings, branch_tokens, branch_logits, entropy, branch_length = LLM.predict(LLM_input, constrain_generation=config.constrain_generation, predict_for_cot_decoding=config.predict_for_cot_decoding, entropy_branching_threshold=config.entropy_branching_threshold)
                        elif config.analyse_top_k_tokens:
                            top_k_token_ids, top_k_token_logits, entropies = LLM.predict(LLM_input, constrain_generation=config.constrain_generation, analyse_top_k_tokens=config.analyse_top_k_tokens)
                        else:
                            response = LLM.predict(LLM_input, constrain_generation=config.constrain_generation, constrained_beamsearch=config.constrained_beamsearch, term_list=terms, relation_list=relationlist)
                    except Exception as e:
                        logger.info(f"Checkpoint saved at {checkpoint_filename} after {counter} dialogues")
                        logger.error(f"Error at dialogue {dial_id} in split {split}")
                        logger.error(f"Error message: {e}")
                        continue

                    if config.predict_for_cot_decoding:
                        output_string += "Step " + str(i) + " response:\n"
                        for j, branch_string in enumerate(branch_strings):
                            output_string += f"Branch {j}:\n{branch_string}\n"
                        current_responses.append(branch_strings[0])

                    elif config.analyse_top_k_tokens:
                        output_string += "Step " + str(i) + " response:\n"
                        for j, top_k_token_id in enumerate(top_k_token_ids):
                            output_string += f"Top {j} token id: {top_k_token_id}\n"
                            output_string += f"Top {j} token: {LLM.tokenizer.decode(top_k_token_id)}\n"
                            output_string += f"Top {j} token logit: {top_k_token_logits[j]}\n"
                            output_string += f"Top {j} token entropy: {entropies[j]}\n"
                        current_responses.append(LLM.tokenizer.decode(top_k_token_ids[0]))
                    else:
                        output_string += "Step " + str(i) + " response:\n" + response + "\n"
                        current_responses.append(response)

                    #print the input and the response only for the first two dialogue in the first split
                    if counter < 3 and split == config.splits[0]:
                        logger.info(f"{counter}th dialogue input and response")
                        if config.predict_for_cot_decoding:
                            logger.info(f"{i}th Input:\n {LLM_input}")
                            for j, branch_string in enumerate(branch_strings):
                                logger.info(f"Branch {j} response branched after {branch_length} tokens:\n {branch_string}")
                        
                        elif config.analyse_top_k_tokens:
                            logger.info(f"{i}th Input:\n {LLM_input}")
                            for j, top_k_token_id in enumerate(top_k_token_ids):
                                logger.info(f"Top {j} token id: {top_k_token_id}")
                                logger.info(f"Top {j} token: {LLM.tokenizer.decode(top_k_token_id)}")
                                logger.info(f"Top {j} token logit: {top_k_token_logits[j]}")
                                logger.info(f"Top {j} token entropy: {entropies[j]}")

                        else:
                            logger.info(f"{i}th Input:\n {LLM_input}")
                            logger.info(f"Step {i} response:\n {response}")

                
                if config.predict_for_cot_decoding:
                    response_per_dialogue[split][dial_id] = (branch_strings, branch_tokens, branch_logits, entropy, branch_length)
                elif config.analyse_top_k_tokens:
                    response_per_dialogue[split][dial_id] = (top_k_token_ids, top_k_token_logits, entropies)
                else:
                    response_per_dialogue[split][dial_id] = output_string
        
                #save checkpoint
                if counter % 10 == 0:
                    if config.predict_for_cot_decoding or config.analyse_top_k_tokens:
                        #save with torch becaues of the tensors of the logits
                        torch.save(response_per_dialogue, checkpoint_filename)
                    else:
                        with Path(checkpoint_filename).open("w", encoding="UTF-8") as file:
                            json.dump(response_per_dialogue, file)
                    logger.info(f"Saved checkpoint after {counter} dialogues")    
            
            
    logger.info(f"Finished inference on {config.dataset} with splits {config.splits} with model {config.model_name}")

    logger.info("Saving results")
    if config.predict_for_cot_decoding or config.analyse_top_k_tokens:
        #save with torch becaues of the tensors of the logits
        torch.save(response_per_dialogue, result_filename)
    else:
        #save the responses as json file
        with Path(result_filename).open("w", encoding="UTF-8") as file: 
            json.dump(response_per_dialogue, file)

        
    logger.info(f"Saved results to {result_filename}")

    #save the config as json file
    logger.info("Saving config")
    with Path(config_filename).open("w", encoding="UTF-8") as file:
        json.dump(config_param_dict, file)
    logger.info(f"Saved config to {config_filename}")

    

if __name__ == "__main__":
    main()







