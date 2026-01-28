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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import networkx as nx

from handle_logging_config import setup_logging, get_git_info
from configs import *

from evaluation_functions import *

from CoT_decoder_instances import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default="bart_multiwoz_validation")
    parser.add_argument('--config_name_list', nargs='*', type=str, default=None)

    #add the arguments for cot decoding, i.e. the aggregation strategy and the disparity threshold
    parser.add_argument('--cot_aggregation_strategy', type=str, default="highest-disparity-branch")
    parser.add_argument('--cot_disparity_threshold', type=float, default=0.5)
    parser.add_argument('--k_branches_to_use', type=int, default=None)

    #argument for seeds
    parser.add_argument('--seed', type=int, default=None)
    

    args = parser.parse_args()

    #setup logging
    if args.config_name_list is not None:
        common_config_prefix = os.path.commonprefix(args.config_name_list)
        common_config_suffix = os.path.commonprefix([c[::-1] for c in args.config_name_list])[::-1]
        args.config_name = args.config_name_list[0] + "_" + "_plus_".join([text.removeprefix(common_config_prefix).removesuffix(common_config_suffix) for text in  args.config_name_list[1:]])
    
    logger = setup_logging("evaluation_" + args.config_name)


    #load the config(s)
    if args.config_name_list is not None:
        print(args.config_name_list)
        logger.info(f"Running with configs: {args.config_name_list}")
        config_list = [get_config(config_name) for config_name in args.config_name_list]
        config_param_dict_list = [config.to_dict() for config in config_list]
        logger.info(f"Loaded configs: {config_param_dict_list}")
        #set a main config for the evaluation
        config = config_list[0]
        config_param_dict = config_param_dict_list[0]
    else:
        logger.info(f"Running with config: {args.config_name}")
        config = get_config(args.config_name)
        config_param_dict = config.to_dict()
        logger.info(f"Loaded config: {config_param_dict}")


    logger.info("Loading dataset")

    with Path("../data/" + config.dataset + "_dialogue_" + "term_dict.json").open("r") as file:
        dialogue_term_dict = json.load(file)
    
    logger.info(f"Loaded dataset with splits: {config.splits}")


    map_to_ontology_values_via_groundtruth_equivalence = True
    map_to_ontology_values = True
    if map_to_ontology_values_via_groundtruth_equivalence:
        logger.info("Mapping predicted values to ontology values via groundtruth refers to same concept relation predictions")
        map_to_ontology_values = False
    
    if map_to_ontology_values:
        logger.info("Mapping predicted values to ontology values via refers to same concept relation predictions")
        

    #if prediction for cot decoding then set a tokenizer
    if config.predict_for_cot_decoding:
        logger.info(f"Loading tokenizer {config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        aggregation_strategy = args.cot_aggregation_strategy
        disparity_threshold = args.cot_disparity_threshold


    
    result_filename = "results/"
    result_filename += "config_"
    result_filename += args.config_name
    if args.seed and not "seed" in args.config_name:
        result_filename += "_seed_" + str(args.seed)
    result_filename += "_LLM_TOD_ontology_inference_results"
    checkpoint_filename = result_filename + "_checkpoint"
    if config.predict_for_cot_decoding:
        result_filename += ".pt"
        checkpoint_filename += ".pt"
    else:
        result_filename += ".json"
        checkpoint_filename += ".json"
    config_filename = "results/" + args.config_name + "_config.json"


    if args.config_name_list is not None:
        #load the different results and put them in a list of dictionaries
        response_per_dialogue_list = []
        for config_name in args.config_name_list:
            result_filename = "results/"
            result_filename += "config_"
            result_filename += config_name
            if args.seed and not "seed" in config_name:
                result_filename += "_seed_" + str(args.seed)
            result_filename += "_LLM_TOD_ontology_inference_results.json"
            checkpoint_filename = result_filename.replace(".json", "_checkpoint.json")
            if config.predict_for_cot_decoding:
                result_filename = result_filename.replace(".json", ".pt")
                checkpoint_filename = checkpoint_filename.replace(".json", ".pt")
            
            if Path(result_filename).exists():
                logger.info(f"Loading results from {result_filename}")
                if config.predict_for_cot_decoding:
                    response_per_dialogue = torch.load(result_filename)
                else:
                    with Path(result_filename).open("r") as file:
                        response_per_dialogue = json.load(file)
                response_per_dialogue_list.append(response_per_dialogue)
            elif Path(checkpoint_filename).exists():
                logger.info(f"Loading results from checkpoint {checkpoint_filename}")
                if config.predict_for_cot_decoding:
                    response_per_dialogue = torch.load(checkpoint_filename)
                else:
                    with Path(checkpoint_filename).open("r") as file:
                        response_per_dialogue = json.load(file)
                response_per_dialogue_list.append(response_per_dialogue)
            else:
                raise ValueError(f"Results file {result_filename} does not exist and checkpoint file {checkpoint_filename} does not exist.")
        
        logger.info(f"Loaded all {len(args.config_name_list)} results")
    else:
        logger.info(f"Loading results from {result_filename}")
        #load the results if they exist
        if Path(result_filename).exists():
            if config.predict_for_cot_decoding:
                response_per_dialogue = torch.load(result_filename)
            else:
                with Path(result_filename).open("r") as file:
                    response_per_dialogue = json.load(file)
        
        elif Path(checkpoint_filename).exists():
            logger.info(f"Loading results from checkpoint {checkpoint_filename}")
            if config.predict_for_cot_decoding:
                response_per_dialogue = torch.load(checkpoint_filename)
            else:
                with Path(checkpoint_filename).open("r") as file:
                    response_per_dialogue = json.load(file)

        else:
            raise ValueError(f"Results file {result_filename} does not exist and checkpoint file {checkpoint_filename} does not exist.")

    logger.info(f"Loaded results from {result_filename}")

    evaluation_dict = {}

    #store the relations of all splits that were considered and evaluate
    all_groundtruth_relations = set()
    #also check the relation present in dialogues separately
    all_groundtruth_relations_present_in_dialogues = set()
    all_predicted_relations = set()

    #first get all equivalence relations from the data
    equivalence_relations = set()
    #also get the ontology values for the mapping in the evaluation
    ontology_values = set()
    for split in config.splits:
        dialogue_texts = dialogue_term_dict[split]
        for dial_id, content_triplet in dialogue_texts.items():
            relations = content_triplet["relational triplets"]
            for head, rel, tail in relations:
                if rel == "refers to same concept as":
                    equivalence_relations.add((head, tail))
                else:
                    ontology_values.add(head)
                    ontology_values.add(tail)

    #put those term pairs that are connected by refers into the same set to results in a set of sets
    G = nx.Graph()
    G.add_edges_from(equivalence_relations)
    #get the connected components
    equivalence_connected_components = list(nx.connected_components(G))
        

    #store the dialogue level results for each split for signficance testing
    dialogue_level_results_present_in_dialogues = {}

    for split in config.splits:
        #only consider a subset of the data for faster inference and thus more experiments
        if config.subset_size is not None:
            logger.info(f"Only considering a subset of {config.subset_size} dialogues for evaluation")
            #get the first subset_size dialogues
            dialogue_texts = dict(list(dialogue_term_dict[split].items())[:config.subset_size])
        else:
            #if it is a checkpoint use the dialogues up to checkpoint length
            #dialogue_texts = dialogue_term_dict[split]
            if args.config_name_list is not None:
                smallest_length = min([len(response_per_dialogue[split]) for response_per_dialogue in response_per_dialogue_list])
                #set manual threshold at 1000 for faster inference
                #smallest_length = min(smallest_length, 1000)
                dialogue_texts = dict(list(dialogue_term_dict[split].items())[:smallest_length])
            else:
                #set manual threshold at 1000 for faster inference
                #length = min(len(response_per_dialogue[split]), 1000)
                length = len(response_per_dialogue[split])
                dialogue_texts = dict(list(dialogue_term_dict[split].items())[:length])

        logger.info(f"Run evaluation on {split} split with {len(dialogue_texts)} dialogues with model {config.model_name}")
        #store the relations for the current split and evaluate
        all_groundtruth_relations_this_split = set()
        all_groundtruth_relations_this_split_present_in_dialogues = set()
        all_predicted_relations_this_split = set()
        not_parsable_count = 0
        not_in_the_right_format = set()

        dialogue_level_results_present_in_dialogues[split] = {}

        for dial_id, content_triplet in tqdm(dialogue_texts.items()):
            text = content_triplet["text"]
            terms = content_triplet["terms"]
            relations = content_triplet["relational triplets"]
            #turn relations to a set of tuples
            relations = set((head, rel, tail) for head, rel, tail in relations)
            all_groundtruth_relations_this_split.update(relations)
            all_groundtruth_relations.update(relations)
            #store the relations present in the dialogues
            relations_present_in_dialogue = set((head, rel, tail) for head, rel, tail in relations if present_in_utterance(head, text) and present_in_utterance(tail, text))

            #get those relations that are present in the dialogues via equivalence relation
            relations_present_in_dialogue_via_equivalence = set()
            for head, rel, tail in relations:
                #first get the relations that are present in the dialogues
                if present_in_utterance(head, text) and present_in_utterance(tail, text):
                    relations_present_in_dialogue_via_equivalence.add((head, rel, tail))
                    continue
				
                if rel == "refers to same concept as":
                    continue

                equivalent_for_head_present = False
                if present_in_utterance(head, text):
                    equivalent_for_head_present = True
                equivalent_for_tail_present = False
                if present_in_utterance(tail, text):
                    equivalent_for_tail_present = True

                for connected_component in equivalence_connected_components:
                    if head in connected_component:
                        for equivalent_head in connected_component:
                            if present_in_utterance(equivalent_head, text):
                                equivalent_for_head_present = True
                                break
                    if tail in connected_component:
                        for equivalent_tail in connected_component:
                            if present_in_utterance(equivalent_tail, text):
                                equivalent_for_tail_present = True
                                break
                    if equivalent_for_head_present and equivalent_for_tail_present:
                        break


                if equivalent_for_head_present and equivalent_for_tail_present:
                    relations_present_in_dialogue_via_equivalence.add((head, rel, tail))


            #count the relations where the terms are present via equivalent terms as the ones that are present in the dialogues
            all_groundtruth_relations_this_split_present_in_dialogues.update(relations_present_in_dialogue_via_equivalence)
            all_groundtruth_relations_present_in_dialogues.update(relations_present_in_dialogue_via_equivalence)


            if config.predict_for_cot_decoding:
                if args.config_name_list is not None:
                    branch_answer_tokens = []
                    branch_answer_logits = []
                    for response_per_dialogue_dict in response_per_dialogue_list:
                        current_response = response_per_dialogue_dict[split][dial_id]
                        branch_answer_tokens = current_response[1]
                        branch_answer_logits = current_response[2]

                        if not branch_answer_tokens or not branch_answer_logits:
                            continue
                        cot_decoder = get_CoTDecoder(tokenizer, aggregation_strategy)
                        predicted_relations = cot_decoder.decode(branch_prediction_ids=branch_answer_tokens, branch_prediction_top_logits=branch_answer_logits, disparity_threshold=disparity_threshold, k=args.k_branches_to_use)
                        predicted_relations = set(tuple(triplet) for triplet in predicted_relations if len(triplet) == 3)
                        predicted_relations = set((" ".join(tokenize(head)), " ".join(tokenize(rel)), " ".join(tokenize(tail)) ) for head, rel, tail in predicted_relations)
                        all_predicted_relations_this_split.update(predicted_relations)
                        all_predicted_relations.update(predicted_relations)

                else:
                    response = response_per_dialogue[split][dial_id]
                    response = response_per_dialogue[split][dial_id]
                    branch_answer_tokens = response[1]
                    branch_answer_logits = response[2]
                    if not branch_answer_tokens or not branch_answer_logits:
                        continue
                    cot_decoder = get_CoTDecoder(tokenizer, aggregation_strategy)
                    predicted_relations = cot_decoder.decode(branch_prediction_ids=branch_answer_tokens, branch_prediction_top_logits=branch_answer_logits, disparity_threshold=disparity_threshold, k=args.k_branches_to_use)
                    predicted_relations = set(tuple(triplet) for triplet in predicted_relations if len(triplet) == 3)
                    predicted_relations = set((" ".join(tokenize(head)), " ".join(tokenize(rel)), " ".join(tokenize(tail)) ) for head, rel, tail in predicted_relations)
                    all_predicted_relations_this_split.update(predicted_relations)
                    all_predicted_relations.update(predicted_relations)

                #get the results for the current dialogue and add them to the dialogue level results dictionary
                current_dialogue_results = evaluate_term_relation_extraction_f1(predicted_relations, relations_present_in_dialogue_via_equivalence, map_terms_to_ontology_referrals=map_to_ontology_values, mapping_based_on_groundtruth=map_to_ontology_values_via_groundtruth_equivalence, ontology_values=ontology_values, groundtruth_equivalence_relations=equivalence_relations)
                dialogue_level_results_present_in_dialogues[split][dial_id] = current_dialogue_results


            else:
                if args.config_name_list is not None:
                    predicted_relation_response = ""
                    for response_per_dialogue_dict in response_per_dialogue_list:
                        predicted_relation_response += response_per_dialogue_dict[split][dial_id] + "\n"

                else:
                    predicted_relation_response = response_per_dialogue[split][dial_id]
                #extract the predicted relations from the response string
                predicted_relations, not_parsable = extract_list_from_string(predicted_relation_response, return_not_parsable=True)
                not_parsable_count += not_parsable
                #turn predictions to a set of tuples
                not_in_the_right_format.update(set([tuple(triplet) for triplet in predicted_relations if len(triplet) != 3]))
                predicted_relations = set(tuple(triplet) for triplet in predicted_relations if len(triplet) == 3)
                all_predicted_relations_this_split.update(predicted_relations)
                all_predicted_relations.update(predicted_relations)

                #get the results for the current dialogue and add them to the dialogue level results dictionary
                current_dialogue_results = evaluate_term_relation_extraction_f1(predicted_relations, relations_present_in_dialogue_via_equivalence, map_terms_to_ontology_referrals=map_to_ontology_values, mapping_based_on_groundtruth=map_to_ontology_values_via_groundtruth_equivalence, ontology_values=ontology_values, groundtruth_equivalence_relations=equivalence_relations)
                dialogue_level_results_present_in_dialogues[split][dial_id] = current_dialogue_results

        
        
    logger.info(f"Number of all groundtruth relations: {len(all_groundtruth_relations)}")
    logger.info(f"Number of all groundtruth relations present in dialogues: {len(all_groundtruth_relations_present_in_dialogues)}")
    logger.info(f"Number of all predicted relations: {len(all_predicted_relations)}")


    #evaluate the term relation extraction for relations present in dialogues
    f1_dict = evaluate_term_relation_extraction_f1(all_predicted_relations, all_groundtruth_relations_present_in_dialogues, map_terms_to_ontology_referrals=map_to_ontology_values, mapping_based_on_groundtruth=map_to_ontology_values_via_groundtruth_equivalence, ontology_values=ontology_values, groundtruth_equivalence_relations=equivalence_relations)
    evaluation_dict["all_present_in_dialogues"] = {"f1": f1_dict}

    logger.info(f"Finished evaluation on {config.dataset} with splits {config.splits} with model {config.model_name} for relations present in dialogues")
    for rel in f1_dict:
        logger.info(f"F1 score for relation {rel}: {f1_dict[rel]['f1']}")
        logger.info(f"Recall for relation {rel}: {f1_dict[rel]['recall']}")
        logger.info(f"Precision for relation {rel}: {f1_dict[rel]['precision']}")
            
            

    evaluation_dict["dialogue_level_results_present_in_dialogue"] = dialogue_level_results_present_in_dialogues

    logger.info(f"Finished evaluation on {config.dataset} with splits {config.splits} with model {config.model_name}")

    evaluation_filename = "evaluation/"
    evaluation_filename += "config_"
    evaluation_filename += args.config_name
    if args.seed and "seed" not in args.config_name:
        evaluation_filename += "_seed_" + str(args.seed)
    if map_to_ontology_values_via_groundtruth_equivalence:
        evaluation_filename += "_mapped_to_ontology_values_via_groundtruth_equivalence"
    elif map_to_ontology_values:
        evaluation_filename += "_terms_mapped_to_ontology_values"
    if config.predict_for_cot_decoding:
        if aggregation_strategy == "highest-disparity-branch":
            aggregation_strategy = "hdb"
        evaluation_filename += "_cot_" + aggregation_strategy + "_disp" + str(disparity_threshold)
    if args.k_branches_to_use:
        evaluation_filename += "_k_branches_" + str(args.k_branches_to_use)
    evaluation_filename += "_LLM_TOD_ontology_evaluation_" + "results.json"
    config_filename = "evaluation/" + args.config_name + "_config.json"

    logger.info("Saving evaluation results")
    with Path(evaluation_filename).open("w", encoding="UTF-8") as file: 
        json.dump(evaluation_dict, file)
        
    logger.info(f"Saved results to {evaluation_filename}")

    #save the config as json file
    logger.info("Saving config")
    with Path(config_filename).open("w", encoding="UTF-8") as file:
        json.dump(config_param_dict, file)
    logger.info(f"Saved config to {config_filename}")

    

if __name__ == "__main__":
    main()