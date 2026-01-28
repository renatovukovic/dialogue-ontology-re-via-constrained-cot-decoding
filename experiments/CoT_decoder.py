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

#class for handling cod decoding after having the predictions with the top 2 logits for each token in each branch

from abc import ABC, abstractmethod
from typing import Union
import torch
from transformers import AutoTokenizer

from evaluation_functions import *

class CoTDecoder(ABC):
	aggregation_strategy: str
	tokenizer: AutoTokenizer

	def __init__(self, tokenizer: AutoTokenizer, aggregation_strategy: str,):
		self.aggregation_strategy = aggregation_strategy
		self.tokenizer = tokenizer
		

	@abstractmethod
	def decode(self,
	    branch_prediction_ids: list[torch.Tensor],
	    branch_prediction_top_logits: list[torch.Tensor],
	) -> set[tuple[str]]:
		#input are the predicted ids of the response branches and the top 2 logits for each token in each branch
		#branch_prediction_ids.shape = [num_branches, seq_len, 1]
		#branch_prediction_top_logits.shape = [num_branches, seq_len, 2]
		#output is the decoded response based on the respective disparity aggregation strategy of the relations
		pass

	#function that get the token ids and the logits as input and should decide on the decoding
	#this function just calculates the disparities for all the answers in the different barnches
	def branch_disparities(self, branch_token_ids: list[torch.Tensor],
					branch_token_top2_logits: list[torch.Tensor],
					answer_start_token="[",
					answer_end_token="]",
					answer_separator_token=",") -> list[torch.Tensor]:
		
		#get the disparity measures for the answer tokens, which is everything between the start and end token except the separator token
		#the disparity measure is the difference between the first and second highest logit for each token
		#for each relation take the average disparity measure across tokens


		#initialise the start, end and separator token ids
		answer_start_token_id = self.tokenizer(answer_start_token).input_ids[-1]
		answer_start_token_id2 = self.tokenizer("\n"+answer_start_token).input_ids[-1]
		answer_start_token_id3 = self.tokenizer(" "+answer_start_token).input_ids[-1]
		answer_start_token_ids = [answer_start_token_id, answer_start_token_id2, answer_start_token_id3]


		answer_end_token_id = self.tokenizer(answer_end_token).input_ids[-1]
		answer_end_token_id2 = self.tokenizer("\n"+answer_end_token).input_ids[-1]
		answer_end_token_id3 = self.tokenizer(" "+answer_end_token).input_ids[-1]
		answer_end_token_ids = [answer_end_token_id, answer_end_token_id2, answer_end_token_id3]


		answer_separator_token_id = self.tokenizer(answer_separator_token).input_ids[-1]
		answer_separator_token_id2 = self.tokenizer("\n"+answer_separator_token).input_ids[-1]
		answer_separator_token_id3 = self.tokenizer(" "+answer_separator_token).input_ids[-1]
		answer_separator_token_ids = [answer_separator_token_id, answer_separator_token_id2, answer_separator_token_id3]

		branch_answer_disparities = []
		branch_answer_relations = []
		branch_answer_strings = []

		for token_ids, token_top2_logits in zip(branch_token_ids, branch_token_top2_logits):
			current_branch_answer_disparities = []
			current_branch_answer_strings = []
			current_branch_answer_relations = []
			start_token_set = False
			for token, top2_logit in zip(token_ids, token_top2_logits):
				#if the start token is found, gather the tokens until the end token is found to form the answer
				#calculate the disparity for the answer term tokens without the separator token and start and end token
				#add the disparity to the list of disparities
				#add the answer string to the list of answer strings

				
				#for token, top2_logit in zip(tokens, top2_logits):
				if token.item() in answer_start_token_ids and not start_token_set:
					answer_term_tokens = []
					answer_term_logits = []
					start_token_set = True
					current_term_logits = []
					current_term_tokens = []
					answer_term_string = [token]
				elif start_token_set and token.item() not in answer_end_token_ids:
					answer_term_string.append(token)
					if token.item() not in answer_separator_token_ids:
						current_term_tokens.append(token)
						current_term_logits.append(top2_logit)
					else:
						answer_term_tokens.append(current_term_tokens)
						answer_term_logits.append(current_term_logits)
						current_term_tokens = []
						current_term_logits = []
						
				elif start_token_set and token.item() in answer_end_token_ids:
					answer_term_tokens.append(current_term_tokens)
					answer_term_logits.append(current_term_logits)
					current_term_tokens = []
					current_term_logits = []
					start_token_set = False
					if type(answer_term_string) == str: #this means something is wrong with the formatting of the predicted relation
						continue
					answer_term_string.append(token)
					answer_term_string = self.tokenizer.decode(answer_term_string, skip_special_tokens=False)
					#check that there are exactly 3 terms/relations in the answer term tokens
					if len(answer_term_tokens) != 3:
						continue
					current_branch_answer_strings.append(answer_term_string)
					current_branch_answer_relations.append(self.tokenizer.batch_decode(answer_term_tokens))
					#average the disparities of the tokens for each term in the answer_term_logits
					try:
						answer_term_disparities = [torch.stack([logit[0]-logit[1] for logit in term_logits]) for term_logits in answer_term_logits]
					except:
						continue
					#average over each list of disparities for each term in the answer
					answer_term_disparities = [torch.mean(disparities) for disparities in answer_term_disparities]
					current_branch_answer_disparities.append(answer_term_disparities)
					
					
				

			branch_answer_disparities.append(current_branch_answer_disparities)
			branch_answer_strings.append(current_branch_answer_strings)
			branch_answer_relations.append(current_branch_answer_relations)

		
		#only keep those branches where relations were predicted in the right format, i.e. remove empty branches
		branch_answer_disparities = [disparities for disparities in branch_answer_disparities if disparities]
		branch_answer_strings = [strings for strings in branch_answer_strings if strings]
		branch_answer_relations = [relations for relations in branch_answer_relations if relations]

		return branch_answer_disparities, branch_answer_strings, branch_answer_relations
		

	#for each relation get all the disparities for the different branches and dialogues
	def get_relation_disparities(self, dialogues_with_branches: dict, 
				token_index=1, 
				logit_index=2,
				num_branches=5,
				) -> dict[tuple[str]]:
		relation_disparities = {}
		for split, dialogue_responses in dialogues_with_branches.items():
			relation_disparities[split] = {}
			for dialogue_id, branches in dialogue_responses.items():
				branch_token_ids = branches[token_index]
				branch_logits = branches[logit_index]
				branch_answer_disparities, branch_answer_strings, branch_answer_relations = self.branch_disparities(branch_token_ids, branch_logits)
				for i in range(len(branch_answer_disparities)):
					for disparities, strings, relations in zip(branch_answer_disparities[i], branch_answer_strings[i], branch_answer_relations[i]):
						if tuple(relations) not in relation_disparities[split]:
							relation_disparities[split][tuple(relations)] = []
						relation_disparities[split][tuple(relations)].append(disparities)



		return relation_disparities
	
	def get_disparities_per_relation(self, 
				  branch_answer_disparities: list[torch.Tensor], 
				  branch_answer_relations: list[list[str]],
				token_index=1, 
				logit_index=2,
				) -> dict[tuple[str]]:
		relation_disparities = {}
		for i in range(len(branch_answer_disparities)):
			for disparities, relations in zip(branch_answer_disparities[i], branch_answer_relations[i]):
				if tuple(relations) not in relation_disparities:
					relation_disparities[tuple(relations)] = []
				relation_disparities[tuple(relations)].append(disparities)



		return relation_disparities
	     

	def get_relations_above_threshold(self,
	    branch_prediction_ids: list[torch.Tensor],
	    branch_prediction_top_logits: list[torch.Tensor],
	    disparity_threshold: float = 0.5,
	    aggregation_function = torch.mean,
	) -> set[tuple[str]]:
		#input are the predicted ids of the response branches and the top 2 logits for each token in each branch
		#output is the decoded response based on the max disparity aggregation strategy of the relations
		branch_answer_disparities, branch_answer_strings, branch_answer_relations = self.branch_disparities(branch_prediction_ids, branch_prediction_top_logits) 

		#aggregate the disparities of each relation in each branch
		aggregated_disparities = []
		for i in range(len(branch_answer_disparities)):
			disparities = aggregation_function(torch.stack([torch.stack(disparity) for disparity in branch_answer_disparities[i]]), dim=-1)
			if aggregation_function != torch.mean:
				disparities = disparities.values
			aggregated_disparities.append(disparities)

		#based on the aggregated disparities choose which relation to keep
		#also average the occurences of the same relation in different branches
		relation_disparities = self.get_disparities_per_relation(aggregated_disparities, branch_answer_relations)
		#average the disparities of the same relation in different branches
		for key in relation_disparities:
			relation_disparities[key] = torch.mean(torch.stack(relation_disparities[key]))

		#choose the relation with the highest disparity over threshold
		relations_to_keep = set()
		for key in relation_disparities:
			if relation_disparities[key] > disparity_threshold:
				relations_to_keep.add(key)

		return relations_to_keep, relation_disparities
	

	def relation_aggregation(self, relation_disparity_tuple: tuple[torch.Tensor], aggregation = "mean") -> torch.Tensor:
		assert aggregation in ["mean", "median", "max", "min"], "Aggregation method not supported"
		if aggregation == "mean":
			return torch.mean(torch.stack(relation_disparity_tuple))
		elif aggregation == "median":
			return torch.median(torch.stack(relation_disparity_tuple))
		elif aggregation == "max":
			return torch.max(torch.stack(relation_disparity_tuple))
		elif aggregation == "min":
			return torch.min(torch.stack(relation_disparity_tuple))

	#aggregate the disparities for each relation and get a dict with the bin as key and the relations with disparities in that bin as value
	def aggregate_disparities(self, relation_disparities: dict[tuple[str]], aggregation = "mean") -> dict[tuple[str]]:
		assert aggregation in ["mean", "median", "max", "min"], "Aggregation method not supported"
		aggregated_disparity_dict = {}
		for split, relations in relation_disparities.items():
			aggregated_disparity_dict[split] = {}
			for relation, disparities in relations.items():
				#aggregate the disparities for each relation
				aggregated_disparities = [self.relation_aggregation(disparity, aggregation) for disparity in disparities]
				#take the mean of the aggregated disparities of the different occurences
				aggregated_disparities = torch.mean(torch.stack(aggregated_disparities))
				#add the relation to the aggregated_disparities dict
				aggregated_disparity_dict[split][relation] = aggregated_disparities

		return aggregated_disparity_dict
    
		     
