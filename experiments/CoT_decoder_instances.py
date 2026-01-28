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

#class for generating prompts for LLM for zero shot ontology relation extraction
import torch
from transformers import AutoTokenizer
from CoT_decoder import CoTDecoder


#getter method for getting the right prompt generator class based on the config
def get_CoTDecoder(tokenizer: AutoTokenizer, aggregation_strategy: str = "mean") -> CoTDecoder:
	#return the right prompt generator class based on the config
	if aggregation_strategy == "max":
		return MaxCoTDecoder(tokenizer, aggregation_strategy)
	elif aggregation_strategy == "min":
		return MinCoTDecoder(tokenizer, aggregation_strategy)
	elif aggregation_strategy == "mean":
		return MeanCoTDecoder(tokenizer, aggregation_strategy)
	elif aggregation_strategy == "median":
		return MedianCoTDecoder(tokenizer, aggregation_strategy)
	# elif config.aggregation_strategy == "random":
	# 	return RandomCoTDecoder(config.aggregation_strategy)
	elif aggregation_strategy == "highest-disparity-branch":
		return HighestDisparityBranchCoTDecoder(tokenizer, aggregation_strategy)
	else:
		raise ValueError("Invalid config for prompt generator")


#classes that implement the decode method for the different aggregation strategies

class MaxCoTDecoder(CoTDecoder):
	def decode(self,
	    branch_prediction_ids: list[torch.Tensor],
	    branch_prediction_top_logits: list[torch.Tensor],
	    disparity_threshold: float = 0.5,
	    k=None, #add k to take less than the branches that were computed for analysis
	) -> set[tuple[str]]:
		#input are the predicted ids of the response branches and the top 2 logits for each token in each branch
		#output is the decoded response based on the max disparity aggregation strategy of the relations
		
		relations_to_keep, _ = self.get_relations_above_threshold(branch_prediction_ids, branch_prediction_top_logits, disparity_threshold=disparity_threshold, aggregation_function=torch.max)

		return relations_to_keep

	

class MinCoTDecoder(CoTDecoder):
	def decode(self,
	    branch_prediction_ids: list[torch.Tensor],
	    branch_prediction_top_logits: list[torch.Tensor],
	    disparity_threshold: float = 0.5,
	    k=None, #add k to take less than the branches that were computed for analysis
	) -> set[tuple[str]]:
		#input are the predicted ids of the response branches and the top 2 logits for each token in each branch
		#output is the decoded response based on the min disparity aggregation strategy of the relations
		
		relations_to_keep, _ = self.get_relations_above_threshold(branch_prediction_ids, branch_prediction_top_logits, disparity_threshold=disparity_threshold, aggregation_function=torch.min)

		return relations_to_keep
	
class MeanCoTDecoder(CoTDecoder):
	def decode(self,
	    branch_prediction_ids: list[torch.Tensor],
	    branch_prediction_top_logits: list[torch.Tensor],
	    disparity_threshold: float = 0.5,
	    k=None, #add k to take less than the branches that were computed for analysis
	) -> set[tuple[str]]:
		#input are the predicted ids of the response branches and the top 2 logits for each token in each branch
		#output is the decoded response based on the mean disparity aggregation strategy of the relations
		
		relations_to_keep, _ = self.get_relations_above_threshold(branch_prediction_ids, branch_prediction_top_logits, disparity_threshold=disparity_threshold, aggregation_function=torch.mean)

		return relations_to_keep
	

class MedianCoTDecoder(CoTDecoder):
	def decode(self,
	    branch_prediction_ids: list[torch.Tensor],
	    branch_prediction_top_logits: list[torch.Tensor],
	    disparity_threshold: float = 0.5,
	    k=None, #add k to take less than the branches that were computed for analysis
	) -> set[tuple[str]]:
		#input are the predicted ids of the response branches and the top 2 logits for each token in each branch
		#output is the decoded response based on the median disparity aggregation strategy of the relations
		
		relations_to_keep, _ = self.get_relations_above_threshold(branch_prediction_ids, branch_prediction_top_logits, disparity_threshold=disparity_threshold, aggregation_function=torch.median)

		return relations_to_keep
	
	
class HighestDisparityBranchCoTDecoder(CoTDecoder): #only choose one branch with the highest disparity instead of considering all the relations in all branches
	def decode(self,
	    branch_prediction_ids: list[torch.Tensor],
	    branch_prediction_top_logits: list[torch.Tensor],
	    aggregation_function = torch.mean,
	    disparity_threshold: float = 0.5,
	    k=None, #add k to take less than the branches that were computed for analysis
	    return_branch_number=False,
	) -> set[tuple[str]]:
		#input are the predicted ids of the response branches and the top 2 logits for each token in each branch
		#output is the decoded response based on the highest disparity relations disparity aggregation strategy of the relations
		
		#get the relation dict for each branch and then choose the branch with the highest average disparity across all relations
		branch_relation_dicts = {}
		iterate_over = len(branch_prediction_ids) if k is None else k
		for i in range(iterate_over):
			_, relation_dict = self.get_relations_above_threshold([branch_prediction_ids[i]], [branch_prediction_top_logits[i]], aggregation_function=aggregation_function)
			branch_relation_dicts[i] = relation_dict

		#get the average disparity for each branch
		branch_disparities = {}
		for branch, relation_dict in branch_relation_dicts.items():
			if relation_dict:
				branch_disparities[branch] = sum(relation_dict.values())/len(relation_dict)
			else:
				branch_disparities[branch] = 0

		#get the branch with the highest average disparity
		highest_disparity_branch = max(branch_disparities, key=branch_disparities.get)

		if return_branch_number:
			return branch_relation_dicts[highest_disparity_branch].keys(), highest_disparity_branch

		#return the relations of the branch with the highest disparity
		return branch_relation_dicts[highest_disparity_branch].keys()
	


			
			

	
