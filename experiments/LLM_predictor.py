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

#predictor class for LLM that takes different LLM models as input and does predictions for a given prompt

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from LLM_prediction_config_class import LLM_prediction_config
from typing import Union
from tqdm import tqdm

class LLM_predictor:
	config: LLM_prediction_config
	device: torch.device
	hugginfcae_auth_token: str
    
	def __init__(self, config, device):
		self.model_name = config.model_name
		self.device = device
		#if there is a gpu use torch dtype float16
		if torch.cuda.is_available():
			#torch.set_default_tensor_type(torch.cuda.HalfTensor)
			#if it is a funetuned model merge it with the adapter for faster inference
			self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
							torch_dtype=torch.float16,)
		else:
			self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, add_prefix_space=True)
		self.tokenizer.truncation_side = "left"

			

		self.model.eval()
		self.model.to(self.device)
		self.max_new_tokens = config.max_new_tokens
		self.max_model_length = min(config.max_model_length, self.tokenizer.model_max_length) #4096 is the max length of the llama model, however the output for model_max_length is some random number
		
		

	
	def predict(self, prompt: str, constrain_generation=False, constrained_beamsearch=False, predict_for_cot_decoding=False, analyse_top_k_tokens=False, entropy_branching_threshold=0., term_list=None, relation_list=["has slot", "has value", "has domain", "refers to same concept as"]):
		#if constrain_generation=True constrain the model on only generating the terms in the list on the first and third position of a tuple and generating a relation on the second position of the tuple

		if constrain_generation and predict_for_cot_decoding:
			branch_strings, branch_tokens, branch_logits, entropy = self.constrained_cot_decoding(prompt, k=5, temperature=1., max_length=self.max_model_length-self.max_new_tokens, batch_size=1, entropy_branching_threshold=entropy_branching_threshold, term_list=term_list, relation_list=relation_list)

			return branch_strings, branch_tokens, branch_logits, entropy

		elif constrain_generation:
			
			outputs = self.constrained_generation(prompt, term_list=term_list, max_length=self.max_model_length-self.max_new_tokens, relation_list=relation_list)
			return outputs
		
		elif predict_for_cot_decoding:
			branch_strings, branch_tokens, branch_logits, entropy, branch_length = self.predict_for_cot_decoding(prompt, k=5, temperature=1., max_length=self.max_model_length-self.max_new_tokens, batch_size=1, entropy_branching_threshold=entropy_branching_threshold) 

			return branch_strings, branch_tokens, branch_logits, entropy, branch_length
		
		elif analyse_top_k_tokens:
			top_k_token_ids, top_k_token_logits, entropies = self.top_k_token_analysis(prompt, k=20, temperature=1., max_length=self.max_model_length-self.max_new_tokens) 

			return top_k_token_ids, top_k_token_logits, entropies
		

		#truncate the input such that the output does not exceed the max model length
		inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_model_length-self.max_new_tokens)
		prompt_length = len(self.tokenizer.decode(
										inputs["input_ids"][0],
										skip_special_tokens=True,
										clean_up_tokenization_spaces=True,
									)
								)
		
		#generate the output
		output = self.model.generate(inputs["input_ids"].to(self.device), 
			       attention_mask=inputs["attention_mask"].to(self.device),
				   num_return_sequences=1, 
				   pad_token_id=self.tokenizer.eos_token_id, 
				   eos_token_id=self.tokenizer.eos_token_id,
				   #do_sample=False,
				   num_beams=1,
				   max_new_tokens=self.max_new_tokens)
		#decode the output
		output = self.tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
		#only return the generated text
		outputs = output[prompt_length:]


		return outputs


	def constrained_generation(self, 
			    input_text: str = "", 
				term_list=None, 
				relation_list=["has slot", "has value", "has domain", "refers to same concept as"], 
				start_token="[", end_token="]", 
				separator_token=",", 
				max_length=100, 
				temperature=1., 
				do_sample=False, 
				compare_greedy_to_constrained_logits=False,
				):
		
		input_ids = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).input_ids[0]
		
		current_token_id = input_ids[-1].item()

		#also add the token ids with new lines or any other token right in front of the token without space as they lead to different token ids
		#these are also used by the model normally
		start_token_id = self.tokenizer("\n" + start_token).input_ids[-1]
		end_token_id = self.tokenizer("\n" + end_token).input_ids[-1]
		separator_token_id = self.tokenizer("\n" + separator_token).input_ids[-1]
		
		#get the token ids of the start and end tokens, separator token and the term and relation lists
		start_token_id2 = self.tokenizer(start_token).input_ids[-1]
		end_token_id2 = self.tokenizer(end_token).input_ids[-1]
		separator_token_id2 = self.tokenizer(separator_token).input_ids[-1]

		#for gemma the token ids are different if there is a whitespace in front of the token
		start_token_id3 = self.tokenizer(" " + start_token).input_ids[-1]
		end_token_id3 = self.tokenizer(" " + end_token).input_ids[-1]
		separator_token_id3 = self.tokenizer(" " + separator_token).input_ids[-1]

		

		start_token_ids = [start_token_id, start_token_id2, start_token_id3]
		end_token_ids = [end_token_id, end_token_id2, end_token_id3]
		separator_token_ids = [separator_token_id, separator_token_id2, separator_token_id3]


		if term_list:
			term_list_ids = self.tokenizer(term_list).input_ids
		if relation_list:
			relation_list_ids = self.tokenizer(relation_list).input_ids

		
		k_initial = 0
		#if there is a bos token at the start of each tokenized term and relation, set the initial k to 1, so that it is skipped
		if self.tokenizer.bos_token_id in term_list_ids[0]:
			#print("bos token found")
			k_initial =  1

		#get the length of the input text
		prompt_length = len(self.tokenizer.decode(
										input_ids,
										skip_special_tokens=True,
										clean_up_tokenization_spaces=True,
									)
								)


		if compare_greedy_to_constrained_logits:
			#TODO compare the logits of the greedy generation to the constrained generation to see whether the logits are off in constrained generation in terms of absolute values
			#i.e. compute the difference of the average highest logits of next tokens in normal generation to the average logits in constrained generation in one or several predictions
			pass

		i = 0
		start_token_set = False
		first_separator = False
		term_id_to_pop = None
		while i < self.max_new_tokens:
		#for i in tqdm(range(self.max_new_tokens)):
			with torch.no_grad():
				outputs = self.model(input_ids.unsqueeze(0).to(self.device))
				
			#apply softmax with temperature to the logits
			soft_temp_logits = torch.softmax(outputs.logits[0, -1, :] / temperature, dim=-1).to("cpu")

			input_ids.to("cpu")
			#if the start token is set, find the highest probability token from the term list
			if current_token_id in start_token_ids:
				input_ids, i, current_token_id, start_token_set, first_separator, term_list_ids, term_id_to_pop = self.do_constrained_generation(logits=soft_temp_logits, input_ids=input_ids, current_token_id=current_token_id, i=i, k_initial=k_initial, term_list_ids=term_list_ids, start_token_id=start_token_id, end_token_id=end_token_id, separator_token_id=separator_token_id, start_token_set=start_token_set, first_separator=first_separator, popped_term_id=term_id_to_pop, compare_greedy_to_constrained_logits=compare_greedy_to_constrained_logits)
					

			elif current_token_id in separator_token_ids and start_token_set:
				if first_separator:
					input_ids, i, current_token_id, start_token_set, first_separator, _, term_id_to_pop = self.do_constrained_generation(logits=soft_temp_logits, input_ids=input_ids, current_token_id=current_token_id, i=i, k_initial=k_initial, term_list_ids=relation_list_ids, start_token_id=start_token_id, end_token_id=end_token_id, separator_token_id=separator_token_id, start_token_set=start_token_set, first_separator=first_separator, popped_term_id=term_id_to_pop, compare_greedy_to_constrained_logits=compare_greedy_to_constrained_logits)
						

				else: #get another term and put the end token
					input_ids, i, current_token_id, start_token_set, first_separator, term_list_ids, term_id_to_pop = self.do_constrained_generation(logits=soft_temp_logits, input_ids=input_ids, current_token_id=current_token_id, i=i, k_initial=k_initial, term_list_ids=term_list_ids, start_token_id=start_token_id, end_token_id=end_token_id, separator_token_id=separator_token_id, start_token_set=start_token_set, first_separator=first_separator, popped_term_id=term_id_to_pop, compare_greedy_to_constrained_logits=compare_greedy_to_constrained_logits)


			elif do_sample:
				#TODO: implement speculative sampling as in the huggingface code, as only with sampling temperature makes a difference
				pass
			
			else:
				#get the highest probability token for the last token in the input and add it to the input
				highest_prob_index = torch.argmax(soft_temp_logits)
				input_ids = torch.cat([input_ids, highest_prob_index.unsqueeze(0)], dim=0)
				current_token_id = highest_prob_index.item()


			if current_token_id == self.tokenizer.eos_token_id:
				break

			i += 1


		#decode the input ids, only take what is after the prompt
		decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

		return decoded_text[prompt_length:]




	def do_constrained_generation(self, logits, input_ids, current_token_id, i, k_initial, term_list_ids, start_token_id, end_token_id, separator_token_id, start_token_set=False, first_separator=False, popped_term_id=None, compare_greedy_to_constrained_logits=False):
		k=k_initial
		#set same token indices to all indices in the term list
		same_token_indices = list(range(len(term_list_ids)))
		while True:
			#find the highest probability token from the term list first tokens, but only consider the indices that are in the same_token_indices
			current_term_token_ids = [ids[k] for i, ids in enumerate(term_list_ids) if i in same_token_indices and len(ids) > k]
			current_term_token_indices = [i for i in range(len(term_list_ids)) if i in same_token_indices and len(term_list_ids[i]) > k]

			term_list_probs = logits[current_term_token_ids]
			highest_prob_index = torch.argmax(term_list_probs).item()
			#in case of a term being part of another term check if the separator token is possibly more probable
			#first check if there are nested terms
			current_terms = [self.tokenizer.decode(ids) for i, ids in enumerate(term_list_ids) if i in same_token_indices]
			#check if there is a term that is not in the current term token ids but present in other terms
			shorter_terms = [self.tokenizer.decode(ids) for i, ids in enumerate(term_list_ids) if i in same_token_indices and len(ids) <= k]
			nested_terms = False
			if any([shorter_term in term for term in current_terms for shorter_term in shorter_terms]):
				nested_terms = True
			if logits[separator_token_id] > term_list_probs[highest_prob_index] and nested_terms:
				#set the separator token as the next token
				input_ids = torch.cat([input_ids, torch.tensor([separator_token_id])], dim=0)
				current_token_id = separator_token_id
				first_separator = True
				break
			current_token_id = current_term_token_ids[highest_prob_index]
			input_ids = torch.cat([input_ids, torch.tensor([current_token_id])], dim=0)


			same_token_indices = [i for i in same_token_indices if len(term_list_ids[i]) > k and all([term_list_ids[i][j] == term_list_ids[current_term_token_indices[highest_prob_index]][j] for j in range(k+1)])]

			k+=1
			if len(same_token_indices) < 2:
				#if the number of tokens for the chosen term is more than k then add the rest of the tokens
				while len(term_list_ids[current_term_token_indices[highest_prob_index]]) > k:
					current_token_id = term_list_ids[current_term_token_indices[highest_prob_index]][k]
					input_ids = torch.cat([input_ids, torch.tensor([current_token_id])], dim=0)
					k+=1
					i+=1 


				if not start_token_set:
					#remove the term from the term list ids in the first step in order to mitigate two same terms in a relation
					popped_term_id = term_list_ids.pop(current_term_token_indices[highest_prob_index])
					#set the separator token as the next token
					input_ids = torch.cat([input_ids, torch.tensor([separator_token_id])], dim=0)
					current_token_id = separator_token_id
					start_token_set = True
					first_separator = True
					i+=1
					break

				else:
					if first_separator:
						#set the separator token as the next token
						input_ids = torch.cat([input_ids, torch.tensor([separator_token_id])], dim=0)
						current_token_id = separator_token_id
						first_separator = False
						i+=1
						break
					else:
						#add the first term token ids that were removed at the start back into the term list ids
						if popped_term_id is not None:
							term_list_ids.append(popped_term_id)
						#reset the popped term id
						popped_term_id = None
						#set the end token as the next token
						input_ids = torch.cat([input_ids, torch.tensor([end_token_id])], dim=0)
						current_token_id = end_token_id
						i+=1
						start_token_set = False
						break
			
			i+=1

		return input_ids, i, current_token_id, start_token_set, first_separator, term_list_ids, popped_term_id
	

	def predict_for_cot_decoding(self, 
		  input_text, 
		  k=2,
		  temperature=1.,
		  max_length=100,
		  batch_size=1,
		  entropy_branching_threshold=0.,
		  ) -> Union[list[str],  list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
		#CoT resaoning without prompting by Wang and Zhou 2024
		#In this function just get the logits for the top 2 tokens for each of the k branches after the first token
		#both lists have length k
		#the output are the chosen token ids decoded and not decoded for each of the k branches plus the top 2 logits for each token of the k branches
		#the resulting list of strings has length k
		#the resulting token id tensor has shape (k, longest_branch_length, 1)
		#the resulting logits tensor has shape (k, longest_branch_length, 2)
		#the shorter branches that contain the eos token are padded with more eos tokens until the longest branch length is reached

		input_ids = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).input_ids[0]
		#current_token_id = input_ids[-1].item()


		#do predictions until the entropy threshold for branching is surpassed
		branch_length = 0
		tokens_until_branch = []
		for i in range(max_length):
			with torch.no_grad():
				outputs = self.model(input_ids.unsqueeze(0).to(self.device))	
			#apply softmax with temperature to the logits
			soft_temp_logits = torch.softmax(outputs.logits[0, -1, :] / temperature, dim=-1).to("cpu")
			#also save the entropy of the first token
			entropy = -torch.sum(soft_temp_logits * torch.log(soft_temp_logits))

			#check if the entropy is below the threshold
			if entropy < entropy_branching_threshold:
				#add the highest probability token to the input ids
				current_token_id = torch.argmax(soft_temp_logits).item()
				tokens_until_branch.append(current_token_id)
				input_ids = torch.cat([input_ids, torch.tensor([current_token_id])], dim=0)
			else:
				branch_length = i
				break

		tokens_until_branch = torch.tensor(tokens_until_branch)
		#now branch out from the input ids

		#get the top k token ids for continuation
		top_k_token_ids = torch.topk(soft_temp_logits, k).indices

		#initialise the list of branches with the input ids plus the first token for each branch
		branches = [torch.cat([input_ids, torch.tensor([token_id])], dim=0) for token_id in top_k_token_ids]

		#initialise the list of logits for each branch
		branch_logits = [torch.tensor([[soft_temp_logits[token_id], 0]]) for token_id in top_k_token_ids] #for the first token the second highest logit is 0 because it was already taken from the top k
		branch_tokens = [torch.tensor([token_id]) for token_id in top_k_token_ids]

		
		#make branch batches of size batch_size
		branch_batches = [branches[i:i+batch_size] for i in range(0, len(branches), batch_size)]
		#make a list of whether the eos token was reached for each branch in each batch
		eos_reached = [[False for _ in range(len(branch_batches[i]))] for i in range(len(branch_batches))]
		

		#run each batch separately until its end
		#then run the next batch until its end
		#print(f"run on {len(branch_batches)} batches of size {batch_size}")
		for i, branch_batch in enumerate(branch_batches):
			#run the model for the batch of branches
			for j in range(self.max_new_tokens - branch_length):
				#run the model for the batch of branches
				with torch.no_grad():
					outputs = self.model(torch.stack(branch_batch).to(self.device))

				#apply softmax with temperature to the logits
				softmax_logits = torch.softmax(outputs.logits[:, -1, :] / temperature, dim=-1).to("cpu")

				#print(softmax_logits.shape)

				#get the top 2 logits for each branch token and also the highest logit token for the continuation of each branch
				top_k_logits = [torch.topk(logits, 2).values for logits in softmax_logits]

				#print(len(top_k_logits))
				#now get the top tokens for each branch, i.e. argmax for each branch
				branch_top_tokens = [torch.argmax(logits, dim=-1) for logits in softmax_logits]

				#print(branch_top_tokens)

				#add the top tokens to the branches
				for branch_index, branch in enumerate(branch_batch):
					current_branch_index = i * batch_size + branch_index
					branch = torch.cat([branch, branch_top_tokens[branch_index].unsqueeze(0)], dim=0)
					branch_batch[branch_index] = branch
					#add the top token to the branch tokens if the eos token was not reached
					if not eos_reached[i][branch_index]:
						branch_tokens[current_branch_index] = torch.cat([branch_tokens[current_branch_index], branch_top_tokens[branch_index].unsqueeze(0)], dim=0)
						branch_logits[current_branch_index] = torch.cat([branch_logits[current_branch_index], top_k_logits[branch_index].unsqueeze(0)], dim=0)
						#check if the end token is the next token in any of the branches
						#if that is the case do not add the tokens to the branch tokens and the logits anymore in later steps
						if branch_top_tokens[branch_index].item() == self.tokenizer.eos_token_id:
							eos_reached[i][branch_index] = True

				#check if all branches have reached the eos token
				if all(eos_reached[i]):
					break


		#return the branches as strings and as token ids and also add the tokens_until_branch before the branches
		branch_strings = [self.tokenizer.decode(torch.cat([tokens_until_branch, branch_tokens[i]], dim=0).int()) for i in range(len(branch_tokens))]


		return branch_strings, branch_tokens, branch_logits, entropy, branch_length



	### analyse the semantics of different branching point top k tokens and compare to entropy
	def top_k_token_analysis(self, 
		  input_text, 
		  k=2,
		  temperature=1.,
		  max_length=100,
		  ) -> Union[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
		#return the top k token ids for the next token in the sequence and the entropy for each token, also return the top k token logits

		input_ids = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).input_ids[0]
		#current_token_id = input_ids[-1].item()

		entropies = []
		top_k_token_id_list = []
		top_k_token_logits = []

		for i in range(self.max_new_tokens):

			with torch.no_grad():
				outputs = self.model(input_ids.unsqueeze(0).to(self.device))
				
			#apply softmax with temperature to the logits
			soft_temp_logits = torch.softmax(outputs.logits[0, -1, :] / temperature, dim=-1).to("cpu")

			#get the top k token ids for continuation
			top_k_token_ids = torch.topk(soft_temp_logits, k).indices
			top_k_token_id_list.append(top_k_token_ids)

			#get the top k token logits
			top_k_token_logits.append(torch.topk(soft_temp_logits, k).values)

			#add the top 1 token to the input_ids
			input_ids = torch.cat([input_ids, top_k_token_ids[0].unsqueeze(0)], dim=0)

			#also save the entropy of the first token
			entropy = torch.sum(-soft_temp_logits * torch.log(soft_temp_logits))
			entropies.append(entropy)

			#check if eos token was reached
			if top_k_token_ids[0].item() == self.tokenizer.eos_token_id:
				break



		



		return top_k_token_id_list, top_k_token_logits, entropies
	


	#function for cot decoding plus constrained generation
	def constrained_cot_decoding(self,
					input_text, 
					k=2,
					temperature=1.,
					max_length=100,
					batch_size=1,
					entropy_branching_threshold=0.,
					term_list=None, 
					relation_list=["has slot", "has value", "has domain", "refers to same concept as"], 
					start_token="[", end_token="]", 
					separator_token=",", 
					do_sample=False, 
					compare_greedy_to_constrained_logits=False
					):
		input_ids = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).input_ids[0]
		#current_token_id = input_ids[-1].item()

		current_token_id = input_ids[-1].item()

		#also add the token ids with new lines or any other token right in front of the token without space as they lead to different token ids
		#these are also used by the model normally
		start_token_id = self.tokenizer("\n" + start_token).input_ids[-1]
		end_token_id = self.tokenizer("\n" + end_token).input_ids[-1]
		separator_token_id = self.tokenizer("\n" + separator_token).input_ids[-1]
		
		#get the token ids of the start and end tokens, separator token and the term and relation lists
		start_token_id2 = self.tokenizer(start_token).input_ids[-1]
		end_token_id2 = self.tokenizer(end_token).input_ids[-1]
		separator_token_id2 = self.tokenizer(separator_token).input_ids[-1]

		#for gemma the token ids are different if there is a whitespace in front of the token
		start_token_id3 = self.tokenizer(" " + start_token).input_ids[-1]
		end_token_id3 = self.tokenizer(" " + end_token).input_ids[-1]
		separator_token_id3 = self.tokenizer(" " + separator_token).input_ids[-1]


		start_token_ids = [start_token_id, start_token_id2, start_token_id3]
		end_token_ids = [end_token_id, end_token_id2, end_token_id3]
		separator_token_ids = [separator_token_id, separator_token_id2, separator_token_id3]

		# [hotel, has slot, price range]


		if term_list:
			term_list_ids = self.tokenizer(term_list).input_ids
		if relation_list:
			relation_list_ids = self.tokenizer(relation_list).input_ids

		
		k_initial = 0
		#if there is a bos token at the start of each tokenized term and relation, set the initial k to 1, so that it is skipped
		if self.tokenizer.bos_token_id in term_list_ids[0]:
			#print("bos token found")
			k_initial =  1


		with torch.no_grad():
			outputs = self.model(input_ids.unsqueeze(0).to(self.device))	
		#apply softmax with temperature to the logits
		soft_temp_logits = torch.softmax(outputs.logits[0, -1, :] / temperature, dim=-1).to("cpu")
		#also save the entropy of the first token
		entropy = -torch.sum(soft_temp_logits * torch.log(soft_temp_logits))

		#get the top k token ids for continuation
		top_k_token_ids = torch.topk(soft_temp_logits, k).indices

		#initialise the list of branches with the input ids plus the first token for each branch
		branches = [torch.cat([input_ids, torch.tensor([token_id])], dim=0) for token_id in top_k_token_ids]

		#initialise the list of logits for each branch
		branch_logits = [torch.tensor([[soft_temp_logits[token_id], 0]]) for token_id in top_k_token_ids] #for the first token the second highest logit is 0 because it was already taken from the top k
		branch_tokens = [torch.tensor([token_id]) for token_id in top_k_token_ids]

		
		

		#run each branch with constrained generation until its end
		for branch_index, branch in enumerate(branches):
			input_ids = branch
			i = 0
			start_token_set = False
			first_separator = False
			term_id_to_pop = None
			while i < self.max_new_tokens:
				with torch.no_grad():
					outputs = self.model(input_ids.unsqueeze(0).to(self.device))
					
				#apply softmax with temperature to the logits
				soft_temp_logits = torch.softmax(outputs.logits[0, -1, :] / temperature, dim=-1).to("cpu")

				#also get the original logits for normalising over the set of term tokens for cot decoding disparity
				original_logits = outputs.logits[0, -1, :].to("cpu")

				input_ids.to("cpu")
				#if the start token is set, find the highest probability token from the term list
				if current_token_id in start_token_ids:
					input_ids, i, current_token_id, start_token_set, first_separator, term_list_ids, term_id_to_pop, top_2_logits, tokens_added = self.do_constrained_cot_generation(logits=soft_temp_logits, input_ids=input_ids, current_token_id=current_token_id, i=i, k_initial=k_initial, term_list_ids=term_list_ids, start_token_id=start_token_id, end_token_id=end_token_id, separator_token_id=separator_token_id, start_token_set=start_token_set, first_separator=first_separator, popped_term_id=term_id_to_pop, compare_greedy_to_constrained_logits=compare_greedy_to_constrained_logits, original_logits=original_logits)

					#turn the top 2 logit list into a tensor and append it to the branch logits
					branch_logits[branch_index] = torch.cat([branch_logits[branch_index], torch.stack(top_2_logits)], dim=0)
					branch_tokens[branch_index] = torch.cat([branch_tokens[branch_index], torch.tensor(tokens_added)], dim=0)
						

				elif current_token_id in separator_token_ids and start_token_set:
					if first_separator:
						input_ids, i, current_token_id, start_token_set, first_separator, _, term_id_to_pop, top_2_logits, tokens_added = self.do_constrained_cot_generation(logits=soft_temp_logits, input_ids=input_ids, current_token_id=current_token_id, i=i, k_initial=k_initial, term_list_ids=relation_list_ids, start_token_id=start_token_id, end_token_id=end_token_id, separator_token_id=separator_token_id, start_token_set=start_token_set, first_separator=first_separator, popped_term_id=term_id_to_pop, compare_greedy_to_constrained_logits=compare_greedy_to_constrained_logits, original_logits=original_logits)

						#turn the top 2 logit list into a tensor and append it to the branch logits
						branch_logits[branch_index] = torch.cat([branch_logits[branch_index], torch.stack(top_2_logits)], dim=0)
						branch_tokens[branch_index] = torch.cat([branch_tokens[branch_index], torch.tensor(tokens_added)], dim=0)
							

					else: #get another term and put the end token
						input_ids, i, current_token_id, start_token_set, first_separator, term_list_ids, term_id_to_pop, top_2_logits, tokens_added = self.do_constrained_cot_generation(logits=soft_temp_logits, input_ids=input_ids, current_token_id=current_token_id, i=i, k_initial=k_initial, term_list_ids=term_list_ids, start_token_id=start_token_id, end_token_id=end_token_id, separator_token_id=separator_token_id, start_token_set=start_token_set, first_separator=first_separator, popped_term_id=term_id_to_pop, compare_greedy_to_constrained_logits=compare_greedy_to_constrained_logits, original_logits=original_logits)

						#turn the top 2 logit list into a tensor and append it to the branch logits
						branch_logits[branch_index] = torch.cat([branch_logits[branch_index], torch.stack(top_2_logits)], dim=0)
						branch_tokens[branch_index] = torch.cat([branch_tokens[branch_index], torch.tensor(tokens_added)], dim=0)

				
				else:
					#get the highest probability token for the last token in the input and add it to the input
					highest_prob_index = torch.argmax(soft_temp_logits)
					#get the top two tokens for the current prediction and add them
					top_2_logits = torch.topk(soft_temp_logits, 2).values
					branch_logits[branch_index] = torch.cat([branch_logits[branch_index], top_2_logits.unsqueeze(0)], dim=0)
					input_ids = torch.cat([input_ids, highest_prob_index.unsqueeze(0)], dim=0)
					branch_tokens[branch_index] = torch.cat([branch_tokens[branch_index], highest_prob_index.unsqueeze(0)], dim=0)
					current_token_id = highest_prob_index.item()


				if current_token_id == self.tokenizer.eos_token_id:
					break



				i += 1
			

		#return the branches as strings and as token ids and also add the tokens_until_branch before the branches
		branch_strings = [self.tokenizer.decode(branch_tokens[i]) for i in range(len(branch_tokens))]


		return branch_strings, branch_tokens, branch_logits, entropy
		
		

	def do_constrained_cot_generation(self, logits, input_ids, current_token_id, i, k_initial, term_list_ids, start_token_id, end_token_id, separator_token_id, start_token_set=False, first_separator=False, popped_term_id=None, compare_greedy_to_constrained_logits=False, original_logits=None, temperature=1.0):
		k=k_initial
		#set same token indices to all indices in the term list
		same_token_indices = list(range(len(term_list_ids)))
		tokens_added = []
		while True:
			#find the highest probability token from the term list first tokens, but only consider the indices that are in the same_token_indices
			current_term_token_ids = [ids[k] for i, ids in enumerate(term_list_ids) if i in same_token_indices and len(ids) > k]
			current_term_token_indices = [i for i in range(len(term_list_ids)) if i in same_token_indices and len(term_list_ids[i]) > k]

			term_list_probs = logits[current_term_token_ids]
			highest_prob_index = torch.argmax(term_list_probs).item()
			#in case of a term being part of another term check if the separator token is possibly more probable
			#first check if there are nested terms
			current_terms = [self.tokenizer.decode(ids) for i, ids in enumerate(term_list_ids) if i in same_token_indices]
			#check if there is a term that is not in the current term token ids but present in other terms
			shorter_terms = [self.tokenizer.decode(ids) for i, ids in enumerate(term_list_ids) if i in same_token_indices and len(ids) <= k]
			nested_terms = False
			if any([shorter_term in term for term in current_terms for shorter_term in shorter_terms]):
				nested_terms = True
			if logits[separator_token_id] > term_list_probs[highest_prob_index] and nested_terms:
				#set the separator token as the next token
				input_ids = torch.cat([input_ids, torch.tensor([separator_token_id])], dim=0)
				current_token_id = separator_token_id
				tokens_added.append(separator_token_id)
				first_separator = True
				break
			current_token_id = current_term_token_ids[highest_prob_index]
			input_ids = torch.cat([input_ids, torch.tensor([current_token_id])], dim=0)
			tokens_added.append(current_token_id)


			#get the term logits from the original logits, normalise them and get the top 2 logits
			original_term_logits = original_logits[current_term_token_ids]
			#normalise the logits
			normalised_term_logits = torch.softmax(original_term_logits/temperature, dim=-1)
			#get the top 2 logits if there is more than one token in the list
			if len(current_term_token_ids) > 1:
				top_2_logits = torch.topk(normalised_term_logits, 2).values
			else:
				#put the one token prob and 0 in a tensor
				top_2_logits = torch.tensor([normalised_term_logits[0], 0.0])




			same_token_indices = [i for i in same_token_indices if len(term_list_ids[i]) > k and all([term_list_ids[i][j] == term_list_ids[current_term_token_indices[highest_prob_index]][j] for j in range(k+1)])]

			k+=1
			if len(same_token_indices) < 2:
				#if the number of tokens for the chosen term is more than k then add the rest of the tokens
				while len(term_list_ids[current_term_token_indices[highest_prob_index]]) > k:
					current_token_id = term_list_ids[current_term_token_indices[highest_prob_index]][k]
					tokens_added.append(current_token_id)
					input_ids = torch.cat([input_ids, torch.tensor([current_token_id])], dim=0)
					k+=1
					i+=1 


				if not start_token_set:
					#remove the term from the term list ids in the first step in order to mitigate two same terms in a relation
					popped_term_id = term_list_ids.pop(current_term_token_indices[highest_prob_index])
					#set the separator token as the next token
					input_ids = torch.cat([input_ids, torch.tensor([separator_token_id])], dim=0)
					current_token_id = separator_token_id
					tokens_added.append(separator_token_id)
					start_token_set = True
					first_separator = True
					i+=1
					break

				else:
					if first_separator:
						#set the separator token as the next token
						input_ids = torch.cat([input_ids, torch.tensor([separator_token_id])], dim=0)
						current_token_id = separator_token_id
						tokens_added.append(separator_token_id)
						first_separator = False
						i+=1
						break
					else:
						#add the first term token ids that were removed at the start back into the term list ids
						if popped_term_id is not None:
							term_list_ids.append(popped_term_id)
						#reset the popped term id
						popped_term_id = None
						#set the end token as the next token
						input_ids = torch.cat([input_ids, torch.tensor([end_token_id])], dim=0)
						current_token_id = end_token_id
						tokens_added.append(end_token_id)
						i+=1
						start_token_set = False
						break
			
			i+=1

		#as there is always a separator token or an end token at the end, we have to include the logits for this token which is 1, as it is the only token that can be chose here
		top_2_logits = [top_2_logits, torch.tensor([1.0, 0.0])]

		return input_ids, i, current_token_id, start_token_set, first_separator, term_list_ids, popped_term_id, top_2_logits, tokens_added


