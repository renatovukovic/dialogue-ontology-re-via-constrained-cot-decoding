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

import re
import ast
import json
import networkx as nx
import copy

def tokenize(utt):
    utt_lower = utt.lower()
    #utt_lower = utt_lower.replace("\n", " ")
    utt_lower = utt_lower.replace("\t", " ")
    utt_tok = utt_to_token(utt_lower)
    return utt_tok

def utt_to_token(utt):
    return [tok for tok in map(lambda x: re.sub(" ", "", x), re.split("(\W+)", utt)) if len(tok) > 0]


def present_in_utterance(term, utterance):
    if term in utterance:
        #now check that each token of the splitted term occurs behind each other in the utterance
        splitted_term = term.split()
        splitted_utterance = utterance.split()
        for i in range(len(splitted_utterance)-len(splitted_term)+1):
            if all([splitted_utterance[i+j] == splitted_term[j] for j in range(len(splitted_term))]):
                return True
        return False
    else:
        return False


#function for extracting lists from strings, i.e. the relational triplets from the LLM answers
def extract_list_from_string(text, return_not_parsable = False):
	#if there happen to be normal brackets, replace them with square brackets
	text = text.replace("(", "[").replace(")", "]")
	#replace the special quotes “ and ” with space and ‘ and ’ with nothing
	text = text.replace("“", " ").replace("”", " ").replace("‘", "").replace("’", "")
    #extract everything between the brackets []
	list_strings =  re.findall(r'\[.*?\]', text)
	#remove quotes inbetween opening brackets
	list_strings = [string.replace("['[", "[[") for string in list_strings]
	#check if there are instances with two or more brackets at the start of the string, only keep one then
	list_strings = [re.sub(r'^\[\[', '[', string) for string in list_strings]
	#add quotes around each element in each list string
	list_strings = [string.replace('[', '["').replace(']', '"]').replace(',', '","') for string in list_strings]
	#list_strings = [string.replace('[', '["').replace(']', '"]').replace(',', '","') for string in list_strings]
	#remove single quotes that are now too much
	list_strings = [string.replace("'", "") for string in list_strings]
	#if there are erroneously two quotes, remove one
	list_strings = [string.replace('""', '"') for string in list_strings]
	#remove quotes inbetween opening brackets
	list_strings = [string.replace('["[', '[[') for string in list_strings]
	list_strings = [string.replace('[" [', '[[') for string in list_strings]
	#check if there are instances with two or more brackets at the start of the string, only keep one then
	list_strings = [re.sub(r'^\[\[', '[', string) for string in list_strings]
	#evaluate the string to get the list	
	lists = []
	#count how often a list is not parsable
	not_parsable = 0
	for string in list_strings:
		try:
			lists.append(ast.literal_eval(string))
		except: #strings that are not parsable are not evaluated, just dropped
			not_parsable += 1
			continue
			#raise ValueError(f"Could not evaluate the string {string}")
	
	#tokenize all the strings in the lists
	for i, l in enumerate(lists):
		lists[i] = [" ".join(tokenize(s)) for s in l]
	if return_not_parsable:
		return lists, not_parsable
	else:
		return lists


#here only one-hop connections to ontology values are considered
def get_referrals_to_ontology_values_for_term(term, ontology_values, relations):
	ontology_values_referrals = set()
	for head, rel, tail in relations:
		if rel == "refers to same concept as" and (head == term or tail == term):
			if head == term and tail in ontology_values:
				ontology_values_referrals.add(tail)
			elif tail == term and head in ontology_values:
				ontology_values_referrals.add(head)
	return ontology_values_referrals

def get_referral_connected_ontology_value_components_for_term(term: str, ontology_values: set, connected_components: list) -> set:
	#for a term get the set of terms that are connected to it via refers to same concept as relations, also include multihop connections
	#then return those terms in the set that are ontology values
	connected_components_for_term = set()
	for connected_component in connected_components:
		if term in connected_component:
			connected_components_for_term.update(connected_component)

	#now get the ontology values from the connected components
	ontology_values_for_term = connected_components_for_term.intersection(ontology_values)
	return ontology_values_for_term
	


def evaluate_term_relation_extraction_f1(prediction: set[tuple[str]], labels: set[tuple[str]], map_terms_to_ontology_referrals=True, mapping_based_on_groundtruth=False, ontology_values=None, groundtruth_equivalence_relations=None) -> dict:
    #evaluate the f1, precision and recall of the predicted terms
	#calculate true positives, false positives and false negatives

	#for different prompts, different relation names were used, so we need to map them to the same relation names
	domain_slot_relation_synonyms = ["has_slot", "hasSlot", "domain - slot - relation", "domain - to - slot"]
	slot_value_relation_synonyms = ["has_value", "hasValue", "slot - value - relation", "slot - to - value"]
	value_domain_relation_synonyms = ["belongs_to_domain", "belongsToDomain", "value - domain - relation", "value - to - domain", "belongs to domain"]
	equivalence_relation_synonyms = ["equivalence", "equivalence - relation", "equivalent_to", "isEquivalentTo", "equivalence - relation", "is_equivalent_to", "is equivalent to", "is equivalent", "equivalent"]

	#map the relation names to the same relation names
	prediction_list = []
	for i, (head, relation, tail) in enumerate(prediction):
		if relation in domain_slot_relation_synonyms + ["has slot"]:
			prediction_list.append((head, "has slot", tail))
		elif relation in slot_value_relation_synonyms + ["has value"]:
			prediction_list.append((head, "has value", tail))
		elif relation in value_domain_relation_synonyms + ["has domain"]:
			prediction_list.append((head, "has domain", tail))
		elif relation in equivalence_relation_synonyms + ["refers to same concept as"]:
			prediction_list.append((head, "refers to same concept as", tail))
		else: #if the relation is none of the above, then do not use it
			continue

	prediction = set(prediction_list)


	#print("equivalence predictions:", len([x for x in prediction if x[1] == "refers to same concept as"]))


	#if map_terms_to_ontology_referrals is True, the function will map the terms to the ontology values based on the refers to same concept as relation predictions
	if map_terms_to_ontology_referrals or mapping_based_on_groundtruth:
		#handle the refers to same relations
		refers_to_same = set()
		if mapping_based_on_groundtruth: #use the groundtruth refers to same concept as relations to map the predictions
			refers_to_same = groundtruth_equivalence_relations
		else: #use the predicted refers to same concept as relations to map the predictions
			for head, relation, tail in prediction:
				if relation == "refers to same concept as":
					refers_to_same.add((head, tail))

		#put those term pairs that are connected by refers into the same set to results in a set of sets
		G = nx.Graph()
		G.add_edges_from(refers_to_same)
		#get the connected components
		prediction_refers_to_same_connected_components = list(nx.connected_components(G))

		#now go through the predictions and map the values in the non-refers to same concept as relations to the ontology values if they are related via refers to same concept as
		prediction_copy = copy.deepcopy(prediction)
		for head, rel, tail in prediction_copy:
			if rel != "refers to same concept as":
				if head not in ontology_values and tail not in ontology_values:
					#check if there is a refers to same concept relation to an ontology value in the prediction
					head_referrals = get_referral_connected_ontology_value_components_for_term(head, ontology_values, prediction_refers_to_same_connected_components)
					tail_referrals = get_referral_connected_ontology_value_components_for_term(tail, ontology_values, prediction_refers_to_same_connected_components)
					if len(head_referrals) > 0 and len(tail_referrals) > 0:
						#remove the current relation
						prediction.remove((head, rel, tail))
						#add the new relations with the referrals as pairs
						for h in head_referrals:
							for t in tail_referrals:
								prediction.add((h, rel, t))
				elif head not in ontology_values:
					#check if there is a refers to same concept relation to an ontology value in the prediction
					head_referrals = get_referral_connected_ontology_value_components_for_term(head, ontology_values, prediction_refers_to_same_connected_components)
					if len(head_referrals) > 0:
						#remove the current relation
						prediction.remove((head, rel, tail))
						#add the new relations with the referrals as pairs
						for h in head_referrals:
							prediction.add((h, rel, tail))
				elif tail not in ontology_values:
					#check if there is a refers to same concept relation to an ontology value in the prediction
					tail_referrals = get_referral_connected_ontology_value_components_for_term(tail, ontology_values, prediction_refers_to_same_connected_components)
					if len(tail_referrals) > 0:
						#remove the current relation
						prediction.remove((head, rel, tail))
						#add the new relations with the referrals as pairs
						for t in tail_referrals:
							prediction.add((head, rel, t))
						


	#turn the tuples for refers to same concept as into sets to ignore the direction of the relation
	prediction_with_refer_sets = set(frozenset((head, rel, tail)) if rel == "refers to same concept as" else (head, rel, tail) for head, rel, tail in prediction)
	labels_with_refer_sets = set(frozenset((head, rel, tail)) if rel == "refers to same concept as" else (head, rel, tail) for head, rel, tail in labels)



	true_positives = prediction_with_refer_sets.intersection(labels_with_refer_sets)
	false_positives = prediction_with_refer_sets - labels_with_refer_sets
	false_negatives = labels_with_refer_sets - prediction_with_refer_sets

	#calculate precision, recall and f1
	if len(true_positives) + len(false_positives) == 0:
		precision = 0.
	else:
		precision = len(true_positives) / (len(true_positives) + len(false_positives))
	if len(true_positives) + len(false_negatives) == 0:
		recall = 1.
	else:
		recall = len(true_positives) / (len(true_positives) + len(false_negatives))

	if len(true_positives) == 0 and len(false_positives) == 0 and len(false_negatives) == 0:
		recall = 1.
		precision = 1.

	if precision + recall == 0:
		f1 = 0.
	else:
		f1 = 2 * (precision * recall) / (precision + recall)

	relation_eval_results = {}

	evaluation_dict = {}
	evaluation_dict["precision"] = precision
	evaluation_dict["recall"] = recall
	evaluation_dict["f1"] = f1
	evaluation_dict["true_positives"] = list(list(entry) for entry in true_positives)
	evaluation_dict["false_positives"] = list(list(entry) for entry in false_positives)
	evaluation_dict["false_negatives"] = list(list(entry) for entry in false_negatives)
	
	relation_eval_results["all relations"] = evaluation_dict

	#calculate precision, recall and f1 for each relation, i.e. has slot, has value and refers to same concept as
	for relation in ["has slot", "has value", "has domain", "refers to same concept as"]:
		if relation == "refers to same concept as": #turn the tuples into sets to ignore the direction of the relation
			prediction_relation = set(frozenset((head, rel, tail)) for head, rel, tail in prediction if rel == relation)
			labels_relation = set(frozenset((head, rel, tail)) for head, rel, tail in labels if rel == relation)
		else:
			prediction_relation = set((head, rel, tail) for head, rel, tail in prediction if rel == relation)
			labels_relation = set((head, rel, tail) for head, rel, tail in labels if rel == relation)

		true_positives = prediction_relation.intersection(labels_relation)
		false_positives = prediction_relation - labels_relation
		false_negatives = labels_relation - prediction_relation

		if len(true_positives) + len(false_positives) == 0:
			precision = 0.
		else:
			precision = len(true_positives) / (len(true_positives) + len(false_positives))
		if len(true_positives) + len(false_negatives) == 0:
			recall = 1.
		else:
			recall = len(true_positives) / (len(true_positives) + len(false_negatives))
		if precision + recall == 0:
			f1 = 0.
		else:
			f1 = 2 * (precision * recall) / (precision + recall)

		evaluation_dict = {}
		evaluation_dict["precision"] = precision
		evaluation_dict["recall"] = recall
		evaluation_dict["f1"] = f1
		evaluation_dict["true_positives"] = list(list(entry) for entry in true_positives)
		evaluation_dict["false_positives"] = list(list(entry) for entry in false_positives)
		evaluation_dict["false_negatives"] = list(list(entry) for entry in false_negatives)
		
		relation_eval_results[relation] = evaluation_dict

	return relation_eval_results




#function for getting the one hop neighbour relations of a term in a list of relations
def get_one_hop_neighbour_relations_for_term(term: str, relation_set: set) -> set:
	#return the set of relations that have the term as head or tail
	neighbour_relations = set()
	for head, relation, tail in relation_set:
		if head == term or tail == term:
			neighbour_relations.add((head, relation, tail))
	return neighbour_relations

#now do that for a list of terms
def get_one_hop_neighbour_relations_for_termlist(terms: set, relation_set: set) -> set:
	#return the set of relations that have the term as head or tail
	neighbour_relations = set()
	for term in terms:
		neighbour_relations = neighbour_relations.union(get_one_hop_neighbour_relations_for_term(term, relation_set))
	return neighbour_relations