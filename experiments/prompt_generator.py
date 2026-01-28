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

from LLM_prediction_config_class import LLM_prediction_config
from abc import ABC, abstractmethod
from typing import Union

class PromptGenerator(ABC):
	config: LLM_prediction_config
	prompt: dict

	def __init__(self, config):
		self.config = config
		

	@abstractmethod
	def generate_prompt(self, step: int, dialogue: str, term_list: list[str], relations_so_far: Union[set[tuple[str]], None]=None, additional_input:Union[list[str], None]=None) -> str:
		#step is the step in the pipeline
		#dialogue is the current dialogue
		#term_list is the current term list input
		#additional_input is the input that is needed for the prompt generation step 2
		#implement prompt generation
		pass


	def get_prompt_dict_path(self) -> str:
		#return the path to the prompt dict
		only_hasslot = "only_hasslot_" if self.config.only_hasslot else ""
		only_hasvalue = "only_hasvalue_" if self.config.only_hasvalue else ""
		only_hasdomain = "only_hasdomain_" if self.config.only_hasdomain else ""
		only_equivalence = "only_equivalence_" if self.config.only_equivalence else ""
		one_shot = "oneshot_" if self.config.oneshot else ""
		exemplar_from_sgd = "sgd_" if self.config.exemplar_from_sgd else ""


		prompt_dict_file_name = only_hasslot + only_hasvalue + only_hasdomain + only_equivalence + one_shot + exemplar_from_sgd + "prompt_dict.json"

		prompt_dict_folder_name = "prompts/"

		few_shot = "few_shot/" if self.config.oneshot else ""
		one_relation_only = "one_relation_only/" if self.config.only_hasslot or self.config.only_hasvalue or self.config.only_hasdomain or self.config.only_equivalence else ""

		prompt_dict_folder_name = prompt_dict_folder_name + few_shot + one_relation_only


		prompt_dict_path = prompt_dict_folder_name + prompt_dict_file_name

		return prompt_dict_path
	
	def set_prompt_dict(self, prompt_dict: dict):
		#set the prompt dict
		self.prompt = prompt_dict