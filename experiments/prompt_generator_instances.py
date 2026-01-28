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
from typing import Union
from prompt_generator import PromptGenerator

llama_prompt_template = """<s>[INST] <<SYS>>
                            {{ system_prompt }}
                            <</SYS>>

                            {{ user_message }} [/INST]""" #user_message is the current instruction and system prompt is a prompt to the system where the prompt could be inserted


gemma_prompt_template = """ 
	<start_of_turn>user \n PROMPT<end_of_turn>\n<start_of_turn>model
	"""

#getter method for getting the right prompt generator class based on the config
def get_prompt_generator(config: LLM_prediction_config) -> PromptGenerator:
	#return the right prompt generator class based on the config
	if config.finetuned_model:
		return FinetunedPromptGenerator(config)
	else:
		return OneshotPromptGenerator(config)
	
	# else:
	# 	raise ValueError("Invalid config for prompt generator")


#classes that implement the generate_prompt method for the PromptGenerator class

class FinetunedPromptGenerator(PromptGenerator):
	#class for generating prompts for LLM for zero shot ontology relation extraction with no memory
	def generate_prompt(self, step: int, dialogue: str, term_list: list[str], relations_so_far: Union[set[tuple[str]], None]=None, additional_input:Union[list[str], None]=None,instruction_prefix = "prompt:", answer_prefix = "completion:") -> str:
		#step is the step in the pipeline
		#dialogue is the current dialogue
		#term_list is the current term list input
		#additional_input is the input that is needed for the prompt generation step 2
		#implement prompt generation

		task_description = self.prompt["task_description"]
		dialogue_input = self.prompt["dialogue"]
		term_input = self.prompt["term_list"]
		output_instruction = self.prompt["output_instruction"]
			
		#step 1, only one step here
		user_text = ""
		user_text += dialogue_input + "\n" + dialogue + "\n"
		user_text += term_input + "\n" + str(term_list)  + "\n"
		user_text += output_instruction + "\n"

		#put the extra tokens in the prompt for the instruction and the answer that were used in training
		LLM_input = instruction_prefix + "\n" + task_description + user_text + "\n" + answer_prefix + "\n"

		return LLM_input

class OneshotPromptGenerator(PromptGenerator):
	#class for generating prompts for LLM for zero shot ontology relation extraction with no memory and reframed
	def generate_prompt(self, step: int, dialogue: str, term_list: list[str], relations_so_far: Union[set[tuple[str]], None]=None, additional_input:Union[list[str], None]=None) -> str:
		#step is the step in the pipeline
		#dialogue is the current dialogue
		#term_list is the current term list input
		#additional_input is the input that is needed for the prompt generation step 2
		#implement prompt generation
		task_description = self.prompt["task_description"]
		dialogue_input = self.prompt["dialogue"]
		term_input = self.prompt["term_list"]
		output_instruction = self.prompt["output_instruction"]
			
		#step 1, only one step here
		user_text = ""
		user_text += dialogue_input + "\n" + dialogue + "\n"
		user_text += term_input + "\n" + str(term_list)  + "\n"
		user_text += output_instruction + "\n"

		if "llama" in self.config.model_name:
			LLM_input = llama_prompt_template.replace("system_prompt", task_description)
			LLM_input = LLM_input.replace("user_message", user_text)
		elif "gemma" in self.config.model_name:
			LLM_input = gemma_prompt_template.replace("PROMPT", task_description + "\n" + user_text)
		else:
			LLM_input = task_description + "\n" + user_text

		return LLM_input

