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
#list of configs for fine-tuning models on DORE
####################################################################################################################

from training.finetuning_config_class import training_config

def get_config(config_name):
	#input: name of the config
	#output: the respective config instance for the names

	#return config based on string name
	if config_name not in globals():
		raise ValueError(f"Config name {config_name} not found")
	return globals()[config_name]


###multiwoz dataset training configs
gemma_2b_multiwoz21_training = training_config(
	model_name="google/gemma-1.1-2b-it",
	dataset_name="multiwoz21",
	eval_steps=5000,
	num_train_epochs=5,
	batch_size=1,
	model_max_length=4096,
	instruction_prefix="<start_of_turn>user ",
	answer_prefix="<end_of_turn> <start_of_turn>model",
)

gemma_2b_sgd_training = training_config(
	model_name="google/gemma-1.1-2b-it",
	dataset_name="sgd",
	eval_steps=5000,
	num_train_epochs=5,
	batch_size=1,
	model_max_length=4096,
	gradient_checkpointing=True,
	instruction_prefix="<start_of_turn>user ",
	answer_prefix="<end_of_turn> <start_of_turn>model",
)


##add some configs with different seeds for statistical significance

#first for the multiwoz training
gemma_2b_multiwoz21_training_seed_42 = training_config(
	model_name="google/gemma-1.1-2b-it",
	dataset_name="multiwoz21",
	eval_steps=5000,
	num_train_epochs=5,
	batch_size=1,
	model_max_length=4096,
	instruction_prefix="<start_of_turn>user ",
	answer_prefix="<end_of_turn> <start_of_turn>model",
	seed=42,
)

gemma_2b_multiwoz21_training_seed_142 = training_config(
	model_name="google/gemma-1.1-2b-it",
	dataset_name="multiwoz21",
	eval_steps=5000,
	num_train_epochs=5,
	batch_size=1,
	model_max_length=4096,
	instruction_prefix="<start_of_turn>user ",
	answer_prefix="<end_of_turn> <start_of_turn>model",
	seed=142,
)

gemma_2b_multiwoz21_training_seed_343 = training_config(
	model_name="google/gemma-1.1-2b-it",
	dataset_name="multiwoz21",
	eval_steps=5000,
	num_train_epochs=5,
	batch_size=1,
	model_max_length=4096,
	instruction_prefix="<start_of_turn>user ",
	answer_prefix="<end_of_turn> <start_of_turn>model",
	seed=343,
)

gemma_2b_multiwoz21_training_seed_442 = training_config(
	model_name="google/gemma-1.1-2b-it",
	dataset_name="multiwoz21",
	eval_steps=5000,
	num_train_epochs=5,
	batch_size=1,
	model_max_length=4096,
	instruction_prefix="<start_of_turn>user ",
	answer_prefix="<end_of_turn> <start_of_turn>model",
	seed=442,
)


#now for the SGD training
gemma_2b_sgd_training_seed_42 = training_config(
	model_name="google/gemma-1.1-2b-it",
	dataset_name="sgd",
	eval_steps=5000,
	num_train_epochs=5,
	batch_size=1,
	model_max_length=4096,
	gradient_checkpointing=True,
	instruction_prefix="<start_of_turn>user ",
	answer_prefix="<end_of_turn> <start_of_turn>model",
	seed=42,
)

gemma_2b_sgd_training_seed_142 = training_config(
	model_name="google/gemma-1.1-2b-it",
	dataset_name="sgd",
	eval_steps=5000,
	num_train_epochs=5,
	batch_size=1,
	model_max_length=4096,
	gradient_checkpointing=True,
	instruction_prefix="<start_of_turn>user ",
	answer_prefix="<end_of_turn> <start_of_turn>model",
	seed=142,
)


gemma_2b_sgd_training_seed_343 = training_config(
	model_name="google/gemma-1.1-2b-it",
	dataset_name="sgd",
	eval_steps=5000,
	num_train_epochs=5,
	batch_size=1,
	model_max_length=4096,
	gradient_checkpointing=True,
	instruction_prefix="<start_of_turn>user ",
	answer_prefix="<end_of_turn> <start_of_turn>model",
	seed=343,
)

gemma_2b_sgd_training_seed_442 = training_config(
	model_name="google/gemma-1.1-2b-it",
	dataset_name="sgd",
	eval_steps=5000,
	num_train_epochs=5,
	batch_size=1,
	model_max_length=4096,
	gradient_checkpointing=True,
	instruction_prefix="<start_of_turn>user ",
	answer_prefix="<end_of_turn> <start_of_turn>model",
	seed=442,
)