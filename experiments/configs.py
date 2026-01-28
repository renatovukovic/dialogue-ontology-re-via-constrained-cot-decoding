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

from LLM_prediction_config_class import LLM_prediction_config


def get_config(config_name):
	#return config based on string name
	if config_name not in globals():
		raise ValueError(f"Config name {config_name} not found")
	return globals()[config_name]




###############################################################################################################

### experiments on MultiWOZ and SGD test set with Gemma 2B model ###

#one shot prompt for each relation individually

gemma2b_multiwoz_test_no_memory_reframed_only_hasslot_oneshot = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasslot=True,
												oneshot=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasvalue_oneshot = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasvalue=True,
												oneshot=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasdomain_oneshot = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasdomain=True,
												oneshot=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_equivalence_oneshot = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_equivalence=True,
												oneshot=True,

)


#sgd

gemma2b_sgd_test_no_memory_reframed_only_hasslot_oneshot = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasslot=True,
												oneshot=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasvalue_oneshot = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasvalue=True,
												oneshot=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasdomain_oneshot = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasdomain=True,
												oneshot=True,

)

gemma2b_sgd_test_no_memory_reframed_only_equivalence_oneshot = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_equivalence=True,
												oneshot=True,

)

#now one shot prompt for each relation individually and constrained decoding

gemma2b_multiwoz_test_no_memory_reframed_only_hasslot_oneshot_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasslot=True,
												oneshot=True,
												constrain_generation=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasvalue_oneshot_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasvalue=True,
												oneshot=True,
												constrain_generation=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasdomain_oneshot_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasdomain=True,
												oneshot=True,
												constrain_generation=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_equivalence_oneshot_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_equivalence=True,
												oneshot=True,
												constrain_generation=True,

)


#sgd

gemma2b_sgd_test_no_memory_reframed_only_hasslot_oneshot_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasslot=True,
												oneshot=True,
												constrain_generation=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasvalue_oneshot_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasvalue=True,
												oneshot=True,
												constrain_generation=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasdomain_oneshot_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasdomain=True,
												oneshot=True,
												constrain_generation=True,

)

gemma2b_sgd_test_no_memory_reframed_only_equivalence_oneshot_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_equivalence=True,
												oneshot=True,
												constrain_generation=True,

)


#now one shot prompt for each relation individually and cot decoding

gemma2b_multiwoz_test_no_memory_reframed_only_hasslot_oneshot_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasslot=True,
												oneshot=True,
												predict_for_cot_decoding=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasvalue_oneshot_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasvalue=True,
												oneshot=True,
												predict_for_cot_decoding=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasdomain_oneshot_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasdomain=True,
												oneshot=True,
												predict_for_cot_decoding=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_equivalence_oneshot_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_equivalence=True,
												oneshot=True,
												predict_for_cot_decoding=True,

)


#sgd

gemma2b_sgd_test_no_memory_reframed_only_hasslot_oneshot_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasslot=True,
												oneshot=True,
												predict_for_cot_decoding=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasvalue_oneshot_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasvalue=True,
												oneshot=True,
												predict_for_cot_decoding=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasdomain_oneshot_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasdomain=True,
												oneshot=True,
												predict_for_cot_decoding=True,

)

gemma2b_sgd_test_no_memory_reframed_only_equivalence_oneshot_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_equivalence=True,
												oneshot=True,
												predict_for_cot_decoding=True,

)

#now one shot prompt for each relation individually and constrained cot decoding

gemma2b_multiwoz_test_no_memory_reframed_only_hasslot_oneshot_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasslot=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasvalue_oneshot_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasvalue=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasdomain_oneshot_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasdomain=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_equivalence_oneshot_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_equivalence=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,

)


#sgd

gemma2b_sgd_test_no_memory_reframed_only_hasslot_oneshot_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasslot=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasvalue_oneshot_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasvalue=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasdomain_oneshot_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasdomain=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,

)

gemma2b_sgd_test_no_memory_reframed_only_equivalence_oneshot_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_equivalence=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,

)



############################################################################################################
############################################################################################################

#now model fine-tuned on MultiWOZ predictions
gemma2b_multiwoz_test_no_memory_reframed_finetuned_on_multiwoz = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)


#now model fine-tuned on MultiWOZ + constrained decoding predictions

gemma2b_multiwoz_test_no_memory_reframed_finetuned_on_multiwoz_constrained_generation = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												constrain_generation=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_constrained_generation = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												constrain_generation=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#now model fine-tuned on MultiWOZ + cot decoding predictions

gemma2b_multiwoz_test_no_memory_reframed_finetuned_on_multiwoz_cot_decoding = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 350,
												finetuned_model=True,
												predict_for_cot_decoding=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_cot_decoding = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 350,
												finetuned_model=True,
												predict_for_cot_decoding=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#now model fine-tuned on MultiWOZ + constrained cot decoding predictions
gemma2b_multiwoz_test_no_memory_reframed_finetuned_on_multiwoz_constrained_cot_decoding = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 350,
												finetuned_model=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_constrained_cot_decoding = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 350,
												finetuned_model=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#now the different seeds for fine-tuning on MultiWOZ
#multiwoz test set

#seed 42
gemma2b_multiwoz_test_no_memory_reframed_finetuned_on_multiwoz_seed_42 = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_42",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#seed 142
gemma2b_multiwoz_test_no_memory_reframed_finetuned_on_multiwoz_seed_142 = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_142",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)



#seed 343
gemma2b_multiwoz_test_no_memory_reframed_finetuned_on_multiwoz_seed_343 = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_343",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#seed 442
gemma2b_multiwoz_test_no_memory_reframed_finetuned_on_multiwoz_seed_442 = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_442",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)


# SGD test set 

#seed 42
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_42 = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_42",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)


#now model fine-tuned on MultiWOZ + constrained decoding predictions
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_42_constrained_generation = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_42",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												constrain_generation=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#now model fine-tuned on MultiWOZ + cot decoding predictions
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_42_cot_decoding = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_42",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 350,
												finetuned_model=True,
												predict_for_cot_decoding=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#now model fine-tuned on MultiWOZ + constrained cot decoding predictions
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_42_constrained_cot_decoding = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_42",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 350,
												finetuned_model=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#seed 142
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_142 = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_142",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)


#now model fine-tuned on MultiWOZ + constrained decoding predictions
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_142_constrained_generation = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_142",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												constrain_generation=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#now model fine-tuned on MultiWOZ + cot decoding predictions
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_142_cot_decoding = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_142",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 350,
												finetuned_model=True,
												predict_for_cot_decoding=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#now model fine-tuned on MultiWOZ + constrained cot decoding predictions
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_142_constrained_cot_decoding = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_142",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 350,
												finetuned_model=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)


#seed 343
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_343 = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_343",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)


#now model fine-tuned on MultiWOZ + constrained decoding predictions
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_343_constrained_generation = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_343",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												constrain_generation=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#now model fine-tuned on MultiWOZ + cot decoding predictions
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_343_cot_decoding = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_343",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 350,
												finetuned_model=True,
												predict_for_cot_decoding=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#now model fine-tuned on MultiWOZ + constrained cot decoding predictions
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_343_constrained_cot_decoding = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_343",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 350,
												finetuned_model=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)


#seed 442
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_442 = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_442",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)


#now model fine-tuned on MultiWOZ + constrained decoding predictions
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_442_constrained_generation = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_442",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												constrain_generation=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#now model fine-tuned on MultiWOZ + cot decoding predictions
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_442_cot_decoding = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_442",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 350,
												finetuned_model=True,
												predict_for_cot_decoding=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#now model fine-tuned on MultiWOZ + constrained cot decoding predictions
gemma2b_sgd_test_no_memory_reframed_finetuned_on_multiwoz_seed_442_constrained_cot_decoding = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_multiwoz21_seed_442",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 350,
												finetuned_model=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)


#now model fine-tuned on SGD predictions
gemma2b_multiwoz_test_no_memory_reframed_finetuned_on_sgd = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_sgd",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

gemma2b_sgd_test_no_memory_reframed_finetuned_on_sgd = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_sgd",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#now model fine-tuned on SGD + constrained decoding predictions, cot decoding and constrained cot decoding respectively
gemma2b_sgd_test_no_memory_reframed_finetuned_on_sgd_constrained_generation = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_sgd",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												constrain_generation=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

gemma2b_sgd_test_no_memory_reframed_finetuned_on_sgd_cot_decoding = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_sgd",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 350,
												finetuned_model=True,
												predict_for_cot_decoding=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

gemma2b_sgd_test_no_memory_reframed_finetuned_on_sgd_constrained_cot_decoding = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_sgd",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 350,
												finetuned_model=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#now the different seeds for fine-tuning on SGD

#seed 42
gemma2b_sgd_test_no_memory_reframed_finetuned_on_sgd_seed_42 = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_sgd_seed_42",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)

#seed 142
gemma2b_sgd_test_no_memory_reframed_finetuned_on_sgd_seed_142 = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_sgd_seed_142",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)





#seed 343
gemma2b_sgd_test_no_memory_reframed_finetuned_on_sgd_seed_343 = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_sgd_seed_343",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)


#seed 442
gemma2b_sgd_test_no_memory_reframed_finetuned_on_sgd_seed_442 = LLM_prediction_config(model_name = "../finetuning/models/google-gemma-1.1-2b-it_trained_on_sgd_seed_442",
					  							dataset = "sgd",
												splits = ["test"],
												max_model_length=4096,
												max_new_tokens = 1000,
												finetuned_model=True,
												instruction_prefix="<start_of_turn>user ",
												answer_prefix="<end_of_turn> <start_of_turn>model",

)




#gemma with oneshot example from sgd and each relation separately

gemma2b_multiwoz_test_no_memory_reframed_only_hasslot_oneshot_from_sgd = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasslot=True,
												oneshot=True,
												exemplar_from_sgd=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasvalue_oneshot_from_sgd = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasvalue=True,
												oneshot=True,
												exemplar_from_sgd=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasdomain_oneshot_from_sgd = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasdomain=True,
												oneshot=True,
												exemplar_from_sgd=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_equivalence_oneshot_from_sgd = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_equivalence=True,
												oneshot=True,
												exemplar_from_sgd=True,

)


#sgd

gemma2b_sgd_test_no_memory_reframed_only_hasslot_oneshot_from_sgd = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasslot=True,
												oneshot=True,
												exemplar_from_sgd=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasvalue_oneshot_from_sgd = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasvalue=True,
												oneshot=True,
												exemplar_from_sgd=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasdomain_oneshot_from_sgd = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasdomain=True,
												oneshot=True,
												exemplar_from_sgd=True,

)

gemma2b_sgd_test_no_memory_reframed_only_equivalence_oneshot_from_sgd = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_equivalence=True,
												oneshot=True,
												exemplar_from_sgd=True,

)

#now model with one shot prompt for each relation individually and constrained decoding

gemma2b_multiwoz_test_no_memory_reframed_only_hasslot_oneshot_from_sgd_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasslot=True,
												oneshot=True,
												constrain_generation=True,
												exemplar_from_sgd=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasvalue_oneshot_from_sgd_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasvalue=True,
												oneshot=True,
												constrain_generation=True,
												exemplar_from_sgd=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasdomain_oneshot_from_sgd_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasdomain=True,
												oneshot=True,
												constrain_generation=True,
												exemplar_from_sgd=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_equivalence_oneshot_from_sgd_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 1000,
												only_equivalence=True,
												oneshot=True,
												constrain_generation=True,
												exemplar_from_sgd=True,

)


#sgd

gemma2b_sgd_test_no_memory_reframed_only_hasslot_oneshot_from_sgd_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasslot=True,
												oneshot=True,
												constrain_generation=True,
												exemplar_from_sgd=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasvalue_oneshot_from_sgd_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasvalue=True,
												oneshot=True,
												constrain_generation=True,
												exemplar_from_sgd=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasdomain_oneshot_from_sgd_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_hasdomain=True,
												oneshot=True,
												constrain_generation=True,
												exemplar_from_sgd=True,

)

gemma2b_sgd_test_no_memory_reframed_only_equivalence_oneshot_from_sgd_constrain_generation = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 1000,
												only_equivalence=True,
												oneshot=True,
												constrain_generation=True,
												exemplar_from_sgd=True,

)


#now model with one shot prompt for each relation individually and cot decoding

gemma2b_multiwoz_test_no_memory_reframed_only_hasslot_oneshot_from_sgd_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasslot=True,
												oneshot=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasvalue_oneshot_from_sgd_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasvalue=True,
												oneshot=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasdomain_oneshot_from_sgd_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasdomain=True,
												oneshot=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_equivalence_oneshot_from_sgd_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_equivalence=True,
												oneshot=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)


#sgd

gemma2b_sgd_test_no_memory_reframed_only_hasslot_oneshot_from_sgd_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasslot=True,
												oneshot=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasvalue_oneshot_from_sgd_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasvalue=True,
												oneshot=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasdomain_oneshot_from_sgd_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasdomain=True,
												oneshot=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)

gemma2b_sgd_test_no_memory_reframed_only_equivalence_oneshot_from_sgd_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_equivalence=True,
												oneshot=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)

#now model with one shot prompt for each relation individually and constrained cot decoding

gemma2b_multiwoz_test_no_memory_reframed_only_hasslot_oneshot_from_sgd_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasslot=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasvalue_oneshot_from_sgd_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasvalue=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_hasdomain_oneshot_from_sgd_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasdomain=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)

gemma2b_multiwoz_test_no_memory_reframed_only_equivalence_oneshot_from_sgd_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "multiwoz21",
												splits = ["test"],
												max_new_tokens = 350,
												only_equivalence=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)


#sgd

gemma2b_sgd_test_no_memory_reframed_only_hasslot_oneshot_from_sgd_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasslot=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasvalue_oneshot_from_sgd_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasvalue=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)

gemma2b_sgd_test_no_memory_reframed_only_hasdomain_oneshot_from_sgd_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_hasdomain=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)

gemma2b_sgd_test_no_memory_reframed_only_equivalence_oneshot_from_sgd_constrained_cot_decoding = LLM_prediction_config(model_name = "google/gemma-1.1-2b-it",
					  							dataset = "sgd",
												splits = ["test"],
												max_new_tokens = 350,
												only_equivalence=True,
												oneshot=True,
												constrain_generation=True,
												predict_for_cot_decoding=True,
												exemplar_from_sgd=True,

)


