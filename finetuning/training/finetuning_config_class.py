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
#class for the training configs
####################################################################################################################

class training_config():
    model_name: str
    dataset_name: str
    num_train_epochs: int
    batch_size: int
    fp16: bool #floating point 16 precision for less memory consumption during training/inference
    eval_steps: int #when to eval/save during training
    eight_bit: int #whether to load the model in 8bit precision for less memory consumption during training/inference
    instruction_prefix: str #prefix for the instruction in the dataset
    answer_prefix: str #prefix for the answer in the dataset
    model_max_length: int #maximum length of the model, if it is not set in the tokenizer
    gradient_checkpointing: bool #whether to use gradient checkpointing for less memory consumption during training
    seed: int #seed for reproducibility/statistical significance
    

    def __init__(self, model_name, dataset_name, num_train_epochs=10, batch_size=8, fp16=True, eval_steps=5000, eight_bit=False, instruction_prefix="prompt:", answer_prefix="completion:", model_max_length=4096, gradient_checkpointing=False, seed=None):
        self.model_name = model_name
        self.dataset_name=dataset_name
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.fp16=fp16
        self.eval_steps=eval_steps
        self.eight_bit=eight_bit
        self.instruction_prefix=instruction_prefix
        self.answer_prefix=answer_prefix
        self.model_max_length=model_max_length
        self.gradient_checkpointing=gradient_checkpointing
        self.seed=seed

    #method for returning the config params as a dict for printing/saving as json
    def to_dict(self):
        return vars(self)