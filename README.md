# Dialogue Ontology Relation Extraction via Constrained Chain-of-Thought Decoding

This is the code for the SIGDIAL 2024 paper Dialogue Ontology Relation Extraction via Constrained Chain-of-Thought Decoding


## Data
We use the [Multi-WOZ 2.1 Data-set](https://github.com/budzianowski/multiwoz) 
and the [Schema-Guided Dialogue data-set](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) 
for which the preprocessed datasets we use can be found in the data folder.

## Requirements
Install requirements using:
```bash
python3 -m pip install -r requirements.txt
```


## Training

Train a model with a config_name given in finetuning/training/finetuning_config_list.py using:
```
cd finetuning
python -m training.autoregressive_finetuning --config_name ${config_name}
```

## Inference

Do inference with a config_name given in LLM_experiments/configs.py using:
```
cd experiments
python -u TOD_ontology_inference.py --config_name ${config_name} 
```

## Evaluation

Evaluate with a config_name given in LLM_experiments/configs.py using:
```
cd experiments
python -u TOD_ontology_evaluation.py --config_name ${config_name} 
```

For combining the predictions of different approaches, e.g. of the different one relation only prompts, use a list of config names as input:
```
cd LLM_experiments
python -u TOD_ontology_evaluation.py --config_name_list ${config_names[@]}
```

For CoT-decoding set the aggregation strategy, number of branches used and the threshold for non highest disparity branch strategies using the following flags:
```
--cot_aggregation_strategy ${aggregation_strategy} 
--cot_disparity_threshold ${disparity_threshold} 
--k_branches_to_use ${k_branches_to_use} 
--cot_disparity_threshold ${disparity_threshold} 
```


## Citation

```
@inproceedings{vukovic-etal-2024-dialogue,
    title = "Dialogue Ontology Relation Extraction via Constrained Chain-of-Thought Decoding",
    author = "Vukovic, Renato  and
      Arps, David  and
      van Niekerk, Carel  and
      Ruppik, Benjamin Matthias  and
      Lin, Hsien-chin  and
      Heck, Michael  and
      Gasic, Milica",
    editor = "Kawahara, Tatsuya  and
      Demberg, Vera  and
      Ultes, Stefan  and
      Inoue, Koji  and
      Mehri, Shikib  and
      Howcroft, David  and
      Komatani, Kazunori",
    booktitle = "Proceedings of the 25th Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = sep,
    year = "2024",
    address = "Kyoto, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.sigdial-1.33",
    doi = "10.18653/v1/2024.sigdial-1.33",
    pages = "370--384",
}

```

## License
This project is licensed under the Apache License, Version 2.0 (the "License");
you may not use the files except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0



