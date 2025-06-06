# Enhancing PyKeen with Multiple Negative Sampling Solutions for Knowledge Graph Embedding Models
[![DOI](https://zenodo.org/badge/971351696.svg)](https://doi.org/10.5281/zenodo.15413074)
[![Docs](https://img.shields.io/badge/docs-online-success.svg)](https://ivandiliso.github.io/refactor-negative-sampler/)

## Documentation

In depth documentation and tutorials are available in the apposite GitHub Page https://ivandiliso.github.io/refactor-negative-sampler/

## Folder Structure

```
📁 data             -> Dataset used during traning, validation and testing
    📁 YAGO4-20     
    📁 FB15K        
    📁 WN18        
    📁 DB50K        
📁 doc              -> Documentations and logs
📁 cached           -> Cached Negative Sampler subsets for faster computation
📁 model
    📁 embedding    -> Embedding models checkpoints
    📁 sampling     -> Checkpoints for models used in dynamic sampling
📁 experiments      -> Experiments results after HPO pipeline
📁 script           -> Single execution files, settings etc     
📁 src              -> Source code
    📁 extension    -> Extensions of PyKeen classes for negative sampling
    📁 utils        -> Utility files, libraries, logging
    📁 notebooks    -> Testing, single exectuion and code evaluation notebooks
    📁 temp         -> Temporary files 
```

## Dataset Stucture

Each dataset is provided with the following folder structure

```
📁 dataset_name
    📁 mapping      
        📄 entity_to_id.json    -> Dictionary mapping entity names (string) to IDs (integer)
        📄 relation_to_id.json  -> Dictionary mapping relation names (string) to IDs (integer)
    📁 metadata
        📄 entity_classes.json          -> Dictionary mapping entity names (string) to classes (list of strings)
        📄 relation_domain_range.json   -> Dictionary mapping relation names (string) to domain  and range classes (string)
    📁 owl          -> Additional schema-level information in OWL format
    📄 train.txt    -> Training Split Triples in TSV format (using string names)
    📄 test.txt     -> Testing Split Triples in TSV format (using string names)
    📄 valid.txt    -> Validation Split Triples in TSV format (using string names)
```

## Extension Structure

```
📁 src/extension
    📄 constants.py    -> Constant variables used across the whole library
    📄 dataset.py      -> Implementation of OnMemoryDataset
    📄 filtering.py    -> Implementation of NullPytonSetFilterer
    📄 sampling.py     -> Implementation of SubsetNegativeSampler and all the specific sampling strategies
    📄 utils.py        -> Utility functions

```

## Instructions

A fully detailed tutorial is provided in `src/tutorial.ipynb`.Detailed instruction are available in https://ivandiliso.github.io/refactor-negative-sampler/

1. Unzip the datasets files
2. Install the dependencies found in the requirements.txt file
3. Manually run the example python files, or use one of the provided scripts in the scripts folder

The library is completely integrated in the PyKEEN ecosystem, if you need a boostrap on using the library on the fly, just 
follow this guide, three example file can be used to run in order a hpo pipeline, a normal pipeline, and the negative sampler
evaluation. If you want to directly run an example configuration, you can find 

#### hpo_pipeline.py
Run a hyperparameter optimization pipeline using the chosen model, can be run using CLI arguments:

```bash
python src/hpo_pipeline.py 
    --dataset dataset_name 
    --model model_name 
    --sampler sampler_name 
    --negatives number_negatives
```

#### pipeline.py
Run a pipeline using the chosen model and static defined parameters, can be run using CLI arguments:

```bash
python src/hpo_pipeline.py 
    --dataset dataset_name 
    --model model_name 
    --sampler sampler_name 
    --negatives number_negatives 
    --l2 regularizer_weight 
    --lr learning_rate 
    --margin loss_margin
```

#### negative_evaluation.py

Example code on how to compute the negative sampler statistic for a specific dataset. This file also contains use examples 
of Dynamic Sampling using a TransE pretained model on YAGO4-20, it provides pre-written prediciton function that work 
with the provided model.




