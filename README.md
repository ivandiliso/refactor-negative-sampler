# Enhancing PyKeen with Multiple Negative Sampling Solutions for Knowledge Graph Embedding Models

## Folder Structure

```
📁 data             -> Dataset used during traning, validation and testing
    📁 YAGO4-20     -> Download from https://drive.google.com/file/d/1XDwdvz23X4V0tmUI9ONvvBWS0W5yW3b-/view?usp=drive_link
    📁 FB15K        -> Download from https://drive.google.com/file/d/11wQRJVez7xBCGeRgf5ioAia5rOPz2Nh-/view?usp=drive_link
    📁 WN18         -> Download from https://drive.google.com/file/d/1kT5rUw1IQYG9i4Kew9cTm1QLt85tRfHN/view?usp=drive_link
    📁 DB50K        -> Download from https://drive.google.com/file/d/1El3i5J2RClkliJcA_lVt5IcZzLP_UzPJ/view?usp=drive_link
📁 doc              -> Documentations and logs
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

1. Download the datasets from the provided links, put them in the data folder in the main directory
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

Example code on how to compute the negative sampler statistic for a specific dataset







