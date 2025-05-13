# [PyKeen Refactor] Empirical Study of Negative Sampling

## Folder Structure

```
📁 data             -> Dataset used during traning, validation and testing
    📁 YAGO4-20
    📁 FB15K
    📁 YAGO4-20
    📁 YAGO4-20 
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

## Todos and Progress

### Data

Progress relative to **YAGO4-20** dataset

- ![](https://geps.dev/progress/100) Custom Dataset
- ![](https://geps.dev/progress/100) Domain Range Metadata
- ![](https://geps.dev/progress/100) Class Memebership Metadata
- ![](https://geps.dev/progress/0) Reasoned Metadata

### Negative Samplers

- ![](https://geps.dev/progress/100) Random (Already Available in PyKeen)
- ![](https://geps.dev/progress/100) Bernoulli (Already Available in PyKeen)
- ![](https://geps.dev/progress/90) Corrupt (Called PositiveInstance in our own work)
- ![](https://geps.dev/progress/90) Typed
- ![](https://geps.dev/progress/90) Relational
- ![](https://geps.dev/progress/90) NearestNeighbour 
- ![](https://geps.dev/progress/90) NearMiss (Called Adversarial in our own work)

#### Relational 

Uses temporary cached subset file for more optimized execution of negative samples search. The main corrupting function uses LRU caching for optimizing the 
retrieval of negatives pools.

#### NearestNeighbour and NearMiss

Missing information in the original paper on the hyperparameters of RESCAL model, I think, reading the paper, they used  the whole dataset for pretraining the model, using
the validation to perform hyperparameter optimization. They only information they provide is:
- On FB15K they use RESCAL pretrained with Typed Negative Sampler with 100 negatives for positive
- On WN18 they use RESCAL pretrained with Corrupt Negative Sampler with 100 negatives for positive

Need to take a decision on how to move on on the parameters, the code will be written so that the NegativeSampler Function exepect to receive a pretrained 
embedding model class (from PyKeen)



### Training 

- ![](https://geps.dev/progress/100) Hyperparameter Optimization
- ![](https://geps.dev/progress/100) Training Loop
- ![](https://geps.dev/progress/0) Evaluation on Testing Loop




## Dataset, Preprocessing and Reasoning

### YAGO4-20

The data is taken from [KelpiePP](https://github.com/rbarile17/kelpiePP) (Barile et al.) additional information on domain and range proprieties is taken from YAGO based dataset from [Sem@K](https://github.com/nicolas-hbt/benchmark-sematk) (Hubert et al.)

- The class membership is taken from the `reasoned/entities.csv` file. 
- Missing data are reported as `"None"`


