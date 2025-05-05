# [PyKeen Refactor] Empirical Study of Negative Sampling

## Folder Structure

```
📁 data
    📁 YAGO4-20
    📁 YAGO39K   
📁 doc          
📁 script       
📁 src          
    📁 extension
    📁 utils
    📁 model
    📁 notebooks
    📁 temp
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
- ![](https://geps.dev/progress/15) NearestNeighbour 
- ![](https://geps.dev/progress/0) NearMiss (Called Adversarial in our own work)

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


**Hyperparameters FIXED for pretraining RECAL Model**
```js
embedding_dim = 200 
negative_sampler = RandomNegativeSampler + PythonSetFilterer
num_negs_per_pos = 100
optimizer = Adam
regularized = LP (2-Norm)
training_loop = sLCWA
loss = MarginRankingLoss (with mean reduction)
evaluation = RankBasedEvaluator
epochs = 100,
stopper = EarlyStopper
stopper_frequency = 5,
stopper_patience = 2,
stopper_relative_delata = 0.002
random_seed = 42
```
**Hyperparameters OPTIMIZED with GRID SEARCH (20 Trials) for pretraining RECAL Model**
```js
loss_margin = [0.5, 1.0, 2.0]
learning_rate = [1e-3, 5e-4, 1e-4]
regularizer_weight = [1e-5, 1e-4, 1e-3]
```


### Training 

- ![](https://geps.dev/progress/0) Hyperparameter Optimization
- ![](https://geps.dev/progress/0) Training Loop
- ![](https://geps.dev/progress/0) Evaluation Loop




## Dataset, Preprocessing and Reasoning

### YAGO4-20

The data is taken from [KelpiePP](https://github.com/rbarile17/kelpiePP) (Barile et al.) additional information on domain and range proprieties is taken from YAGO based dataset from [Sem@K](https://github.com/nicolas-hbt/benchmark-sematk) (Hubert et al.)

- The class membership is taken from the `reasoned/entities.csv` file. 
- Missing data are reported as `"None"`


