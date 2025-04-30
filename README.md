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
- ![](https://geps.dev/progress/10) NearestNeighbour 
- ![](https://geps.dev/progress/0) NearMiss (Called Adversarial in our own work)


### Training 

- ![](https://geps.dev/progress/0) Hyperparameter Optimization
- ![](https://geps.dev/progress/0) Training Loop
- ![](https://geps.dev/progress/0) Evaluation Loop




## Dataset, Preprocessing and Reasoning

### YAGO4-20

The data is taken from [KelpiePP](https://github.com/rbarile17/kelpiePP) (Barile et al.) additional information on domain and range proprieties is taken from YAGO based dataset from [Sem@K](https://github.com/nicolas-hbt/benchmark-sematk) (Hubert et al.)

- The class membership is taken from the `reasoned/entities.csv` file. 
- Missing data are reported as `"None"`


