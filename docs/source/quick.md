# Quick Start Guide

This tutorial will guide you in a quickstarting you in using the new negative samplers in a PyKEEN pipeline. A Jupyter Notebook Version of this resouce is available in the repository main folder. This tutorial is designed as a progressive guide to help you understand and use the core features of the library. Each section builds upon the previous ones, so we recommend following them in order.

## Installing requirements

Please be sure to have the following requirements installed in your system

- Python 3.x
- pip

Regarding libraries, the extension is built entirely around PyKEEN, and does not need other external libraries.
In case an exact replica of the installed packed and version is needed, you can use the provided requirements file, installable with:

```bash
pip install -r requirements.txt
```

## Import the extension and setting paths

Let's import all the required libraries to use our extension within the PyKEEN ecosystem, here we import all the samplers, filterer and dataset classes. 

```python
from pathlib import Path

import pykeen
import torch
from pykeen.pipeline import pipeline
from pykeen.sampling import (
    BasicNegativeSampler,
    BernoulliNegativeSampler,
    negative_sampler_resolver,
)
from pykeen.sampling.filtering import filterer_resolver
from pykeen.training import SLCWATrainingLoop

from extension.dataset import OnMemoryDataset
from extension.filtering import NullPythonSetFilterer
from extension.sampling import (
    CorruptNegativeSampler,
    NearestNeighbourNegativeSampler,
    NearMissNegativeSampler,
    RelationalNegativeSampler,
    SubSetNegativeSampler,
    TypedNegativeSampler,
)
import extension.constants as const
```

## Loading a Dataset

A dataset can be loaded referencing the main folder where it is located. Using our custom dataloader the fixed entity and id mappings will be loaded, and we can choose wheter to load or not additional metadata. The data loading process handles the mapping of names to ids in the loaded metadata files. We choose YAGO4-20 in this tutorial since it is provided with additional metadata.

```python
dataset = OnMemoryDataset(
    data_path="/<custom_path_to_dataset_folder>/YAGO4-20", load_domain_range=True, load_entity_classes=True
)
```

The additional metadata are stored in two data properties:

```python
dataset.entity_id_to_classes  # Dictionary with mapping from entity ID to list of class names
dataset.relation_id_to_domain_range # Dictionary with mappings from entity ID to domain and range class
```


## Using Static Negative Samplers

In this tutorial we will use the Typed negative sampler, levaraging the additional metadata loaded from YAGO4-20, you can instantiate the negative sampler using the standard PyKEEN negative sampler interface, adding the additional arguments needed for the specific sampler, in this case we will provide the entity to classes mapping and the relation to domain and range mapping. In order to leverage fully the functionalities of this sampler, we also use the `NullPythonSetFilterer`, instantiate like any other filterer in the PyKEEN ecosystem:

```python
filterer = NullPythonSetFilterer(mapped_triples=dataset.training.mapped_triples)
sampler = TypedNegativeSampler(
        mapped_triples=dataset.training.mapped_triples,
        filtered=True,
        filterer=filterer,
        num_negs_per_pos=5,
        entity_classes_dict=dataset.entity_id_to_classes,
        relation_domain_range_dict=dataset.relation_id_to_domain_range,
        integrate=False,
    )
```

Then after instantiation the sampler can be used like any other PyKEEN negative sampler:

```python
sampler.sample(dataset.training.mapped_triples)
```

## Using Dynamic Negative Samplers

Dynamic negative samplers require some addiotional configuration before hand, first, lets train a model like `TransE` on our chosen dataset, this will be used as the auxiliary sampling model in the corruption procedure.

```python
model = pykeen.models.TransE(triples_factory=dataset.training, embedding_dim=10)

loop = SLCWATrainingLoop(
    model=model, triples_factory=dataset.training, optimizer="Adam"
)

loop.train(triples_factory=dataset.training, num_epochs=2, batch_size=256)
```

Then, lets define the custom function used to get the predicted entity vector representation, this function expect to receive the model, a batch, and the targets for corruption (list of head or tail in ID form)

```python
def sampling_model_prediction(model, hrt_batch, targets):
    out = torch.zeros(
        (hrt_batch.size(0), model.entity_representations[0]().size(1)),
        device=hrt_batch.device,
    )
    out[targets == 0] = model.entity_representations[0](
        hrt_batch[targets == 0, 2]
    ) - model.relation_representations[0](hrt_batch[targets == 0, 1])
    out[targets == 2] = model.entity_representations[0](
        hrt_batch[targets == 2, 0]
    ) + model.relation_representations[0](hrt_batch[targets == 2, 1])

    return out
```
Now we can instantiate our dynamic negative sampler providing the additional pretrained model and prediction function. Since the adversarial sampler uses a nearest neighbour algorithm internally, we can also define the K parameter with the `num_query_results` argument.

```python
sampler = NearMissNegativeSampler(
    mapped_triples=dataset.training.mapped_triples,
    prediction_function=sampling_model_prediction,
    sampling_model=model,
    num_negs_per_pos=5,
    num_query_results=10,
    filtered=True,
    filterer=filterer,
)
```

Then after instantiation the sampler can be used like any other PyKEEN negative sampler:

```python
sampler.sample(dataset.training.mapped_triples)
```

## Using Negative Samplers in PyKEEN Pipelines

Using our custom samplers in PyKEEN pipelines is even easier, we just need to provied them as input to the pipeline, or even easier, register them in the PyKEEN namespace, allowing to refer to them only using their name, like other available negative samplers in the library.

```python
negative_sampler_resolver.register(element=CorruptNegativeSampler)
filterer_resolver.register(element=NullPythonSetFilterer)

pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    negative_sampler="corrupt",
    negative_sampler_kwargs=dict(filtered=True, filterer="nullpythonset"),
    training_loop="sLCWA",
)
```

## Writing you own custom Negative Samplers

One core contribution in our proposed extension is abstraction on negative sampling computation, providing an abstract class for static samplers that hides all the need computation, allowing developers to define their own samplers implementing only the core logic of the negative sampling process. In this tutorial we will show a new simple negative sampler implemented using our proposed abstaction. 

This sampler, named `TutorialSampler` defined a negative pool for a triple, as the least occuring entities for its relation as head (if corrupting the head entity) or tail (if corrupting the tail entity). So the negative pool for each triple will depend solely on its relation. Let's see how to implement this strategy:

```python
class TutorialSampler(SubSetNegativeSampler):
    def __init__(self, *args, top_k=100, **kwargs):
        # We define this variable before super, to it can be available in the subset generation
        object.__setattr__(self, "top_k", top_k)
        super().__init__(*args, **kwargs)

    # Precompute the entity set for head and tail for each relation
    def generate_subset(self, mapped_triples, **kwargs):
        subset = dict()

        for r in range(self.num_relations):

            subset[r] = {0: None, 2: None}

            for target in [const.HEAD, const.TAIL]:
                data, counts = torch.unique(
                    self.mapped_triples[self.mapped_triples[:, const.REL] == r, target],
                    return_counts=True,
                )
                ordered_data = data[torch.sort(counts, descending=False)[1]][
                    : self.top_k
                ]

                subset[r][target] = ordered_data

        return subset

    # Now lets define the negative pool for each triple
    def strategy_negative_pool(self, h, r, t, target):
        return self.subset[r][const.TARGET_TO_INDEX[target]]
```

As you can see implementing a new negative sampler is extremely easy, requiring only the core logic for the subset, and the core logic for extracting the negative pool. Our abstraction hides all the computation on batching, interleaving for multiple negative per positive, filtering, and random selection from the negative pool. If necessary, a developer could also override the `choose_from_pool` function, providing specific functionality on how to sample negative form the negative pool.

After this this new negative sampler can be used like any other one.

```python
sampler = TutorialSampler(
    mapped_triples=dataset.training.mapped_triples,
    top_k=5,
    num_negs_per_pos=100,
)
```

## Integration of random negatives 

Using the previous example as negative sampler, given, for example, the top 50 least occuring entities for each relation, each negative pool will have exaclty 50 different entities. If we set a number of `num_negs_per_pos` (number of negative triple to generate for each positive triple) higher than this number, like 100, we will have a lot of dupliate negative in our negative pools. This situation, in general, arises when the negative pool strategy is not able to produce enough negatives, in this case, we can set the `integrate` argument to `True`, and our implementation will automatically detect when a specific negative pool doesn't reach the desired number of negatives and supplement it with randomly sampled entities, this can be done as simple as:

```python
sampler = TutorialSampler(
    mapped_triples=dataset.training.mapped_triples,
    top_k=5,
    num_negs_per_pos=100,
    integrate=True
)
```

## Generating Negative Sampler Statistics

Additionaly you can compute the dataset statistics directly using our provided functions, this can take some time, since this computation as to be computed for each `<h,r,*>` and `<*,r,t>` combination, in our notebook tutorial we test it on a subset of the training triples, in order to speed up the computation.

```python
sampler = TutorialSampler(
    mapped_triples=dataset.training.mapped_triples, top_k=5, num_negs_per_pos=5
)

sampler.average_pool_size(dataset.training.mapped_triples)
```

The function produces the average number of entities in each negative pool (checking if there are false negative), and then in order, the number of triples that have less than 0, 2, 10, 40, 100 entities in their negative pool.