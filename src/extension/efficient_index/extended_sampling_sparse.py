import math
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from sys import getsizeof
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast

import torch
import tqdm as tqdm
from collections.abc import Callable
from extension.utils import SimpleLogger
from pykeen.sampling import NegativeSampler
from pykeen.triples import CoreTriplesFactory
from pykeen.typing import BoolTensor, EntityMapping, LongTensor, MappedTriples, Target
from torch.utils.data import Dataset
from functools import lru_cache
from pykeen.models import TransE, RESCAL, ERModel
from scipy.spatial import KDTree
import numpy as np
from extension.constants import (
    HEAD,
    TAIL,
    REL,
    TARGET_TO_INDEX,
    INDEX_TO_TARGET,
    SWAP_TARGET,
    SWAP_TARGET_ID,
)
from torch.utils.data import DataLoader


class SubSetNegativeSampler(NegativeSampler, ABC):
    """Abstract Class Handling static negative sampling, requires implementing
    a method able to calculate the correct subset pool of negative for each
    entity in the triples set
    """

    def __init__(
        self,
        *,
        mapped_triples,
        num_entities=None,
        num_relations=None,
        num_negs_per_pos=None,
        filtered=False,
        filterer=None,
        filterer_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            mapped_triples=mapped_triples,
            num_entities=num_entities,
            num_relations=num_relations,
            num_negs_per_pos=num_negs_per_pos,
            filtered=filtered,
            filterer=filterer,
            filterer_kwargs=filterer_kwargs,
        )

        self.mapped_triples = mapped_triples
        self.subset = self._generate_subset(mapped_triples, **kwargs)

    @abstractmethod
    def _generate_subset(self, mapped_triples: MappedTriples, **kwargs):
        """Generated the supporting subset to corrupt the triple

        Args:
            mapped_triples (MappedTriples): Base triples to generate the subset
        """
        raise NotImplementedError

    def corrupt_batch(self, positive_batch: MappedTriples) -> MappedTriples:
        """Subset batch corruptor. Uniform corruption between head and tail.
        Corrupts each triple using the generated subset

        Args:
            positive_batch (MappedTriples): Batch of positive triples

        Returns:
            MappedTriples: Batch of negative triples of size (positive_size * num_neg_per_pos, 3)
        """

        batch_shape = positive_batch.shape[:-1]

        # Clone Negative for corruption (cloned the number of negative per positive )
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(
            self.num_negs_per_pos, dim=0
        )

        num_negative_batch = negative_batch.size(0)

        targets = torch.full(size=(num_negative_batch,), fill_value=False)
        targets[torch.randperm(num_negative_batch)[: num_negative_batch // 2]] = True

        head_negative_ids = self._choose_from_pools(negative_batch[targets], "head")
        tail_negative_ids = self._choose_from_pools(negative_batch[~targets], "tail")

        negative_batch[targets, HEAD] = head_negative_ids
        negative_batch[~targets, TAIL] = tail_negative_ids

        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)


    @abstractmethod
    def _choose_from_pools(self, triple, target, target_size) -> torch.tensor:
        raise NotImplementedError

   

class CorruptNegativeSampler(SubSetNegativeSampler):
    """Negative sampler from "Richard Socher, Danqi Chen, Christopher D Manning,
    and Andrew Ng. 2013. Reasoning With Neural Tensor Networks for Knowledge
    Base Completion." Corrupt head and tails based on the subset of entities seen
    as head or tail of the specific relation

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _generate_subset(self, mapped_triples):
        subset = dict()

        subset_head = torch.tensor([], dtype=torch.int16)
        subset_tail = torch.tensor([], dtype=torch.int16)
        lens = torch.zeros(size=(self.num_relations, 2), dtype=torch.int16)

        for r in range(self.num_relations):
            mask = mapped_triples[mapped_triples[:, REL] == r]

            head_negatives = torch.unique(mask[:, HEAD])
            tail_negatives = torch.unique(mask[:, TAIL])
         
            subset_head = torch.cat([subset_head, head_negatives])
            subset_tail = torch.cat([subset_tail, tail_negatives])

            lens[r, 0] = len(head_negatives)
            lens[r, 1] = len(tail_negatives)

        subset["head"] = subset_head
        subset["tail"] = subset_tail
        subset["lens"] = lens
        subset["lens_cumsum"] = torch.cumsum(lens, dim=0) - lens

        return subset


    def _choose_from_pools(self, negative_batch, target) -> torch.tensor:
        
        rels = negative_batch[:, REL]
        target_id = 0 if target == "head" else 1

        lens = self.subset["lens"][rels, target_id]
        
        negative_ids = (
            torch.tensor(np.random.randint(0, lens), dtype=torch.int16)
            + self.subset["lens_cumsum"][rels, target_id]
        )

        negative_ids = self.subset[target][negative_ids]
            
        return negative_ids

    
class TypedNegativeSampler(SubSetNegativeSampler):
    """Type-Constrained Negative sampler from "Krompaß, D., Baier, S., Tresp, V.: Type-constrained representation
    learning in knowledge graphs. In: The Semantic Web-ISWC 2015". Produces the subsed of available negatives using only
    entities that appear as domain (for corruptiong head) and range (for corrupting tails) of a triple relation.
    Need additional information on triples, a dict with domain and range for each relation (mapped to IDS) and a
    dictionary of class memebership for each entity (mapped to IDS)
    """

    def __init__(self, *, relation_domain_range_dict, entity_classes_dict, **kwargs):

        object.__setattr__(self, "entity_classes", entity_classes_dict)
        object.__setattr__(self, "relation_domain_range", relation_domain_range_dict)

        super().__init__(**kwargs)

        self.mapping = {"head": "domain", "tail": "range"}


    def _choose_from_pools(self, negative_batch, target):


        rels = negative_batch[:, REL]
        target_id = 0 if target == "head" else 1

        relation_classes = self.subset["domain_range"][rels, target_id]
        
        lens = self.subset["lens"][relation_classes]
        lens_cumsum = self.subset["lens_cumsum"][relation_classes]
        available_pools = (lens != 0)

        
        negative_ids = (
            torch.tensor(np.random.randint(0, lens[available_pools]), dtype=int)
            + lens_cumsum[available_pools]
        )


        out = torch.full(size=(len(negative_batch),), fill_value=-1, dtype=int)
        out[available_pools] = self.subset["classes"][negative_ids]


        return out



    def _generate_subset(self, mapped_triples, **kwargs):

        
        domain_range = np.zeros((self.num_relations, 2), dtype=object)
        for rel in range(self.num_relations):
            domain_range[rel][0] = self.relation_domain_range[rel]["domain"]
            domain_range[rel][1] = self.relation_domain_range[rel]["range"]

        unique_names = np.unique(domain_range)
        classes_mapping = {k:v for k,v in zip(unique_names.tolist(), np.arange(len(unique_names)).tolist())}
        classes_dict = {i:[] for i in np.arange(len(unique_names)).tolist()}

        for rel in range(self.num_relations):
            domain_range[rel][0] = classes_mapping[domain_range[rel][0]]
            domain_range[rel][1] = classes_mapping[domain_range[rel][1]]

        for ent in range(self.num_entities):
            classes = self.entity_classes[ent]
            for c in classes:
                if c in unique_names:
                    classes_dict[classes_mapping[c]].append(ent)

        subset_corrupt = torch.tensor([], dtype=int)
        lens = torch.tensor([], dtype=torch.int16)
        

        for class_id in range(len(unique_names)):
            subset_corrupt = torch.cat([subset_corrupt, torch.tensor(classes_dict[class_id], dtype=int)])
            lens = torch.cat([lens, torch.tensor([len(classes_dict[class_id])])])

        subset = dict()

        subset["classes"] = subset_corrupt
        subset["domain_range"] = torch.tensor(domain_range.astype(int), dtype=int)
        subset["lens"] = lens
        subset["lens_cumsum"] = torch.cumsum(lens, dim=0) - lens

        return subset
    

class CorruptNegativeSampler(SubSetNegativeSampler):
    """Negative sampler from "Richard Socher, Danqi Chen, Christopher D Manning,
    and Andrew Ng. 2013. Reasoning With Neural Tensor Networks for Knowledge
    Base Completion." Corrupt head and tails based on the subset of entities seen
    as head or tail of the specific relation

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _generate_subset(self, mapped_triples):
        subset = dict()

        subset_head = torch.tensor([], dtype=torch.int16)
        subset_tail = torch.tensor([], dtype=torch.int16)
        lens = torch.zeros(size=(self.num_relations, 2), dtype=torch.int16)

        for r in range(self.num_relations):
            mask = mapped_triples[mapped_triples[:, REL] == r]

            head_negatives = torch.unique(mask[:, HEAD])
            tail_negatives = torch.unique(mask[:, TAIL])
         
            subset_head = torch.cat([subset_head, head_negatives])
            subset_tail = torch.cat([subset_tail, tail_negatives])

            lens[r, 0] = len(head_negatives)
            lens[r, 1] = len(tail_negatives)

        subset["head"] = subset_head
        subset["tail"] = subset_tail
        subset["lens"] = lens
        subset["lens_cumsum"] = torch.cumsum(lens, dim=0) - lens

        return subset


    def _choose_from_pools(self, negative_batch, target) -> torch.tensor:
        
        rels = negative_batch[:, REL]
        target_id = 0 if target == "head" else 1

        lens = self.subset["lens"][rels, target_id]
        
        negative_ids = (
            torch.tensor(np.random.randint(0, lens), dtype=torch.int16)
            + self.subset["lens_cumsum"][rels, target_id]
        )

        negative_ids = self.subset[target][negative_ids]
            
        return negative_ids
    

class RelationalNegativeSampler(SubSetNegativeSampler):
    """Relational constrained Negative Sampler from "Kotnis, B., Nastase, V.: Analysis of the impact of negative
    sampling on link prediction in knowledge graphs".
    If follows the assuption that each head,tail pair are connected
    by only one relation, so, fixed the head (tail) we take all the tail (head) elements that appear in the triple with
    a relation different from the original one.
    """

    def __init__(
        self,
        *args,
        local_file=None,
        **kwargs,
    ):

        object.__setattr__(self, "local_file", Path(local_file))

        super().__init__(
            *args,
            **kwargs,
        )

    def _generate_subset(self, mapped_triples, **kwargs):

        subset = dict()

        if self.local_file.is_file():
            print("[RelationalNegativeSampler] Loading Pre-Computed Subset")
            with open(self.local_file, "rb") as f:
                subset = torch.load(f, weights_only=False)

        else:
            print("[RelationalNegativeSampler] Generating Subset")
            for entity_id in tqdm.tqdm(range(self.num_entities)):

                entity_dict = {
                    "head": mapped_triples[mapped_triples[:, HEAD] == entity_id],
                    "tail": mapped_triples[mapped_triples[:, TAIL] == entity_id],
                }

                lens = torch.zeros(size=(2,self.num_entities, self.num_relations))
                head_subset = torch.tensor([], dtype=int)
                tail_subset = torch.tensor([], dtype=int)

                for e in 
                

                subset[entity_id] = entity_dict

            with open(self.local_file, "wb") as f:
                torch.save(subset, f)

            print(f"[RelationalNegativeSampler] Saved Subset as {self.local_file}")

        return subset

    def _strategy_negative_pool(self, h, r, t, target):

        # If corrupting HEAD we take the TAIL entity to use as a pivot for the subset
        # If corrupting TAIL we take the HEAD entity to use as a pivot for the subset d

        print(f"corrupt {h} {r} {t} on {target}")

        match target:
            case "head":
                negative_pool = self._get_subset(t, r, target)
            case "tail":
                negative_pool = self._get_subset(h, r, target)

        negative_pool = negative_pool if len(negative_pool) > 0 else torch.tensor([-1])

        return negative_pool

    @lru_cache(maxsize=1024, typed=None)
    def _get_subset(self, entity, rel, target):

        pivot_entity_position = SWAP_TARGET[target]
        subset = self.subset[entity][pivot_entity_position]
        subset = subset[subset[:, REL] != rel, TARGET_TO_INDEX[target]]

        return subset
