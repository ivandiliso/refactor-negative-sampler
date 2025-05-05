import math
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from sys import getsizeof
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast

import torch
import tqdm as tqdm
from pykeen.constants import TARGET_TO_INDEX
from pykeen.sampling import NegativeSampler
from pykeen.triples import CoreTriplesFactory
from pykeen.typing import BoolTensor, EntityMapping, LongTensor, MappedTriples, Target
from torch.utils.data import Dataset
from functools import lru_cache
from pykeen.models import TransE, RESCAL, ERModel
from scipy.spatial import KDTree


INDEX_TO_TARGET = {v: k for k, v in TARGET_TO_INDEX.items()}
SWAP_TARGET = {"head": "tail", "tail": "head"}
HEAD = 0
REL = 1
TAIL = 2


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

        self.subset = self._generate_subset(mapped_triples, **kwargs)

    @abstractmethod
    def _corrupt_triple(self, triple: torch.LongTensor, target: Target):
        """Corrupt the selected triple using the generated subset. The triple
        has to be corrupted in-place

        Args:
            triple (torch.LongTensor): Triple in (h,r,t) format
            target (Target | int): Corrupt head ("head" or 0) or tail ("tail" or 2)
        """
        raise NotImplementedError

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

        # Create tarket mask
        target = torch.full((negative_batch.size(0),), fill_value=2)
        target[: int(negative_batch.size(0) * 0.5)] = 0
        target = target[torch.randperm(target.size(0))]

        total_num_negatives = negative_batch.shape[0]

        for i in range(total_num_negatives):
            self._corrupt_triple(negative_batch[i], INDEX_TO_TARGET[int(target[i])])

        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)

    def _dummy_corrupt_triple(self, triple: torch.LongTensor):
        """Trasform triple in place to a dummy corrupted triple [-1, -1, -1]

        Args:
            triple (torch.LongTensor): Triple in (h,r,t) format
        """
        triple[:] = torch.tensor([-1, -1, -1], device=triple.device, dtype=triple.dtype)

    def _choose_from_pool(self, pool: torch.tensor) -> torch.tensor:
        """Choose one element from the input pool of elements

        Args:
            pool (torch.tensor): Pool of elements to choose from

        Returns:
            torch.tensor: Chosen element in singleton tensor
        """
        return pool[torch.randperm(len(pool))[:1]]


class CorruptNegativeSampler(SubSetNegativeSampler):
    """Negative sampler from "Richard Socher, Danqi Chen, Christopher D Manning,
    and Andrew Ng. 2013. Reasoning With Neural Tensor Networks for Knowledge
    Base Completion." Corrupt head and tails based on the subset of entities seen
    as head or tail of the specific relation

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

    def _generate_subset(self, mapped_triples):
        relations = torch.unique(mapped_triples[:, REL]).tolist()
        subset = dict()
        for r in relations:
            mask = mapped_triples[mapped_triples[:, REL] == r]
            subset[r] = {
                "head": torch.unique(mask[:, HEAD]),
                "tail": torch.unique(mask[:, TAIL]),
            }
        return subset

    def _corrupt_triple(self, triple: torch.LongTensor, target: Target):
        rel = int(triple[REL])
        negative_pool = self.subset[rel][target]
        triple[TARGET_TO_INDEX[target]] = self._choose_from_pool(negative_pool)


class TypedNegativeSampler(SubSetNegativeSampler):
    """Type-Constrained Negative sampler from "Krompaß, D., Baier, S., Tresp, V.: Type-constrained representation
    learning in knowledge graphs. In: The Semantic Web-ISWC 2015". Produces the subsed of available negatives using only
    entities that appear as domain (for corruptiong head) and range (for corrupting tails) of a triple relation.
    Need additional information on triples, a dict with domain and range for each relation (mapped to IDS) and a
    dictionary of class memebership for each entity (mapped to IDS)
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
        relation_domain_range_dict=None,
        entity_classes_dict=None,
    ):
        super().__init__(
            mapped_triples=mapped_triples,
            num_entities=num_entities,
            num_relations=num_relations,
            num_negs_per_pos=num_negs_per_pos,
            filtered=filtered,
            filterer=filterer,
            filterer_kwargs=filterer_kwargs,
            relation_domain_range_dict=relation_domain_range_dict,
            entity_classes_dict=entity_classes_dict,
        )

        self.relation_domain_range = relation_domain_range_dict
        self.mapping = {"head": "domain", "tail": "range"}

    def _corrupt_triple(self, triple, target):

        rel = int(triple[REL])
        target_class = self.relation_domain_range[rel][self.mapping[target]]

        if target_class != "None":
            negative_pool = self.subset[target_class]

            if len(negative_pool) > 0:
                triple[TARGET_TO_INDEX[target]] = self._choose_from_pool(negative_pool)
            else:
                self._dummy_corrupt_triple(triple)
        else:
            self._dummy_corrupt_triple(triple)

    def _generate_subset(self, mapped_triples, **kwargs):

        entity_classes = kwargs.get("entity_classes_dict")
        relation_domain_range = kwargs.get("relation_domain_range_dict")

        classes_dict = dict()

        for _, domain_range_dict in relation_domain_range.items():
            for classes_name in domain_range_dict.values():
                if classes_name != "None":
                    classes_dict[classes_name] = []

        for entity_id, classes_names in entity_classes.items():
            for class_name in classes_names:
                if class_name in classes_dict:
                    classes_dict[class_name].append(entity_id)

        for class_name, entity_ids in classes_dict.items():
            classes_dict[class_name] = torch.unique(torch.tensor(entity_ids))

        return classes_dict


class RelationalNegativeSampler(SubSetNegativeSampler):
    """Relational constrained Negative Sampler from "Kotnis, B., Nastase, V.: Analysis of the impact of negative
    sampling on link prediction in knowledge graphs".
    If follows the assuption that each head,tail pair are connected
    by only one relation, so, fixed the head (tail) we take all the tail (head) elements that appear in the triple with
    a relation different from the original one.
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
        local_file=None,
        **kwargs,
    ):

        self.local_file = Path(local_file)

        super().__init__(
            mapped_triples=mapped_triples,
            num_entities=num_entities,
            num_relations=num_relations,
            num_negs_per_pos=num_negs_per_pos,
            filtered=filtered,
            filterer=filterer,
            filterer_kwargs=filterer_kwargs,
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

                subset[entity_id] = entity_dict

            with open(self.local_file, "wb") as f:
                torch.save(subset, f)

            print(f"[RelationalNegativeSampler] Saved Subset as {self.local_file}")

        return subset

    def _corrupt_triple(self, triple, target):

        # If corrupting HEAD we take the TAIL entity to use as a pivot for the subset
        # If corrupting TAIL we take the HEAD entity to use as a pivot for the subset
        pivot_entity = int(triple[TARGET_TO_INDEX[SWAP_TARGET[target]]])
        rel = int(triple[REL])

        print("____________________")
        print(f" Corruping {triple} on {target} so taking {pivot_entity} as pivot")

        # Get the subset of element with
        negative_pool = self._get_subset(pivot_entity, rel, target)

        print(negative_pool)

        if len(negative_pool) > 0:
            triple[TARGET_TO_INDEX[target]] = self._choose_from_pool(negative_pool)
        else:
            self._dummy_corrupt_triple(triple)

    @lru_cache(maxsize=1024, typed=False)
    def _get_subset(self, entity, rel, target):
        # All the triples that
        # oppure se appare come tail se il target è head
        pivot_entity_as_inv_t = self.subset[entity][SWAP_TARGET[target]]
        return pivot_entity_as_inv_t[
            pivot_entity_as_inv_t[:, REL] != rel, TARGET_TO_INDEX[target]
        ]


class NearestNeighbourNegativeSampler(SubSetNegativeSampler):

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
        local_file=None,
        sampling_model: ERModel = None,
        **kwargs,
    ):

        object.__setattr__(self, "local_file", Path(local_file))
        object.__setattr__(self, "sampling_model", sampling_model)

        super().__init__(
            mapped_triples=mapped_triples,
            num_entities=num_entities,
            num_relations=num_relations,
            num_negs_per_pos=num_negs_per_pos,
            filtered=filtered,
            filterer=filterer,
            filterer_kwargs=filterer_kwargs,
            **kwargs,
        )

    def _generate_subset(self, mapped_triples, **kwargs):

        subset = dict()
        subset["positive_triples"] = mapped_triples
        subset["entity_representations"] = (
            self.sampling_model.entity_representations[0]().cpu().detach().numpy()
        )

        return subset

    def _corrupt_triple(self, triple, target):

        negative_pool, kdtree = self._get_subset(
            entity=int(triple[TARGET_TO_INDEX[SWAP_TARGET[target]]]),
            rel=int(triple[REL]),
            target=target,
        )

        search_entity_id = int(triple[TARGET_TO_INDEX[target]])
        search_entity = self.subset["entity_representations"][search_entity_id]

        distances, indices = kdtree.query(search_entity, k=self.num_negs_per_pos)
        
        negative_pool = negative_pool[indices[indices != len(negative_pool)]]

        triple[TARGET_TO_INDEX[target]] = self._choose_from_pool(negative_pool)


    @lru_cache(maxsize=1024, typed=False)
    def _get_subset(self, entity, rel, target):
        """Returns all the real negative entity given a entity and the constructed kdtree, a relation and the target
        for the corruption

        """

        pivot_id = TARGET_TO_INDEX[SWAP_TARGET[target]]
        target_id = TARGET_TO_INDEX[target]

        positive_pool = self.subset["positive_triples"][
            self.subset["positive_triples"][:, pivot_id] == entity
        ]
        positive_pool = positive_pool[positive_pool[:, REL] == rel][:, target_id]

        negative_pool = torch.arange(0, self.num_entities, step=1)

        mask = torch.full_like(negative_pool, fill_value=True, dtype=torch.bool)
        mask[positive_pool] = False

        negative_pool = negative_pool[mask].numpy()

        kdtree = KDTree(
            data=self.subset["entity_representations"][negative_pool],
            leafsize=self.num_negs_per_pos,
        )

        return negative_pool, kdtree
