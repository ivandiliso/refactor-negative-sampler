import math
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from sys import getsizeof
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast

import torch
import tqdm as tqdm
from collections.abc import Callable
from test_utils import SimpleLogger
from pykeen.sampling import NegativeSampler
from pykeen.triples import CoreTriplesFactory
from pykeen.typing import BoolTensor, EntityMapping, LongTensor, MappedTriples, Target
from torch.utils.data import Dataset
from functools import lru_cache
from pykeen.models import TransE, RESCAL, ERModel
from scipy.spatial import KDTree
import numpy as np
from extended_constants import (
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

        for i in tqdm.tqdm(range(total_num_negatives)):
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

    @lru_cache(maxsize=1024, typed=False)
    def _triple_full_negative_pool(self, entity, rel, target):
        """Returns all the real negatives given an entity, a relation, and the taget for corruption.
        if target == "head" returns the full availabile negative entities for (*, rel, entity)
        if target == "tail" returns the full availabile negative entities for (entity, rel, *)
        """

        entity_target_id = TARGET_TO_INDEX[SWAP_TARGET[target]]
        target_id = TARGET_TO_INDEX[target]

        positive_pool = self.mapped_triples[
            self.mapped_triples[:, entity_target_id] == entity
        ]

        positive_pool = positive_pool[positive_pool[:, REL] == rel][:, target_id]

        negative_pool = torch.arange(0, self.num_entities, step=1)

        mask = torch.full_like(negative_pool, fill_value=True, dtype=torch.bool)
        mask[positive_pool] = False

        negative_pool = negative_pool[mask]

        return negative_pool


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
        relation_domain_range_dict=None,
        entity_classes_dict=None,
        **kwargs
    ):
        
        object.__setattr__(self, "entity_classes", entity_classes_dict)
        object.__setattr__(self, "relation_domain_range", relation_domain_range_dict)

        super().__init__(
            **kwargs
        )

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

        entity_classes = self.entity_classes
        relation_domain_range = self.relation_domain_range

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

        # Get the subset of element with
        negative_pool = self._get_subset(pivot_entity, rel, target)

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
        *args,
        sampling_model: ERModel = None,
        **kwargs,
    ):
        object.__setattr__(self, "sampling_model", sampling_model)

        super().__init__(
            *args,
            **kwargs,
        )

    def _generate_subset(self, mapped_triples, **kwargs):

        subset = dict()
        subset["positive_triples"] = mapped_triples
        subset["kdtree"] = KDTree(
            self.sampling_model.entity_representations[0]().cpu().detach().numpy(),
            leafsize=self.num_entities,
        )
        subset["entity_representations"] = (
            self.sampling_model.entity_representations[0]().cpu().detach().numpy()
        )

        return subset

    def _corrupt_triple(self, triple, target):

        negative_pool = self._get_subset(
            entity=int(triple[TARGET_TO_INDEX[SWAP_TARGET[target]]]),
            rel=int(triple[REL]),
            target=target,
        )

        search_entity_id = int(triple[TARGET_TO_INDEX[target]])

        k_nearest_entities = self._query_kdtree(search_entity_id)

        chosen_negatives = k_nearest_entities[
            np.isin(k_nearest_entities, negative_pool)
        ][: self.num_negs_per_pos]

        triple[TARGET_TO_INDEX[target]] = self._choose_from_pool(chosen_negatives)

    @lru_cache(maxsize=1024, typed=False)
    def _query_kdtree(self, entity_id):
        """TODO Optimize

        Args:
            entity_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        search_entity = self.subset["entity_representations"][entity_id]
        _, indices = self.subset["kdtree"].query(
            search_entity, k=self.num_negs_per_pos * 10
        )
        return indices

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

        negative_pool = negative_pool[mask]

        return negative_pool.numpy()


class NearMissNegativeSampler(SubSetNegativeSampler):
    """Auxiliary Model based Negative Sampler from "Kotnis, B., & Nastase, V. (2017). Analysis of the impact of
    negative sampling on link prediction in knowledge graphs. arXiv preprint arXiv:1708.06816." Uses a pretrained model
    on the same dataset to produce harder negatives. Given the predicted entity embedding for each triple, a Nearest
    Neighbour algorithm is used to produce negatives that could be predicted as positive but in reality are negatives.
    """

    def __init__(
        self,
        *,
        sampling_model: ERModel = None,
        prediction_function: Callable[
            [ERModel, MappedTriples, torch.tensor], torch.tensor
        ] = None,
        num_query_results: int = None,
        **kwargs,
    ):
        """Inizialite the NearMissNegativeSampler

        Args:
            sampling_model (ERModel, optional): Auxiliary model used to predict the target embedding. Defaults to None.
            prediction_function (Callable[ [ERModel, MappedTriples, torch.tensor], torch.tensor ], optional): Function that produces the predicted entity in tensor format. Defaults to None.
            num_query_results (int, optional): The K to be used in K Nearest Neighbours search. Defaults to None.
        """

        object.__setattr__(self, "sampling_model", sampling_model)
        object.__setattr__(self, "prediction_function", prediction_function)
        object.__setattr__(self, "num_query_results", num_query_results)

        super().__init__(
            **kwargs,
        )

    def _generate_subset(self, mapped_triples: MappedTriples, **kwargs) -> Dict:
        """Generate the auxiliary subset to aid in triple corruption. Specifically
        it creates the BallTree structure with the filtering triples (in Numpy format)

        Args:
            mapped_triples (MappedTriples): Triples used for filtering

        Returns:
            Dict: Dictionary with auxiliary data
        """

        subset = dict()
        subset["kdtree"] = KDTree(
            self.sampling_model.entity_representations[0]().cpu().detach().numpy(),
            leafsize=self.num_entities,
        )

        return subset

    def corrupt_batch(self, positive_batch: MappedTriples) -> MappedTriples:
        """Subset batch corruptor. Uniform corruption between head and tail.
        Corrupts each triple using the generated subset

        Args:
            positive_batch (MappedTriples): Batch of positive triples

        Returns:
            MappedTriples: Batch of negative triples of size (positive_size * num_neg_per_pos, 3)
        """

        log = SimpleLogger()
        batch_shape = positive_batch.shape[:-1]

        # Entity embeddings from pretrained model
        # Head prediction and tail predicions are tensor data
        ################################################################################

        log.start(
            f"[NS {self._get_name()}] Calculating HEAD prediction with {self.sampling_model._get_name()} pretrained model"
        )
        head_prediction = (
            self.prediction_function(
                self.sampling_model,
                positive_batch,
                torch.full((positive_batch.size(0),), fill_value=0),
            )
            .cpu()
            .detach()
            .numpy()
        )
        log.end()

        log.start(
            f"[NS {self._get_name()}] Calculating TAIL prediction with {self.sampling_model._get_name()} pretrained model"
        )
        tail_prediction = (
            self.prediction_function(
                self.sampling_model,
                positive_batch,
                torch.full((positive_batch.size(0),), fill_value=2),
            )
            .cpu()
            .detach()
            .numpy()
        )
        log.end()

        # Head and Tail K-Nearest Neighbours from BallTree Query
        # The head_query_negative_pool and tail_query_negative_pool
        # contain the IDs of the entities
        ################################################################################

        log.start(f"[NS {self._get_name()}] Querying KDTREE for HEAD predictions")
        _, head_query_negative_pool = self.subset["kdtree"].query(
            head_prediction, k=self.num_query_results
        )
        log.end()

        log.start(f"[NS {self._get_name()}] Querying KDTREE for TAIL predictions")
        _, tail_query_negative_pool = self.subset["kdtree"].query(
            tail_prediction, k=self.num_query_results
        )
        log.end()

        # Triples Corruption
        ################################################################################

        # Clone Negative for corruption (cloned the number of negative per positive )
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(
            self.num_negs_per_pos, dim=0
        )

        # Create tarket mask
        target = torch.full((negative_batch.size(0),), fill_value=2)
        target[: int(negative_batch.size(0) * 0.5)] = 0
        target = target[torch.randperm(target.size(0))]

        total_num_negatives = negative_batch.shape[0]

        # Here we do i // self.num_negs_per_pos so we can get the ID of the original
        # non interleave triple (since triples are repeated for the nuber of negatives required)

        for i in tqdm.tqdm(range(total_num_negatives)):
            if target[i] == 0:  # HEAD
                self._corrupt_triple(
                    negative_batch[i],
                    INDEX_TO_TARGET[int(target[i])],
                    head_query_negative_pool[i // self.num_negs_per_pos],
                )
            elif target[i] == 2:  # TAIL
                self._corrupt_triple(
                    negative_batch[i],
                    INDEX_TO_TARGET[int(target[i])],
                    tail_query_negative_pool[i // self.num_negs_per_pos],
                )

        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)

    def _corrupt_triple(self, triple, target, query_pool):

        # We get the REAL negatives for the triple
        
        negative_pool = self._triple_full_negative_pool(
            entity=int(triple[TARGET_TO_INDEX[SWAP_TARGET[target]]]),
            rel=int(triple[REL]),
            target=target,
        )
        
        # We then take the elements from the BallTree query results that are real negatives
        
        chosen_negatives = query_pool[
            torch.isin(torch.tensor(query_pool), negative_pool)
        ]
    
        chosen_negatives = query_pool

        if len(chosen_negatives) > 0:
            triple[TARGET_TO_INDEX[target]] = self._choose_from_pool(chosen_negatives)
        else:
            self._dummy_corrupt_triple(triple)





