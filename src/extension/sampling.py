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
        mapped_triples: MappedTriples,
        num_entities: int = None,
        num_relations: int = None,
        num_negs_per_pos: int = None,
        filtered: int = False,
        filterer: str = None,
        filterer_kwargs: dict = None,
        integrate: bool = False,
        **kwargs,
    ):
        """_summary_

        Args:
            mapped_triples (MappedTriples): Triples used for computation of subsets and filtering
            num_entities (int, optional): Number of entities. Defaults to None.
            num_relations (int, optional): Number of relations. Defaults to None.
            num_negs_per_pos (int, optional): Negative triples generated per positive ones. Defaults to None.
            filtered (int, optional): Where to use a filterer. Defaults to False.
            filterer (str, optional): Filterer to use is filtered is se to True. Defaults to None.
            filterer_kwargs (dict, optional): Dictionary arguments for filterer . Defaults to None.
            integrate (bool, optional): Integrate negative pool with random instances if they are below the required size. Defaults to False.
        """
        super().__init__(
            mapped_triples=mapped_triples,
            num_entities=num_entities,
            num_relations=num_relations,
            num_negs_per_pos=num_negs_per_pos,
            filtered=filtered,
            filterer=filterer,
            filterer_kwargs=filterer_kwargs,
        )

        self.integrate = integrate
        self.mapped_triples = mapped_triples
        self.subset = self._generate_subset(mapped_triples, **kwargs)

    @abstractmethod
    def _generate_subset(self, mapped_triples: MappedTriples, **kwargs):
        """Generated the supporting subset to corrupt the triple

        Args:
            mapped_triples (MappedTriples): Base triples to generate the subset
        """
        raise NotImplementedError

    @abstractmethod
    @lru_cache(maxsize=1024, typed=False)
    def _strategy_negative_pool(
        self, h: int, r: int, t: int, target: str
    ) -> torch.tensor:
        """Compute the negative pool for a triple and the target for corruption

        Args:
            h (int): Head entity ID
            r (int): Relation ID
            t (int): Tail entity ID
            target (str): "head" or "tail" corruption

        Returns:
            torch.tensor: Tensor with computed negative entities IDs
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

        for i in range(0, positive_batch.size(0)):

            batch_start = i * self.num_negs_per_pos
            batch_end = batch_start + self.num_negs_per_pos

            if self.num_negs_per_pos > 1:
                targets = torch.full(size=(self.num_negs_per_pos,), fill_value=False)
                targets[
                    torch.randperm(self.num_negs_per_pos)[: self.num_negs_per_pos // 2]
                ] = True

                num_head_negatives = targets.sum()
                num_tail_negatives = self.num_negs_per_pos - num_head_negatives

                # Head Corruption
                negative_batch[batch_start:batch_end][targets, HEAD] = (
                    self._choose_from_pools(
                        positive_batch[i], "head", num_head_negatives
                    )
                )

                # Tail Corruption
                negative_batch[batch_start:batch_end][~targets, TAIL] = (
                    self._choose_from_pools(
                        positive_batch[i], "tail", num_tail_negatives
                    )
                )

            else:
                target = np.random.choice(["head", "tail"])
                negative_pool = self._choose_from_pools(positive_batch[i], target, 1)
                negative_batch[i, TARGET_TO_INDEX[target]] = negative_pool[0]

        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)

    def _choose_from_pools(
        self, triple: torch.tensor, target: str, target_size: int
    ) -> torch.tensor:
        """Sample negatives from the negative pool

        Args:
            triple (torch.tensor): Triple for corruption
            target (str): Target of corruption
            target_size (int): Number of negatives to produce

        Returns:
            torch.tensor: Chosen negatives from the negative pool
        """
        negative_pool = self._strategy_negative_pool(
            int(triple[HEAD]), int(triple[REL]), int(triple[TAIL]), target
        )

        if self.integrate:
            available_len = len(negative_pool)
            if negative_pool[0] == -1:
                # We cannot have a negative pool
                negatives = torch.randint(0, self.num_entities, size=(target_size,))
            elif available_len < target_size:
                # The negative pool does not reach the desired size
                negatives = torch.cat(
                    [
                        negative_pool,
                        torch.randint(
                            0, self.num_entities, size=(target_size - available_len,)
                        ),
                    ],
                    dim=0,
                )
            else:
                # The negative pool exists and has enough negatives
                negatives = negative_pool[
                    torch.randint(0, len(negative_pool), size=(target_size,))
                ]
        else:
            negatives = negative_pool[
                torch.randint(0, len(negative_pool), size=(target_size,))
            ]

        return negatives

    @lru_cache(maxsize=1024, typed=False)
    def _get_positive_pool(self, e: int, r: int, target: str) -> torch.tensor:
        """Returns all the real negatives given an entity, a relation, and the taget for corruption.
        if target == "head" returns the full availabile negative entities for (*, rel, entity)
        if target == "tail" returns the full availabile negative entities for (entity, rel, *)

        Args:
            e (int): Entity ID
            r (int): Relation ID
            target (str): Target of corruption

        Returns:
            torch.tensor: Positive istances IDs
        """

        e_position = TARGET_TO_INDEX[SWAP_TARGET[target]]

        positive_pool = self.mapped_triples[self.mapped_triples[:, e_position] == e]
        positive_pool = positive_pool[
            positive_pool[:, REL] == r, TARGET_TO_INDEX[target]
        ]

        return positive_pool

    def average_pool_size(self, check_triples: MappedTriples) -> Tuple[int, dict]:
        """Compute the average pool size for every h,r combination and r,t combination

        Args:
            check_triples (MappedTriples): Triples used for computating the pool size

        Returns:
            Tuple[int, dict]: Average pool size, and dictionary with number of triples with less than X negative (from 2 to 100)
        """
        head_relation = torch.unique(check_triples[:, [HEAD, REL]], dim=0)
        tail_relation = torch.unique(check_triples[:, [TAIL, REL]], dim=0)

        return self._compute_poolsize_aggregate(head_relation, tail_relation)

    def _compute_poolsize_aggregate(
        self, head_relation: torch.tensor, tail_relation: torch.tensor
    ) -> Tuple[int, dict]:
        """Compute the average pool size for every h,r combination and r,t combination, strategy specific implementation

        Args:
            head_relation (torch.tensor): Head, Relation tensor
            tail_relation (torch.tensor): Tail, Relation tensor

        Returns:
            Tuple[int, dict]: Average pool size, and dictionary with number of triples with less than X negative (from 2 to 100)
        """

        total = 0
        less_dict = {0: 0, 2: 0, 10: 0, 40: 0, 100: 0}
        total_len = len(head_relation) + len(tail_relation)

        print("[SubsetNegativeSampler] Computing <h,r,*> Negative Pools")
        for comb in tqdm.tqdm(head_relation):
            e = int(comb[0])
            r = int(comb[1])
            negative_pool = self._strategy_negative_pool(e, r, -1, "tail")

            if -1 in negative_pool:
                if self.integrate:
                    pool_size = self.num_entities - len(
                        self._get_positive_pool(e, r, "tail")
                    )
                else:
                    pool_size = 0
            else:
                positive_pool = self._get_positive_pool(e, r, "tail")
                pool_size = int(
                    torch.isin(negative_pool, positive_pool, invert=True).sum()
                )

            total += pool_size
            for k in list(less_dict.keys()):
                if pool_size <= k:
                    less_dict[k] += 1

        print("[SubsetNegativeSampler] Computing <*,r,t> Negative Pools")
        for comb in tqdm.tqdm(tail_relation):
            e = int(comb[0])
            r = int(comb[1])
            negative_pool = self._strategy_negative_pool(-1, r, e, "head")
            if -1 in negative_pool:
                if self.integrate:
                    pool_size = self.num_entities - len(
                        self._get_positive_pool(e, r, "head")
                    )
                else:
                    pool_size = 0
            else:
                positive_pool = self._get_positive_pool(e, r, "head")
                pool_size = int(
                    torch.isin(negative_pool, positive_pool, invert=True).sum()
                )

            total += pool_size
            for k in list(less_dict.keys()):
                if pool_size < k:
                    less_dict[k] += 1

        for k, v in less_dict.items():
            less_dict[k] = (v, float(v / total_len))

        return int(total / total_len), less_dict


class CorruptNegativeSampler(SubSetNegativeSampler):
    """Negative sampler from "Richard Socher, Danqi Chen, Christopher D Manning,
    and Andrew Ng. 2013. Reasoning With Neural Tensor Networks for Knowledge
    Base Completion." Corrupt head and tails based on the subset of entities seen
    as head or tail of the specific relation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    @lru_cache(maxsize=1024)
    def _strategy_negative_pool(self, h, r, t, target):
        return self.subset[r][target]


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

    @lru_cache(maxsize=1024)
    def _strategy_negative_pool(self, h, r, t, target):

        target_class = self.relation_domain_range[r][self.mapping[target]]

        negative_pool = (
            self.subset[target_class] if target_class != "None" else torch.tensor([-1])
        )

        negative_pool = negative_pool if len(negative_pool) > 0 else torch.tensor([-1])

        return negative_pool

    def _generate_subset(self, mapped_triples, **kwargs):

        classes_dict = dict()

        for i in range(self.num_relations):
            if i not in self.relation_domain_range.keys():
                self.relation_domain_range[i] = {"domain": "None", "range": "None"}

        for _, domain_range_dict in self.relation_domain_range.items():
            for classes_name in domain_range_dict.values():
                if classes_name != "None":
                    classes_dict[classes_name] = []

        for entity_id, classes_names in self.entity_classes.items():
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

                subset[entity_id] = entity_dict

            with open(self.local_file, "wb") as f:
                torch.save(subset, f)

            print(f"[RelationalNegativeSampler] Saved Subset as {self.local_file}")

        return subset

    def _strategy_negative_pool(self, h, r, t, target):

        # If corrupting HEAD we take the TAIL entity to use as a pivot for the subset
        # If corrupting TAIL we take the HEAD entity to use as a pivot for the subset d

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


class NearestNeighbourNegativeSampler(SubSetNegativeSampler):
    """Nearest Neighbour Negative Sampler from "Kotnis, B., Nastase, V.: Analysis of the impact of negative
    sampling on link prediction in knowledge graphs". Uses the entity embedding from a pretrained KGE input model to compute
    the entity K-Nearest neighbours to be used as negatives.
    """

    def __init__(
        self,
        *args,
        sampling_model: ERModel = None,
        num_query_results: int = None,
        **kwargs,
    ):
        """Inizialite the NearestNeighbourNegativeSampler

        Args:
            sampling_model (ERModel, optional): Auxiliary pretrained model used to get entity embeddings. Defaults to None.
            num_query_results (int, optional): The K to be used in K Nearest Neighbours search. Defaults to None.
        """
        object.__setattr__(self, "sampling_model", sampling_model)
        object.__setattr__(self, "num_query_results", num_query_results)

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

    def _strategy_negative_pool(self, h, r, t):

        head_positive_pool, tail_positive_pool = self._get_positive_pool(h, r, t)

        head_negative_pool = torch.tensor(self._query_kdtree(h))
        tail_negative_pool = torch.tensor(self._query_kdtree(t))

        head_negative_pool = head_negative_pool[
            torch.isin(head_negative_pool, head_positive_pool, invert=True)
        ]
        tail_negative_pool = tail_negative_pool[
            torch.isin(tail_negative_pool, tail_positive_pool, invert=True)
        ]

        head_negative_pool = (
            head_negative_pool if len(head_negative_pool) > 0 else torch.tensor([-1])
        )
        tail_negative_pool = (
            tail_negative_pool if len(tail_negative_pool) > 0 else torch.tensor([-1])
        )

        return head_negative_pool, tail_negative_pool

    @lru_cache(maxsize=1024, typed=False)
    def _query_kdtree(self, entity_id):

        search_entity = self.subset["entity_representations"][entity_id]
        _, indices = self.subset["kdtree"].query(
            search_entity, k=self.num_query_results
        )

        return indices


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
            sampling_model (ERModel, optional): Auxiliary pretrained model used to predict the target embedding. Defaults to None.
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

        batch_shape = positive_batch.shape[:-1]

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
        _, head_negative_pool = self.subset["kdtree"].query(
            head_prediction, k=self.num_query_results
        )
        log.end()

        log.start(f"[NS {self._get_name()}] Querying KDTREE for TAIL predictions")
        _, tail_negative_pool = self.subset["kdtree"].query(
            tail_prediction, k=self.num_query_results
        )
        log.end()

        self.subset["head_negative_pool"] = torch.tensor(head_negative_pool)
        self.subset["tail_negative_pool"] = torch.tensor(tail_negative_pool)

        # Clone Negative for corruption (cloned the number of negative per positive )
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(
            self.num_negs_per_pos, dim=0
        )

        for i in tqdm.tqdm(range(0, positive_batch.size(0))):

            batch_start = i * self.num_negs_per_pos
            batch_end = batch_start + self.num_negs_per_pos

            triple_batch = negative_batch[batch_start:batch_end]
            negative_heads, negative_tails = self._choose_from_pools(
                positive_batch[i], i
            )

            # Head Corruption
            triple_batch[: self.num_negs_per_pos // 2][:, HEAD] = negative_heads

            # Tail Corruption
            triple_batch[self.num_negs_per_pos // 2 :][:, TAIL] = negative_tails

        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)

    def _strategy_negative_pool(self, h, r, t, internal_id):

        head_positive_pool, tail_positive_pool = self._get_positive_pool(h, r, t)

        head_negative_pool = self.subset["head_negative_pool"][internal_id]
        tail_negative_pool = self.subset["tail_negative_pool"][internal_id]

        head_negative_pool = head_negative_pool[
            torch.isin(head_negative_pool, head_positive_pool, invert=True)
        ]
        tail_negative_pool = tail_negative_pool[
            torch.isin(tail_negative_pool, tail_positive_pool, invert=True)
        ]

        head_negative_pool = (
            head_negative_pool if len(head_negative_pool) > 0 else torch.tensor([-1])
        )
        tail_negative_pool = (
            tail_negative_pool if len(tail_negative_pool) > 0 else torch.tensor([-1])
        )

        return head_negative_pool, tail_negative_pool

    def _choose_from_pools(self, triple, internal_id) -> torch.tensor:
        head_negative_pool, tail_negative_pool = self._strategy_negative_pool(
            int(triple[HEAD]), int(triple[REL]), int(triple[TAIL]), internal_id
        )

        num_head_negatives = self.num_negs_per_pos // 2
        num_tail_negatives = self.num_negs_per_pos - num_head_negatives

        negative_heads = head_negative_pool[
            torch.randint(0, len(head_negative_pool), size=(num_head_negatives,))
        ]
        negativs_tails = tail_negative_pool[
            torch.randint(0, len(tail_negative_pool), size=(num_tail_negatives,))
        ]

        return negative_heads, negativs_tails
