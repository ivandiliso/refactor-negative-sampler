import math
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from sys import getsizeof
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast

import torch
import tqdm as tqdm
from collections.abc import Callable
from extension.test_utils import SimpleLogger
from pykeen.sampling import NegativeSampler
from pykeen.triples import CoreTriplesFactory
from pykeen.typing import BoolTensor, EntityMapping, LongTensor, MappedTriples, Target
from torch.utils.data import Dataset
from functools import lru_cache
from pykeen.models import TransE, RESCAL, ERModel
from scipy.spatial import KDTree
import numpy as np
from extension.extended_constants import (
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

       