from typing import Dict, Mapping, Optional, Sequence, Set, Tuple, Union, List, cast
import torch
from torch.utils.data import Dataset

from pykeen.typing import EntityMapping, MappedTriples, Target
from pykeen.sampling import NegativeSampler
from pykeen.triples import CoreTriplesFactory
from abc import ABC, abstractmethod
from pykeen.typing import LongTensor, BoolTensor, MappedTriples
import math
from pykeen.constants import TARGET_TO_INDEX


INDEX_TO_TARGET = {v:k for k,v in TARGET_TO_INDEX.items()}


class SubSetNegativeSampler(NegativeSampler, ABC):
    """ Abstract Class Handling static negative sampling, requires implementing
    a method able to calculate the correct subset pool of negative for each
    entity in the triples set
    """
    def __init__(self, *, mapped_triples, num_entities = None, num_relations = None, num_negs_per_pos = None, filtered = False, filterer = None, filterer_kwargs = None, **kwargs):
        super().__init__(mapped_triples=mapped_triples, num_entities=num_entities, num_relations=num_relations, num_negs_per_pos=num_negs_per_pos, filtered=filtered, filterer=filterer, filterer_kwargs=filterer_kwargs)
        
        self.subset = self._generate_subset(mapped_triples, **kwargs)


    @abstractmethod
    def _corrupt_triple(self, triple: torch.LongTensor, target: Target):
        """ Corrupt the selected triple using the generated subset. The triple
        has to be corrupted in-place

        Args:
            triple (torch.LongTensor): Triple in (h,r,t) format
            target (Target | int): Corrupt head ("head" or 0) or tail ("tail" or 2)
        """
        raise NotImplementedError


    @abstractmethod
    def _generate_subset(self, mapped_triples: MappedTriples, **kwargs):
        """ Generated the supporting subset to corrupt the triple

        Args:
            mapped_triples (MappedTriples): Base triples to generate the subset
        """
        raise NotImplementedError
    

    def corrupt_batch(self, positive_batch: MappedTriples) -> MappedTriples:
        """ Subset batch corruptor. Uniform corruption between head and tail.
        Corrupts each triple using the generated subset

        Args:
            positive_batch (MappedTriples): Batch of positive triples

        Returns:
            MappedTriples: Batch of negative triples of size (positive_size * num_neg_per_pos, 3)
        """
        
        batch_shape = positive_batch.shape[:-1]

        # Clone Negative for corruption (cloned the number of negative per positive )
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(self.num_negs_per_pos, dim=0)

        # Create tarket mask
        target = torch.full((negative_batch.size(0),), fill_value=2)
        target[:int(negative_batch.size(0)*0.5)] = 0
        target = target[torch.randperm(target.size(0))]

        total_num_negatives = negative_batch.shape[0]

        for i in range(total_num_negatives):
            self._corrupt_triple(negative_batch[i], INDEX_TO_TARGET[int(target[i])])

        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)



class CorruptNegativeSampler(SubSetNegativeSampler):
    """ Negative sampler from "Richard Socher, Danqi Chen, Christopher D Manning, 
    and Andrew Ng. 2013. Reasoning With Neural Tensor Networks for Knowledge 
    Base Completion." Corrupt head and tails based on the subset of entities seen
    as head or tail of the specific relation 

    """
    def __init__(self, *, mapped_triples, num_entities=None, num_relations=None, num_negs_per_pos=None, filtered=False, filterer=None, filterer_kwargs=None):
        super().__init__(mapped_triples=mapped_triples, num_entities=num_entities, num_relations=num_relations, num_negs_per_pos=num_negs_per_pos, filtered=filtered, filterer=filterer, filterer_kwargs=filterer_kwargs)
    

    def _generate_subset(self, mapped_triples):
        relations = torch.unique(mapped_triples[:,1]).tolist()
        subset = dict()
        for r in relations:
            mask = mapped_triples[mapped_triples[:,1] == r]
            subset[r] = {
                "head" : torch.unique(mask[:,0]),
                "tail" : torch.unique(mask[:,2])
            }
        return subset
    
    
    def _corrupt_triple(self, triple: torch.LongTensor, target: Target):
        negative_pool = self.subset[int(triple[1])][target]
        triple[TARGET_TO_INDEX[target]] = negative_pool[torch.randperm(len(negative_pool))[:1]]



class TypedNegativeSampler(SubSetNegativeSampler):
    def __init__(self, *, mapped_triples, num_entities=None, num_relations=None, num_negs_per_pos=None, filtered=False, filterer=None, filterer_kwargs=None, domain_range_dict=None, entity_classes_dict=None):
        super().__init__(mapped_triples=mapped_triples, num_entities=num_entities, num_relations=num_relations, num_negs_per_pos=num_negs_per_pos, filtered=filtered, filterer=filterer, filterer_kwargs=filterer_kwargs, domain_range_dict=domain_range_dict, entity_classes_dict=entity_classes_dict)
  

    def _generate_subset(self, mapped_triples, **kwargs):
        print(kwargs)


