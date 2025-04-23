import pykeen.constants
import torch
import numpy as np
from typing import Dict, List, Tuple
from pykeen.triples import TriplesFactory
from pykeen.sampling import BernoulliNegativeSampler
from pykeen.sampling.filtering import PythonSetFilterer
from extended_filtering import NullPythonSetFilterer
from extended_sampling import SubSetNegativeSampler, CorruptNegativeSampler, TypedNegativeSampler



torch.manual_seed(42)
np.random.seed(42)

entities = [
    "NYC", "LA", "Chicago", "Houston",  # cities (0-3)
    "USA", "California", "Illinois", "Texas",  # states/country (4-7)
    "English", "Spanish",  # languages (8-9)
    "Pizza", "Burger", "Taco", "HotDog"  # foods (10-13)
]

relations = [
    "located_in",
    "speaks",
    "known_for"
]

# Create entity and relation mappings
entity_to_id = {entity: idx for idx, entity in enumerate(entities)}
relation_to_id = {relation: idx for idx, relation in enumerate(relations)}


mapped_triples = [
    (0, 0, 4),  # NYC located_in USA
    (1, 0, 5),  # LA located_in California
    (1, 0, 4),  # LA located_in USA
    (2, 0, 6),  # Chicago located_in Illinois
    (2, 0, 4),  # Chicago located_in USA
    (3, 0, 7),  # Houston located_in Texas
    (3, 0, 4),  # Houston located_in USA
    (5, 0, 4),  # California located_in USA
    (6, 0, 4),  # Illinois located_in USA
    (7, 0, 4),  # Texas located_in USA

    (0, 1, 8),  # NYC speaks English
    (1, 1, 8),  # LA speaks English
    (1, 1, 9),  # LA speaks Spanish
    (2, 1, 8),  # Chicago speaks English
    (3, 1, 8),  # Houston speaks English
    (3, 1, 9),  # Houston speaks Spanish

    (0, 2, 10),  # NYC known_for Pizza
    (2, 2, 10),  # Chicago known_for Pizza
    (1, 2, 12),  # LA known_for Taco
    (3, 2, 11),  # Houston known_for Burger
    (7, 2, 13),  # Texas known_for HotDog
]

# Convert to PyKeen format
triples_tensor = torch.tensor(mapped_triples, dtype=torch.long)

triples_factory = TriplesFactory(
    mapped_triples=triples_tensor,
    entity_to_id=entity_to_id,
    relation_to_id=relation_to_id,
    num_entities=14,  # 0-13
    num_relations=3,  # 0-2
    create_inverse_triples=False
)

print(f"Knowledge Graph Overview:")
print(f"Number of entities: {triples_factory.num_entities}")
print(f"Number of relations: {triples_factory.num_relations}")
print(f"Number of triples: {triples_factory.num_triples}")

# Create entity subset mapping
entity_subset_mapping = {
    # Cities can only have other cities as negatives
    0: [1, 2, 3],  # NYC -> [LA, Chicago, Houston]
    1: [0, 2, 3],  # LA -> [NYC, Chicago, Houston]
    2: [0, 1, 3],  # Chicago -> [NYC, LA, Houston]
    3: [0, 1, 2],  # Houston -> [NYC, LA, Chicago]
    
    # States can only have other states as negatives
    4: [5, 6, 7],  # USA -> [California, Illinois, Texas]
    5: [4, 6, 7],  # California -> [USA, Illinois, Texas]
    6: [4, 5, 7],  # Illinois -> [USA, California, Texas]
    7: [4, 5, 6],  # Texas -> [USA, California, Illinois]
    
    # Languages can only have other languages as negatives
    8: [9],        # English -> [Spanish]
    9: [8],        # Spanish -> [English]
    
    # Foods can only have other foods as negatives
    10: [11, 12, 13],  # Pizza -> [Burger, Taco, HotDog]
    11: [10, 12, 13],  # Burger -> [Pizza, Taco, HotDog]
    12: [10, 11, 13],  # Taco -> [Pizza, Burger, HotDog]
    13: [10, 11, 12],  # HotDog -> [Pizza, Burger, Taco]
}




sampler = CorruptNegativeSampler(
    mapped_triples=triples_factory.mapped_triples,
    filtered=True,
    filterer=NullPythonSetFilterer(mapped_triples=triples_factory.mapped_triples),
    num_negs_per_pos = 2
)

#print(sampler.subset)

negatives = sampler.sample(triples_factory.mapped_triples)

print(negatives)










