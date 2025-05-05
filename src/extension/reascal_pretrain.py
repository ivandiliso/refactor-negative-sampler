"""
Author: Ivan Diliso
Description: Pretrain test for Rescal Models
"""


import pykeen
from pykeen.triples import TriplesFactory
from pathlib import Path
from tabulate import tabulate
from test_utils import *
from pykeen.pipeline import pipeline
from pykeen.sampling.filtering import PythonSetFilterer
from extended_filtering import NullPythonSetFilterer
from extended_sampling import CorruptNegativeSampler, TypedNegativeSampler, RelationalNegativeSampler
from extended_dataset import OnMemoryDataset
import torch
from datetime import datetime
from pykeen.models import RESCAL, TransE
from pykeen.sampling import BasicNegativeSampler
from pykeen.pipeline import pipeline
from pykeen.losses import MarginRankingLoss
from pykeen.evaluation import RankBasedEvaluator
from pykeen.utils import set_random_seed
from pykeen.hpo import hpo_pipeline
from pykeen.stoppers import EarlyStopper
from pykeen.optimizers import Adam
from pykeen.regularizers import LpRegularizer
import gc
import signal
import sys


# Python and Torch Configuration
################################################################################

set_random_seed(42)

torch.cuda.empty_cache()

def cleanup():
    print("\n Cleaning up GPU memory")
    gc.collect()
    torch.cuda.empty_cache()
    print("Cleanup Complete")

def signal_handler(sig, frame):
    print("Exiting gracefully")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)



# Initial Global Variables Configuration 
################################################################################
config = {
    "home_path"     : Path().cwd(),
    "dataset_name"  : "YAGO4-20"
}
config["data_path"] = config["home_path"] / "data" / config["dataset_name"]


# Dataset Loading
################################################################################
dataset = OnMemoryDataset(config["data_path"])

pretty_print("t", "Dataset Information")
print(f"Statistics: {dataset.num_entities} entities, {dataset.num_relations} relations")
print(tabulate(
    [[dataset.training.num_triples, dataset.validation.num_triples, dataset.testing.num_triples]],
    headers=["Train", "Valid", "Test"],
    tablefmt="github"
))


# Model Pre Training
################################################################################


"""
model = RESCAL(
    triples_factory=dataset.training,
    embedding_dim=128,
    entity_initializer=torch.nn.init.xavier_uniform_,
    relation_initializer=torch.nn.init.xavier_uniform_
)
"""



model = TransE(
    triples_factory = dataset.training,
    embedding_dim=128,
    scoring_fct_norm=2
)



negative_sampler = BasicNegativeSampler(
    mapped_triples = dataset.training.mapped_triples,
    filtered=True,
    filterer=NullPythonSetFilterer(
        mapped_triples=dataset.training.mapped_triples
    ),
    num_negs_per_pos=5,
)


result = pipeline(
    dataset=dataset,
    model=model,
    training_loop="sLCWA",
    loss=MarginRankingLoss(margin=10, reduction="mean"),
    optimizer=Adam(params=model.parameters(), lr=0.0098),
    negative_sampler=negative_sampler,
    evaluator=RankBasedEvaluator,
    training_kwargs=dict(
        num_epochs=59,
        batch_size=2048
    ),
    device=torch.device("mps"),
    random_seed=42,
    evaluation_kwargs=dict(batch_size=2048),
    evaluator_kwargs=dict(
        filtered=True,
        batch_size=1024),

)


result.save_to_directory('model/transe_yago420')



"""
result = hpo_pipeline(

    n_trials=20,
    direction="maximize",

    dataset=dataset,
    model=model,
    training_loop="sLCWA",

    loss=MarginRankingLoss(margin=1.0, reduction="mean"),
    loss_kwargs_ranges=dict(
        margin=dict(type='categorical', choices=[0.5, 1.0, 2.0])
    ),

    regularizer=LpRegularizer,
    regularizer_kwargs_ranges=dict(
        weight=dict(type='categorical', choices=[1e-5, 1e-4, 1e-3]),
        p=dict(type='categorical', choices=[2]),
    ), 

    optimizer=Adam,
    optimizer_kwargs_ranges=dict(
        lr=dict(type='categorical', choices=[1e-3, 5e-4, 1e-4])
    ), 

    
    negative_sampler=negative_sampler,
    evaluator=RankBasedEvaluator,
    epochs=100,
    device=torch.device("cuda"),
    stopper=EarlyStopper,
    stopper_kwargs={
        'frequency': 5,
        'patience': 2,
        'relative_delta': 0.002,
    }
)
"""