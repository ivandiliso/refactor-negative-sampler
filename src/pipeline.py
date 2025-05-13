"""
Author: Ivan Diliso
Description: Experimentations with HPO Pipeline
"""

import gc
import signal
import sys
from datetime import datetime
from pathlib import Path

import pykeen
import pykeen.models
import torch
from extension.dataset import OnMemoryDataset
from extension.filtering import NullPythonSetFilterer
from extension.sampling import (
    CorruptNegativeSampler,
    RelationalNegativeSampler,
    TypedNegativeSampler,
    NearestNeighbourNegativeSampler,
    NearMissNegativeSampler,
)

from optuna.samplers import RandomSampler
from pykeen.regularizers import LpRegularizer
from types import SimpleNamespace
from pykeen.losses import MarginRankingLoss
from pykeen.sampling import BernoulliNegativeSampler, BasicNegativeSampler
from pykeen.models import TransE, TransH, TransR
from pykeen.pipeline import pipeline
from pykeen.sampling.filtering import PythonSetFilterer
from pykeen.triples import TriplesFactory
from tabulate import tabulate
from extension.utils import automatic_backend_chooser, set_random_seed_all
import argparse
from pykeen.pipeline import pipeline
from pykeen.sampling import negative_sampler_resolver
from pykeen.sampling.filtering import filterer_resolver


from pykeen.optimizers import Adam
import json

params = SimpleNamespace()

# Arguments Parsing
################################################################################

parser = argparse.ArgumentParser("Experiments Configurations")
parser.add_argument(
    "--model", type=str, choices=["transe", "transh", "transr"], required=True
)
parser.add_argument(
    "--sampler",
    type=str,
    choices=["random", "bernoulli", "corrupt", "typed", "relational", "nearmiss"],
    required=True,
)

parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--l2", type=float, required=True)
parser.add_argument("--margin", type=int, required=True)

parser.add_argument("--negatives", type=int, choices=[2, 10, 40, 100], required=True)
parser.add_argument("--dataset", type=str, choices=["yago4-20", "db50k", "fb15k", "wn18"])

args = parser.parse_args()

print(f"[Pipeline] Starting Experiment with configuration:")
print(f"[Pipeline] Dataset: {args.dataset.capitalize()}")
print(f"[Pipeline] Model: {args.model.capitalize()}")
print(f"[Pipeline] Negative Sampler: {args.sampler.capitalize()}")
print(f"[Pipeline] Negatives per Positive: {args.negatives}")
print(f"[Pipeline] Learning Rate: {args.lr}")
print(f"[Pipeline] L2 Regularizer Weight: {args.l2}")
print(f"[Pipeline] Loss Margin: {args.margin}")


params.experiment_name = f"NoHPO_{args.dataset.upper()}_{args.sampler.upper()}_{args.model.upper()}_{args.negatives}_{str(round(datetime.now().timestamp() * 1000))}"

params.dataset_name = args.dataset
params.model_name = args.model
params.negative_sampler_name = args.sampler
params.num_neg_per_pos = args.negatives

print(f"[Pipeline] Experiment named {params.experiment_name}")

params.data_path = Path.cwd() / "data" / params.dataset_name.upper()
params.experiment_path = Path.cwd() / "experiments" / params.experiment_name

print(f"[Pipeline] Experiment will be saved in {params.experiment_path}")
print(f"[Pipeline] Using dataset in folder {params.data_path}")


# Python and Torch Configuration
################################################################################

params.random_seed = 42
set_random_seed_all(params.random_seed)

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

params.device = automatic_backend_chooser()

# Experiment Fixed Hyperparameters
################################################################################

params.epochs = 100
params.embedding_dim = 100
params.relation_dim = 100
params.scoring_fct_norm = 2
params.batch_size = 500
params.regularizer_p = 2
params.loss = "marginloss"

params.margin = args.margin
params.learning_rate = args.lr
params.regularizer_weight = args.l2

# Dataset Loading
################################################################################

match params.dataset_name:
    case "yago4-20":
        dataset = OnMemoryDataset(
            data_path=params.data_path, load_domain_range=True, load_entity_classes=True
        )
    case "wn18":
        dataset = OnMemoryDataset(
            data_path=params.data_path, load_domain_range=False, load_entity_classes=False
        )
    case "db50k":
        dataset = OnMemoryDataset(
            data_path=params.data_path, load_domain_range=True, load_entity_classes=True
        )
    case "fb15k":
        dataset = OnMemoryDataset(
            data_path=params.data_path, load_domain_range=False, load_entity_classes=False
        )

print("[Data Loader] Dataset Information")
print(
    tabulate(
        [
            [
                dataset.training.num_triples,
                dataset.validation.num_triples,
                dataset.testing.num_triples,
            ]
        ],
        headers=["Train", "Valid", "Test"],
        tablefmt="github",
    )
)
print(
    f"[Data Loader] Statistics: {dataset.num_entities} entities, {dataset.num_relations} relations"
)


# Initialization of Negative Sampler
################################################################################


negative_sampler_resolver.register(element=CorruptNegativeSampler)
negative_sampler_resolver.register(element=TypedNegativeSampler)
negative_sampler_resolver.register(element=RelationalNegativeSampler)
filterer_resolver.register(element=NullPythonSetFilterer)

match params.negative_sampler_name:
    case "random":
        params.negative_sampler = "basic"
        params.negative_sampler_kwargs = dict(
            filtered=True,
            filterer="nullpythonset",
            num_negs_per_pos=params.num_neg_per_pos,
        )
    case "bernoulli":
        params.negative_sampler = "bernoulli"
        params.negative_sampler_kwargs = dict(
            filtered=True,
            filterer="nullpythonset",
            num_negs_per_pos=params.num_neg_per_pos,
        )
    case "corrupt":
        params.negative_sampler = "corrupt"
        params.negative_sampler_kwargs = dict(
            filtered=True,
            filterer="nullpythonset",
            num_negs_per_pos=params.num_neg_per_pos,
        )
    case "typed":
        params.negative_sampler = "typed"
        params.negative_sampler_kwargs = dict(
            filtered=True,
            filterer="nullpythonset",
            num_negs_per_pos=params.num_neg_per_pos,
            entity_classes_dict=dataset.entity_id_to_classes,
            relation_domain_range_dict=dataset.relation_id_to_domain_range,
        )
    case "relational":
        params.negative_sampler = "relational"
        params.negative_sampler_kwargs = dict(
            filtered=True,
            filterer="nullpythonset",
            num_negs_per_pos=params.num_neg_per_pos,
            local_file=params.data_path / "relational_cached.bin"
        )

print(f"[Negative Sampler] {params.negative_sampler}")


# Inizialization of Model
################################################################################


match params.model_name:
    case "transe":
        params.model = TransE(
            triples_factory = dataset.training,
            embedding_dim=100,
            regularizer=LpRegularizer,
            regularizer_kwargs=dict(
                p = params.regularizer_p,
                weight = params.regularizer_weight
            ),
        )
    case "transh":
        params.model = TransH(
            triples_factory = dataset.training,
            embedding_dim=100,
            regularizer=LpRegularizer,
            regularizer_kwargs=dict(
                p = params.regularizer_p,
                weight = params.regularizer_weight
            ),
        )
    case "transr":
        params.model = TransR(
            triples_factory = dataset.training,
            embedding_dim=100
        )


        
print(f"[Embedding Model] {params.model}")



# HPO Pipeline
################################################################################


pipeline_result = pipeline(

    training=dataset.training,
    testing=dataset.testing,
    validation=dataset.validation,

    model=params.model,

    negative_sampler=params.negative_sampler,
    negative_sampler_kwargs=params.negative_sampler_kwargs,

    training_loop="sLCWA",
    training_kwargs=dict(
        num_epochs=50,
        batch_size=1000
    ),

    loss=MarginRankingLoss,
    loss_kwargs=dict(
        margin=params.margin
    ),

    optimizer=Adam,
    optimizer_kwargs=dict(
        lr=params.learning_rate
    ),

    device=params.device,

    evaluation_kwargs=dict(
        batch_size = 500
    )
)


# Saving HPO Pipeline Results
################################################################################

pipeline_result.save_to_directory(
    directory =params.experiment_path,
    save_metadata = True,
    save_training = False
    )