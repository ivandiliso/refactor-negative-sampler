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
from pykeen.models import TransE
from pykeen.pipeline import pipeline
from pykeen.sampling.filtering import PythonSetFilterer
from pykeen.triples import TriplesFactory
from tabulate import tabulate
from extension.utils import automatic_backend_chooser, set_random_seed_all
import argparse
from pykeen.hpo import hpo_pipeline
from pykeen.sampling import negative_sampler_resolver
from pykeen.sampling.filtering import filterer_resolver


from pykeen.optimizers import Adam
import json

params = SimpleNamespace()
params.dataset = "YAGO4-20"

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
parser.add_argument("--negatives", type=int, choices=[2, 10, 40, 100], required=True)
parser.add_argument("--dataset", type=str, choices=["yago4-20", "db50k", "fb15k", "wn18"])

args = parser.parse_args()

print(f"[HPO Pipeline] Starting Experiment with configuration:")
print(f"[HPO Pipeline] Dataset: {args.dataset.capitalize()}")
print(f"[HPO Pipeline] Model: {args.model.capitalize()}")
print(f"[HPO Pipeline] Negative Sampler: {args.sampler.capitalize()}")
print(f"[HPO Pipeline] Negatives per Positive: {args.negatives}")

params.experiment_name = f"{args.sampler.upper()}_{args.model.upper()}_{args.negatives}_{str(round(datetime.now().timestamp() * 1000))}"

params.dataset_name = args.dataset
params.model_name = args.model
params.negative_sampler_name = args.sampler
params.num_neg_per_pos = args.negatives

print(f"[HPO Pipeline] Experiment named {params.experiment_name}")

params.data_path = Path.cwd() / "data" / params.dataset_name.upper()
params.experiment_path = Path.cwd() / "experiments" / params.experiment_name

print(f"[HPO Pipeline] Experiment will be saved in {params.experiment_path}")
print(f"[HPO Pipeline] Using dataset in folder {params.data_path}")


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
params.hpo_timeout = 4
params.hpo_strategy = "random"
params.loss = "marginloss"


# Hyparameters Ranges for HPO Pipeline
################################################################################

params.hpo_margin = dict(type="categorical", choices=[1, 2, 5, 10])
params.hpo_learning_rate = dict(type=float, low=1e-5, high=1e-2, log=True)
params.hpo_regularizer_weight = dict(type=float, low=1e-6, high=1e-2, log=True)


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
        params.model = TransE


        
print(f"[Embedding Model] {params.model}")



# HPO Pipeline
################################################################################


hpo_pipeline_result = hpo_pipeline(

    timeout = params.hpo_timeout * 60 * 60,

    training=dataset.training,
    testing=dataset.testing,
    validation=dataset.validation,

    model=params.model,
    model_kwargs=dict(
        embedding_dim=params.embedding_dim,
        random_seed = params.random_seed,
        scoring_fct_norm=params.scoring_fct_norm
    ),

    negative_sampler=params.negative_sampler,
    negative_sampler_kwargs=params.negative_sampler_kwargs,

    training_loop="sLCWA",
    training_kwargs=dict(
        num_epochs=params.epochs,
        batch_size=1000
    ),

    loss=MarginRankingLoss,
    loss_kwargs_ranges=dict(
        margin=params.hpo_margin
    ),

    regularizer=LpRegularizer,
    regularizer_kwargs=dict(
        p = params.regularizer_p
    ),
    regularizer_kwargs_ranges=dict(
        weight = params.hpo_regularizer_weight
    ),

    optimizer=Adam,
    optimizer_kwargs_ranges=dict(
        lr=params.hpo_learning_rate
    ),
    device=params.device,

    evaluation_kwargs=dict(
        batch_size = 500
    ),
    save_model_directory = params.experiment_path / "models_checkpoints"
)


# Saving HPO Pipeline Results
################################################################################

out_dict = {
    "hpo_optimized_params" : hpo_pipeline_result.study.best_params,
    "best_trial_number" : hpo_pipeline_result.study.best_trial.number,
    "model" : str(params.model),
    "negative_sampler" : params.negative_sampler,
    "epochs": params.epochs,
    "embedding_dim": params.embedding_dim,
    "relation_dim": params.relation_dim,
    "scoring_fct_norm": params.scoring_fct_norm,
    "batch_size": params.batch_size,
    "regularizer_p": params.regularizer_p,
    "hpo_timeout": params.hpo_timeout,
    "hpo_strategy": params.hpo_strategy,
    "loss": params.loss,
    "hpo_margin": {k:v for k,v in params.hpo_margin.items()} ,
    "hpo_learning_rate":{k:v for k,v in params.hpo_learning_rate.items() if k != "type"},
    "hpo_regularizer_weight":{k:v for k,v in params.hpo_regularizer_weight.items() if k != "type"},
    "random_seed": params.random_seed,
    "num_neg_per_pos": params.num_neg_per_pos,
    "dataset": params.dataset,
    "training_loop" : "sLCWA",
    "pykeen_version" : pykeen.version.get_version(),
    "python_verison" : sys.version
}

with open(params.experiment_path / "best_trial_params.json", "w") as f:
    json.dump(out_dict, f, indent=4)
