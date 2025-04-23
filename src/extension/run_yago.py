import pykeen
from pykeen.triples import TriplesFactory
from pathlib import Path
from tabulate import tabulate
from test_utils import *
from pykeen.pipeline import pipeline
from pykeen.sampling.filtering import PythonSetFilterer
from extended_filtering import NullPythonSetFilterer
from extended_sampling import CorruptNegativeSampler

# Initial Global Variables Configuration 
################################################################################
config = {
    "home_path"     : Path().cwd() / "refactor",
    "dataset_name"  : "YAGO4-20"
}
config["data_path"] = config["home_path"] / "data" / config["dataset_name"]


# Dataset Loading
################################################################################
data = load_dataset_from_file(config["data_path"])

pretty_print("t", "Dataset Information")
print(f"Statistics: {data["train"].num_entities} entities, {data["train"].num_relations} relations")
print(tabulate(
    [[data["train"].num_triples, data["valid"].num_triples, data["test"].num_triples]],
    headers=["Train", "Valid", "Test"],
    tablefmt="github"
))


# Training
################################################################################


negative_sampler = CorruptNegativeSampler(
        mapped_triples=data["train"].mapped_triples,
        filtered=True,
        filterer=NullPythonSetFilterer(mapped_triples=data["train"].mapped_triples),
        num_negs_per_pos=5
)


result = pipeline(
    training=data["train"],
    testing=data["test"],
    validation=data["valid"],
    model="TransE",
    model_kwargs=dict(
        embedding_dim=100,
        scoring_fct_norm=1
    ),
    negative_sampler= negative_sampler,
    training_kwargs=dict(
        num_epochs=1,
        batch_size=256,
        label_smoothing=0.0,
    ),
    evaluation_kwargs=dict(batch_size=1024),
    evaluator_kwargs=dict(batch_size=1024),
    device="mps",
    random_seed=42
)


# Evaluation
################################################################################
metrics = result.metric_results.to_dict()
result.save_to_directory('transe_model')