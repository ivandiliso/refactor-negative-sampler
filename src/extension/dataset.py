import pykeen
import pykeen.datasets
from pykeen.datasets.base import Dataset
from pykeen.triples import TriplesFactory
from pathlib import Path
import json
import ast
import pandas as pd
from pykeen.datasets import get_dataset

ENTITY_TO_ID_FILENAME = "mapping/entity_to_id.json"
RELATION_TO_ID_FILENAME = "mapping/relation_to_id.json"
TRAIN_SPLIT_FILENAME = "train.txt"
VALID_SPLIT_FILENAME = "valid.txt"
TEST_SPLIT_FILENAME = "test.txt"
DOMAIN_RANGE_METATDATA_FILENAME = "metadata/relation_domain_range.json"
CLASS_MEMBERSHIP_METADATA_FILENAME = "metadata/entity_classes.json"


class OnMemoryDataset(Dataset):
    """Dataset located on memory, requires already splitted data in RDF triple
    format. The folder should contain the following files

    ### Folder Structure

    - train.txt : Training triples in "h r t" format using RDF names
    - test.txt : Testing triples in "h r t" format using RDF names
    - valid.txt : Validation triples in "h r t" format using RDF names
    - entity_to_id.json: Tab separated file for id to entity name mapping
    - relation_to_id.json: Tab separeted file for id to relation name mapping
    - entities_classes.json : Additional metadata of class memebership for each entity, need to have format

    ```json
    {
        "<ENTITY_NAME>" : [
            "<CLASS_NAME_1>"
            ...
            "<CLASS_NAME_N>"
        ]
    }
    ```

    - relation_domain_range.json : Additional metadata of domain and range classes for each relation, needs to have format:

    ```json
    {
        "<RELATION_NAME>" : {
            "domain" : "<CLASS_NAME_DOMAIN>" OR "None"
            "range"  : "<CLASS_NAME_RANGE>" OR "None"
        }
    }
    ```
    """

    def __init__(
        self,
        data_path: str | Path = None,
        load_entity_classes: bool = True,
        load_domain_range: bool = True,
        **kwargs
    ):
        """Initialize dataset from on disk folder

        Args:
            data_path (str | Path, optional): Dataset folder path. Defaults to None.
            load_entity_classes (bool, optional): Load the entity class memebership metadata. Defaults to True.
            load_domain_range (bool, optional): Load the relation domain and range classes metadata. Defaults to True.
        """
        self.data_path = Path(data_path)

       
        with open(self.data_path / ENTITY_TO_ID_FILENAME, "r") as f:
            entity_id_mapping = json.load(f)

        with open(self.data_path / RELATION_TO_ID_FILENAME, "r") as f:
            relation_id_mapping = json.load(f)
        

        self.training = TriplesFactory.from_path(
            path=self.data_path / TRAIN_SPLIT_FILENAME,
            create_inverse_triples=False,
            entity_to_id=entity_id_mapping,
            relation_to_id=relation_id_mapping
        )

        self.testing = TriplesFactory.from_path(
            path=self.data_path / TEST_SPLIT_FILENAME,
            create_inverse_triples=False,
            entity_to_id=entity_id_mapping,
            relation_to_id=relation_id_mapping,
        )

        self.validation = TriplesFactory.from_path(
            path=self.data_path / VALID_SPLIT_FILENAME,
            create_inverse_triples=False,
            entity_to_id=entity_id_mapping,
            relation_to_id=relation_id_mapping,
        )

        if load_entity_classes:
            self.entity_id_to_classes = self._load_entity_classes()

        if load_domain_range:
            self.relation_id_to_domain_range = self._load_relation_domain_range()

    def _load_entity_classes(self) -> dict:
        """Load the entity class membership metadata from the provided JSON file.
        Entity names are trasfomed to IDs.

        Returns:
            dict: Dictionary of entity id to list of class names
        """
        with open(self.data_path / CLASS_MEMBERSHIP_METADATA_FILENAME, "r") as f:
            data = json.load(f)

        return {self.entity_to_id[k]: v for k, v in data.items()}

    def _load_relation_domain_range(self) -> dict:
        """Load the relation domain and range classes from the provided JSON file.
        Relation names are transformed to IDs.

        Returns:
            dict: Dictionary of relation is to dict with domain and range classes
        """
        with open(self.data_path / DOMAIN_RANGE_METATDATA_FILENAME, "r") as f:
            data = json.load(f)

        return {
            self.relation_to_id[k]: v
            for k, v in data.items()
            if k in self.relation_to_id.keys()
        }
