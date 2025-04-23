import pykeen
from pykeen.triples import TriplesFactory



def load_dataset_from_file(path):
    data = dict()

    data["train"] = TriplesFactory.from_path(
        path = path / "train.txt",
        create_inverse_triples = False
    )

    entity_to_id = data["train"].entity_to_id
    relation_to_id = data["train"].relation_to_id

    data["test"] = TriplesFactory.from_path(
        path = path / "test.txt",
        create_inverse_triples = False,
        entity_to_id = entity_to_id,
        relation_to_id = relation_to_id
    )

    data["valid"] = TriplesFactory.from_path(
        path = path / "valid.txt",
        create_inverse_triples = False,
        entity_to_id = entity_to_id,
        relation_to_id = relation_to_id
    )

    return data

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def pretty_print(type, str):
    match type:
        case "t":
            print(f"[{color.RED}{str}{color.END}]")


def additional_data_loader(triples_factory: TriplesFactory, relation_domain_range: dict, entity_classes: dict):
    """_summary_

    Args:
        triples_factory (TriplesFactory): _description_
        relation_domain_range (dict): _description_
        entity_classes (dict): _description_
    """

