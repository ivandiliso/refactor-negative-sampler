from pykeen.constants import TARGET_TO_INDEX

INDEX_TO_TARGET = {v: k for k, v in TARGET_TO_INDEX.items()}
SWAP_TARGET = {"head": "tail", "tail": "head"}
SWAP_TARGET_ID = {0:2, 2:0}
HEAD = 0
REL = 1
TAIL = 2
ENTITY_TO_ID_FILENAME = "mapping/entity_to_id.json"
RELATION_TO_ID_FILENAME = "mapping/relation_to_id.json"
TRAIN_SPLIT_FILENAME = "train.txt"
VALID_SPLIT_FILENAME = "valid.txt"
TEST_SPLIT_FILENAME = "test.txt"
DOMAIN_RANGE_METATDATA_FILENAME = "metadata/relation_domain_range.json"
CLASS_MEMBERSHIP_METADATA_FILENAME = "metadata/entity_classes.json"