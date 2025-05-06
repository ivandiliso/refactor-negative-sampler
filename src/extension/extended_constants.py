from pykeen.constants import TARGET_TO_INDEX

INDEX_TO_TARGET = {v: k for k, v in TARGET_TO_INDEX.items()}
SWAP_TARGET = {"head": "tail", "tail": "head"}
SWAP_TARGET_ID = {0:2, 2:0}
HEAD = 0
REL = 1
TAIL = 2