import pickle
from pathlib import Path
import json

home = Path().cwd()  / "data" / "YAGO4-20"


def clean_name(name):
    return name.split("/")[-1]


with open(home / "rel2id.pkl", "rb") as f:
    rel2id = pickle.load(f)

with open(home / "class2id.pkl", "rb") as f:
    class2id = pickle.load(f)

id2rel = {v:clean_name(k) for k,v in rel2id.items()}
id2class = {v:clean_name(k) for k,v in class2id.items()}

print(id2rel)
print(id2class)


with open(home / "rel2dom.pkl", "rb") as f:
    rel2dom = pickle.load(f)


with open(home / "rel2range.pkl", "rb") as f:
    rel2range = pickle.load(f)


data = {}


for rel_id, rel_name in id2rel.items():


    dom_id = rel2dom.get(rel_id, None)

    if dom_id:
        domain = id2class[dom_id]
    else:
        domain = "None"

    range_id = rel2range.get(rel_id, None)

    if range_id:
        range = id2class[range_id]
    else:
        range = "None"

    data[rel_name] = {
        "domain" : domain,
        "range"  : range
    }


with open(home / "relation_domain_range.json", "w") as f:
    f.write(json.dumps(data, indent=4))



import csv
import ast

instances = set([])
concepts = set([])
instanceof = dict([])

with open(home / "reasoned/entities.csv", "r") as f:
    csv_data = csv.reader(f)
    for index, line in enumerate(csv_data):
        if index != 0:
            ent = line[0].strip()
            classes = ast.literal_eval(line[1].strip())
            classes = [clean_name(c) for c in classes]
            instances.add(ent)
            concepts.update(classes)
            instanceof[ent] = classes

import json

with open(home / "entities_classes.json", "w") as f:
    f.write(json.dumps(instanceof, indent=4))
