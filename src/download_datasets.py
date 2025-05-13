from extension.dataset import OnCloudDataset
from pathlib import Path

data_path = Path().cwd() / "data"
datasets_names = ["yago4-20", "db50k", "wn18", "fb15k"]

for d_name in datasets_names:
    OnCloudDataset(data_path=data_path, dataset_name=d_name, load_domain_range=True, load_entity_classes=True)