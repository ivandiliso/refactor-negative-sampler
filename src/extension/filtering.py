import pykeen
import torch
from pykeen.sampling.filtering import PythonSetFilterer


class NullPythonSetFilterer(PythonSetFilterer):
    """Extensiion of Python Set based filterer that also check for manually inserted invalid negatives entities
    with negative indices.
    """

    def __init__(self, mapped_triples):
        super().__init__(mapped_triples)

    def contains(self, batch):
        return torch.as_tensor(
            data=[
                ((-1 in tuple(triple)) or (tuple(triple) in self.triples))
                for triple in batch.view(-1, 3).tolist()
            ],
            dtype=torch.bool,
            device=batch.device,
        ).view(*batch.shape[:-1])
