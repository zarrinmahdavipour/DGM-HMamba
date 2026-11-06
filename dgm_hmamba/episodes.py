"""N-way K-shot episodic sampling utilities."""

import random
from typing import Dict, List, Sequence, Tuple
import torch


class ByClassIndex:
    def __init__(self, labels: Sequence[int]):
        self.by_class: Dict[int, List[int]] = {}
        for idx, y in enumerate(labels):
            self.by_class.setdefault(int(y), []).append(idx)

    @property
    def classes(self) -> List[int]:
        return list(self.by_class.keys())

    def sample_episode(
        self,
        n_way: int,
        k_shot: int,
        q_query: int,
        rng: random.Random = None,
    ) -> Tuple[List[int], List[int], List[int]]:
        rng = rng or random
        classes = self.classes
        if len(classes) < n_way:
            raise ValueError(f"Need >= {n_way} classes, found {len(classes)}")
        episode_classes = rng.sample(classes, n_way)

        support_idx, query_idx = [], []
        for c in episode_classes:
            pool = list(self.by_class[c])
            rng.shuffle(pool)
            needed = k_shot + q_query
            if len(pool) < needed:
                raise ValueError(
                    f"Class {c} has only {len(pool)} samples, needs {needed}"
                )
            support_idx.extend(pool[:k_shot])
            query_idx.extend(pool[k_shot:needed])
        return support_idx, query_idx, episode_classes


def gather_episode(
    patches: torch.Tensor,
    labels: torch.Tensor,
    support_idx: List[int],
    query_idx: List[int],
    episode_classes: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    remap = {c: i for i, c in enumerate(episode_classes)}
    support_patches = patches[support_idx]
    query_patches = patches[query_idx]
    support_labels = torch.tensor(
        [remap[int(labels[i])] for i in support_idx], dtype=torch.long
    )
    query_labels = torch.tensor(
        [remap[int(labels[i])] for i in query_idx], dtype=torch.long
    )
    return support_patches, support_labels, query_patches, query_labels
