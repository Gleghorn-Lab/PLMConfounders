import pandas as pd
import torch
from random import random
from torch.utils.data import Dataset as TorchDataset
from typing import List, Tuple, Dict, Any, Optional


class BiogridDataset(TorchDataset):
    def __init__(self, df: pd.DataFrame, seq_dict: Dict[str, str], eval_mode: bool = False):
        self.seq_dict = seq_dict
        self.ids_a = df['IdA'].tolist()
        self.ids_b = df['IdB'].tolist()
        self.labels = df['labels'].tolist()
        self.eval_mode = eval_mode

    def __len__(self) -> int:
        return len(self.ids_a)

    def __getitem__(self, idx: int) -> Tuple[str, str, int, str, str]:
        ida = self.ids_a[idx]
        idb = self.ids_b[idx]
        label = int(self.labels[idx])
        if random() < 0.5 and not self.eval_mode:
            ida, idb = idb, ida
        seq_a = self.seq_dict[ida]
        seq_b = self.seq_dict[idb]
        return seq_a, seq_b, label


class BaseCollator:
    def _pad_embeds(self, embeddings_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select a random section of the embedding matrix
        Build attention mask
        """
        max_len = max(emb.size(0) for emb in embeddings_list)
        final_embeds, masks = [], []
        for emb in embeddings_list:
            emb_len = emb.size(0)
            padding_len = max_len - emb_len
            if padding_len > 0:
                padding = torch.zeros(padding_len, self.embed_dim, device=emb.device)
                emb = torch.cat([emb, padding], dim=0)
            final_embeds.append(emb)
            mask = torch.ones(max_len, device=emb.device)
            mask[emb_len:] = 0
            masks.append(mask)
        return torch.stack(final_embeds), torch.stack(masks)
    

class BiogridCollator(BaseCollator):
    def __init__(
            self,
            embed_dim: int = 1280,
            max_length: int = 2048,
            embedding_dict: Optional[Dict[str, torch.Tensor]] = None
        ):
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.embedding_dict = embedding_dict

    def __call__(
            self,
            batch: List[Tuple[str, str, float]]
        ) -> Dict[str, Any]:
        seqs_a, seqs_b, labels = zip(*batch)
        seqs_a = [seq[:self.max_length] for seq in seqs_a]
        seqs_b = [seq[:self.max_length] for seq in seqs_b]
        labels = torch.tensor(labels, dtype=torch.float)

        A = [self.embedding_dict[seq].reshape(-1, self.embed_dim) for seq in seqs_a]
        B = [self.embedding_dict[seq].reshape(-1, self.embed_dim) for seq in seqs_b]
        A, a_mask = self._pad_embeds(A)
        B, b_mask = self._pad_embeds(B)

        return {
            'a': A,
            'b': B,
            'a_mask': a_mask,
            'b_mask': b_mask,
            'labels': labels,
        }
