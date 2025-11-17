import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as copy
from typing import Optional, Tuple
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from dataclasses import dataclass

from .utils import LinearLayer
from .blocks import PTransformerBlock, TransformerBlock
from .attention import AttentionPooler
from .mixin import PPIEmbeddingMixin


@dataclass
class PPIOutput(BaseModelOutput):
    a: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    ad_logits_a: Optional[torch.Tensor] = None
    ad_logits_b: Optional[torch.Tensor] = None
    ad_labels_a: Optional[torch.Tensor] = None
    ad_labels_b: Optional[torch.Tensor] = None
    ad_loss: Optional[torch.Tensor] = None


class PPIConfig(PretrainedConfig):
    model_type = "ppi"
    def __init__(
        self,
        input_size: int = 1280,
        hidden_size: int = 512,
        output_size: int = 512,
        expansion_ratio: float = 4.0,
        n_tokens: int = 32,
        n_heads: int = 8,
        n_layers: int = 1,
        dropout: float = 0.2,
        rotary: bool = True,
        block_type: str = "transformer",
        spectral_norm: bool = False,
        plm_path: str = 'Synthyra/ESM2-650M',
        adversarial_num_labels: int = 11,
        adversarial: bool = True,
        add_block_0: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.expansion_ratio = expansion_ratio
        self.n_tokens = n_tokens
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.rotary = rotary
        self.block_type = block_type
        self.spectral_norm = spectral_norm
        self.plm_path = plm_path
        self.adversarial_num_labels = adversarial_num_labels
        self.adversarial = adversarial
        self.add_block_0 = add_block_0


class Proj(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.2):
        super(Proj, self).__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    

class PPIModel(PreTrainedModel):
    config_class = PPIConfig
    def __init__(self, config: PPIConfig):
        super(PPIModel, self).__init__(config)
        self.config = config
        hidden_size = config.hidden_size
        self.a_encoder = LinearLayer(config.input_size, hidden_size, spectral_norm=config.spectral_norm)
        self.b_encoder = LinearLayer(config.input_size, hidden_size, spectral_norm=config.spectral_norm)

        pooler_heads = hidden_size // 128
        self.a_pooler = AttentionPooler(hidden_size, n_tokens=config.n_tokens, n_heads=pooler_heads, spectral_norm=config.spectral_norm)
        self.b_pooler = AttentionPooler(hidden_size, n_tokens=config.n_tokens, n_heads=pooler_heads, spectral_norm=config.spectral_norm)

        block_cls = PTransformerBlock if config.block_type == "ptransformer" else TransformerBlock

        block_0_config = copy(config)
        block_1_config = copy(config)
        block_1_config.hidden_size = hidden_size
        block_1_config.n_heads = max(1, hidden_size // 128)
        block_2_config = copy(config)
        block_2_config.hidden_size = hidden_size // 2
        block_2_config.n_heads = max(1, hidden_size // 2 // 128)
        block_3_config = copy(config)
        block_3_config.hidden_size = hidden_size // 4
        block_3_config.n_heads = max(1, hidden_size // 4 // 128)
        block_4_config = copy(config)
        block_4_config.hidden_size = config.output_size
        block_4_config.n_heads = max(1, config.output_size // 128)

        self.a_block_0 = block_cls(block_0_config) # happens before a_pooler
        self.b_block_0 = block_cls(block_0_config) # happens before b_pooler

        self.block_1 = block_cls(block_1_config)
        self.block_12_proj = Proj(hidden_size, hidden_size // 2, config.dropout)
        self.block_2 = block_cls(block_2_config)
        self.block_23_proj = Proj(hidden_size // 2, hidden_size // 4, config.dropout)
        self.block_3 = block_cls(block_3_config)
        self.block_34_proj = Proj(hidden_size // 4, config.output_size, config.dropout)
        self.block_4 = block_cls(block_4_config)
        self.final_proj = Proj(config.output_size, 1, config.dropout)

    def featurize_a(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.a_encoder(x)
        x = self.a_block_0(x, mask)
        x = self.a_pooler(x, mask)
        return x
    
    def featurize_b(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.b_encoder(x)
        x = self.b_block_0(x, mask)
        x = self.b_pooler(x, mask)
        return x

    def forward(
            self,
            a: torch.Tensor,
            b: torch.Tensor,
            a_mask: Optional[torch.Tensor] = None,
            b_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> PPIOutput:
        a = self.featurize_a(a, a_mask) # (b, n, d)
        b = self.featurize_b(b, b_mask) # (b, n, d)
        
        x = torch.cat([a, b], dim=1) # (b, 2n, d)
        x = self.block_1(x) # (b, 2n, d)
        x = self.block_12_proj(x) # (b, 2n, d)
        x = self.block_2(x) # (b, 2n, d)
        x = self.block_23_proj(x) # (b, 2n, d)
        x = self.block_3(x) # (b, 2n, d)
        x = self.block_34_proj(x) # (b, 2n, d)
        x = self.block_4(x) # (b, 2n, d)
        x = self.final_proj(x) # (b, 2n, 2)
        logits = x.mean(dim=1) # (b, 1)

        return PPIOutput(
            a=a,
            b=b,
            logits=logits,
        )


class PPIForEmbedding(PPIModel, PPIEmbeddingMixin):
    def __init__(self, config: PPIConfig):
        super(PPIForEmbedding, self).__init__(config)
        super(PPIEmbeddingMixin, self).__init__()
        self.config = config

    # To work with EmbeddingMixin, assign _embed_a or _embed_b -> _embed before calling, then do the other one
    # Build an embedding_dict for each (embedding_dict_a, embedding_dict_b)
    def _embed_a(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.featurize_a(x, attention_mask)

    def _embed_b(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.featurize_b(x, attention_mask)

    def _embed(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process pre-computed embeddings through the PPI model.
        
        This method should be assigned to either _embed_a or _embed_b functionality before calling.
        Use embed_fn parameter in embed_from_embeddings to specify which one to use.
        
        Args:
            embeddings: Pre-computed embeddings tensor of shape (batch_size, seq_len, embedding_dim)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Processed embeddings of shape (batch_size, n_tokens, output_size)
        """
        raise NotImplementedError("Make sure to assign _embed_a or _embed_b before calling embed_from_embeddings")

    def forward(self, **kwargs):
        raise NotImplementedError("Should be using _embed not forward")