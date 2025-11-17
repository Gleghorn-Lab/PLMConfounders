import torch
import numpy as np
import argparse


def set_seed(seed):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 


class AutoGradClipper:
    # Auto gradient clipping that adapts based on gradient history.
    # adapted from https://github.com/pseeth/autoclip/tree/master
    def __init__(self, model, clip_percentile=10, history_length=1000000):
        self.model = model
        self.clip_percentile = clip_percentile
        self.history_length = history_length
        self.grad_history = []
    
    def clip_gradients(self):
        """Clip gradients based on percentile of gradient history."""
        obs_grad_norm = _get_grad_norm(self.model)
        self.grad_history.append(obs_grad_norm)
        
        # Keep history length manageable
        if len(self.grad_history) > self.history_length:
            self.grad_history = self.grad_history[-self.history_length:]
        
        # Only start clipping after we have some history
        if len(self.grad_history) >= 10:
            clip_value = np.percentile(self.grad_history, self.clip_percentile)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            return clip_value
        return 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Synthyra Trainer")
    # logistics
    parser.add_argument("--token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--plm_path", type=str, default="Synthyra/ESMplusplus_large", help="Path to the PLM to use for training")
    parser.add_argument("--save_path", type=str, default="lhallee/pstring_test", help="Path to save the model and report to wandb")
    parser.add_argument("--wandb_project", type=str, default="P-STRING", help="Wandb project name")

    # training hypers
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size for the model")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--clip_grad", action="store_true", help="Clip gradients")
    parser.add_argument("--save_every", type=int, default=1000, help="Save the model every n steps and evaluate every n/2 steps")
    parser.add_argument("--patience", type=int, default=100, help="Number of epochs to wait before stopping training")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    parser.add_argument("--skip_violations", action="store_true", help="Skip violation checking for debugging (faster startup)")
    parser.add_argument("--one_hot", action="store_true", help="Use one-hot embeddings")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs")

    # adversarial training
    parser.add_argument("--adversarial", action="store_true", help="Use adversarial training")
    parser.add_argument("--adversarial_alpha_lr", type=float, default=0.001, help="Learning rate for adaptive adversarial alpha updates")
    parser.add_argument("--adversarial_loss_window", type=int, default=100, help="Window size for averaging adversarial losses for adaptive alpha")
    parser.add_argument("--adversarial_initial_alpha", type=float, default=0.0, help="Initial alpha value for adversarial training")

    # model hypers
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size of the model")
    parser.add_argument("--output_size", type=int, default=128, help="Output size of the model")
    parser.add_argument("--expansion_ratio", type=float, default=8/3, help="Expansion ratio of the model")
    parser.add_argument("--n_tokens", type=int, default=32, help="Number of tokens in the model")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate of the model")
    parser.add_argument("--spectral_norm", action="store_true", help="Use spectral normalization")
    parser.add_argument("--block_type", type=str, default="transformer", help="Block type to use for training")
    parser.add_argument("--add_block_0", action="store_true", help="Add a block before the first attention pooler")

    # data
    parser.add_argument("--species_ids", nargs='+', default=['9606'], help="Species IDs of the dataset to use for training")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length of embeddings gathered")
    parser.add_argument("--minimum_test_size", type=int, default=5000, help="Minimum number of test rows")
    parser.add_argument("--similarity_threshold", type=float, default=0.4, help="Similarity threshold for clustering")
    parser.add_argument("--minimum_confidence_train", type=int, default=150, help="Minimum confidence for training")
    parser.add_argument("--minimum_confidence_eval", type=int, default=150, help="Minimum confidence for evaluation")
    parser.add_argument("--no_update_alpha", action="store_true", help="Do not update the alpha value for adversarial training")

    args = parser.parse_args()
    return args
