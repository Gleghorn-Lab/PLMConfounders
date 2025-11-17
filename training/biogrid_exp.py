#! /usr/bin/env python3
# py -m training.biogrid_exp
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Set matplotlib backend before any imports that might use it
import matplotlib
matplotlib.use('Agg')
import random
import torch
import datetime
import logging
import numpy as np
from torch.utils.data import DataLoader
from huggingface_hub import login
from tqdm import tqdm
from typing import Optional, Dict
from transformers import AutoModel, get_scheduler
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef
)

from data.biogrid import get_biogrid_data
from training.utils import set_seed, parse_args, AutoGradClipper
from model.ppi_model import PPIConfig, PPIModel
from data.data import (
    BiogridDataset,
    BiogridCollator,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


global WANDB_AVAILABLE
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class BiogridBinaryTrainer:
    def __init__(self, args, matching_orgs: bool, seed: int, run_idx: int, preloaded_data: Optional[dict] = None):
        self.args = args
        self.matching_orgs = matching_orgs
        self.seed = seed
        self.run_idx = run_idx
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Deterministic seeds per run
        set_seed(seed)

        # Load BIOGRID data with specified negative sampling behavior
        if preloaded_data is not None:
            self.train_df = preloaded_data['train_df']
            self.valid_df = preloaded_data['valid_df']
            self.test_df = preloaded_data['test_df']
            self.seq_dict = preloaded_data['seq_dict']
            self.interaction_set = preloaded_data['interaction_set']
        else:
            self.train_df, self.valid_df, self.test_df, self.seq_dict, self.interaction_set = get_biogrid_data(
                similarity_threshold=args.similarity_threshold,
                min_rows=args.minimum_test_size,
                n=2 if not args.bugfix else 5,
                save=True,
                matching_orgs=matching_orgs,
                sample_rows=100000 if args.bugfix else None,
                rebuild_negatives=False,
            )
        print(f'Train: {len(self.train_df)}, Val: {len(self.valid_df)}, Test: {len(self.test_df)}')

        # Sequences for embedding
        self.all_seqs = list(set(self.seq_dict.values()))
        print(f'All sequences: {len(self.all_seqs)}')

        # Basic trainer fields
        self.batch_size = args.batch_size
        tag = 'match' if matching_orgs else 'nomatch'
        base_results_dir = os.path.join('results', 'biogrid_species_experiment')
        os.makedirs(base_results_dir, exist_ok=True)
        self.save_dir = os.path.join(base_results_dir, f"biogrid_{tag}_seed{seed}")
        self.best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        self.skip_violations = True  # not used in binary setting
        os.makedirs(self.save_dir, exist_ok=True)

    def prep_for_training(self):
        self.get_embeddings()
        self.get_data_loaders()
        self.get_model()
        self.get_optimizers()
        self.get_loss_fct()
        # Ensure logging and save directory are set up before determining best model path
        self.setup_logging()
        # Do not overwrite if a subclass (e.g., binary trainer) already set a specific path
        if not hasattr(self, 'best_model_path') or self.best_model_path in (None, ''):
            self.best_model_path = os.path.join(self.save_dir, 'best_model.pth')

    def setup_logging(self):
        """Setup logging to file for metrics"""
        if not hasattr(self, 'save_dir'):
            # Create a default save_dir if not set
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = f"logs_{timestamp}"
        
        os.makedirs(self.save_dir, exist_ok=True)
        log_file = os.path.join(self.save_dir, 'metrics.log')
        
        # Setup file logger
        self.metrics_logger = logging.getLogger('metrics_logger')
        self.metrics_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.metrics_logger.handlers[:]:
            self.metrics_logger.removeHandler(handler)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.metrics_logger.addHandler(file_handler)
        
        print(f"Metrics will be logged to: {log_file}")

    def log_metrics(self, metrics: Dict[str, float], prefix: str = ''):
        """Log metrics to file and filter for WANDB"""
        # Filter metrics for WANDB (roc_auc and mcc metrics for best model tracking)
        wandb_metrics = {k: v for k, v in metrics.items() if any(metric in k.lower() for metric in ['roc_auc', 'mcc'])}
        
        # Log all metrics to file
        self.metrics_logger.info(f"=== {prefix.upper()} METRICS ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                self.metrics_logger.info(f"{k}: {v:.4f}")
            else:
                self.metrics_logger.info(f"{k}: {v}")

        self.metrics_logger.info("=" * 50)
        
        # Send roc_auc and mcc metrics to WANDB
        if WANDB_AVAILABLE and wandb_metrics:
            log_metrics_wandb = {f'{prefix}/{k}' if prefix else k: v for k, v in wandb_metrics.items()}
            wandb.log(log_metrics_wandb)
        
        return wandb_metrics

    def get_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=1000,
            num_training_steps=self.args.num_epochs * len(self.train_loader)
        )
        if self.args.clip_grad:
            self.auto_grad_clipper = AutoGradClipper(self.model)
        else:
            self.auto_grad_clipper = None

    def get_embeddings(self):
        species_id_str = self.args.species_ids[0]
        model_name = self.args.plm_path.split('/')[-1].strip().lower()
        plm = AutoModel.from_pretrained(self.args.plm_path, trust_remote_code=True).to(self.device).eval()
        if not self.args.bugfix:
            plm = torch.compile(plm)
        self.embed_dim = plm.config.hidden_size
        embed_path = f'{model_name}_{species_id_str}_embeddings.pth'
        self.embed_dict = plm.embed_dataset(
            sequences=self.all_seqs,
            tokenizer=plm.tokenizer,
            batch_size=self.args.patch_size,
            max_len=self.args.max_length,
            full_embeddings=True,
            embed_dtype=torch.float32,
            num_workers=0 if self.args.bugfix else 4,
            sql=False,
            save=True,
            save_path=embed_path,
        )
        plm.cpu()
        del plm
        torch.cuda.empty_cache()

    def get_model(self):
        config = PPIConfig(
            plm_path=self.args.plm_path,
            input_size=self.embed_dim,
            hidden_size=self.args.hidden_size,
            output_size=self.args.output_size,
            expansion_ratio=self.args.expansion_ratio,
            n_tokens=self.args.n_tokens,
            dropout=self.args.dropout,
            rotary=True,
            block_type=self.args.block_type,
            spectral_norm=self.args.spectral_norm,
            adversarial=False,
            adversarial_num_labels=1,
            add_block_0=False,
        )
        print(f'Config: \n{config}')
        self.model = PPIModel(config).to(self.device)
        if not self.args.bugfix:
            self.model = torch.compile(self.model)

    def get_data_loaders(self):
        train_dataset = BiogridDataset(self.train_df, self.seq_dict, eval_mode=False)
        valid_dataset = BiogridDataset(self.valid_df, self.seq_dict, eval_mode=True)
        test_dataset = BiogridDataset(self.test_df, self.seq_dict, eval_mode=True)
        data_collator = BiogridCollator(
            embed_dim=self.embed_dim,
            max_length=self.args.max_length,
            embedding_dict=self.embed_dict,
        )

        num_workers = 0 if self.args.bugfix else 4
        prefetch_factor = None if self.args.bugfix else 2

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    def get_loss_fct(self):
        pos = int((self.train_df['labels'] > 0).sum())
        neg = int((self.train_df['labels'] == 0).sum())
        pos_weight = torch.tensor(neg / max(1, pos), device=self.device, dtype=torch.float32)
        self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')

    def _metrics_helper(self, logits: np.ndarray, labels: np.ndarray, prefix: Optional[str] = '') -> Dict[str, float]:
        metrics = {}
        
        # Validate input arrays
        if len(logits) == 0 or len(labels) == 0:
            print(f"Warning: Empty arrays provided to _metrics_helper (logits: {len(logits)}, labels: {len(labels)})")
            return metrics
        
        # Check if we have both classes for ROC AUC calculation
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            print(f"Warning: Only one class found in labels: {unique_labels}, cannot calculate ROC AUC")
            return metrics
        
        roc_auc = roc_auc_score(labels, logits)
        precisions, recalls, thresholds = precision_recall_curve(labels, logits)
        pr_auc = auc(recalls, precisions)

        # Only sample 100 thresholds for speed
        if len(thresholds) > 100:
            idxs = np.linspace(0, len(thresholds) - 1, 100, dtype=int)
            sampled_thresholds = thresholds[idxs]
        else:
            sampled_thresholds = thresholds

        f1_scores = []
        for t in sampled_thresholds:
            preds = logits > t
            f1 = f1_score(labels, preds, average='macro')
            f1_scores.append(f1)

        best_idx = np.argmax(np.array(f1_scores))
        best_threshold = sampled_thresholds[best_idx]

        best_preds = (logits >= best_threshold).astype(int)
        f1 = f1_score(labels, best_preds, average='weighted')
        precision = precision_score(labels, best_preds, average='weighted')
        recall = recall_score(labels, best_preds, average='weighted')
        accuracy = accuracy_score(labels, best_preds)
        mcc = matthews_corrcoef(labels, best_preds)

        metrics[f'{prefix}_roc_auc'] = roc_auc
        metrics[f'{prefix}_pr_auc'] = pr_auc
        metrics[f'{prefix}_f1'] = f1
        metrics[f'{prefix}_precision'] = precision
        metrics[f'{prefix}_recall'] = recall
        metrics[f'{prefix}_accuracy'] = accuracy
        metrics[f'{prefix}_mcc'] = mcc
        metrics[f'{prefix}_threshold'] = best_threshold

        print(f"{prefix}_Confusion matrix:")
        print(confusion_matrix(labels, best_preds))
        return metrics

    def calculate_metrics(self, logits: torch.Tensor, labels: torch.Tensor, string_labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        if isinstance(logits, torch.Tensor):
            logits = logits.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        if string_labels is not None:
            if isinstance(string_labels, torch.Tensor):
                string_labels = string_labels.numpy()
                string_labels = string_labels.flatten()

        logits = logits.flatten()
        labels = labels.flatten()

        # Filter out violation labels (-100.0) and keep only valid labels (0 or 1)
        valid_mask = (labels == 0) | (labels == 1)
        if not valid_mask.any():
            print("Warning: No valid labels found (all are violations). Returning default metrics.")
            return {
                'ratio': 1.0,
                'avg_pos': 0.5,
                'avg_neg': 0.5,
                'roc_auc': 0.5,
                'pr_auc': 0.5,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0,
                'mcc': 0.0,
                'threshold': 0.5,
            }
        
        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]

        # Balance the dataset: sample as many negatives as there are positives
        pos_indices = np.where(valid_labels == 1)[0]
        neg_indices = np.where(valid_labels == 0)[0]
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        # Randomly sample negatives to match the number of positives
        if n_neg > n_pos:
            sampled_neg_indices = np.random.choice(neg_indices, size=n_pos, replace=False)
        else:
            sampled_neg_indices = neg_indices

        balanced_indices = np.concatenate([pos_indices, sampled_neg_indices])
        np.random.shuffle(balanced_indices)

        logits_bal = valid_logits[balanced_indices]
        labels_bal = valid_labels[balanced_indices]

        pos_values = logits_bal[labels_bal == 1]
        neg_values = logits_bal[labels_bal == 0]
        pos_avg = pos_values.mean()
        neg_avg = neg_values.mean()
        ratio = pos_avg / neg_avg if neg_avg != 0 else float('inf')

        metrics = self._metrics_helper(logits_bal, labels_bal, prefix='balanced')

        if string_labels is not None:
            string_labels = string_labels[valid_mask]
            string_labels = string_labels[balanced_indices]
            string_thresholds = [1.5, 4.0, 7.0, 9.0]
            for t in string_thresholds:
                string_mask = string_labels > t
                logits_string = logits_bal[string_mask]
                labels_string = labels_bal[string_mask]
                p = str(t).replace('.', '') + '0'
                if len(logits_string) == 0:
                    continue
                string_metrics = self._metrics_helper(logits_string, labels_string, prefix=f'string_{p}')
                metrics.update(string_metrics)

        return metrics

    def train_step(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        output = self.model(**batch)
        logits = output.logits.view(-1)
        labels = batch['labels']
        loss = self.loss_fct(logits.view(-1), labels.view(-1))
        probs = logits.detach().sigmoid().cpu()
        labels_cpu = labels.detach().cpu()
        if (labels_cpu == 1).any():
            avg_pos_value = probs[labels_cpu == 1].mean()
        else:
            avg_pos_value = torch.tensor(0.5)
        if (labels_cpu == 0).any():
            avg_neg_value = probs[labels_cpu == 0].mean()
        else:
            avg_neg_value = torch.tensor(0.5)
        return loss, avg_pos_value, avg_neg_value

    @torch.no_grad()
    def eval_step(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        output = self.model(**batch)
        logits = output.logits.view(-1)
        labels = batch['labels']
        loss = self.loss_fct(logits.view(-1), labels.view(-1))
        logits = logits.detach().cpu().sigmoid()
        labels = labels.detach().cpu()
        return loss, logits, labels

    def evaluate(self, data_loader, prefix: str = 'test'):
        print("Starting evaluation (binary)...")
        self.model.eval()
        total_loss, total_logits, total_labels = 0.0, [], []
        progress_bar = tqdm(total=len(data_loader), desc="Evaluating")

        batch_count = 0
        for batch in data_loader:
            loss, logits, labels = self.eval_step(batch)
            total_loss += loss.item()
            total_logits.append(logits)
            total_labels.append(labels)
            batch_count += 1
            progress_bar.update(1)

        progress_bar.close()
        if batch_count == 0:
            return {}
        total_loss /= batch_count
        total_logits = torch.cat(total_logits)
        total_labels = torch.cat(total_labels)
        metrics = self.calculate_metrics(total_logits, total_labels)
        metrics['loss'] = total_loss
        self.log_metrics(metrics, prefix)
        print(f"{prefix} metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
        return metrics

    def train(self):
        best_val_mcc, patience_counter, global_step = 0.0, 0, 0
        # Track best validation MCC for this trainer instance
        self.best_val_mcc = 0.0
        # Initial test eval
        _ = self.evaluate(self.test_loader, prefix='test')

        grad_accum = self.args.grad_accum
        for epoch in range(self.args.num_epochs):
            self.model.train()
            losses, avg_pos_values, avg_neg_values = [], [], []
            progress_bar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.args.num_epochs}")

            for step_idx, batch in enumerate(self.train_loader, start=1):
                if patience_counter >= self.args.patience:
                    print(f"Early stopping triggered after {patience_counter} evaluations without improvement")
                    break

                loss, avg_pos_value, avg_neg_value = self.train_step(batch)
                loss = loss / grad_accum
                losses.append(loss.item() * grad_accum)
                avg_pos_values.append(avg_pos_value)
                avg_neg_values.append(avg_neg_value)
                loss.backward()

                if (step_idx % grad_accum == 0) or (step_idx == len(self.train_loader)):
                    if self.args.clip_grad and hasattr(self, 'auto_grad_clipper') and self.auto_grad_clipper:
                        _ = self.auto_grad_clipper.clip_gradients()
                    self.optimizer.step()
                    self.scheduler.step()
                    global_step += 1
                    self.model.zero_grad(set_to_none=True)
                    progress_bar.update(1)

                    if global_step % 100 == 0 and global_step > 0:
                        avg_loss = sum(losses) / len(losses)
                        avg_pos_value = sum(avg_pos_values) / len(avg_pos_values)
                        avg_neg_value = sum(avg_neg_values) / len(avg_neg_values)
                        train_metrics = {
                            "loss": avg_loss,
                            "avg_pos": float(avg_pos_value),
                            "avg_neg": float(avg_neg_value),
                        }
                        if WANDB_AVAILABLE:
                            wandb.log({f"train/{k}": v for k, v in train_metrics.items()}, step=global_step)
                        losses, avg_pos_values, avg_neg_values = [], [], []

                    if global_step % self.args.save_every == 0 and global_step > 0:
                        metrics = self.evaluate(self.valid_loader, prefix='valid')
                        mcc = metrics.get('balanced_mcc', 0.0)
                        if mcc > best_val_mcc:
                            print(f"New best validation MCC: {mcc:.4f}")
                            best_val_mcc = mcc
                            self.best_val_mcc = mcc
                            patience_counter = 0
                            torch.save(self.model.state_dict(), self.best_model_path)
                        else:
                            patience_counter += 1
            progress_bar.close()

        # After training epochs, evaluate validation once more to ensure best is recorded
        # and then load best model for final test evaluation
        final_valid = self.evaluate(self.valid_loader, prefix='valid')
        final_mcc = final_valid.get('balanced_mcc', 0.0)
        if final_mcc > best_val_mcc:
            print(f"New best validation MCC (final pass): {final_mcc:.4f}")
            best_val_mcc = final_mcc
            self.best_val_mcc = final_mcc
            torch.save(self.model.state_dict(), self.best_model_path)

        # Load best model if saved
        if os.path.exists(self.best_model_path):
            print(f"Loading best model from: {self.best_model_path}")
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        else:
            print("Warning: No best model found, using current model state for final evaluation")

        test_metrics = self.evaluate(self.test_loader, prefix='test')
        return test_metrics


if __name__ == "__main__":
    args = parse_args()
    n_runs = args.n_runs

    if args.token is not None:
        login(args.token)

    if args.bugfix:
        args.plm_path = "Synthyra/ESM2-8M"
        args.batch_size = 2
        args.max_length = 16
        args.save_every = 100
        args.hidden_size = 128
        args.output_size = 16
        args.expansion_ratio = 1.0
        args.n_tokens = 4
        args.dropout = 0.2
        args.grad_accum = 1
        args.patience = 1
        args.minimum_test_size = 100
        args.similarity_threshold = 0.95
        args.num_epochs = 1
        args.n_runs = 2

    seeds = [random.randint(0, 1000) for _ in range(n_runs)]

    # Prebuild and cache both datasets (matching_orgs False and True) once
    print("Prebuilding cached BIOGRID splits for matching_orgs=False and matching_orgs=True ...")
    preloaded_nomatch_tuple = get_biogrid_data(
        similarity_threshold=args.similarity_threshold,
        min_rows=args.minimum_test_size,
        n=2 if not args.bugfix else 5,
        save=True,
        matching_orgs=False,
        sample_rows=100000 if args.bugfix else None,
        rebuild_negatives=False,
    )
    preloaded_match_tuple = get_biogrid_data(
        similarity_threshold=args.similarity_threshold,
        min_rows=args.minimum_test_size,
        n=2 if not args.bugfix else 5,
        save=True,
        matching_orgs=True,
        sample_rows=100000 if args.bugfix else None,
        rebuild_negatives=False,
    )
    preloaded_nomatch = {
        'train_df': preloaded_nomatch_tuple[0],
        'valid_df': preloaded_nomatch_tuple[1],
        'test_df': preloaded_nomatch_tuple[2],
        'seq_dict': preloaded_nomatch_tuple[3],
        'interaction_set': preloaded_nomatch_tuple[4],
    }
    preloaded_match = {
        'train_df': preloaded_match_tuple[0],
        'valid_df': preloaded_match_tuple[1],
        'test_df': preloaded_match_tuple[2],
        'seq_dict': preloaded_match_tuple[3],
        'interaction_set': preloaded_match_tuple[4],
    }

    # Matching orgs = False runs with identical seeds
    for i, seed in enumerate(seeds):
        if WANDB_AVAILABLE:
            run_name = f"biogrid_exp_nomatch_{i}_seed{seed}"
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        set_seed(seed)
        trainer = BiogridBinaryTrainer(args, matching_orgs=False, seed=seed, run_idx=i, preloaded_data=preloaded_nomatch)
        trainer.prep_for_training()
        test_metrics = trainer.train()
        del trainer
        torch.cuda.empty_cache()
        if WANDB_AVAILABLE:
            wandb.finish()

        # Matching orgs = True runs
        if WANDB_AVAILABLE:
            run_name = f"biogrid_exp_match_{i}_seed{seed}"
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        set_seed(seed)
        trainer = BiogridBinaryTrainer(args, matching_orgs=True, seed=seed, run_idx=i, preloaded_data=preloaded_match)
        trainer.prep_for_training()
        test_metrics = trainer.train()
        del trainer
        torch.cuda.empty_cache()
        if WANDB_AVAILABLE:
            wandb.finish()

    # Training complete. Per-run metrics are logged to each run directory under metrics.log.
    print("Training completed. To generate plots from metrics.log files, run: py -m training.biogrid_logs_to_plots")
