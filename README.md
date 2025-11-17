# Protein Language Models are Accidental Taxonomists

[![Paper](https://img.shields.io/badge/Paper-Preprint-blue)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

## Overview

Protein-protein interactions (PPIs) are fundamental to nearly all biological processes, and computational methods using protein language models (pLMs) have shown promising results. However, we uncovered a critical confounding factor: **pLM-based models can "cheat" by learning to distinguish taxonomic origin rather than genuine interaction features**.

### Key Findings

- **Multi-species PPI datasets have inherent phylogenetic biases**: When negatives are randomly sampled, only ~31% share the same species as positives, while real PPIs are almost always intra-species.
- **pLMs encode phylogenetic information**: Models can distinguish whether two proteins share taxonomic origin with 0.87 F1 score.
- **Strategic sampling prevents cheating**: Restricting negative examples to same-species pairs eliminates the taxonomic shortcut.
- **Multi-species data still helps**: When properly curated, multi-species models outperform single-species models (0.37 vs 0.30 MCC on rigorous test sets).

### The "Accidental Taxonomist" Problem

Traditional PPI datasets use random negative sampling, which creates an unintended signal:
- ✅ **Positive PPIs**: Protein A and B from *same species* → Label = 1
- ❌ **Negative PPIs**: Protein A and B from *different species* (~70% of the time) → Label = 0

Models learn this taxonomic shortcut instead of (or in addition to) genuine PPI features, resulting in misleadingly high performance metrics.

## Installation

### Prerequisites

- Python 3.8+
- Docker or Docker Desktop (required for Windows users)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Gleghorn-Lab/PLMConfounders.git
cd PLMConfounders
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

or utilize a **Python virtual environment**
```bash
chmod +x setup_bioenv.sh
./setup_bioenv.sh
source ~/bioenv/bin/activate
```

3. **Install and start Docker**:
   - Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - **Windows users**: Ensure Docker Desktop is running before executing training scripts
   - The training pipeline uses Docker containers for embedding generation and data processing

## Usage

### Reproducing the Experiments

To reproduce the BioGRID PPI experiments from the paper:

```bash
py -m training.biogrid_exp
```

This will:
1. Download and process BioGRID data
2. Generate two training conditions:
   - **Normal Sampling (NS)**: Negatives randomly sampled from entire dataset
   - **Strategic Sampling (SS)**: Negatives sampled from within same species
3. Train 5 models for each condition with different random seeds
4. Evaluate on rigorous test sets (C3 splits, 40% CD-HIT clustering)
5. Save results to `results/biogrid_species_experiment/`

**Note**: Full training requires significant computational resources (tested on GH200 GPU). For quick testing, use the `--bugfix` flag:

```bash
py -m training.biogrid_exp --bugfix
```

### Command-Line Arguments

Key arguments for `biogrid_exp.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--plm_path` | `esmc_600m` | Protein language model to use for embeddings |
| `--batch_size` | `128` | Training batch size |
| `--max_length` | `512` | Maximum sequence length |
| `--num_epochs` | `1` | Number of training epochs |
| `--n_runs` | `5` | Number of training runs with different seeds |
| `--similarity_threshold` | `0.4` | CD-HIT clustering threshold |
| `--save_every` | `5000` | Evaluation frequency (steps) |
| `--patience` | `5` | Early stopping patience |
| `--wandb_project` | `biogrid_ppi` | Weights & Biases project name |
| `--bugfix` | `False` | Quick test mode with reduced dataset and model size |

## Project Structure

```
PLMConfounders/
├── data/
│   ├── biogrid.py           # BioGRID data loading and processing
│   └── data.py              # Dataset and collator classes
├── model/
│   ├── ppi_model.py         # Main PPI prediction model
│   ├── attention.py         # Attention mechanisms
│   ├── blocks.py            # Transformer blocks
│   ├── rotary.py            # Rotary positional embeddings
│   └── utils.py             # Model utilities
├── training/
│   ├── biogrid_exp.py       # Main training script (NS vs SS experiment)
│   └── utils.py             # Training utilities
├── processed_datasets/      # Cached preprocessed datasets
├── results/                 # Training outputs and checkpoints
├── sequence_data/           # Raw BioGRID data and sequences
├── preprint/                # Figures and analysis from the paper
│   ├── plot_biogrid.py      # Visualization scripts
│   └── preprint.ipynb       # Analysis notebook
└── requirements.txt         # Python dependencies
```

## Key Results

### Performance Comparison: NS vs SS

| Metric | Normal Sampling (NS) | Strategic Sampling (SS) |
|--------|----------------------|-------------------------|
| Validation MCC | **0.71** | 0.39 |
| Test MCC | 0.23 | **0.37** |

The NS models achieve high validation scores by exploiting phylogenetic distances, but fail on properly sampled test sets. SS models maintain consistent performance and **outperform single-species state-of-the-art** (0.30 MCC on Bernett et al. dataset).

### Taxonomic Classification Performance

pLM embeddings can predict taxonomic origin with surprising accuracy:

| Level | # Classes | Best F1 Score | Model |
|-------|-----------|---------------|-------|
| Domain | 3 | 0.97 | ProtT5 |
| Kingdom | 17 | 0.69 | ESMC-600M |
| Phylum | 68 | 0.49 | ProtT5 |
| Species | 618 | 0.18 | ESMC-600M |
| **Different Species (Binary)** | **2** | **0.87** | **ESMC-600M** |

The high performance on the binary "different species" task (0.87 F1) explains why models can cheat on PPI datasets.

## Implications

This work has broad implications for protein machine learning:

1. **PPI Prediction**: Always use strategic sampling for multi-species datasets
2. **Any Supervised Protein Task**: Check for taxonomic biases in label distributions
3. **Dataset Design**: Consider phylogenetic distances when constructing train/test splits
4. **Model Evaluation**: Test on rigorous same-species splits to detect taxonomic shortcuts
5. **Benchmark Caution**: High multi-species performance may indicate confounding, not genuine learning

### Recommended Best Practices

✅ **Do:**
- Sample negatives from within the same species
- Use C3 splits (no sequence overlap between train/val/test)
- Cluster at 40% similarity to prevent memorization
- Report performance on both same-species and multi-species test sets
- Check label distributions across taxonomic ranks

❌ **Don't:**
- Randomly shuffle protein pairs to generate negatives
- Trust >90% accuracy on multi-species PPI datasets
- Ignore phylogenetic distances in dataset construction
- Use subcellular localization-based negatives without controls

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{hallee2025accidental,
  title={Protein Language Models are Accidental Taxonomists},
  author={Hallee, Logan and Peleg, Tamar and Rafailidis, Nikolaos and Gleghorn, Jason P.},
  journal={bioRxiv},
  year={2025}
}
```

## Data Availability

- **Processed datasets**: Available in `processed_datasets/` after running training scripts
- **BioGRID source**: Downloaded from [thebiogrid.org](https://thebiogrid.org/)
- **Model weights**: Saved to `results/` after training
- **Protify datasets / code**: Available at [Protify repository](https://github.com/Gleghorn-Lab/Protify)

## Contributing

We welcome contributions! If you find additional confounding factors or improvements to dataset construction, please open an issue or pull request.

## Authors

- **Logan Hallee** - University of Delaware & Synthyra - [lhallee@udel.edu](mailto:lhallee@udel.edu)
- **Tamar Peleg** - University of Delaware - [tamarp@udel.edu](mailto:tamarp@udel.edu)
- **Nikolaos Rafailidis** - University of Delaware - [nrafaili@udel.edu](mailto:nrafaili@udel.edu)
- **Jason P. Gleghorn** - University of Delaware & Synthyra - [gleghorn@udel.edu](mailto:gleghorn@udel.edu)

## Acknowledgements

This work was supported by:
- University of Delaware Graduate College (Unidel Distinguished Graduate Scholar Award)
- National Science Foundation (NAIRR pilot 240064)
- National Institutes of Health (NIGMS T32GM142603, R01HL178817, R01HL133163, R01HL145147)

Special thanks to Katherine M. Nelson, Ph.D., for reviewing and commenting on drafts of the manuscript.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Conflict of Interest

LH and JPG are co-founders of and have an equity stake in Synthyra, PBLLC.

---

**Questions?** Open an issue or contact [lhallee@udel.edu](mailto:lhallee@udel.edu)

**Found this useful?** ⭐ Star the repository to help others discover this work!
