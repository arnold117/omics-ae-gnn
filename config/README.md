# Configuration Files

This directory contains YAML configuration files for the multi-omics integration pipeline.

## Files

- **config.yaml**: Main configuration file - data paths, model architecture, training parameters
- **paths.yaml**: File path definitions (optional, can be merged into config.yaml)
- **hardware.yaml**: Hardware-specific settings (GPU/CPU, memory limits)

## Quick Start

1. **Copy the template:**
   ```bash
   cp config/config.yaml config/my_project.yaml
   ```

2. **Edit your configuration:**
   - Replace sample definitions in `data.samples.groups`
   - Update data file paths in `data.transcriptome.path` and `data.metabolome.path`
   - Adjust model parameters if needed

3. **Run with your config:**
   ```bash
   python scripts/run_pipeline.py --config config/my_project.yaml
   ```

## Key Configuration Sections

### Data Configuration

**REQUIRED**: You must update these sections with your own data:

```yaml
data:
  samples:
    total: 20  # Your total sample count
    groups:
      - name: "your_group_name"
        samples: ["S01", "S02", ...]  # Your sample IDs
        tissue: "your_tissue_type"
        condition: "your_condition"
```

### Model Architecture

The default architecture works for most datasets, but can be tuned:

- `latent_dim`: Dimensionality of latent space (recommend 64-256 for genes, 32-128 for metabolites)
- `hidden_dims`: List of hidden layer sizes (decreasing from input to latent)
- `dropout`: Regularization strength (0.1-0.3 typical)

### Training Parameters

- `epochs`: Increase if loss hasn't plateaued
- `learning_rate`: Decrease if training is unstable (0.0001-0.01 range)
- `batch_size`: Adjust based on GPU memory
- `early_stopping_patience`: Number of epochs to wait before stopping

## Example Configurations

### Small Dataset (<1000 genes)
```yaml
models:
  gene_autoencoder:
    latent_dim: 32
    hidden_dims: [512, 128]
```

### Large Dataset (>50,000 genes)
```yaml
models:
  gene_autoencoder:
    latent_dim: 256
    hidden_dims: [8192, 2048, 512]
```

## Notes

- All file paths can be absolute or relative to the project root
- Comments starting with `# REPLACE` indicate fields you should customize
- Keep `random_seed` fixed for reproducible results
