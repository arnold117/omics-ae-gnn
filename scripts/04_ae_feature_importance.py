#!/usr/bin/env python3
"""
Step 4: AutoEncoder Feature Importance + GNN Gene Mapping

Computes gene importance scores using trained AutoEncoder and GNN models,
closing the loop between deep learning and gene candidate selection.

Pipeline:
  1. Load or retrain Gene AutoEncoder (checkpoint recovery)
  2. Compute reconstruction-based gene importance
  3. Compute gradient-based gene importance
  4. Compute latent-correlation importance (fallback, no checkpoint needed)
  5. Retrain GNN and map latent importance back to gene space
  6. Export combined importance scores

Output:
  - outputs/ae_importance/gene_ae_importance.csv
  - outputs/ae_importance/gene_gnn_importance.csv
  - outputs/ae_importance/top50_ae_importance.png
  - outputs/ae_importance/importance_distribution.png
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.config_loader import ConfigLoader
from core.device_manager import DeviceManager
from core.logger import setup_logger
from models.autoencoder import AutoEncoder
from models.gnn import MultiOmicsGNN
from training.data_loader import MatrixDataset, MultiOmicsGraphDataset
from training.trainer import AutoEncoderTrainer, load_checkpoint
from analysis.explainability import AutoEncoderAnalyzer


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find latest checkpoint file in directory"""
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("best_model_epoch*.pt"))
    return checkpoints[-1] if checkpoints else None


def train_ae_if_needed(
    matrix_path: Path,
    checkpoint_dir: Path,
    configs: dict,
    device: torch.device,
    logger
) -> AutoEncoder:
    """Load AE from checkpoint or retrain if missing"""

    dataset = MatrixDataset(matrix_path)
    input_dim = dataset.n_features

    ae_params = configs['config']['models']['autoencoder']
    model = AutoEncoder(
        input_dim=input_dim,
        latent_dim=ae_params['latent_dim'],
        dropout_rate=ae_params['dropout']
    )

    # Try to load existing checkpoint
    ckpt_path = find_latest_checkpoint(checkpoint_dir)
    if ckpt_path is not None:
        logger.info(f"Loading AE checkpoint: {ckpt_path.name}")
        model, epoch, score = load_checkpoint(ckpt_path, model, device=device)
        model = model.to(device)
        return model

    # Retrain
    logger.info("No AE checkpoint found, retraining...")
    logger.info(f"Dataset: {dataset.n_samples} samples x {input_dim:,} features")

    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_batch_size = max(2, min(n_train // 2,
        configs['hardware']['training']['batch_size']['autoencoder']))
    val_batch_size = n_val

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size,
        shuffle=True, pin_memory=False, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size,
        shuffle=False, pin_memory=False
    )

    train_params = configs['config']['training']['autoencoder']
    trainer = AutoEncoderTrainer(
        model=model, device=device,
        learning_rate=train_params['learning_rate'],
        kl_weight=train_params['kl_weight']
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_params['epochs'],
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=train_params['early_stopping_patience']
    )

    logger.info("AE retraining complete")
    return model


def train_gnn_if_needed(
    gene_latent: torch.Tensor,
    metab_latent: torch.Tensor,
    metadata: pd.DataFrame,
    checkpoint_dir: Path,
    configs: dict,
    device: torch.device,
    logger
) -> MultiOmicsGNN | None:
    """Load GNN from checkpoint or retrain if missing"""

    dataset = MultiOmicsGraphDataset(
        gene_latent=gene_latent,
        metab_latent=metab_latent,
        metadata=metadata,
        similarity_threshold=configs['config']['models']['gnn']['similarity_threshold']
    )

    node_features, adjacency, labels = dataset[0]

    gnn_params = configs['config']['models']['gnn']
    model = MultiOmicsGNN(
        input_dim=node_features.shape[1],
        hidden_dims=gnn_params['hidden_dims'],
        n_heads=gnn_params['n_heads'],
        dropout=gnn_params['dropout']
    )

    # Try to load existing checkpoint
    ckpt_path = find_latest_checkpoint(checkpoint_dir)
    if ckpt_path is not None:
        logger.info(f"Loading GNN checkpoint: {ckpt_path.name}")
        model, epoch, score = load_checkpoint(ckpt_path, model, device=device)
        model = model.to(device)
        return model

    # Retrain
    logger.info("No GNN checkpoint found, retraining...")
    model = model.to(device)

    node_features_gpu = node_features.unsqueeze(0).to(device)
    adjacency_gpu = adjacency.unsqueeze(0).to(device)
    labels_gpu = {k: v.to(device) for k, v in labels.items()}

    train_params = configs['config']['training']['gnn']
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = float('inf')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(train_params['epochs']):
        model.train()
        embeddings, tissue_logits, geo_logits, cult_logits = model(node_features_gpu, adjacency_gpu)

        tissue_logits = tissue_logits.squeeze(0)
        geo_logits = geo_logits.squeeze(0)
        cult_logits = cult_logits.squeeze(0)

        total_loss = (
            train_params['task_weights']['tissue'] * criterion(tissue_logits, labels_gpu['tissue']) +
            train_params['task_weights']['geography'] * criterion(geo_logits, labels_gpu['geography']) +
            train_params['task_weights']['cultivation'] * criterion(cult_logits, labels_gpu['cultivation'])
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': model.get_config()
            }, checkpoint_dir / f"best_model_epoch{epoch+1}.pt")

        if (epoch + 1) % 50 == 0 or epoch == 0:
            with torch.no_grad():
                tissue_acc = (tissue_logits.argmax(dim=-1) == labels_gpu['tissue']).float().mean()
                geo_acc = (geo_logits.argmax(dim=-1) == labels_gpu['geography']).float().mean()
            logger.info(
                f"  Epoch {epoch+1}/{train_params['epochs']} - "
                f"loss: {total_loss.item():.4f}, "
                f"tissue: {tissue_acc:.2f}, geo: {geo_acc:.2f}"
            )

    logger.info(f"GNN retraining complete (best loss: {best_loss:.4f})")
    return model


def compute_ae_importance(
    model: AutoEncoder,
    gene_matrix: pd.DataFrame,
    device: torch.device,
    logger
) -> pd.DataFrame:
    """
    Compute gene importance using AutoEncoder reconstruction and gradient methods.

    Returns DataFrame with columns: gene, recon_importance, grad_importance, combined_ae_importance
    """
    logger.info("Computing AE-based gene importance...")

    # Prepare data: transpose to (n_samples, n_features)
    data_np = gene_matrix.T.values.astype(np.float32)
    data_tensor = torch.FloatTensor(data_np).to(device)
    gene_names = list(gene_matrix.index)

    analyzer = AutoEncoderAnalyzer(model, device)

    # Method 1: Reconstruction importance
    logger.info("  Computing reconstruction importance...")
    recon_importance = analyzer.compute_reconstruction_importance(data_tensor)
    logger.info(f"  Reconstruction importance: shape {recon_importance.shape}")

    # Method 2: Gradient importance (may fail on MPS for some ops)
    logger.info("  Computing gradient importance...")
    try:
        grad_importance = analyzer.compute_gradient_importance(
            torch.FloatTensor(data_np).to(device)
        )
        logger.info(f"  Gradient importance: shape {grad_importance.shape}")
    except Exception as e:
        logger.warning(f"  Gradient importance failed ({e}), using reconstruction only")
        grad_importance = recon_importance.copy()

    # Rank-normalize both scores
    def rank_normalize(arr):
        ranks = np.argsort(np.argsort(arr)).astype(float)
        return ranks / len(ranks)

    recon_norm = rank_normalize(recon_importance)
    grad_norm = rank_normalize(grad_importance)
    combined = (recon_norm + grad_norm) / 2.0

    result = pd.DataFrame({
        'gene': gene_names,
        'recon_importance': recon_importance,
        'grad_importance': grad_importance,
        'combined_ae_importance': combined
    })

    result = result.sort_values('combined_ae_importance', ascending=False).reset_index(drop=True)

    logger.info(f"  Top 5 AE-important genes:")
    for _, row in result.head(5).iterrows():
        logger.info(f"    {row['gene']}: {row['combined_ae_importance']:.4f}")

    return result


def compute_latent_correlation_importance(
    gene_matrix: pd.DataFrame,
    gene_latent: pd.DataFrame,
    logger
) -> pd.DataFrame:
    """
    Fallback: compute gene importance from latent representation correlations.
    For each gene j: importance[j] = sum_k(var(latent_k) * |corr(gene_j, latent_k)|)

    This method does not require model checkpoints.
    """
    logger.info("Computing latent-correlation importance (no checkpoint needed)...")

    gene_names = list(gene_matrix.index)

    # gene_matrix: (n_genes, n_samples), gene_latent: (n_samples, n_latent_dims)
    gene_vals = gene_matrix.values
    latent_vals = gene_latent.values

    # Variance of each latent dimension across samples
    latent_var = np.var(latent_vals, axis=0)

    # Standardize for correlation computation
    gene_std = (gene_vals - gene_vals.mean(axis=1, keepdims=True))
    gene_norms = np.sqrt((gene_std ** 2).sum(axis=1, keepdims=True)) + 1e-8
    gene_std = gene_std / gene_norms

    latent_std = (latent_vals - latent_vals.mean(axis=0, keepdims=True))
    latent_norms = np.sqrt((latent_std ** 2).sum(axis=0, keepdims=True)) + 1e-8
    latent_std = latent_std / latent_norms

    # Correlation matrix: (n_genes, n_samples) @ (n_samples, n_latent) = (n_genes, n_latent)
    corr_matrix = gene_std @ latent_std

    # Weighted importance: sum_k(var_k * |corr_jk|)
    importance = np.abs(corr_matrix) @ latent_var

    result = pd.DataFrame({
        'gene': gene_names,
        'latent_corr_importance': importance
    })

    result = result.sort_values('latent_corr_importance', ascending=False).reset_index(drop=True)

    logger.info(f"  Computed importance for {len(result):,} genes")
    logger.info(f"  Top 5 latent-correlation-important genes:")
    for _, row in result.head(5).iterrows():
        logger.info(f"    {row['gene']}: {row['latent_corr_importance']:.4f}")

    return result, corr_matrix


def compute_gnn_gene_importance(
    gnn_model: MultiOmicsGNN,
    gene_latent_tensor: torch.Tensor,
    metab_latent_tensor: torch.Tensor,
    adjacency: torch.Tensor,
    gene_latent_corr_matrix: np.ndarray,
    gene_names: list,
    device: torch.device,
    logger
) -> pd.DataFrame:
    """
    Map GNN latent-space importance back to individual genes.

    Strategy:
    1. Compute per-latent-dimension importance via input perturbation
    2. Map to gene space using gene-latent correlation matrix

    gnn_gene_importance[j] = sum_k(latent_importance[k] * |corr(gene_j, latent_k)|)
    """
    logger.info("Computing GNN-to-gene importance mapping...")

    gnn_model.eval()

    node_features = torch.cat([gene_latent_tensor, metab_latent_tensor], dim=-1)
    node_features_gpu = node_features.unsqueeze(0).to(device)
    adjacency_gpu = adjacency.unsqueeze(0).to(device)

    # Baseline prediction
    with torch.no_grad():
        base_emb, base_t, base_g, base_c = gnn_model(node_features_gpu, adjacency_gpu)
        base_output = torch.cat([
            base_t.squeeze(0),
            base_g.squeeze(0),
            base_c.squeeze(0)
        ], dim=-1)

    # Perturbation-based importance for gene-latent dimensions (first N dims)
    n_gene_latent = gene_latent_tensor.shape[1]
    latent_importance = np.zeros(n_gene_latent)

    for k in range(n_gene_latent):
        perturbed = node_features.clone()
        perturbed[:, k] = 0.0  # Zero out latent dimension k

        with torch.no_grad():
            perturbed_gpu = perturbed.unsqueeze(0).to(device)
            _, pt, pg, pc = gnn_model(perturbed_gpu, adjacency_gpu)
            perturbed_output = torch.cat([
                pt.squeeze(0), pg.squeeze(0), pc.squeeze(0)
            ], dim=-1)

        # Importance = change in output when dimension k is zeroed
        diff = (base_output - perturbed_output).abs().mean().item()
        latent_importance[k] = diff

    logger.info(f"  Latent importance range: [{latent_importance.min():.4f}, {latent_importance.max():.4f}]")

    # Map to gene space: (n_genes, n_latent) @ (n_latent,) = (n_genes,)
    gnn_gene_imp = np.abs(gene_latent_corr_matrix) @ latent_importance

    result = pd.DataFrame({
        'gene': gene_names,
        'gnn_importance': gnn_gene_imp
    })

    result = result.sort_values('gnn_importance', ascending=False).reset_index(drop=True)

    logger.info(f"  Top 5 GNN-important genes:")
    for _, row in result.head(5).iterrows():
        logger.info(f"    {row['gene']}: {row['gnn_importance']:.4f}")

    return result


def visualize_importance(
    ae_importance: pd.DataFrame,
    gnn_importance: pd.DataFrame | None,
    output_dir: Path,
    logger
):
    """Generate importance visualizations"""
    logger.info("Generating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Top 30 AE important genes
    ax = axes[0, 0]
    top30 = ae_importance.head(30)
    ax.barh(range(len(top30)), top30['combined_ae_importance'],
            color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels([str(g)[:25] for g in top30['gene']], fontsize=7)
    ax.set_xlabel('AE Importance Score')
    ax.set_title('Top 30 Genes by AutoEncoder Importance', fontsize=12)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')

    # Plot 2: AE importance distribution
    ax = axes[0, 1]
    ax.hist(ae_importance['combined_ae_importance'], bins=100,
            color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('AE Importance Score')
    ax.set_ylabel('Number of Genes')
    ax.set_title('AE Importance Distribution', fontsize=12)
    ax.axvline(ae_importance['combined_ae_importance'].quantile(0.99),
               color='red', linestyle='--', label='Top 1%')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Reconstruction vs Gradient importance
    ax = axes[1, 0]
    sample_idx = np.random.choice(len(ae_importance), min(5000, len(ae_importance)), replace=False)
    sample = ae_importance.iloc[sample_idx]
    ax.scatter(sample['recon_importance'], sample['grad_importance'],
               alpha=0.3, s=5, color='steelblue')
    ax.set_xlabel('Reconstruction Importance')
    ax.set_ylabel('Gradient Importance')
    ax.set_title('Reconstruction vs Gradient Importance', fontsize=12)
    ax.grid(alpha=0.3)

    # Plot 4: GNN importance (if available) or latent correlation
    ax = axes[1, 1]
    if gnn_importance is not None:
        top30_gnn = gnn_importance.head(30)
        ax.barh(range(len(top30_gnn)), top30_gnn['gnn_importance'],
                color='coral', alpha=0.7)
        ax.set_yticks(range(len(top30_gnn)))
        ax.set_yticklabels([str(g)[:25] for g in top30_gnn['gene']], fontsize=7)
        ax.set_xlabel('GNN Importance Score')
        ax.set_title('Top 30 Genes by GNN Importance', fontsize=12)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'GNN importance\nnot available',
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_title('GNN Importance (N/A)')
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plot_path = output_dir / "importance_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {plot_path}")


def main():
    print("\n")
    print("=" * 80)
    print("  Step 4: AutoEncoder Feature Importance + GNN Gene Mapping")
    print("=" * 80)
    print()

    # Load configurations
    config_loader = ConfigLoader(project_root=PROJECT_ROOT)
    configs = config_loader.load_all()

    # Setup
    log_dir = PROJECT_ROOT / configs['paths']['output']['logs']
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        name="ae_importance",
        log_file=log_dir / "04_ae_feature_importance.log",
        level=20
    )

    device_manager = DeviceManager.from_config(configs['hardware'])
    device = device_manager.get_device()
    logger.info(f"Device: {device}")

    matrices_dir = PROJECT_ROOT / configs['paths']['processed']['matrices']
    latent_dir = PROJECT_ROOT / configs['paths']['processed']['latent']
    output_dir = PROJECT_ROOT / "outputs/ae_importance"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # =====================================================================
        # PART 1: AutoEncoder Gene Importance
        # =====================================================================
        logger.info("")
        logger.info("=" * 80)
        logger.info("PART 1: AUTOENCODER GENE IMPORTANCE")
        logger.info("=" * 80)

        gene_matrix_path = matrices_dir / "gene_expression_matrix.csv"
        gene_matrix = pd.read_csv(gene_matrix_path, index_col=0)
        logger.info(f"Gene matrix: {gene_matrix.shape}")

        # Load or retrain AE
        gene_ckpt_dir = PROJECT_ROOT / configs['paths']['output']['checkpoints'] / "gene"
        ae_model = train_ae_if_needed(
            gene_matrix_path, gene_ckpt_dir, configs, device, logger
        )

        # Compute AE importance
        ae_importance = compute_ae_importance(ae_model, gene_matrix, device, logger)

        # Save
        ae_output_path = output_dir / "gene_ae_importance.csv"
        ae_importance.to_csv(ae_output_path, index=False)
        logger.info(f"Saved AE importance: {ae_output_path}")

        # =====================================================================
        # PART 2: Latent-Correlation Importance (fallback method)
        # =====================================================================
        logger.info("")
        logger.info("=" * 80)
        logger.info("PART 2: LATENT-CORRELATION IMPORTANCE")
        logger.info("=" * 80)

        gene_latent_path = latent_dir / "gene_latent.csv"
        gene_latent_df = pd.read_csv(gene_latent_path, index_col=0)

        latent_corr_importance, gene_latent_corr_matrix = \
            compute_latent_correlation_importance(gene_matrix, gene_latent_df, logger)

        latent_corr_path = output_dir / "gene_latent_corr_importance.csv"
        latent_corr_importance.to_csv(latent_corr_path, index=False)
        logger.info(f"Saved latent-correlation importance: {latent_corr_path}")

        # =====================================================================
        # PART 3: GNN Gene Importance Mapping
        # =====================================================================
        logger.info("")
        logger.info("=" * 80)
        logger.info("PART 3: GNN GENE IMPORTANCE MAPPING")
        logger.info("=" * 80)

        gnn_importance = None

        metab_latent_path = latent_dir / "metabolite_latent.csv"
        metadata_path = matrices_dir / "sample_metadata.csv"

        if metab_latent_path.exists() and metadata_path.exists():
            gene_latent_tensor = torch.FloatTensor(gene_latent_df.values)
            metab_latent_df = pd.read_csv(metab_latent_path, index_col=0)
            metab_latent_tensor = torch.FloatTensor(metab_latent_df.values)
            metadata = pd.read_csv(metadata_path)

            # Load or retrain GNN
            gnn_ckpt_dir = PROJECT_ROOT / configs['paths']['output']['checkpoints'] / "gnn"
            gnn_model = train_gnn_if_needed(
                gene_latent_tensor, metab_latent_tensor, metadata,
                gnn_ckpt_dir, configs, device, logger
            )

            if gnn_model is not None:
                # Build adjacency for inference
                dataset = MultiOmicsGraphDataset(
                    gene_latent=gene_latent_tensor,
                    metab_latent=metab_latent_tensor,
                    metadata=metadata,
                    similarity_threshold=configs['config']['models']['gnn']['similarity_threshold']
                )
                _, adjacency, _ = dataset[0]

                gnn_importance = compute_gnn_gene_importance(
                    gnn_model, gene_latent_tensor, metab_latent_tensor,
                    adjacency, gene_latent_corr_matrix,
                    list(gene_matrix.index), device, logger
                )

                gnn_output_path = output_dir / "gene_gnn_importance.csv"
                gnn_importance.to_csv(gnn_output_path, index=False)
                logger.info(f"Saved GNN importance: {gnn_output_path}")
        else:
            logger.warning("Metabolite latent or metadata not found, skipping GNN analysis")

        # =====================================================================
        # PART 4: Visualization
        # =====================================================================
        logger.info("")
        logger.info("=" * 80)
        logger.info("PART 4: VISUALIZATION")
        logger.info("=" * 80)

        visualize_importance(ae_importance, gnn_importance, output_dir, logger)

        # =====================================================================
        # Summary
        # =====================================================================
        logger.info("")
        logger.info("=" * 80)
        logger.info("FEATURE IMPORTANCE ANALYSIS COMPLETE!")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Outputs:")
        logger.info(f"  AE importance: {output_dir / 'gene_ae_importance.csv'}")
        logger.info(f"  Latent-corr importance: {output_dir / 'gene_latent_corr_importance.csv'}")
        if gnn_importance is not None:
            logger.info(f"  GNN importance: {output_dir / 'gene_gnn_importance.csv'}")
        logger.info(f"  Visualization: {output_dir / 'importance_analysis.png'}")
        logger.info("")
        logger.info("Next: run scripts/05_multi_evidence_ranking.py")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
