#!/usr/bin/env python3
"""
Step 2: Model Training

Trains AutoEncoders and GNN for multi-omics integration.

Pipeline:
  1. Train gene expression AutoEncoder (132K → 64-dim latent)
  2. Train metabolite AutoEncoder (7K → 64-dim latent)
  3. Extract latent representations
  4. Train GNN on combined latents (gene + metabolite → sample embeddings)
  5. Save trained models and latent representations

Hardware:
  - Uses MPS acceleration on macOS (Metal Performance Shaders)
  - Can switch to CUDA by changing hardware.yaml
  - Automatically falls back to CPU if needed
"""

import sys
from pathlib import Path
import pandas as pd
import torch

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.config_loader import ConfigLoader
from core.device_manager import DeviceManager
from core.logger import setup_logger
from models.autoencoder import AutoEncoder
from models.gnn import MultiOmicsGNN
from training.data_loader import (
    create_autoencoder_dataloader,
    create_graph_from_latents,
    MatrixDataset
)
from training.trainer import AutoEncoderTrainer, load_checkpoint


def train_autoencoder(
    matrix_path: Path,
    output_name: str,
    configs: dict,
    device_manager: DeviceManager,
    logger
) -> tuple:
    """
    Train AutoEncoder for dimensionality reduction

    Args:
        matrix_path: Path to processed matrix
        output_name: Output prefix (e.g., "gene" or "metabolite")
        configs: Configuration dict
        device_manager: DeviceManager instance
        logger: Logger instance

    Returns:
        Tuple of (model, latent_representations)
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"TRAINING {output_name.upper()} AUTOENCODER")
    logger.info("=" * 80)

    # Load data
    dataset = MatrixDataset(matrix_path)
    input_dim = dataset.n_features

    # Create dataloaders (use all data for training, small dataset)
    # Split: 80% train, 20% validation
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # For small datasets, use appropriate batch sizes
    # BatchNorm requires batch_size >= 2
    train_batch_size = max(2, min(n_train // 2, configs['hardware']['training']['batch_size']['autoencoder']))
    val_batch_size = n_val  # Use full validation set at once (model.eval() mode)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=False,  # MPS doesn't support pin_memory
        drop_last=True  # Drop last incomplete batch to avoid BatchNorm issues
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=False
    )

    logger.info(f"Dataset: {len(dataset)} samples ({n_train} train, {n_val} val)")
    logger.info(f"Input dimension: {input_dim:,}")
    logger.info(f"Batch sizes: train={train_batch_size}, val={val_batch_size}")

    # Create model
    ae_params = configs['config']['models']['autoencoder']
    model = AutoEncoder(
        input_dim=input_dim,
        latent_dim=ae_params['latent_dim'],
        dropout_rate=ae_params['dropout']
    )

    device = device_manager.get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Create trainer
    train_params = configs['config']['training']['autoencoder']
    trainer = AutoEncoderTrainer(
        model=model,
        device=device,
        learning_rate=train_params['learning_rate'],
        kl_weight=train_params['kl_weight']
    )

    # Train
    checkpoint_dir = PROJECT_ROOT / configs['paths']['output']['checkpoints'] / output_name
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_params['epochs'],
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=train_params['early_stopping_patience']
    )

    # Save training history
    history_dir = PROJECT_ROOT / configs['paths']['output']['logs']
    trainer.save_history(
        history,
        history_dir / f"{output_name}_autoencoder_history.json"
    )

    # Extract latent representations (use full dataset)
    logger.info("")
    logger.info("Extracting latent representations...")
    model.eval()
    with torch.no_grad():
        full_data = dataset.get_full_data().to(device)
        latent = model.encode(full_data)
        latent = latent.cpu()

    logger.info(f"  Latent shape: {latent.shape}")

    return model, latent


def train_gnn(
    gene_latent: torch.Tensor,
    metab_latent: torch.Tensor,
    metadata: pd.DataFrame,
    configs: dict,
    device_manager: DeviceManager,
    logger
) -> torch.nn.Module:
    """
    Train GNN for multi-omics integration

    Args:
        gene_latent: Gene latent representations
        metab_latent: Metabolite latent representations
        metadata: Sample metadata
        configs: Configuration dict
        device_manager: DeviceManager instance
        logger: Logger instance

    Returns:
        Trained GNN model
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING MULTI-OMICS GNN")
    logger.info("=" * 80)

    # Create graph dataset
    from training.data_loader import MultiOmicsGraphDataset

    dataset = MultiOmicsGraphDataset(
        gene_latent=gene_latent,
        metab_latent=metab_latent,
        metadata=metadata,
        similarity_threshold=configs['config']['models']['gnn']['similarity_threshold']
    )

    # Get graph data
    node_features, adjacency, labels = dataset[0]

    logger.info(f"Graph: {node_features.shape[0]} nodes × {node_features.shape[1]} features")
    logger.info(f"Adjacency density: {adjacency.sum() / (adjacency.shape[0] ** 2):.2%}")

    # Create model
    gnn_params = configs['config']['models']['gnn']
    model = MultiOmicsGNN(
        input_dim=node_features.shape[1],
        hidden_dims=gnn_params['hidden_dims'],
        n_heads=gnn_params['n_heads'],
        dropout=gnn_params['dropout']
    )

    device = device_manager.get_device()
    model = model.to(device)

    logger.info(f"Device: {device}")
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Move data to device
    node_features = node_features.unsqueeze(0).to(device)  # Add batch dimension
    adjacency = adjacency.unsqueeze(0).to(device)
    labels = {k: v.to(device) for k, v in labels.items()}

    # Training setup
    train_params = configs['config']['training']['gnn']
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    logger.info("")
    logger.info(f"Training for {train_params['epochs']} epochs...")

    best_loss = float('inf')
    checkpoint_dir = PROJECT_ROOT / configs['paths']['output']['checkpoints'] / 'gnn'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(train_params['epochs']):
        model.train()

        # Forward pass
        embeddings, tissue_logits, geo_logits, cult_logits = model(node_features, adjacency)

        # Remove batch dimension for loss calculation
        tissue_logits = tissue_logits.squeeze(0)
        geo_logits = geo_logits.squeeze(0)
        cult_logits = cult_logits.squeeze(0)

        # Multi-task loss
        loss_tissue = criterion(tissue_logits, labels['tissue'])
        loss_geo = criterion(geo_logits, labels['geography'])
        loss_cult = criterion(cult_logits, labels['cultivation'])

        # Weighted combination
        total_loss = (
            train_params['task_weights']['tissue'] * loss_tissue +
            train_params['task_weights']['geography'] * loss_geo +
            train_params['task_weights']['cultivation'] * loss_cult
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Calculate accuracies
            with torch.no_grad():
                tissue_acc = (tissue_logits.argmax(dim=-1) == labels['tissue']).float().mean()
                geo_acc = (geo_logits.argmax(dim=-1) == labels['geography']).float().mean()
                cult_acc = (cult_logits.argmax(dim=-1) == labels['cultivation']).float().mean()

            logger.info(
                f"Epoch {epoch+1}/{train_params['epochs']} - "
                f"loss: {total_loss.item():.4f}, "
                f"tissue_acc: {tissue_acc:.3f}, "
                f"geo_acc: {geo_acc:.3f}, "
                f"cult_acc: {cult_acc:.3f}"
            )

        # Save best model
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            checkpoint_path = checkpoint_dir / f"best_model_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': model.get_config()
            }, checkpoint_path)

    logger.info("")
    logger.info(f"Best loss: {best_loss:.4f}")

    return model


def main():
    """Main execution function"""

    # Load configurations
    print("Loading configurations...")
    config_loader = ConfigLoader(project_root=PROJECT_ROOT)
    configs = config_loader.load_all()

    # Setup logging
    log_dir = PROJECT_ROOT / configs['paths']['output']['logs']
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        name="model_training",
        log_file=log_dir / "02_train_models.log",
        level=20  # INFO
    )

    logger.info("=" * 80)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info("")

    # Initialize device manager
    device_manager = DeviceManager.from_config(configs['hardware'])
    logger.info(f"Using device: {device_manager.get_device()}")
    logger.info("")

    # Create output directories
    matrices_dir = PROJECT_ROOT / configs['paths']['processed']['matrices']
    latent_dir = PROJECT_ROOT / configs['paths']['processed']['latent']
    latent_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Train gene expression AutoEncoder
        gene_matrix_path = matrices_dir / "gene_expression_matrix.csv"
        gene_model, gene_latent = train_autoencoder(
            matrix_path=gene_matrix_path,
            output_name="gene",
            configs=configs,
            device_manager=device_manager,
            logger=logger
        )

        # Save gene latent
        gene_latent_path = latent_dir / "gene_latent.csv"
        gene_samples = pd.read_csv(gene_matrix_path, index_col=0).columns
        pd.DataFrame(
            gene_latent.numpy(),
            index=gene_samples,
            columns=[f"latent_{i}" for i in range(gene_latent.shape[1])]
        ).to_csv(gene_latent_path)
        logger.info(f"Saved gene latent to: {gene_latent_path}")

        # 2. Train metabolite AutoEncoder
        metab_matrix_path = matrices_dir / "metabolite_matrix.csv"
        metab_model, metab_latent = train_autoencoder(
            matrix_path=metab_matrix_path,
            output_name="metabolite",
            configs=configs,
            device_manager=device_manager,
            logger=logger
        )

        # Save metabolite latent
        metab_latent_path = latent_dir / "metabolite_latent.csv"
        metab_samples = pd.read_csv(metab_matrix_path, index_col=0).columns
        pd.DataFrame(
            metab_latent.numpy(),
            index=metab_samples,
            columns=[f"latent_{i}" for i in range(metab_latent.shape[1])]
        ).to_csv(metab_latent_path)
        logger.info(f"Saved metabolite latent to: {metab_latent_path}")

        # 3. Train GNN
        metadata_path = matrices_dir / "sample_metadata.csv"
        metadata = pd.read_csv(metadata_path)

        gnn_model = train_gnn(
            gene_latent=gene_latent,
            metab_latent=metab_latent,
            metadata=metadata,
            configs=configs,
            device_manager=device_manager,
            logger=logger
        )

        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("MODEL TRAINING COMPLETE!")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Trained models:")
        logger.info(f"  Gene AutoEncoder: {gene_model.count_parameters():,} parameters")
        logger.info(f"    Latent dim: {gene_latent.shape}")
        logger.info(f"  Metabolite AutoEncoder: {metab_model.count_parameters():,} parameters")
        logger.info(f"    Latent dim: {metab_latent.shape}")
        logger.info(f"  Multi-Omics GNN: {gnn_model.count_parameters():,} parameters")
        logger.info("")
        logger.info("Outputs:")
        logger.info(f"  Gene latent: {gene_latent_path}")
        logger.info(f"  Metabolite latent: {metab_latent_path}")
        logger.info(f"  Checkpoints: {PROJECT_ROOT / configs['paths']['output']['checkpoints']}")
        logger.info("")
        logger.info("✓ Step 2 complete! Ready for Step 3 (analysis)")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error("=" * 80)
        logger.error("ERROR: Model training failed")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.error("")

        import traceback
        logger.error("Traceback:")
        logger.error(traceback.format_exc())

        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
