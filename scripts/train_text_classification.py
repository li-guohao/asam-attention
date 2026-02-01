"""
Training Script for Text Classification with ASAM
==================================================

Trains ASAM models on long document classification tasks.
Includes: IMDB, ArXiv, ListOps, and custom datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from asam import ASAMLayer, ASAMConfig, ASAMEncoder
from datasets.text_dataset import get_dataloader, SimpleCharTokenizer


class TextClassifier(nn.Module):
    """Text classification model with ASAM."""
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_length: int = 4096,
        dropout: float = 0.1,
        pattern_type: str = "hierarchical",
        use_adaptive_gate: bool = True,
        pooling: str = "mean"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.max_length = max_length
        self.pooling = pooling
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, max_length, dim) * 0.02)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Encoder
        config = ASAMConfig(
            dim=dim,
            num_heads=num_heads,
            dim_head=dim // num_heads,
            dropout=dropout,
            pattern_type=pattern_type,
            use_adaptive_gate=use_adaptive_gate,
        )
        
        self.encoder = ASAMEncoder(config, num_layers=num_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len]
        seq_len = x.size(1)
        
        # Embedding
        x = self.token_embedding(x)
        x = x + self.position_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Encoder
        x = self.encoder(x)
        
        # Pooling
        if self.pooling == "mean":
            x = x.mean(dim=1)
        elif self.pooling == "max":
            x = x.max(dim=1)[0]
        elif self.pooling == "first":
            x = x[:, 0]
        elif self.pooling == "last":
            x = x[:, -1]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classification
        x = self.norm(x)
        logits = self.classifier(x)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Trainer:
    """Training manager."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        args
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args
        
        self.criterion = nn.CrossEntropyLoss()
        self.writer = SummaryWriter(args.log_dir) if args.log_dir else None
        
        self.best_val_acc = 0.0
        self.global_step = 0
        
        # Create checkpoint directory
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        start_time = time.time()
        
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            logits = self.model(inputs)
            loss = self.criterion(logits, labels)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.max_grad_norm
                )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * inputs.size(0)
            pred = logits.argmax(dim=-1)
            total_correct += (pred == labels).sum().item()
            total_samples += inputs.size(0)
            
            # Logging
            if self.writer and batch_idx % self.args.log_interval == 0:
                self.writer.add_scalar('train/loss_step', loss.item(), self.global_step)
                self.writer.add_scalar('train/acc_step', (pred == labels).float().mean().item(), self.global_step)
            
            self.global_step += 1
            
            if batch_idx % self.args.print_interval == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        if self.scheduler:
            self.scheduler.step()
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'time': epoch_time
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for inputs, labels in self.val_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(inputs)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item() * inputs.size(0)
            pred = logits.argmax(dim=-1)
            total_correct += (pred == labels).sum().item()
            total_samples += inputs.size(0)
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        # Logging
        if self.writer:
            self.writer.add_scalar('val/loss', avg_loss, epoch)
            self.writer.add_scalar('val/accuracy', avg_acc, epoch)
        
        # Save best model
        if avg_acc > self.best_val_acc:
            self.best_val_acc = avg_acc
            self.save_checkpoint('best_model.pt', epoch, avg_acc)
            print(f"✓ New best model saved with accuracy: {avg_acc:.4f}")
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc
        }
    
    def save_checkpoint(self, filename: str, epoch: int, accuracy: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'args': vars(self.args)
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = os.path.join(self.args.checkpoint_dir, filename)
        torch.save(checkpoint, path)
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"Time: {train_metrics['time']:.2f}s")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"Best: {self.best_val_acc:.4f}")
            print()
            
            # Save checkpoint
            if epoch % self.args.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch, val_metrics['accuracy'])
        
        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.4f}")
        
        if self.writer:
            self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train ASAM on text classification')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='listops',
                        choices=['imdb', 'arxiv', 'listops', 'synthetic'],
                        help='Dataset name')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model
    parser.add_argument('--dim', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--pattern_type', type=str, default='hierarchical',
                        choices=['local', 'strided', 'random', 'clustered', 'hierarchical'],
                        help='Sparse attention pattern type')
    parser.add_argument('--use_adaptive_gate', action='store_true', default=True,
                        help='Use adaptive gating')
    parser.add_argument('--no_adaptive_gate', dest='use_adaptive_gate', action='store_false')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='runs',
                        help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval')
    parser.add_argument('--print_interval', type=int, default=50,
                        help='Print interval')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Checkpoint save interval')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Dataset info
    dataset_info = {
        'imdb': {'vocab_size': 256, 'num_classes': 2},
        'arxiv': {'vocab_size': 30000, 'num_classes': 8},
        'listops': {'vocab_size': 30, 'num_classes': 10},
        'synthetic': {'vocab_size': 100, 'num_classes': 100},
    }
    
    info = dataset_info[args.dataset]
    
    # Data loaders
    print(f"Loading {args.dataset} dataset...")
    train_loader = get_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        split='train',
        num_workers=args.num_workers
    )
    
    val_loader = get_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        split='test',
        num_workers=args.num_workers
    )
    
    # Model
    print("Creating model...")
    model = TextClassifier(
        vocab_size=info['vocab_size'],
        num_classes=info['num_classes'],
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_length=args.max_length,
        dropout=args.dropout,
        pattern_type=args.pattern_type,
        use_adaptive_gate=args.use_adaptive_gate,
    )
    model = model.to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Cosine annealing with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return 0.5 * (1 + torch.cos(torch.tensor((step - args.warmup_steps) / (len(train_loader) * args.epochs - args.warmup_steps) * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Trainer
    trainer = Trainer(
        model, train_loader, val_loader,
        optimizer, scheduler, device, args
    )
    
    # Train
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
