"""
Plot training curves from training history.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def plot_training_curves(history_path, output_path=None):
    """
    Plot training curves from history file.
    
    Args:
        history_path: Path to training_history.json
        output_path: Path to save plot (if None, saves to same directory as history)
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    learning_rate = history['learning_rate']
    field_errors = history['field_errors']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Loss curves (train and validation)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # 2. Learning rate
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epochs, learning_rate, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Field errors (only epochs where errors were computed)
    ax3 = plt.subplot(2, 2, 3)
    field_colors = {
        'U': 'blue',
        'p': 'red',
        'k': 'green',
        'epsilon': 'orange',
        'nut': 'purple'
    }
    
    for field, errors in field_errors.items():
        # Filter out None values and get corresponding epochs
        valid_errors = [(e, epochs[i]) for i, e in enumerate(errors) if e is not None]
        if valid_errors:
            error_vals, error_epochs = zip(*valid_errors)
            ax3.plot(error_epochs, error_vals, 'o-', label=field, 
                    color=field_colors.get(field, 'black'), linewidth=2, markersize=4)
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Field Error', fontsize=12)
    ax3.set_title('Per-Field Errors (computed every 10 epochs)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Loss difference (overfitting indicator)
    ax4 = plt.subplot(2, 2, 4)
    loss_diff = np.array(val_loss) - np.array(train_loss)
    ax4.plot(epochs, loss_diff, 'm-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Val Loss - Train Loss', fontsize=12)
    ax4.set_title('Overfitting Indicator', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.fill_between(epochs, 0, loss_diff, where=(loss_diff > 0), alpha=0.3, color='red', label='Overfitting')
    ax4.fill_between(epochs, 0, loss_diff, where=(loss_diff <= 0), alpha=0.3, color='green', label='Underfitting')
    ax4.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        history_file = Path(history_path)
        output_path = history_file.parent / 'training_curves.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {output_path}")
    plt.close()


def plot_field_errors_detailed(history_path, output_path=None):
    """
    Create detailed plot of field errors.
    
    Args:
        history_path: Path to training_history.json
        output_path: Path to save plot (if None, saves to same directory as history)
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = history['epoch']
    field_errors = history['field_errors']
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    field_colors = {
        'U': 'blue',
        'p': 'red',
        'k': 'green',
        'epsilon': 'orange',
        'nut': 'purple'
    }
    
    field_names = ['U', 'p', 'k', 'epsilon', 'nut']
    
    for idx, field in enumerate(field_names):
        ax = axes[idx]
        errors = field_errors[field]
        
        # Filter out None values
        valid_errors = [(e, epochs[i]) for i, e in enumerate(errors) if e is not None]
        if valid_errors:
            error_vals, error_epochs = zip(*valid_errors)
            ax.plot(error_epochs, error_vals, 'o-', color=field_colors.get(field, 'black'), 
                   linewidth=2, markersize=5)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Error', fontsize=11)
            ax.set_title(f'{field} Error', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, f'No data for {field}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{field} Error', fontsize=12, fontweight='bold')
    
    # Remove last subplot if odd number of fields
    if len(field_names) < 6:
        axes[-1].remove()
    
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        history_file = Path(history_path)
        output_path = history_file.parent / 'field_errors_detailed.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Field errors plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--history', type=str, default='checkpoints/training_history.json',
                       help='Path to training_history.json file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (default: same directory as history)')
    parser.add_argument('--detailed', action='store_true',
                       help='Also create detailed field errors plot')
    
    args = parser.parse_args()
    
    if not Path(args.history).exists():
        print(f"Error: History file not found: {args.history}")
        print("Make sure you've run training first, which saves training_history.json")
        return
    
    print(f"Loading training history from {args.history}...")
    plot_training_curves(args.history, args.output)
    
    if args.detailed:
        plot_field_errors_detailed(args.history)


if __name__ == '__main__':
    main()
