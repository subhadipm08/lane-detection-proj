import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.menu import select_model

def plot_history():
    base_dir = '/home/subha08/my_projects/lane-detection-proj'
    model_name = select_model()
    
    csv_path = os.path.join(base_dir, 'logs', model_name, 'history.csv')
    out_dir = os.path.join(base_dir, 'output_visualizations', model_name)
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    # Read the CSV. Note: Because we changed metric names mid-run, 
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    # Filter out any duplicate header rows that might have been appended
    df = df[df['epoch'] != 'epoch']
    
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
        
    # Standardize column names (map old _metric names to new short names if they exist)
    rename_map = {
        'f1_metric': 'f1', 'val_f1_metric': 'val_f1',
        'iou_metric': 'iou', 'val_iou_metric': 'val_iou',
        'fnr_metric': 'fnr', 'val_fnr_metric': 'val_fnr',
        'fpr_metric': 'fpr', 'val_fpr_metric': 'val_fpr'
    }
    df.rename(columns=rename_map, inplace=True)
    
    epochs = df['epoch']
    
    # Set up the plot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Loss
    ax = axes[0, 0]
    ax.plot(epochs, df['loss'], label='Train Loss', color='blue', linewidth=2)
    if 'val_loss' in df.columns:
        ax.plot(epochs, df['val_loss'], label='Val Loss', color='orange', linewidth=2)
    ax.set_title('Training & Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (Combo)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: IoU
    ax = axes[0, 1]
    if 'iou' in df.columns:
        ax.plot(epochs, df['iou'], label='Train IoU', color='green', linewidth=2)
    if 'val_iou' in df.columns:
        ax.plot(epochs, df['val_iou'], label='Val IoU', color='lightgreen', linewidth=2)
    ax.set_title('Intersection over Union (IoU)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: F1 Score
    ax = axes[1, 0]
    if 'f1' in df.columns:
        ax.plot(epochs, df['f1'], label='Train F1', color='purple', linewidth=2)
    if 'val_f1' in df.columns:
        ax.plot(epochs, df['val_f1'], label='Val F1', color='violet', linewidth=2)
    ax.set_title('F1 Score')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: FNR & FPR
    ax = axes[1, 1]
    if 'fnr' in df.columns:
        ax.plot(epochs, df['fnr'], label='Train FNR (Missed Lanes)', color='red', linestyle='-', linewidth=2)
    if 'val_fnr' in df.columns:
        ax.plot(epochs, df['val_fnr'], label='Val FNR', color='salmon', linestyle='--', linewidth=2)
        
    if 'fpr' in df.columns:
        ax.plot(epochs, df['fpr'], label='Train FPR (Fake Lanes)', color='blue', linestyle='-', linewidth=2)
    if 'val_fpr' in df.columns:
        ax.plot(epochs, df['val_fpr'], label='Val FPR', color='cyan', linestyle='--', linewidth=2)
        
    ax.set_title('False Negative vs False Positive Rates')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rate')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'training_history_plot.png')
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved successfully to: {save_path}")

if __name__ == '__main__':
    plot_history()
