import os
import io
import numpy as np
import matplotlib.pyplot as plt
import gcsfs

GCS_BUCKET = "thesis-project-bucket-rh-ucl"
MONITORING_DIR = f"gs://{GCS_BUCKET}/monitoring"

def get_fs():
    """Return an authenticated GCSFileSystem."""
    return gcsfs.GCSFileSystem(project=GCS_BUCKET)


def load_npy_from_gcs(gcs_path: str) -> np.ndarray:
    """
    Load a .npy file directly from GCS into memory.
    gcs_path: full gs:// path, e.g. 'gs://bucket/folder/file.npy'
    """
    fs = get_fs()
    with fs.open(gcs_path, 'rb') as f:
        data = np.load(f)
    print(f"Loaded {gcs_path} — shape: {data.shape}")
    return data


def save_plot_to_gcs(fig, filename: str):
    """
    Save a matplotlib figure to GCS monitoring folder.
    filename: e.g. 'loss_epoch_010.png'
    """
    fs = get_fs()
    gcs_path = f"{MONITORING_DIR}/{filename}"
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    with fs.open(gcs_path, 'wb') as f:
        f.write(buf.read())
    print(f"Saved plot to {gcs_path}")


def plot_losses(train_losses, val_losses, epoch, save_to_gcs=True):
    """
    Plot train and val losses. Displays inline and optionally saves to GCS.
    Called every epoch from the training loop.
    """
    from IPython.display import clear_output
    clear_output(wait=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label='train', color='steelblue')
    ax.plot(val_losses,   label='val',   color='coral')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'DDPM Training — Epoch {epoch}')
    ax.legend()
    plt.tight_layout()
    plt.show()

    if save_to_gcs and len(train_losses) > 0:
        save_plot_to_gcs(fig, f'loss_epoch_{epoch:04d}.png')

    plt.close(fig)


def save_final_loss_plot(train_losses, val_losses):
    """Save the final loss curve to GCS at end of training."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label='train', color='steelblue')
    ax.plot(val_losses,   label='val',   color='coral')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('DDPM Training — Final')
    ax.legend()
    plt.tight_layout()
    save_plot_to_gcs(fig, 'loss_final.png')
    plt.close(fig)
    print("Final loss plot saved to GCS.")
