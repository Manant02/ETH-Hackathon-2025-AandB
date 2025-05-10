import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import jax.numpy as jnp

def load_and_visualize_states(data_dir):
    """
    Lädt alle Noisy-Wigner Pickle-Dateien aus dem angegebenen Verzeichnis und visualisiert sie.
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Verzeichnis {data_dir} existiert nicht!")
        return
    
    # Lade alle .pickle Dateien
    for file in data_path.glob('*.pickle'):
        if 'noisy_wigner' not in file.name:
            continue
            
        try:
            print(f"\nLade Datei: {file.name}")
            with open(file, 'rb') as f:
                state = pickle.load(f)
            
            print(type(state))
            print(type(state[0]))
            print(type(state[1]))
            print(np.shape(state[2]))

            print(np.max(state[0]), np.min(state[0]))
            print(np.max(state[1]), np.min(state[1]))
            print(np.max(state[2]), np.min(state[2]))

            # Konvertiere JAX-Array zu NumPy-Array falls nötig
            
            # Visualisiere den State
            plt.figure(figsize=(15, 6))
            
            # (Fake-) Probability
            plt.imshow(state[2], cmap='RdBu')
            plt.colorbar(label='P(x,p)')
            plt.title(f'P(x,p): {file.stem}')
            plt.tight_layout()
            plt.show()
            
            # Zeige zusätzliche Informationen
            print(f"Shape des States: {state[2].shape}")
            print(f"Minimaler Wert: {np.min(np.abs(state[2]))}")
            print(f"Maximaler Wert: {np.max(np.abs(state[2]))}")
            
        except Exception as e:
            print(f"Fehler beim Verarbeiten von {file}: {e}")
            continue

if __name__ == "__main__":
    # Pfad zu den Pickle-Dateien
    data_dir = "../../data/synthetic"
    load_and_visualize_states(data_dir) 