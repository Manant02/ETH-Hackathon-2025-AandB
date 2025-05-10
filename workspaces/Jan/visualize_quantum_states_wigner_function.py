import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dynamiqs import wigner
import jax.numpy as jnp

def load_and_visualize_states(data_dir):
    """
    Lädt alle Quantum-State Pickle-Dateien aus dem angegebenen Verzeichnis und visualisiert sie.
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Verzeichnis {data_dir} existiert nicht!")
        return
    
    # Lade alle .pickle Dateien
    for file in data_path.glob('*.pickle'):
        if 'quantum_state' not in file.name:
            continue
            
        try:
            print(f"\nLade Datei: {file.name}")
            with open(file, 'rb') as f:
                state = pickle.load(f)
            
            # Visualisiere den State
            wigner_function = wigner(state, xmax=6, ymax=6, npixels=50)
            plt.contourf(wigner_function[0], wigner_function[1], wigner_function[2], cmap='seismic', vmax=np.pi/2, vmin=-np.pi/2)
            plt.tight_layout()
            plt.show()
            
            # Zeige zusätzliche Informationen
            #print(f"Shape des States: {state.shape}")
            #print(f"Minimaler Wert: {np.min(np.abs(state))}")
            #print(f"Maximaler Wert: {np.max(np.abs(state))}")
            
        except Exception as e:
            print(f"Fehler beim Verarbeiten von {file}: {e}")

if __name__ == "__main__":
    # Pfad zu den Pickle-Dateien
    data_dir = "../../data/synthetic"
    load_and_visualize_states(data_dir) 