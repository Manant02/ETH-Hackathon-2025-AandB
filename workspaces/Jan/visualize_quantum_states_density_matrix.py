import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
            
            print(np.shape(state))
            print(state[40][20])
            # Konvertiere JAX-Array zu NumPy-Array falls nötig
            if hasattr(state, '_value'):
                state = np.array(state._value)
            elif isinstance(state, tuple):
                state = np.array(state[0])
            
            # Visualisiere den State
            plt.figure(figsize=(15, 6))
            


            
            # Realteil
            plt.subplot(1, 2, 1)
            # Normalisiere den Realteil
            real_state = np.real(state)
            real_state_norm = real_state / np.max(np.abs(real_state))
            plt.imshow(real_state_norm, cmap='RdBu')
            plt.colorbar(label='Realteil (normalisiert)')
            plt.title(f'Realteil: {file.stem}')
            
            # Imaginärteil
            plt.subplot(1, 2, 2)
            imag_state = np.imag(state)
            imag_state_norm = imag_state / np.max(np.abs(imag_state))
            plt.imshow(np.imag(imag_state_norm), cmap='RdBu')
            plt.colorbar(label='Imaginärteil')
            plt.title(f'Imaginärteil: {file.stem}')
            
            plt.tight_layout()
            plt.show()
            
            # Zeige zusätzliche Informationen
            print(f"Shape des States: {state.shape}")
            print(f"Minimaler Wert: {np.min(np.abs(state))}")
            print(f"Maximaler Wert: {np.max(np.abs(state))}")
            
        except Exception as e:
            print(f"Fehler beim Verarbeiten von {file}: {e}")

if __name__ == "__main__":
    # Pfad zu den Pickle-Dateien
    data_dir = "../../data/synthetic"
    load_and_visualize_states(data_dir) 