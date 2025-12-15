"""
Synthetic data generation for information thermodynamics aging framework.
"""
import pandas as pd
import numpy as np
from .aging_model import InformationThermodynamicsModel

class AgingDataGenerator:
    """
    Generate synthetic aging data based on the information thermodynamics model.
    """
    
    def __init__(self, model_params=None, noise_level=0.05, random_seed=42):
        """
        Initialize data generator.
        
        Parameters:
        - model_params: dict, parameters for the aging model
        - noise_level: float, level of Gaussian noise to add (fraction of signal)
        - random_seed: int, random seed for reproducibility
        """
        self.model = InformationThermodynamicsModel(model_params)
        self.noise_level = noise_level
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_synthetic_data(self, include_intervention=False):
        """
        Generate comprehensive synthetic aging dataset.
        
        Parameters:
        - include_intervention: bool, whether to include intervention scenario
        
        Returns:
        - pandas DataFrame with synthetic aging data
        """
        # Generate baseline aging data
        baseline = self.model.simulate()
        
        # Add noise to simulate biological variability
        noisy_I = self._add_noise(baseline['information_fidelity'], self.noise_level)
        noisy_E = self._add_noise(baseline['error_correction'], self.noise_level)
        noisy_D = self._add_noise(baseline['damage'], self.noise_level)
        noisy_S = self._add_noise(baseline['entropy_production'], self.noise_level * 0.5)
        
        # Create baseline dataframe
        df_baseline = pd.DataFrame({
            'age': baseline['time'],
            'information_fidelity': noisy_I,
            'error_correction': noisy_E,
            'molecular_damage': noisy_D,
            'entropy_production': noisy_S,
            'scenario': 'baseline'
        })
        
        if include_intervention:
            # Generate intervention data
            intervention = self.model.simulate_intervention()
            noisy_I_int = self._add_noise(intervention['information_fidelity'], self.noise_level)
            noisy_E_int = self._add_noise(intervention['error_correction'], self.noise_level)
            noisy_D_int = self._add_noise(intervention['damage'], self.noise_level)
            noisy_S_int = self._add_noise(intervention['entropy_production'], self.noise_level * 0.5)
            
            df_intervention = pd.DataFrame({
                'age': intervention['time'],
                'information_fidelity': noisy_I_int,
                'error_correction': noisy_E_int,
                'molecular_damage': noisy_D_int,
                'entropy_production': noisy_S_int,
                'scenario': 'intervention'
            })
            
            # Combine datasets
            df = pd.concat([df_baseline, df_intervention], ignore_index=True)
        else:
            df = df_baseline
        
        return df
    
    def _add_noise(self, signal, noise_level):
        """Add Gaussian noise to signal."""
        noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
        noisy_signal = signal + noise
        # Ensure non-negative values for physical quantities
        noisy_signal = np.maximum(noisy_signal, 0)
        return noisy_signal
    
    def save_data(self, filename, include_intervention=False):
        """Save synthetic data to CSV file."""
        data = self.generate_synthetic_data(include_intervention)
        data.to_csv(filename, index=False)
        print(f"Synthetic data saved to {filename}")
        return data
    
    def load_data(self, filename):
        """Load synthetic data from CSV file."""
        return pd.read_csv(filename)