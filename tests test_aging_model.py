"""
Unit tests for the aging model.
"""
import unittest
import numpy as np
import pandas as pd
from src.aging_model import InformationThermodynamicsModel
from src.data_generator import AgingDataGenerator
from src.validation import AgingValidation

class TestAgingModel(unittest.TestCase):
    
    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = InformationThermodynamicsModel()
        self.assertIsInstance(model.params, dict)
        self.assertEqual(model.params['alpha'], 0.02)
    
    def test_simulation(self):
        """Test basic simulation functionality."""
        model = InformationThermodynamicsModel()
        results = model.simulate(t_span=(0, 10))
        
        self.assertIn('time', results)
        self.assertIn('information_fidelity', results)
        self.assertIn('error_correction', results)
        self.assertIn('damage', results)
        self.assertIn('entropy_production', results)
        
        # Check that information fidelity decreases over time
        self.assertLess(results['information_fidelity'][-1], results['information_fidelity'][0])
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        generator = AgingDataGenerator()
        data = generator.generate_synthetic_data()
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('age', data.columns)
        self.assertIn('information_fidelity', data.columns)
        self.assertIn('molecular_damage', data.columns)
    
    def test_temporal_precedence(self):
        """Test temporal precedence validation."""
        generator = AgingDataGenerator()
        data = generator.generate_synthetic_data()
        
        validator = AgingValidation()
        results = validator.temporal_precedence_test(data)
        
        self.assertIn('temporal_precedence', results)
        self.assertIn('info_crossing_age', results)
        self.assertIn('damage_crossing_age', results)

if __name__ == '__main__':
    unittest.main()