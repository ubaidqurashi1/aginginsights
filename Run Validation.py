from src.validation import AgingValidation
from src.data_generator import AgingDataGenerator

# Generate or load data
generator = AgingDataGenerator()
data = generator.generate_synthetic_data()

# Run validation
validator = AgingValidation()
results = validator.run_full_validation(data)
validator.plot_results()