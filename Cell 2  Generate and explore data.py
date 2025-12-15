# Generate synthetic data
generator = AgingDataGenerator(noise_level=0.02)
data = generator.generate_synthetic_data(include_intervention=True)

# Basic exploration
print(data.head())
print(f"Data shape: {data.shape}")
print(f"Scenarios: {data['scenario'].unique()}")