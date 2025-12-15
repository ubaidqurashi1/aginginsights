from src.data_generator import AgingDataGenerator

generator = AgingDataGenerator()
synthetic_data = generator.generate_synthetic_data()
generator.save_data('data/synthetic_data.csv')