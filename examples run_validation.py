"""
Example script demonstrating the complete validation workflow.
"""
import os
import pandas as pd
from src.data_generator import AgingDataGenerator
from src.validation import AgingValidation
from src.visualization import AgingVisualization

def main():
    """Run complete validation example."""
    print("🧪 Running Information Thermodynamics Aging Validation")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Generate synthetic data
    print("🔄 Generating synthetic aging data...")
    generator = AgingDataGenerator(noise_level=0.03, random_seed=42)
    data = generator.generate_synthetic_data(include_intervention=True)
    generator.save_data('data/synthetic_data.csv', include_intervention=True)
    print(f"✅ Generated {len(data)} data points")
    
    # Run validation
    print("🔍 Running validation suite...")
    validator = AgingValidation()
    results = validator.run_full_validation(data)
    
    # Display results
    print("\n📊 VALIDATION RESULTS:")
    print("-" * 40)
    
    tp = results['temporal_precedence']
    print(f"Temporal Precedence: {tp['temporal_precedence']}")
    if tp['temporal_precedence']:
        print(f"  • Information loss at age: {tp['info_crossing_age']:.1f}")
        print(f"  • Damage accumulation at age: {tp['damage_crossing_age']:.1f}")
        print(f"  • Time difference: {tp['time_difference']:.1f} years")
    
    gc = results['granger_causality']
    print(f"\nGranger Causality:")
    print(f"  • Information → Damage R²: {gc['r2_info_prediction']:.3f}")
    print(f"  • Damage → Damage R²: {gc['r2_damage_prediction']:.3f}")
    print(f"  • Information better predictor: {gc['r2_info_prediction'] > gc['r2_damage_prediction']}")
    
    ir = results['intervention_response']
    if ir['intervention_performed']:
        print(f"\nIntervention Response (age {ir['intervention_age']:.0f}):")
        print(f"  • Information fidelity change: {ir['information_fidelity_change_pct']:+.1f}%")
        print(f"  • Entropy production change: {ir['entropy_production_change_pct']:+.1f}%")
        print(f"  • Molecular damage change: {ir['molecular_damage_change_pct']:+.1f}%")
    
    # Create visualizations
    print("\n📈 Generating visualizations...")
    viz = AgingVisualization()
    
    fig1 = viz.plot_synthetic_data(data, 'results/synthetic_data.png')
    fig2 = viz.plot_validation_results(results, 'results/validation_results.png')
    
    print("✅ Validation complete! Results saved to 'results/' directory")
    
    return results

if __name__ == "__main__":
    main()