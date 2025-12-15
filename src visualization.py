"""
Visualization functions for the information thermodynamics aging framework.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class AgingVisualization:
    """Visualization suite for aging framework results."""
    
    def __init__(self, figsize=(12, 10)):
        self.figsize = figsize
    
    def plot_synthetic_data(self, data, save_path=None):
        """Plot synthetic aging data."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Synthetic Aging Data: Information Thermodynamics Framework', 
                    fontsize=16, fontweight='bold')
        
        scenarios = data['scenario'].unique()
        
        # Information Fidelity
        for scenario in scenarios:
            df = data[data['scenario'] == scenario]
            axes[0, 0].plot(df['age'], df['information_fidelity'], 
                           label=scenario, linewidth=2)
        axes[0, 0].set_xlabel('Age (years)')
        axes[0, 0].set_ylabel('Information Fidelity')
        axes[0, 0].set_title('Information Fidelity Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Entropy Production
        for scenario in scenarios:
            df = data[data['scenario'] == scenario]
            axes[0, 1].plot(df['age'], df['entropy_production'], 
                           label=scenario, linewidth=2)
        axes[0, 1].set_xlabel('Age (years)')
        axes[0, 1].set_ylabel('Entropy Production')
        axes[0, 1].set_title('Entropy Production Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Molecular Damage
        for scenario in scenarios:
            df = data[data['scenario'] == scenario]
            axes[1, 0].plot(df['age'], df['molecular_damage'], 
                           label=scenario, linewidth=2)
        axes[1, 0].set_xlabel('Age (years)')
        axes[1, 0].set_ylabel('Molecular Damage')
        axes[1, 0].set_title('Molecular Damage Accumulation')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined view
        df_baseline = data[data['scenario'] == 'baseline']
        axes[1, 1].plot(df_baseline['age'], df_baseline['information_fidelity'], 
                       'b-', label='Information Fidelity', linewidth=2)
        axes[1, 1].plot(df_baseline['age'], df_baseline['molecular_damage'], 
                       'r--', label='Molecular Damage', linewidth=2)
        axes[1, 1].set_xlabel('Age (years)')
        axes[1, 1].set_ylabel('Normalized Values')
        axes[1, 1].set_title('Information Loss vs Damage Accumulation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_validation_results(self, validation_results, save_path=None):
        """Plot validation results."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Validation Results: Information Thermodynamics Framework', 
                    fontsize=16, fontweight='bold')
        
        # Temporal Precedence
        tp = validation_results['temporal_precedence']
        if tp['temporal_precedence']:
            axes[0].bar(['Information Loss', 'Molecular Damage'], 
                       [tp['info_crossing_age'], tp['damage_crossing_age']], 
                       color=['blue', 'red'])
            axes[0].set_ylabel('Age (years)')
            axes[0].set_title(f'Temporal Precedence\nΔt = {tp["time_difference"]:.1f} years')
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'No temporal\nprecedence found', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Temporal Precedence')
        
        # Granger Causality
        gc = validation_results['granger_causality']
        r2_values = [gc['r2_info_prediction'], gc['r2_damage_prediction']]
        axes[1].bar(['Information → Damage', 'Damage → Damage'], r2_values, 
                   color=['green', 'orange'])
        axes[1].set_ylabel('R²')
        axes[1].set_title('Granger Causality\n(Prediction R²)')
        axes[1].grid(True, alpha=0.3)
        
        # Intervention Response
        ir = validation_results['intervention_response']
        if ir['intervention_performed']:
            changes = [
                ir['information_fidelity_change_pct'],
                ir['entropy_production_change_pct'],
                ir['molecular_damage_change_pct']
            ]
            labels = ['Information\nFidelity', 'Entropy\nProduction', 'Molecular\nDamage']
            colors = ['green' if x > 0 else 'red' for x in changes]
            axes[2].bar(labels, changes, color=colors)
            axes[2].set_ylabel('Change (%)')
            axes[2].set_title(f'Intervention Response\n(Age {ir["intervention_age"]:.0f})')
            axes[2].grid(True, alpha=0.3)
            # Add horizontal line at 0
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        else:
            axes[2].text(0.5, 0.5, 'No intervention\ndata available', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Intervention Response')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig