"""
Validation functions for the information thermodynamics aging framework.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests
from .aging_model import InformationThermodynamicsModel

class AgingValidation:
    """
    Validation suite for testing predictions of the information thermodynamics framework.
    """
    
    def __init__(self, critical_thresholds=None):
        """
        Initialize validation with critical thresholds.
        
        Parameters:
        - critical_thresholds: dict with 'information' and 'damage' thresholds
        """
        if critical_thresholds is None:
            self.critical_thresholds = {
                'information': 0.65,
                'damage': 0.45
            }
        else:
            self.critical_thresholds = critical_thresholds
    
    def temporal_precedence_test(self, data, scenario='baseline'):
        """
        Test if information loss precedes molecular damage accumulation.
        
        Parameters:
        - data: pandas DataFrame with aging data
        - scenario: str, which scenario to analyze
        
        Returns:
        - dict with test results
        """
        # Filter data by scenario
        df = data[data['scenario'] == scenario].copy()
        df = df.sort_values('age')
        
        # Find crossing points
        info_crossing = self._find_crossing_point(
            df['age'].values, 
            df['information_fidelity'].values, 
            self.critical_thresholds['information'], 
            direction='below'
        )
        
        damage_crossing = self._find_crossing_point(
            df['age'].values, 
            df['molecular_damage'].values, 
            self.critical_thresholds['damage'], 
            direction='above'
        )
        
        if info_crossing is not None and damage_crossing is not None:
            precedence = info_crossing < damage_crossing
            time_difference = damage_crossing - info_crossing
        else:
            precedence = False
            time_difference = None
            info_crossing = None
            damage_crossing = None
        
        return {
            'temporal_precedence': precedence,
            'info_crossing_age': info_crossing,
            'damage_crossing_age': damage_crossing,
            'time_difference': time_difference,
            'info_threshold': self.critical_thresholds['information'],
            'damage_threshold': self.critical_thresholds['damage']
        }
    
    def granger_causality_test(self, data, scenario='baseline', max_lag=3):
        """
        Test Granger causality between information and damage.
        
        Parameters:
        - data: pandas DataFrame with aging data
        - scenario: str, which scenario to analyze
        - max_lag: int, maximum lag for Granger test
        
        Returns:
        - dict with causality results
        """
        df = data[data['scenario'] == scenario].copy()
        df = df.sort_values('age')
        
        # Prepare time series data
        info_series = df['information_fidelity'].values
        damage_series = df['molecular_damage'].values
        
        # Test if information Granger-causes damage
        info_to_damage = self._granger_causality_test(
            cause_series=info_series,
            effect_series=damage_series,
            max_lag=max_lag
        )
        
        # Test if damage Granger-causes itself (autocorrelation baseline)
        damage_to_damage = self._granger_causality_test(
            cause_series=damage_series,
            effect_series=damage_series,
            max_lag=max_lag
        )
        
        # Calculate R-squared values for comparison
        r2_info_cause = self._calculate_r2_prediction(info_series, damage_series, max_lag)
        r2_damage_cause = self._calculate_r2_prediction(damage_series[:-1], damage_series[1:], 1)
        
        return {
            'info_granger_causes_damage': info_to_damage['significant'],
            'damage_autocorrelation': damage_to_damage['significant'],
            'info_to_damage_p_value': info_to_damage['p_value'],
            'damage_to_damage_p_value': damage_to_damage['p_value'],
            'r2_info_prediction': r2_info_cause,
            'r2_damage_prediction': r2_damage_cause,
            'max_lag': max_lag
        }
    
    def intervention_response_analysis(self, data):
        """
        Analyze the response to information restoration intervention.
        
        Parameters:
        - data: pandas DataFrame with both baseline and intervention scenarios
        
        Returns:
        - dict with intervention analysis results
        """
        # Get intervention age from data
        intervention_ages = data[data['scenario'] == 'intervention']['age']
        if len(intervention_ages) == 0:
            return {'intervention_performed': False}
        
        # Find intervention time point (minimum age where intervention starts)
        intervention_age = intervention_ages.min()
        
        # Get data points just before and after intervention
        baseline_at_intervention = data[
            (data['scenario'] == 'baseline') & 
            (np.abs(data['age'] - intervention_age) < 1.0)
        ]
        
        intervention_at_intervention = data[
            (data['scenario'] == 'intervention') & 
            (np.abs(data['age'] - intervention_age) < 1.0)
        ]
        
        if len(baseline_at_intervention) == 0 or len(intervention_at_intervention) == 0:
            return {'intervention_performed': False}
        
        # Calculate changes
        baseline_I = baseline_at_intervention['information_fidelity'].mean()
        intervention_I = intervention_at_intervention['information_fidelity'].mean()
        
        baseline_S = baseline_at_intervention['entropy_production'].mean()
        intervention_S = intervention_at_intervention['entropy_production'].mean()
        
        baseline_D = baseline_at_intervention['molecular_damage'].mean()
        intervention_D = intervention_at_intervention['molecular_damage'].mean()
        
        # Calculate percentage changes
        I_change_pct = ((intervention_I - baseline_I) / baseline_I) * 100
        S_change_pct = ((intervention_S - baseline_S) / baseline_S) * 100
        D_change_pct = ((intervention_D - baseline_D) / baseline_D) * 100
        
        return {
            'intervention_performed': True,
            'intervention_age': intervention_age,
            'information_fidelity_change_pct': I_change_pct,
            'entropy_production_change_pct': S_change_pct,
            'molecular_damage_change_pct': D_change_pct,
            'intervention_successful': I_change_pct > 0 and S_change_pct < 0
        }
    
    def run_full_validation(self, data):
        """
        Run complete validation suite.
        
        Parameters:
        - data: pandas DataFrame with aging data
        
        Returns:
        - dict with all validation results
        """
        results = {}
        
        # Temporal precedence
        results['temporal_precedence'] = self.temporal_precedence_test(data)
        
        # Granger causality
        results['granger_causality'] = self.granger_causality_test(data)
        
        # Intervention analysis (if intervention data exists)
        if 'intervention' in data['scenario'].values:
            results['intervention_response'] = self.intervention_response_analysis(data)
        else:
            results['intervention_response'] = {'intervention_performed': False}
        
        return results
    
    def _find_crossing_point(self, x, y, threshold, direction='above'):
        """Find the x-value where y crosses threshold."""
        if direction == 'above':
            mask = y >= threshold
        else:
            mask = y <= threshold
        
        if not np.any(mask):
            return None
        
        crossing_idx = np.where(mask)[0][0]
        if crossing_idx == 0:
            return x[0]
        
        # Linear interpolation for more precise crossing point
        x1, x2 = x[crossing_idx-1], x[crossing_idx]
        y1, y2 = y[crossing_idx-1], y[crossing_idx]
        
        if y2 == y1:
            return x2
        
        t = (threshold - y1) / (y2 - y1)
        return x1 + t * (x2 - x1)
    
    def _granger_causality_test(self, cause_series, effect_series, max_lag=3):
        """Perform Granger causality test."""
        try:
            # Create DataFrame for statsmodels
            data = pd.DataFrame({
                'effect': effect_series,
                'cause': cause_series
            })
            
            # Remove any NaN values
            data = data.dropna()
            
            if len(data) < max_lag * 2:
                return {'significant': False, 'p_value': 1.0}
            
            # Perform Granger causality test
            result = grangercausalitytests(data[['effect', 'cause']], max_lag, verbose=False)
            
            # Get p-value from the best lag (lowest p-value)
            p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
            min_p_value = min(p_values)
            
            return {
                'significant': min_p_value < 0.05,
                'p_value': min_p_value
            }
        except Exception as e:
            print(f"Granger causality test failed: {e}")
            return {'significant': False, 'p_value': 1.0}
    
    def _calculate_r2_prediction(self, predictor, target, lag=1):
        """Calculate R-squared for prediction with given lag."""
        if len(predictor) <= lag or len(target) <= lag:
            return 0.0
        
        X = predictor[:-lag].reshape(-1, 1)
        y = target[lag:]
        
        if len(X) == 0 or len(y) == 0:
            return 0.0
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)
            return max(0.0, r2)  # Ensure non-negative
        except:
            return 0.0