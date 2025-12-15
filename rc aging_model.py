"""
Mathematical implementation of the information thermodynamics aging framework.
"""
import numpy as np
from scipy.integrate import solve_ivp

class InformationThermodynamicsModel:
    """
    Mathematical model implementing the coupled differential equations
    for information thermodynamics of aging.
    """
    
    def __init__(self, params=None):
        """
        Initialize the model with default or custom parameters.
        
        Default parameters (biologically plausible values):
        - alpha: 0.02 (information degradation rate)
        - beta: 0.015 (environmental stress factor)
        - E0: 1.0 (initial error correction capacity)
        - delta: 0.018 (entropy sensitivity)
        - gamma: 0.1 (metabolic entropy coefficient)
        - eta: 0.8 (thermodynamic efficiency)
        - mu: 0.03 (damage accumulation rate)
        - nu: 0.02 (autocatalytic damage rate)
        """
        if params is None:
            self.params = {
                'alpha': 0.02,
                'beta': 0.015,
                'E0': 1.0,
                'delta': 0.018,
                'gamma': 0.1,
                'eta': 0.8,
                'mu': 0.03,
                'nu': 0.02
            }
        else:
            self.params = params
    
    def coupled_odes(self, t, y):
        """
        Coupled differential equations for the aging model.
        
        State variables:
        y[0] = I (information fidelity)
        y[1] = E (error correction capacity)
        y[2] = D (molecular damage)
        """
        I, E, D = y
        alpha = self.params['alpha']
        beta = self.params['beta']
        delta = self.params['delta']
        gamma = self.params['gamma']
        eta = self.params['eta']
        mu = self.params['mu']
        nu = self.params['nu']
        
        # Information degradation dynamics
        dI_dt = -alpha * I + beta * I * E
        
        # Entropy production
        S_prod = gamma * (1 - I) + (1/eta) * alpha * I
        
        # Error correction capacity dynamics
        dE_dt = -delta * S_prod * E
        
        # Molecular damage accumulation
        dD_dt = mu * (1 - I) + nu * D
        
        return [dI_dt, dE_dt, dD_dt]
    
    def simulate(self, t_span=(0, 100), t_eval=None, initial_conditions=None):
        """
        Simulate the aging model over time.
        
        Parameters:
        - t_span: tuple, time span for simulation (default: 0 to 100 years)
        - t_eval: array, specific time points to evaluate (default: weekly intervals)
        - initial_conditions: list, [I0, E0, D0] (default: [1.0, 1.0, 0.01])
        
        Returns:
        - Dictionary with time, information_fidelity, error_correction, damage, entropy_production
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) * 52))
        
        if initial_conditions is None:
            initial_conditions = [1.0, self.params['E0'], 0.01]
        
        # Solve ODEs
        sol = solve_ivp(
            self.coupled_odes,
            t_span,
            initial_conditions,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Calculate entropy production at each time point
        I_vals = sol.y[0]
        S_prod_vals = (self.params['gamma'] * (1 - I_vals) + 
                      (1/self.params['eta']) * self.params['alpha'] * I_vals)
        
        return {
            'time': sol.t,
            'information_fidelity': sol.y[0],
            'error_correction': sol.y[1],
            'damage': sol.y[2],
            'entropy_production': S_prod_vals
        }
    
    def simulate_intervention(self, intervention_age=60, restoration_efficiency=0.6, 
                             t_span=(0, 100)):
        """
        Simulate aging with information restoration intervention.
        
        Parameters:
        - intervention_age: age at which intervention occurs
        - restoration_efficiency: fraction of lost information restored (0-1)
        - t_span: time span for simulation
        
        Returns:
        - Dictionary with intervention results
        """
        # Simulate up to intervention age
        pre_intervention = self.simulate(t_span=(0, intervention_age))
        
        # Apply intervention
        I_before = pre_intervention['information_fidelity'][-1]
        I_restored = I_before + restoration_efficiency * (1 - I_before)
        
        # Continue simulation with restored information
        initial_post = [
            I_restored,
            pre_intervention['error_correction'][-1],
            pre_intervention['damage'][-1]
        ]
        
        post_intervention = self.simulate(
            t_span=(intervention_age, t_span[1]),
            initial_conditions=initial_post
        )
        
        # Combine results
        combined_time = np.concatenate([pre_intervention['time'], post_intervention['time'][1:]])
        combined_I = np.concatenate([pre_intervention['information_fidelity'], post_intervention['information_fidelity'][1:]])
        combined_E = np.concatenate([pre_intervention['error_correction'], post_intervention['error_correction'][1:]])
        combined_D = np.concatenate([pre_intervention['damage'], post_intervention['damage'][1:]])
        
        # Calculate entropy production for combined timeline
        S_prod_combined = (self.params['gamma'] * (1 - combined_I) + 
                          (1/self.params['eta']) * self.params['alpha'] * combined_I)
        
        return {
            'time': combined_time,
            'information_fidelity': combined_I,
            'error_correction': combined_E,
            'damage': combined_D,
            'entropy_production': S_prod_combined,
            'intervention_age': intervention_age
        }