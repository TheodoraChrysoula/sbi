import numpy as np
import scipy.optimize import minimize
from src.simulator import simulator

class OMCInference:
    def __init__(self, y, u_dim, regularization=1e-8):
        '''
        Initilize the OMCInference class
        
        Parameters:
            y (float or ndarray): Observed data to match
            u_dim (int): Dimensionality of the latent variable u.
            regularization (float): Regularization factor for the Jacobian matrix
        '''

        self.y = y
        self.u_dim = u_dim
        self.regularization = regularization

    def objective_function(self, u, theta):
        '''
        Objective function for optimization
        
        Parameters:
            u (ndarray): Latent variable be optimized.
            theta (ndarray): Parameter vector.
            
        Returns:
            float: The objective value (distance between simulation and observatiom).
        '''
        sim = simulator(theta, u)
        return np.linalg.norm(sim - self.y)
    
    def compute_jacobian(self, theta, u):
        '''
        Compute the Jacobian matrix of the simulator with respect to theta.
        
        Parameters:
            theta (ndarray): Parameter vector.
            u (ndarray): Latent variable
            
        Returns:
            ndarray: Jacobian matrix.
        '''
        return np.array([u] * len(theta))
    
    def compute_weight(self, jacobian_matrix):
        '''
        Compute the weight for the given Jacobian matrix using its determinant.
        
        Parameters:
            jacobian_matrix (ndarray): Jacobian matrix.
            
        Returns:
            float: The computed weight.
        '''
        JTJ = np.dot(jacobian_matrix.T, jacobian_matrix) + np.eye(jacobian_matrix.shape[1]) * self.regularization
        det = np.linalg.det(JTJ)
        return (1.0 / np.sqrt(det)) if det > 0 else 0
    
    def infer(self, theta_sampels):
        '''
        Perform the inference process to accept samples and calculate weights.
        
        Parameters:
            theta_samples (ndarray): Array of parameter samples from the prior.
            
        Returns:
            tuple: Arrays of accepted samples and their associated weights.
        '''

        accepted_samples = []
        weights = []

        for theta_i in theta_samples:
            # Initial guess for u
            u_init = np.random.uniform(-1, 1, self.u_dim)

            # Minimize the objective function to find the optimal u
            result = minimize(self.objective_function, u_init, args=(theta_i,), method = 'L-BFGS-B')
            u_star = result.x

            # Compute simulation result and weight
            sim = simulator(theta_i, u_star)
            jacobian_matrix = self.compute_jacobian(theta_i, u_star)
            weight = self.compute_weight(jacobian_matrix)

            accepted_samples.append(theta_i)
            weights.append(weight)

        return np.array(accepted_samples), np.array(weights)
