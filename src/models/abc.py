import numpy as np
#from simulators import simulator, simulator2

def abc_inference(y, x, theta, eps, simulator_func):
    '''
    Perform Approximate Bayesian Computation (ABC) inference.
    
    Parameters:
    y (float): The observed value.
    x (ndarray): The input data
    theta (ndarray): Samples from the prior distribution.
    e (float): The acceptance threshold
    
    Retunrs:
    ndarray: Accepted samples that approximate the posterior distribution.
    '''

    accepted_samples = []
    for i in range(len(theta)):
        sim = simulator_func(theta[i, :], x)
        dis = np.linalg.norm(sim-y)
        if dis < eps:
            accepted_samples.append(theta[i,:])
    return np.array(accepted_samples)

def abc_method1(N, Nsamples, simulator_func, prior, dim, x, obs):
    """
    Perfom Approximate Bayesian Computation 
    Parameters:
    - N: int, the number of accepted samples,
    - Nsamples: int, the number of generated samples,
    - simulation_func: function, generates simulated data given parameters,
    - prior: callable: generates samples from the prior,
    - dim: int, dimensions of the parameter space,
    - x: ndarray, fixed variable,
    - obs: array, the true observed data for comparison
    
    Returns:
    - samples_pos: array, best N accepted samples
    """

    # Generate samples from the prior
    thetas = prior(Nsamples, dim)

    # Simulate data for all thetas
    sim = simulator_func(thetas, x)

    # Compute the distances between obs and sim_data
    distances = np.abs(obs-sim)

    # Find the indices of N smallest distances
    best_indices = np.argsort(distances)[:N]

    # Select the top N thetas
    samples_pos = thetas[best_indices]

    return samples_pos

def abc_method2(N, eps, simulator_func, prior, dim, x, obs, batch_size=100):
    """
    Approximate Bayesian Computation with batch processing and early stopping
    
    Parameters:
    - N: int, total number of samples to generate from the prior
    - eps: float,tolerance threshold for the distance metric,
    - simulator_func: function, generates simulated data given the parameters,
    - prior: callable, generates samples from the prior distribution,
    - dim: int, dimensions of the parameter space,
    - x: array, fixed variable,
    - obs: array, the true observed data for comparison,
    - batch_size: int , the number of samples to process in each batch
    
    Returns:
    - samples_pos: array, first 100 accepted samples (posterior distribution)
    """

    samples_pos = []

    # Initialize step
    step=0

    while len(samples_pos) < batch_size and step < N:

        # Generate a batch of candidate samples
        thetas=prior(N, dim)

        # Simulated data for all parameters in the batch
        sim_data = simulator_func(thetas, x) # Shape: (batch_size,)

        # Compute the distances
        distances = np.abs(obs-sim_data) # Shape: (batch_size, )

        # Identify indices of accepted samples
        accepted_indices = np.where(distances < eps)[0]

        # Extact accepted samples
        accepted_samples = thetas[accepted_indices]

        # Add accepted samples to samples_pos
        samples_pos.extend(accepted_samples)
        
        # Stop if we have collected enough samples
        if len(samples_pos) >= batch_size:
            samples_pos[:batch_size]
            break

        # Increment the step count
        step += batch_size

    if len(samples_pos)<=batch_size:
        raise ValueError(f"Only {len(samples_pos)} accepted samples")

    return np.array(samples_pos)



    