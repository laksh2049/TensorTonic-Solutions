import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    is_scalar = np.isscalar(x)
    
    # Convert input to a NumPy array of floats
    x_arr = np.asarray(x, dtype=float)
    
    # Compute the elementwise sigmoid expression
    result = 1 / (1 + np.exp(-x_arr))
    
    # Return a native float for scalar inputs, otherwise return the array
    return float(result) if is_scalar else result
    