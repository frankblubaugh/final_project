import math
import numpy as np

def convert_vals_freqs(eigen_value):
    """
    Converts the value to the natural frequency
        .. math::
        \lambda = omega^2
        f = \frac{omega}{2*\pi};
    Parameters
    ----------
    eigen_value:

    Returns
    -------
    natural frequency following lambda
    """
    return np.sqrt(eigen_value) / (2.0 * np.pi)