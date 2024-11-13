import numpy as np


def layer_wise_dot_product(*matrices: np.ndarray) -> np.ndarray:
    """
    Performs the sequential layer-wise dot product of multiple 3D matrices along the last dimension.

    Args:
    *matrices: A variable number of 3D numpy arrays.

    Returns:
    A 3D numpy array containing the sequential dot product results for each layer in the last dimension.
    """
    if not all(matrix.shape == matrices[0].shape for matrix in matrices):
        raise ValueError("All matrices must have the same dimensions.")

    result = np.zeros_like(matrices[0], dtype="complex")

    _, _, num_layers = matrices[0].shape

    for layer_i in range(num_layers):
        # Start the product with the identity matrix for the first multiplication
        dot_product = np.eye(result.shape[0])
        for matrix in matrices:
            dot_product = np.dot(dot_product, matrix[:, :, layer_i])

        result[:, :, layer_i] = dot_product

    return result


def bessel(z):
    """
    Bessel function aproximation for air radiation impedance
    """
    bessel_sum = 0
    for k in range(25):
        bessel_i = ((-1) ** k * (z / 2) ** (2 * k + 1)) / (
            np.math.factorial(k) * np.math.factorial(k + 1)
        )
        bessel_sum = bessel_sum + bessel_i
    return bessel_sum


def struve(z):
    """
    Srtuve function aproximation for air radiation impedance
    """
    struve_sum = 0
    for k in range(25):
        struve_i = (((-1) ** k * (z / 2) ** (2 * k + 2))) / (
            np.math.factorial(int(k + 1 / 2)) * np.math.factorial(int(k + 3 / 2))
        )
        struve_sum = struve_sum + struve_i
    return struve_sum


def to_db(x):
    """
    decibel calculus
    """
    db = 20 * np.log10(np.abs(x))
    return db
