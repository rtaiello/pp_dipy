import numpy as np


def coordinates(domain):
    domain = domain
    tensor: np.ndarray = np.mgrid[0.0 : domain[1], 0.0 : domain[3], 0.0 : domain[5]]

    homogenous: np.ndarray = np.zeros((4, tensor[0].size))
    homogenous[0]: np.ndarray = tensor[0].flatten()
    homogenous[1]: np.ndarray = tensor[1].flatten()
    homogenous[2]: np.ndarray = tensor[2].flatten()
    homogenous[3]: np.ndarray = np.ones(tensor[0].size)

    return tensor, homogenous


def transform(tensor, homogenous, p: np.ndarray) -> np.ndarray:
    """
    An affine transformation of coordinates.
    :param p: model parameters
    :return: deformation coordinates
    """

    T = p

    displacement = np.dot(np.linalg.inv(T), homogenous) - homogenous

    shape = tensor[0].shape

    return np.array([displacement[1].reshape(shape), displacement[0].reshape(shape)])


tensor, homogenous = coordinates((0, 120, 0, 110, 0, 140))
print(tensor[0].shape)
