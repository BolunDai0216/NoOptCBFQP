import numpy as np


def cbfqp(h: float, dhdx: np.ndarray, u_0: np.ndarray, α: float = 1.0) -> np.ndarray:
    """
    Solves the CBF-QP problem in closed form.

    Args:
        h: CBF value
        dhdx: Gradient of the CBF, shape (1, n_u)
        u_0: Initial control input, shape (n_u, 1)
        α: CBF weight

    Returns:
        u_optimal: Optimal control input, shape (n_u, 1)
    """
    lhs = dhdx @ u_0 + α * h

    if lhs[0, 0] >= 0:
        u_optimal = u_0
    else:
        u_optimal = u_0 - (lhs / (dhdx @ dhdx.T))[0, 0] * dhdx.T

    return u_optimal
