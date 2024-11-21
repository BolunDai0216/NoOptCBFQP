import numpy as np
import proxsuite

from nooptcbfqp import cbfqp


def test_cbfqp() -> None:
    dhdx = np.array([[2.0, 0.7]])
    α = 10.0
    h = -4.5
    u_0 = np.array([[2.0], [5.0]])

    u_optimal = cbfqp(h, dhdx, u_0, α=α)
    print(f"nooptcbfqp u_optimal:\n {u_optimal}")

    H = np.eye(2)
    g = -u_0
    C = dhdx
    lb = -np.array([[α * h]])

    n = 2
    n_ieq = 1
    n_eq = 0
    qp = proxsuite.proxqp.dense.QP(n, n_eq, n_ieq)

    qp.init(H=H, g=g, C=C, l=lb)
    qp.settings.eps_abs = 1.0e-6
    qp.settings.max_iter = 20
    qp.solve()
    u_optimal_proxsuite = qp.results.x
    print(f"proxsuite u_optimal:\n {u_optimal_proxsuite}")

    np.testing.assert_allclose(u_optimal.flatten(), u_optimal_proxsuite, atol=1.0e-4)


def main():
    test_cbfqp()


if __name__ == "__main__":
    main()
