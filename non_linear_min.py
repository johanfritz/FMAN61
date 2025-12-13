import numpy as np
from typing import Callable, Tuple
from rosenbrock import *
from grad import grad_c


def non_linear_min(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    method: str,
    tol: float,
    restart: bool,
    printout: bool,
) -> Tuple[np.ndarray, int, int, float]:
    """
    Nonlinear optimization using DFP or BFGS quasi-Newton methods.

    Args:
        f : Callable
            Objective function
        x0 : np.ndarray
            Initial guess
        method : str
            "DFP" or "BFGS"
        tol : float
            Convergence tolerance for gradient norm
        restart : bool
            If True, reset Hessian approximation to identity if something goes wrong
        printout : bool
            If True, print iteration information

    Returns:
        x : np.ndarray
            Minimum x*
        n_iter : int
            Number of iterations
        n_fval : int
            Number of function evaluations
        gnorm : float
            Norm of function gradient at x*
    """
    
    c1 = 1e-4
    c2 = 0.9

    n_iter = 0
    max_iter = 1000

    calls=0
    # HELPER FUNCTIONS
    norm = np.linalg.norm

    def _hesse(s, y, H, mthd) -> np.ndarray:
        if mthd == "dfp":
            Hy = H @ y
            sTy = np.dot(s, y)
            yTHy = np.dot(y, Hy)

            if sTy > 1e-12:
                return H - np.outer(Hy, Hy) / yTHy + np.outer(s, s) / sTy
            return None  # if not pos. def.
        if mthd == "bfgs":
            rho = 1.0 / np.dot(y, s)

            if rho > 0:
                I_minus_rhosyT = np.eye(n) - rho * np.outer(s, y)
                return I_minus_rhosyT @ H @ I_minus_rhosyT.T + rho * np.outer(s, s)
            return None  # if not pos. def.
        raise ValueError(f"Unknown update formula: {mthd}")

    def _line_search(f, x, p, g, lcalls) -> float:
        """Line search function that satisfies Wolfe conditions, from Nocedal&Wright 2006 p. 59-60
        Args:
            f: Objective function
            x: Current point
            p: Search direction
            fx: Objective function value at x
            g: Objective function gradient at x
        """
        alpha = 1
        phi = lambda alpha: f(x + alpha * p)

        def _zoom(alpha_lo, alpha_hi,phi_alpha_lo, phi_0, dphi_0, zcalls):

            for _ in range(max_iter):
                alpha_j = (alpha_lo + alpha_hi) / 2
                phi_j = phi(alpha_j)
                zcalls+=1
                if phi_j > phi_0 + c1 * alpha_j * dphi_0 or phi_j >= phi_alpha_lo:
                    alpha_hi = alpha_j
                else:
                    dphi_j = np.dot(grad_c(f, x + alpha_j * p), p)
                    zcalls+=2*n
                    dphi_lo = np.dot(grad_c(f, x + alpha_lo * p), p)
                    zcalls+=2*n
                    if np.abs(dphi_j) <= -c2 * dphi_lo:
                        return alpha_j, zcalls

                    if dphi_j * (alpha_hi - alpha_lo) >= 0:
                        alpha_hi = alpha_lo

                    alpha_lo = alpha_j

            return alpha_lo, zcalls

        dphi_0 = np.dot(g, p)
        alpha_prev = 0.0
        phi_prev = phi_0 = phi(0)
        lcalls+=1
        for i in range(max_iter):
            phi_i = phi(alpha)
            lcalls+=1
            if phi_i > phi_0 + c1 * alpha * dphi_0 or (i > 0 and phi_i >= phi_prev):
                return _zoom(alpha_prev, alpha,phi_prev, phi_0,dphi_0, lcalls)

            dphi_i = np.dot(grad_c(f, x + alpha * p), p)
            lcalls+=2*n
            if np.abs(dphi_i) <= -c2 * dphi_0:
                return alpha, lcalls

            if dphi_i >= 0:
                return _zoom(alpha, alpha_prev,phi_prev,phi_0, dphi_0, lcalls)

            alpha_prev = alpha
            phi_prev = phi_i
            alpha = alpha * 1.5

        return alpha, lcalls

    x = x0.copy()
    n = len(x)

    g = grad_c(f, x)
    calls+=2*n

    H = np.eye(n)

    if printout:
        print(f"Iteration {n_iter}: f(x) = {f(x):.6e}, ||g|| = {np.linalg.norm(g):.6e}")
        calls+=1

    # Main optimization loop
    while norm(g) > tol and n_iter < max_iter:
        p = -H @ g
        alpha,calls= _line_search(f, x, p, g, calls)

        x_new = x + alpha * p
        g_new = grad_c(f, x_new)
        calls+=2*n

        s = alpha * p
        y = g_new - g

        H = _hesse(s, y, H, method)
        if H is None:
            if restart:
                H = np.eye(n)
            else:
                raise RuntimeError("Hesse matrix is not pos. def, consider restarting")

        x = x_new
        g = g_new
        n_iter += 1

        if printout:
            print(
                f"Iteration {n_iter}: f(x) = {f(x):.6e}, ||g|| = {np.linalg.norm(g):.6e}, alpha = {alpha}"
            )
            calls+=1
    if printout:
        if n_iter==max_iter:
            print("="*10+"Minimization terminated unsucessfully" + "="*10)
        else:
            print("="*10+"Minimization finished" + "="*10)
            print(f"x*={x}")
            print(f"f(x*)={f(x)}")
            calls+=1
            print(f"2-norm of g={norm(g)}")
            print(f"Minimization steps: {n_iter}")
            print(f"Function calls: {calls+1}")
            print("="*41)
    return (
        x,
        calls+1,
        n_iter,
        f(x),
    )


if __name__ == "__main__":
    f = lambda x: np.sum(x**2)
    print(non_linear_min(rosenbrock, np.random.rand(2) * 100, "bfgs", 1e-5, True, True))
