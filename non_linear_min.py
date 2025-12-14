from typing import Callable, Tuple
import numpy as np
from grad import grad_c

class _FuncCounter:
    """Counts objective evaluations (including those inside finite-difference gradients)."""
    def __init__(self, f: Callable[[np.ndarray], float]):
        self.f = f
        self.calls = 0

    def __call__(self, x: np.ndarray) -> float:
        self.calls += 1
        return float(self.f(x))

def _strong_wolfe_line_search(
    f: _FuncCounter,
    x: np.ndarray,
    fx: float,
    g: np.ndarray,
    p: np.ndarray,
    c1: float = 1e-4,
    c2: float = 0.9,
    alpha0: float = 1.0,
    alpha_max: float = 50.0,
    max_iter: int = 25,
    max_zoom: int = 25,
) -> Tuple[float, float, np.ndarray, int]:
    """
    wolfe line search (nocedal & wright style), using numerical gradients.
    returns (alpha, f_new, g_new, ls_fun_evals).
    """
    calls_before = f.calls

    phi0 = fx
    dphi0 = float(np.dot(g, p))

    #ensure descent direction, otherwise line search assumptions break.
    if dphi0 >= 0.0:  #enforce descent
        p = -g
        dphi0 = -float(np.dot(g, g))

    def phi(alpha: float) -> float:
        return f(x + alpha * p)

    def dphi(alpha: float) -> Tuple[float, np.ndarray]:
        x_a = x + alpha * p
        g_a = grad_c(f, x_a)
        return float(np.dot(g_a, p)), g_a

    def zoom(a_lo: float, a_hi: float, phi_lo: float) -> Tuple[float, float, np.ndarray]:
        for _ in range(max_zoom):
            a_j = 0.5 * (a_lo + a_hi)  #bisection (its robust)
            phi_j = phi(a_j)

            if (phi_j > phi0 + c1 * a_j * dphi0) or (phi_j >= phi_lo):
                a_hi = a_j
            else:
                dphi_j, g_j = dphi(a_j)
                if abs(dphi_j) <= -c2 * dphi0:
                    return a_j, phi_j, g_j
                #if derivative has wrong sign, swap bracket endpoint
                if dphi_j * (a_hi - a_lo) >= 0.0:
                    a_hi = a_lo
                a_lo = a_j
                phi_lo = phi_j

        #fallback: return best known low point
        dphi_lo, g_lo = dphi(a_lo)
        return a_lo, phi(a_lo), g_lo  #recompute phi for consistency

    a_prev = 0.0
    phi_prev = phi0
    a = float(alpha0)

    for i in range(max_iter):
        phi_a = phi(a)

        if (phi_a > phi0 + c1 * a * dphi0) or (i > 0 and phi_a >= phi_prev):
            a_star, f_star, g_star = zoom(min(a_prev, a), max(a_prev, a), phi_prev if a_prev < a else phi_a)  # FIX: ordered bracket
            return a_star, f_star, g_star, f.calls - calls_before

        dphi_a, g_a = dphi(a)

        if abs(dphi_a) <= -c2 * dphi0:
            return a, phi_a, g_a, f.calls - calls_before

        if dphi_a >= 0.0:
            a_star, f_star, g_star = zoom(min(a, a_prev), max(a, a_prev), phi_a if a < a_prev else phi_prev)  # FIX: ordered bracket
            return a_star, f_star, g_star, f.calls - calls_before

        a_prev = a
        phi_prev = phi_a
        a = min(2.0 * a, alpha_max)  #grow step

    #if we get here, accept last tried step (best-effort)
    dphi_a, g_a = dphi(a)
    return a, phi(a), g_a, f.calls - calls_before


def non_linear_min(
    f_orig: Callable[[np.ndarray], float],
    x0: np.ndarray,
    method: str,
    tol: float,
    restart: bool,
    printout: bool,
) -> Tuple[np.ndarray, int, int, float]:
    """
    quasi-newton minimiser using DFP or BFGS.
    returns (x, N_eval, N_iter, normg).  #matches assignment.
    """
    f = _FuncCounter(f_orig)  #single authoritative evaluation counter
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1:
        raise TypeError("non_linear_min: x0 must be a 1D numpy array.")
    if tol <= 0:
        raise ValueError("non_linear_min: tol must be positive.")

    m = method.strip().upper()  #accept 'DFP'/'BFGS' in any case
    if m not in {"DFP", "BFGS"}:
        raise ValueError("non_linear_min: method must be 'DFP' or 'BFGS'.")

    n = x.size
    H = np.eye(n)  #inverse-Hessian approximation

    max_iter = 1000
    eps = 1e-12

    #"restart regularly" now means periodic resets, not only on failure
    restart_period = 10 * max(1, n)

    fx = f(x)
    g = grad_c(f, x)
    normg = float(np.linalg.norm(g))

    if printout:
        print("iter  x  f(x)  norm(grad)  ls fun evals  lamb")  #matches required printout fields
        print(f"{0:4d}  {x}  {fx:.6e}  {normg:.6e}  {0:11d}  {0.0:.6e}")

    k = 0
    while normg > tol and k < max_iter:
        if restart and (k > 0) and (k % restart_period == 0):
            H = np.eye(n)  #periodic restart

        #search direction
        p = -H @ g
        if float(np.dot(p, g)) >= 0.0:
            H = np.eye(n)  #force descent if numerical issues
            p = -g

        #strong wolfe line search (returns g at new point so we reuse it)
        alpha, fx_new, g_new, ls_evals = _strong_wolfe_line_search(f, x, fx, g, p)

        s = alpha * p
        y = g_new - g

        sTy = float(np.dot(s, y))

        #update H (inverse hessian)
        if m == "BFGS":
            if sTy > eps:
                rho = 1.0 / sTy
                I = np.eye(n)
                V = I - rho * np.outer(s, y)
                H = V @ H @ V.T + rho * np.outer(s, s)
            else:
                if restart:
                    H = np.eye(n)  #if curvature fails, restart instead of corrupting H
        else:  #DFP
            Hy = H @ y
            yTHy = float(np.dot(y, Hy))
            if sTy > eps and yTHy > eps:
                H = H + (np.outer(s, s) / sTy) - (np.outer(Hy, Hy) / yTHy)
            else:
                if restart:
                    H = np.eye(n)  #curvature/denominator safety

        #advance
        x, fx, g = x + s, fx_new, g_new
        normg = float(np.linalg.norm(g))
        k += 1

        if printout:
            print(f"{k:4d}  {x}  {fx:.6e}  {normg:.6e}  {ls_evals:11d}  {alpha:.6e}")

    N_eval = int(f.calls)
    N_iter = int(k)
    return x, N_eval, N_iter, normg  #returns normg (not f(x))
