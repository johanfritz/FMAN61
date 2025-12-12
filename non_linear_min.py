from rosenbrock import rosenbrock
from grad import grad_c, jacobian_c
from typing import Tuple, Callable
import numpy as np
from warnings import warn


def non_linear_min(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    method: str,
    tol: float,
    restart: bool,
    printout: bool,
    plot=False,
) -> Tuple[np.ndarray, int, int, float]:
    """Minimizes an objective function using different non-linear methods.

    Implements quasi-Newton methods for unconstrained optimization. The function
    minimizes a scalar-valued objective function starting from an initial guess
    using either DFP or BFGS updating schemes.

    Args:
        f: Objective function to be minimized. Must accept a 1D numpy array
           and return a float.
        x0: Initial guess for the minimum. Should be a 1D numpy array.
        method: Minimization method to use. Options are:
            - "DFP": Davidon-Fletcher-Powell method
            - "BFGS": Broyden-Fletcher-Goldfarb-Shanno method
        tol: Termination tolerance. Algorithm stops when the gradient norm
             falls below this value.
        restart: Whether to restart the minimization if the Hessian approximation
                 becomes non-positive definite. If True, resets the Hessian to
                 identity and continues.
        printout: If True, prints iteration progress and final results.

    Returns:
        A tuple containing:
            - sol: Solution x* that minimizes the objective function
            - N_eval: Number of function evaluations performed
            - N_iter: Number of iterations completed
            - normg: 2-norm of the gradient at the solution x*

    Raises:
        ValueError: If method is not "DFP" or "BFGS".
        ValueError: If x0 is not array-like
        RuntimeError: If maximum iterations exceeded without convergence.

    Examples:


    Notes:

    """
    if method not in ["DFP", "BFGS"]:
        raise ValueError(f"Unsupported optimization method {method}")
    try:
        x0 = np.array(
            x0, float
        )  # needs float conversion or the provided gradient function will not work
    except Exception:
        raise TypeError("Initial guess is unknown or cannot be converted to float")
    if plot:
        import matplotlib.pyplot as plt

        # Create grid
        x_vals = np.linspace(-2, 2, 100)
        y_vals = np.linspace(-1, 3, 100)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Evaluate function at each grid point
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

        # Plot
        plt.figure(figsize=(10, 8))
        contour = plt.contour(X, Y, Z, levels=30, cmap="viridis")
    # variable initiazation:
    max_iter = 100
    iteration = 0
    H = np.eye(x0.shape[0])
    g = grad_c(f, x0)
    x = x0.copy()
    norm = np.linalg.norm

    def _linesearch(
        f: Callable[[np.ndarray], float],
        x: np.ndarray,
        p: np.ndarray,
        g: np.ndarray,
    ) -> float:
        """Helper function for line search
        Args:
            f: Objective function
            x: Starting point
            p: Search direction
            g: Gradient at x
            mthd: Search method. Options:
                -"armijo": based on Armijo condition (default)
                -"wolfe": based on Wolfe condition

        Returns:
            alpha: step size as a result of line seach
        Raises:
            RuntimeError: If step size is not found
        """
        m = np.sum(p * g)
        if m > 0:
            warn(
                f"Invalid descent direction encountered during line search p.T@g={m:2f}>0",
                RuntimeWarning,
            )
        alpha = 1
        tau = 0.5
        c=1e-4
        f_0 = f(x)
        while f(x + alpha * p) > f_0 + c * m * alpha:
            alpha *= tau
        return alpha

    def _hesse(s: np.ndarray, y: np.ndarray, H: np.ndarray, mthd: str) -> np.ndarray:
        """Helper function to update the Hesse matrix by some rule
        Args:
            s: Step difference
            y: Gradient difference
            H: Previous (inverse) Hesse matrix
            mthd: Update method, "DFP" or "BFGS"

        Returns:
            H: Updated (inverse) Hesse matrix
        """
        if mthd not in ["DFP", "BFGS"]:
            raise ValueError("Only DFP and BFGS are supported QN schemes")
        curve = s.T @ y

        if mthd == "DFP":
            return (
                H - H @ np.outer(y, y) @ H / (y.T @ H @ y) + np.outer(s, s) / (curve)
            )
        if mthd == "BFGS":
            rho = 1 / (curve)
            I = np.eye(s.shape[0])
            # first=(np.outer(s, y)+np.outer(y, H@y))@np.outer(s,s)/((s@y)**2)
            # last=(H@np.outer(y, s)+np.outer(s, y)@H)/(s@y)
            # return H+first-last
            return (I - rho * s @ y.T) @ H @ (I - rho * y @ s.T) + rho * np.outer(s, s)
        raise ValueError("Only DFP and BFGS are supported")

    # Main iteration loop:
    for _ in range(max_iter):
        iteration += 1
        d = -H @ g
        if d.T@g>0:
            H=np.eye(x.shape[0])
            d=-g
            print(f"Reset Hesse matrix due to invalid descent direction") if printout else None
        alpha = _linesearch(f, x, d, g)
        new_x = x + alpha * d
        if plot:
            plt.plot([x[0], new_x[0]], [x[1], new_x[1]], "-b")
        new_g = grad_c(f, new_x)
        s = alpha*d
        y = new_g - g
        #Curvature condition check:
        curve=s.T@y
        if curve>0:
            H = _hesse(s, y, H, method)
            print(f"Updated Hess matrix according to {method} method") if printout else None
        elif restart:
            H=np.eye(x.shape[0])
            print("Curvature condition failed, reset Hesse matrix to identity") if printout else None
        else:
            print("Curvature condition falied!!") if printout else None
        x = new_x
        g = grad_c(f, x)
        if norm(g) < tol:
            # optional plot:
            if plot:
                plt.plot(x[0], x[1], "x")
                plt.show()
            return x, 0, 0, np.linalg.norm(g)
    raise RuntimeError(f"QN did not converge after {max_iter} iterations")


if __name__ == "__main__":
    for _ in range(1):
        start = np.random.rand(2) * 10
        tol = 1e-5
        f = lambda x: np.sum(x**2)
        print(non_linear_min(rosenbrock, start, "DFP", tol, True, True, False))
        print(non_linear_min(rosenbrock, start, "BFGS", tol, True, True, False))
