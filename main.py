"""
Assignment: Research and Development / AI
Task: Estimate Unknown Parameters in a Parametric Curve

Author: [Your Name]
Institution: [Your College Name]
Date: November 2025

Description:
-------------
This script estimates the unknown parameters θ, M, and X
in the given parametric equations of a curve using provided (x, y) data.

Parametric Equations:
---------------------
x = t * cos(θ) - e^(M|t|) * sin(0.3t) * sin(θ) + X
y = 42 + t * sin(θ) + e^(M|t|) * sin(0.3t) * cos(θ)

Parameter Ranges:
-----------------
0° < θ < 50°
-0.05 < M < 0.05
0 < X < 100
6 < t < 60

Optimization Criterion:
-----------------------
The objective minimizes the L1 distance between predicted and observed (x, y)
to estimate θ, M, and X that best fit the given dataset.
"""

# ----------------------------- IMPORT LIBRARIES ----------------------------- #
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ------------------------------- LOAD DATA --------------------------------- #
# Ensure the xy_data.csv file is in the same directory as this script.
data = pd.read_csv("xy_data.csv")

# The CSV contains x and y columns. We infer t as uniform in range (6, 60).
x_obs = data["x"].values
y_obs = data["y"].values
n = len(data)
t = np.linspace(6, 60, n)

# ---------------------------- DEFINE MODEL --------------------------------- #
def parametric_model(theta, M, X, t):
    """
    Computes the parametric curve for given parameters.
    """
    exp_term = np.exp(M * np.abs(t))
    sin_term = np.sin(0.3 * t)

    x = t * np.cos(theta) - exp_term * sin_term * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + exp_term * sin_term * np.cos(theta)
    return x, y

# ------------------------ DEFINE OBJECTIVE FUNCTION ------------------------ #
def l1_loss(params):
    """
    Objective: Sum of L1 distances between observed and predicted (x, y)
    """
    theta, M, X = params
    x_pred, y_pred = parametric_model(theta, M, X, t)
    err = np.abs(x_obs - x_pred) + np.abs(y_obs - y_pred)
    return np.sum(err)

# ------------------------------ BOUNDS ------------------------------------ #
theta_bounds = (np.deg2rad(0), np.deg2rad(50))
M_bounds = (-0.05, 0.05)
X_bounds = (0, 100)

# Initial guess (approximate midpoints)
initial_guess = [np.deg2rad(25), 0.0, 50.0]

# ----------------------------- OPTIMIZATION ------------------------------- #
print("Starting optimization...")

result = minimize(
    l1_loss,
    initial_guess,
    method="Nelder-Mead",
    options={"maxiter": 5000, "fatol": 1e-8, "disp": True},
)

theta_est, M_est, X_est = result.x

# ----------------------------- RESULTS ------------------------------------ #
theta_deg = np.rad2deg(theta_est)
x_pred, y_pred = parametric_model(theta_est, M_est, X_est, t)
avg_L1 = np.mean(np.abs(x_obs - x_pred) + np.abs(y_obs - y_pred))
max_err_x = np.max(np.abs(x_obs - x_pred))
max_err_y = np.max(np.abs(y_obs - y_pred))

print("\n--- Final Estimated Parameters ---")
print(f"Theta (radians): {theta_est:.6f}")
print(f"Theta (degrees): {theta_deg:.6f}")
print(f"M: {M_est:.8f}")
print(f"X: {X_est:.6f}")
print(f"\nAverage L1 Error: {avg_L1:.4f}")
print(f"Max Absolute Error (x): {max_err_x:.4f}")
print(f"Max Absolute Error (y): {max_err_y:.4f}")

# ----------------------------- EQUATION OUTPUT ---------------------------- #
print("\n--- Final Parametric Equation (LaTeX Format) ---")
print(
    rf"\left(t\cos({theta_est:.6f}) - e^{{{M_est:.6f}|t|}}\sin(0.3t)\sin({theta_est:.6f}) + {X_est:.6f}, "
    rf"42 + t\sin({theta_est:.6f}) + e^{{{M_est:.6f}|t|}}\sin(0.3t)\cos({theta_est:.6f})\right)"
)

# ------------------------------ PLOTTING ---------------------------------- #
plt.figure(figsize=(8, 6))
plt.plot(x_obs, y_obs, 'o', label="Observed Data", alpha=0.7)
plt.plot(x_pred, y_pred, '-', linewidth=2, label="Predicted Curve", color='r')
plt.title("Observed vs Predicted Curve")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fitted_curve.png")
plt.show()

# -------------------------- SAVE RESULTS ---------------------------------- #
with open("results_summary.txt", "w") as f:
    f.write("=== Estimated Parameters ===\n")
    f.write(f"Theta (radians): {theta_est:.6f}\n")
    f.write(f"Theta (degrees): {theta_deg:.6f}\n")
    f.write(f"M: {M_est:.8f}\n")
    f.write(f"X: {X_est:.6f}\n")
    f.write("\n=== Performance Metrics ===\n")
    f.write(f"Average L1 Error: {avg_L1:.6f}\n")
    f.write(f"Max Error X: {max_err_x:.6f}\n")
    f.write(f"Max Error Y: {max_err_y:.6f}\n")

print("\nResults saved in 'results_summary.txt' and curve plot as 'fitted_curve.png'.")
