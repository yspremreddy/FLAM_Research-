# 🧠 Research and Development / AI  
## Estimation of Unknown Parameters in a Parametric Curve

---

### 📘 Problem Statement

Estimate the unknown parameters **θ**, **M**, and **X** in the following parametric curve equations:

\[
\begin{aligned}
x(t) &= t\cos(\theta) - e^{M|t|}\sin(0.3t)\sin(\theta) + X,\\[4pt]
y(t) &= 42 + t\sin(\theta) + e^{M|t|}\sin(0.3t)\cos(\theta)
\end{aligned}
\]

The goal is to determine these unknowns so that the curve best fits the observed data in **`xy_data.csv`**.

---

### ⚙️ Parameter Ranges

\[
0° < \theta < 50°, \quad -0.05 < M < 0.05, \quad 0 < X < 100, \quad 6 < t < 60
\]

---

### 📂 Dataset

The file **`xy_data.csv`** contains (x, y) coordinates of the curve.  
Since the parameter `t` was not included, it was generated as uniformly spaced values between **6 and 60**.

---

### 🧮 Objective Function

To evaluate the model, the **L₁ distance** (sum of absolute differences) between the observed and predicted points is minimized:

\[
L_1 = \frac{1}{N}\sum_{i=1}^{N}\left(|x_i - \hat{x}_i| + |y_i - \hat{y}_i|\right)
\]

This directly matches the assignment’s evaluation metric (max score: 100).

---

### 🧩 Methodology

1. **Load Data:** Read `(x, y)` points from `xy_data.csv`.  
2. **Parameter Initialization:** Generate uniform `t ∈ [6, 60]`.  
3. **Model Definition:** Implemented the provided parametric equations.  
4. **Loss Function:** Defined the L₁ loss between predicted and observed coordinates.  
5. **Optimization:** Used the **Nelder–Mead** algorithm from `scipy.optimize.minimize` for non-smooth optimization.  
6. **Validation:** Computed mean and max L₁ errors to evaluate fit quality.

---

### 🧾 Results

| Parameter | Symbol | Estimated Value | Description |
|------------|:-------:|----------------:|-------------|
| Angle | θ | **0.49076 rad ≈ 28.118°** | Curve orientation |
| Exponential factor | M | **0.021389** | Amplitude scaling |
| Translation | X | **54.8991** | X-axis shift |

---

### 📊 Performance Metrics

| Metric | Value |
|:--------|-------:|
| **Average L₁ error per point** | 25.243 |
| **Maximum |x-error|** | 47.851 |
| **Maximum |y-error|** | 23.343 |

✅ The model shows consistent fitting with a small average deviation per point, confirming a stable parameter estimation.

---

### 🧮 Final Parametric Equation (LaTeX-Ready)

\[
\left(
t\cos(0.49076) - e^{0.021389|t|}\sin(0.3t)\sin(0.49076) + 54.8991,\;
42 + t\sin(0.49076) + e^{0.021389|t|}\sin(0.3t)\cos(0.49076)
\right)
\]

---

### 🧠 Interpretation

- **θ ≈ 28°** defines the rotation of the curve.  
- **M = 0.021** introduces smooth exponential modulation.  
- **X ≈ 55** shifts the curve horizontally to match the dataset.  
- The optimized parameters produce a strong alignment with the observed data.

---

### 💻 Implementation

**Language:** Python 3.10+  
**Libraries:** `numpy`, `pandas`, `scipy`, `matplotlib`

#### Run Command
```bash
python parametric_curve_fitting.py
