# High-Performance Monte Carlo Pricing & Risk Engine

Hi, I’m Yoni Arbel — combining Pure Mathematics at Sorbonne University and Business Administration at IE University, with a strong focus on quantitative finance. 

This repository contains a production-grade pricing and risk engine for European Vanilla Options. While standard academic projects often stop at pricing, this engine goes further by implementing a robust risk management framework, calculating first and second-order Greeks ($\Delta, \Gamma, \nu$) using Central Finite Differences and Common Random Numbers (CRN) for variance reduction.

## 🚀 Key Features
* **Vectorized Monte Carlo Engine:** Simulates 1,000,000 price paths in ~0.02 seconds using pure `numpy` vectorization (zero `for` loops).
* **Mathematical Benchmarking:** Real-time cross-validation against exact Black-Scholes analytical prices to compute absolute pricing errors.
* **Advanced Risk Analytics:** Computes Delta, Gamma, and Vega using the "Bump-and-Reset" method applied over stochastic trajectories.
* **Visual Diagnostics:** Generates 3D and 2D convergence dashboards to analyze stochastic paths, the Law of Large Numbers, and payoff probability distributions.

---

## 🧮 Mathematical Framework

### 1. Geometric Brownian Motion (GBM)
The underlying asset price $S_t$ is modeled under the risk-neutral measure $\mathbb{Q}$ using the following Stochastic Differential Equation (SDE):
$$dS_t=r S_t dt+\sigma S_t dW_t$$

Using Ito's Lemma, the exact solution for the terminal stock price at maturity $T$ is:
$$S_T=S_0\exp((r-\frac{1}{2}\sigma^2)T+\sigma\sqrt{T}Z)$$
Where $Z \sim \mathcal{N}(0,1)$.

### 2. Pricing the Option
The price of the European Call is the discounted expected value of the terminal payoff:
$$C_0=e^{-rT} \mathbb{E}^{\mathbb{Q}}[\max(S_T-K, 0)]$$

---

## ⚙️ The Computational Engine: Vectorization & CRN

A naive Monte Carlo implementation uses nested loops, resulting in heavy computational overhead. This engine leverages array programming to compute millions of paths concurrently. 

Furthermore, to calculate the Greeks reliably, the engine utilizes **Common Random Numbers (CRN)**. When shocking (bumping) the spot price or volatility to calculate sensitivities, using the *same* random matrix $Z$ isolates the effect of the parameter change, eliminating statistical noise.

### Calculating The Greeks (Central Finite Differences)
The engine calculates risk metrics using two-sided numerical differentiation:

* **Delta ($\Delta$):** Directional risk.
$$\Delta \approx \frac{P(S_0+\delta S, \sigma)-P(S_0-\delta S, \sigma)}{2\delta S}$$

* **Gamma ($\Gamma$):** Convexity risk.
$$\Gamma \approx \frac{P(S_0+\delta S, \sigma)-2P(S_0, \sigma)+P(S_0-\delta S, \sigma)}{(\delta S)^2}$$

* **Vega ($\nu$):** Volatility risk (scaled to 1%).
$$\nu \approx \frac{P(S_0, \sigma+\delta \sigma)-P(S_0, \sigma-\delta \sigma)}{2\delta \sigma}$$

---

## 📊 Visual Analytics & Dashboards

### 1. Stochastic Trajectories (GBM)
Monte-Carlo-Risk-Engine/trajectories.png
*Visualization of 100 simulated asset paths across 252 trading days, demonstrating the exponential growth and variance dispersion over time.*

### 2. Monte Carlo Convergence
![Convergence](assets/convergence.png)
*Tracking the running mean of the Monte Carlo estimator as it asymptotically converges to the exact Black-Scholes analytical price (Law of Large Numbers).*

### 3. Payoff Probability Distribution
![Distribution](assets/distribution.png)
*Histogram showing the asymmetric risk profile of the discounted Call payoffs. The massive spike at zero represents all Out-of-The-Money (OTM) scenarios.*

---

## 💻 Tech Stack & Usage
* **Language:** Python 3.10+
* **Libraries:** `NumPy` (Vectorized Math), `SciPy` (Statistics & Benchmarking), `Matplotlib` (Data Visualization).

To run the engine locally:
```bash
python mc_pricer.py
