# Quantitative Portfolio Optimization & Asset Allocation Engine

An automated Python data pipeline and Streamlit terminal that calculates the Markowitz Efficient Frontier and extracts the Maximum Sharpe Ratio weights for any given basket of equities.

## Core Architecture
By leveraging live historical asset data via the Yahoo Finance API, this application maps the Efficient Frontier, calculates annualized returns and covariance matrices, and deploys Non-Gradient Optimization (Simulated Annealing & Evolutionary Strategy) alongside SLSQP to find the optimal asset allocation.

* **Data Ingestion:** Vectorized financial data extraction via `yfinance` and `pandas`.
* **Matrix Computation:** Multi-dimensional covariance tensors and expected return modeling via `NumPy`.
* **Optimization Engine:** Non-linear optimization utilizing `SciPy` (SLSQP & Stochastic Metaheuristics).
* **Visualization:** Renders dynamic 3D topology and Efficient Frontier charts via `Plotly` and `Streamlit`.

## How to Run Locally
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt