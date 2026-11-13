# Financial Portfolio Optimisation & Asset Allocation

**Student:** Yog Dhiraj Chopada (PRN: 20240802178)  
**Course:** B.Tech Computer Science Engineering  
**Project Focus:** Quantitative Finance & Non-Gradient Optimization (SLSQP / Monte Carlo Simulation)

## Overview
This capstone project implements an automated portfolio optimization agent. By leveraging historical asset data via the Yahoo Finance API, the application maps the Markowitz Efficient Frontier, calculates annualized returns and covariance matrices, and uses Sequential Least Squares Programming (SLSQP) to find the optimal asset allocation weights maximizing the Sharpe Ratio.

## Tech Stack
* **Python** (Core Logic)
* **Streamlit** (Interactive UI & Dashboard)
* **SciPy** (Non-Gradient Optimization - SLSQP)
* **Pandas & NumPy** (Vectorized Financial Calculations)
* **Plotly** (Interactive Visualizations & Efficient Frontier Plots)

## How to Run Locally
1. Clone the repository and navigate to the `capstone` folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt