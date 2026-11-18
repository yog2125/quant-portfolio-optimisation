import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from scipy.optimize import minimize, differential_evolution

# ==========================================
# 1. PAGE CONFIGURATION & QUANT TERMINAL UI
# ==========================================
st.set_page_config(
    page_title="QuantOptima Terminal | Portfolio Optimization Engine",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark Cyber-Quant Terminal Theme */
    .stApp {
        background: radial-gradient(circle at top left, #0a0f1d 0%, #05070e 100%);
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid #1e293b;
    }
    /* High-Tech Telemetry Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-left: 4px solid #00f2fe;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800 !important;
        font-size: 1.55rem !important;
    }
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.4px;
    }
    .terminal-banner {
        background: linear-gradient(90deg, rgba(14,165,233,0.15) 0%, rgba(99,102,241,0.05) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 22px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #0b1120;
        padding: 8px;
        border-radius: 10px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="terminal-banner">
    <h1 style="margin:0; font-size: 2.1rem;">⚡ QUANTOPTIMA // STOCHASTIC ASSET ALLOCATION ENGINE</h1>
    <p style="margin:6px 0 0 0; color:#38bdf8; font-family:monospace; font-size:0.95rem;">
        Non-Gradient Metaheuristic Search (Simulated Annealing & Evolutionary Strategy) • Tail-Risk CVaR • Live US Execution Engine
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR: HYPERPARAMETERS & CONSTRAINTS
# ==========================================
st.sidebar.markdown("### 🎛️ 1. Asset Universe (US Equities)")

US_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "BRK-B", "JPM", "V", 
    "JNJ", "WMT", "PG", "MA", "XOM", "UNH", "HD", "CVX", "LLY", "ABBV", 
    "BAC", "KO", "PFE", "COST", "CSCO"
]

default_picks = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "COST", "LLY", "META"]
selected_tickers = st.sidebar.multiselect("Select Portfolio Constituents", US_UNIVERSE, default=default_picks)

if len(selected_tickers) < 3:
    st.warning("⚠️ Select at least 3 US assets in the sidebar to initialize multi-dimensional covariance & non-gradient search.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### 💼 2. Real-World Mandate")
capital_usd = st.sidebar.number_input("Deployable Capital ($ USD)", min_value=10000, max_value=10000000, value=100000, step=10000)
lookback_years = st.sidebar.slider("Historical Lookback Window (Years)", 1, 5, 3)
risk_free_rate = st.sidebar.number_input("US 10-Year Treasury Yield (%)", value=4.40, step=0.10) / 100.0

min_floor = round(1.0 / len(selected_tickers) + 0.05, 2)
max_weight_cap = st.sidebar.slider(
    "Max Single-Stock Concentration Cap (%)", 
    min_value=int(min(min_floor * 100, 95)), 
    max_value=100, 
    value=min(40, max(int(min_floor * 100) + 5, 35)), 
    step=5
) / 100.0

objective_mode = st.sidebar.selectbox(
    "Optimization Objective Function",
    [
        "Maximize Sharpe Ratio (Risk-Adjusted Return)",
        "Maximize Sortino Ratio (Downside Risk Only)",
        "Minimize Tail Risk (95% CVaR / Expected Shortfall)",
        "Minimize Global Portfolio Variance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧬 3. Non-Gradient Hyperparameters")
sa_iterations = st.sidebar.slider("Simulated Annealing Iterations", 300, 2500, 1000, step=100)
sa_temp = st.sidebar.slider("Initial Thermal Temperature (T₀)", 1.0, 20.0, 10.0, step=1.0)
evo_generations = st.sidebar.slider("Evolutionary Strategy Generations", 50, 400, 150, step=25)
evo_pop_size = st.sidebar.slider("Evolutionary Population Size (λ)", 20, 100, 40, step=10)

# ==========================================
# 3. LIVE DATA INGESTION & QUANT METRICS
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(tickers, years):
    raw = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close']
    else:
        prices = raw[['Close']] if 'Close' in raw.columns else raw
    prices.dropna(how='all', inplace=True)
    prices.ffill(inplace=True)
    prices.dropna(inplace=True)
    return prices

with st.spinner("🔄 Synchronizing US Market Historical OHLCV & Computing Covariance Tensors..."):
    prices = fetch_market_data(selected_tickers, lookback_years)

if prices.empty or len(prices.columns) < 3:
    st.error("⚠️ Failed to retrieve sufficient overlapping history for the selected tickers.")
    st.stop()

clean_names = [str(col) for col in prices.columns]
prices.columns = clean_names
num_assets = len(clean_names)

returns = prices.pct_change().dropna()
mean_returns = returns.mean().values * 252
cov_matrix = returns.cov().values * 252
returns_matrix = returns.values

# Core Quantitative Evaluation Engine
def evaluate_portfolio(weights):
    w = np.array(weights)
    port_daily = returns_matrix @ w
    ann_ret = np.sum(mean_returns * w)
    ann_vol = np.sqrt(w.T @ cov_matrix @ w)
    sharpe = (ann_ret - risk_free_rate) / (ann_vol + 1e-9)
    
    # Downside Deviation & Sortino Ratio
    downside_diff = np.minimum(0, port_daily - (risk_free_rate / 252.0))
    downside_vol = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252)
    sortino = (ann_ret - risk_free_rate) / (downside_vol + 1e-9)
    
    # 95% Historical Value at Risk (VaR) & Conditional VaR (Expected Shortfall)
    var_95_daily = -np.percentile(port_daily, 5)
    tail_losses = port_daily[port_daily <= -var_95_daily]
    cvar_95_daily = -np.mean(tail_losses) if len(tail_losses) > 0 else var_95_daily
    cvar_95_ann = cvar_95_daily * np.sqrt(252)
    
    return ann_ret, ann_vol, sharpe, sortino, var_95_daily, cvar_95_daily, cvar_95_ann

def repair_weights(w, max_cap):
    """Projects any weight vector onto the simplex sum(w)=1 with 0 <= w_i <= max_cap"""
    w = np.maximum(0.0, np.array(w, dtype=float))
    if np.sum(w) <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w /= np.sum(w)
    for _ in range(12):
        over = w > max_cap
        if not np.any(over):
            break
        excess = np.sum(w[over] - max_cap)
        w[over] = max_cap
        under = ~over
        if np.any(under):
            room = max_cap - w[under]
            if np.sum(room) > 0:
                w[under] += excess * (room / np.sum(room))
    w /= np.sum(w)
    return w

def objective_loss(weights):
    ann_ret, ann_vol, sharpe, sortino, _, _, cvar_ann = evaluate_portfolio(weights)
    if "Sharpe" in objective_mode:
        return -sharpe
    elif "Sortino" in objective_mode:
        return -sortino
    elif "Tail Risk" in objective_mode:
        return cvar_ann - 0.25 * ann_ret
    else:
        return ann_vol

# ==========================================
# 4. CUSTOM NON-GRADIENT SEARCH ALGORITHMS
# ==========================================
@st.cache_data(show_spinner=False)
def run_non_gradient_engines(ret_hash, max_cap, obj_mode, sa_iter, sa_t0, evo_gen, evo_pop, rf_rate):
    np.random.seed(42)
    
    # --- ENGINE 1: SIMULATED ANNEALING (Metropolis-Hastings Search) ---
    current_w = repair_weights(np.ones(num_assets) / num_assets, max_cap)
    current_loss = objective_loss(current_w)
    best_sa_w = current_w.copy()
    best_sa_loss = current_loss
    
    sa_history = []
    sa_temp_schedule = []
    
    for step in range(sa_iter):
        T = sa_t0 * ((0.005 / sa_t0) ** (step / max(1, sa_iter - 1)))
        sa_temp_schedule.append(T)
        
        # Non-gradient stochastic perturbation in weight simplex
        perturbation = np.random.normal(0, 0.08 * (0.3 + 0.7 * (1 - step / sa_iter)), num_assets)
        candidate_w = repair_weights(current_w + perturbation, max_cap)
        candidate_loss = objective_loss(candidate_w)
        
        delta = candidate_loss - current_loss
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-6)):
            current_w = candidate_w
            current_loss = candidate_loss
            if current_loss < best_sa_loss:
                best_sa_loss = current_loss
                best_sa_w = current_w.copy()
        sa_history.append(-best_sa_loss if "Minimize" not in obj_mode else best_sa_loss)

    # --- ENGINE 2: (μ + λ) EVOLUTIONARY STRATEGY ---
    mu = max(4, evo_pop // 4)
    population = [repair_weights(np.random.dirichlet(np.ones(num_assets)), max_cap) for _ in range(evo_pop)]
    sigmas = [0.12 for _ in range(evo_pop)]
    evo_history = []
    
    tau = 1.0 / np.sqrt(2.0 * num_assets)
    for gen in range(evo_gen):
        fitness = np.array([objective_loss(ind) for ind in population])
        elite_idx = np.argsort(fitness)[:mu]
        elites = [population[i].copy() for i in elite_idx]
        elite_sigmas = [sigmas[i] for i in elite_idx]
        
        best_gen_loss = fitness[elite_idx[0]]
        evo_history.append(-best_gen_loss if "Minimize" not in obj_mode else best_gen_loss)
        
        new_pop = elites.copy()
        new_sigmas = elite_sigmas.copy()
        while len(new_pop) < evo_pop:
            p1, p2 = np.random.choice(mu, 2, replace=False)
            alpha = np.random.rand()
            child_sigma = max(0.005, (alpha * elite_sigmas[p1] + (1 - alpha) * elite_sigmas[p2]) * np.exp(tau * np.random.randn()))
            child_w = alpha * elites[p1] + (1 - alpha) * elites[p2] + np.random.normal(0, child_sigma, num_assets)
            new_pop.append(repair_weights(child_w, max_cap))
            new_sigmas.append(child_sigma)
        population = new_pop
        sigmas = new_sigmas
        
    final_fitness = np.array([objective_loss(ind) for ind in population])
    best_evo_w = population[np.argmin(final_fitness)]
    
    # --- ENGINE 3: QUASI-NEWTON SLSQP BENCHMARK ---
    init_w = np.ones(num_assets) / num_assets
    bounds = tuple((0.0, max_cap) for _ in range(num_assets))
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    slsqp_res = minimize(objective_loss, init_w, method='SLSQP', bounds=bounds, constraints=cons)
    best_slsqp_w = repair_weights(slsqp_res.x, max_cap)
    
    # --- MONTE CARLO CLOUD (3,000 Feasible Portfolios) ---
    mc_count = 3000
    mc_vols, mc_rets, mc_sharpes, mc_cvars = np.zeros(mc_count), np.zeros(mc_count), np.zeros(mc_count), np.zeros(mc_count)
    for i in range(mc_count):
        w_mc = repair_weights(np.random.dirichlet(np.ones(num_assets)), max_cap)
        r_m, v_m, sh_m, _, _, _, cv_m = evaluate_portfolio(w_mc)
        mc_rets[i], mc_vols[i], mc_sharpes[i], mc_cvars[i] = r_m, v_m, sh_m, cv_m
        
    return best_sa_w, best_evo_w, best_slsqp_w, sa_history, sa_temp_schedule, evo_history, (mc_vols, mc_rets, mc_sharpes, mc_cvars)

ret_fingerprint = float(np.sum(mean_returns) + np.sum(cov_matrix))
best_sa_w, best_evo_w, best_slsqp_w, sa_hist, sa_temps, evo_hist, mc_cloud = run_non_gradient_engines(
    ret_fingerprint, max_weight_cap, objective_mode, sa_iterations, sa_temp, evo_generations, evo_pop_size, risk_free_rate
)

# Select winning metaheuristic portfolio
sa_score = objective_loss(best_sa_w)
evo_score = objective_loss(best_evo_w)
primary_w = best_evo_w if evo_score <= sa_score else best_sa_w
winning_algo = "Evolutionary Strategy (μ + λ)" if evo_score <= sa_score else "Simulated Annealing"

opt_ret, opt_vol, opt_sharpe, opt_sortino, opt_var95, opt_cvar95, _ = evaluate_portfolio(primary_w)
eq_w = np.ones(num_assets) / num_assets
eq_ret, eq_vol, eq_sharpe, eq_sortino, eq_var95, eq_cvar95, _ = evaluate_portfolio(eq_w)

# ==========================================
# 5. TOP EXECUTIVE TELEMETRY BAR
# ==========================================
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Expected Annual Return", f"{opt_ret*100:.2f}%", f"{(opt_ret - eq_ret)*100:+.2f}% vs Equal-Wt")
m2.metric("Annualized Volatility", f"{opt_vol*100:.2f}%", f"{(opt_vol - eq_vol)*100:+.2f}% vs Equal-Wt", delta_color="inverse")
m3.metric("Sharpe Ratio", f"{opt_sharpe:.2f}", f"{opt_sharpe - eq_sharpe:+.2f} Alpha")
m4.metric("Sortino Ratio", f"{opt_sortino:.2f}", f"{opt_sortino - eq_sortino:+.2f} Downside")
m5.metric("95% Daily VaR (USD)", f"${opt_var95 * capital_usd:,.0f}", f"-{opt_var95*100:.2f}% Cutoff", delta_color="off")
m6.metric("95% Tail CVaR (USD)", f"${opt_cvar95 * capital_usd:,.0f}", f"Winner: {winning_algo[:12]}", delta_color="off")

# ==========================================
# 6. MULTI-TAB QUANTITATIVE WORKSPACE
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧬 1. Non-Gradient Convergence Lab",
    "🎯 2. Efficient Frontier & 3D Surface",
    "💼 3. Live US Equity Order Book",
    "📈 4. Out-of-Sample Backtest & Drawdowns",
    "🔥 5. Correlation & Black-Swan Stress Test"
])

# ------------------------------------------
# TAB 1: NON-GRADIENT CONVERGENCE LAB
# ------------------------------------------
with tab1:
    st.subheader("Stochastic Metaheuristic Convergence vs. Gradient Benchmark")
    st.markdown(
        f"**Active Objective:** `{objective_mode}` | **Position Cap Constraint:** `w_i ≤ {max_weight_cap*100:.0f}%` | "
        f"**Winning Engine:** `{winning_algo}`"
    )
    
    col_conv1, col_conv2 = st.columns([3, 2])
    
    with col_conv1:
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            y=sa_hist, mode='lines', name='Simulated Annealing Trajectory',
            line=dict(color='#00f2fe', width=2.5)
        ))
        evo_x = np.linspace(0, len(sa_hist) - 1, len(evo_hist))
        fig_conv.add_trace(go.Scatter(
            x=evo_x, y=evo_hist, mode='lines+markers', name='Evolutionary Strategy (μ+λ) Elites',
            line=dict(color='#f43f5e', width=2.5), marker=dict(size=4)
        ))
        fig_conv.update_layout(
            title="Optimization Convergence Curve (Objective Fitness vs. Search Evaluations)",
            xaxis_title="Search Evaluations / Cooling Steps",
            yaxis_title="Objective Fitness Score",
            template="plotly_dark", height=420,
            paper_bgcolor='rgba(15,23,42,0.7)', plot_bgcolor='rgba(15,23,42,0.4)',
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_conv, use_container_width=True)
        
    with col_conv2:
        algos = {
            "Simulated Annealing": best_sa_w,
            "Evolutionary Strategy (μ+λ)": best_evo_w,
            "SLSQP Quasi-Newton": best_slsqp_w,
            "Equal-Weight (1/N)": eq_w
        }
        comp_rows = []
        for name, w_vec in algos.items():
            r_a, v_a, sh_a, so_a, vr_a, cv_a, _ = evaluate_portfolio(w_vec)
            comp_rows.append({
                "Algorithm": name,
                "Ann. Return (%)": round(r_a * 100, 2),
                "Volatility (%)": round(v_a * 100, 2),
                "Sharpe": round(sh_a, 3),
                "Sortino": round(so_a, 3),
                "Daily CVaR 95% (%)": round(cv_a * 100, 2)
            })
        df_comp = pd.DataFrame(comp_rows)
        st.markdown("#### 🏆 Multi-Solver Benchmark Table")
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
        
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            y=sa_temps, mode='lines', fill='tozeroy',
            line=dict(color='#f59e0b', width=2), name='Temperature (T)'
        ))
        fig_temp.update_layout(
            title="Simulated Annealing Metropolis Cooling Schedule (T₀ → T_final)",
            xaxis_title="Iteration", yaxis_title="Temperature (T)",
            template="plotly_dark", height=215,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(15,23,42,0.7)', plot_bgcolor='rgba(15,23,42,0.4)'
        )
        st.plotly_chart(fig_temp, use_container_width=True)

# ------------------------------------------
# TAB 2: EFFICIENT FRONTIER & 3D SURFACE
# ------------------------------------------
with tab2:
    st.subheader("Constrained Markowitz Efficient Frontier & Tail-Risk Topology")
    mc_vols, mc_rets, mc_sharpes, mc_cvars = mc_cloud
    
    ef_c1, ef_c2 = st.columns([3, 2])
    with ef_c1:
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=mc_vols * 100, y=mc_rets * 100, mode='markers',
            marker=dict(
                color=mc_sharpes, colorscale='Turbo', showscale=True, size=6, opacity=0.75,
                colorbar=dict(title="Sharpe")
            ),
            name="Feasible Dirichlet Cloud"
        ))
        indiv_vols = np.sqrt(np.diag(cov_matrix)) * 100
        indiv_rets = mean_returns * 100
        fig_ef.add_trace(go.Scatter(
            x=indiv_vols, y=indiv_rets, mode='markers+text',
            text=clean_names, textposition="top center",
            marker=dict(color='#e2e8f0', size=10, symbol='diamond', line=dict(width=1, color='#000')),
            name="Individual US Stocks"
        ))
        fig_ef.add_trace(go.Scatter(
            x=[opt_vol * 100], y=[opt_ret * 100], mode='markers',
            marker=dict(color='#00ff87', size=18, symbol='star', line=dict(width=2, color='#000')),
            name=f"Optimal ({winning_algo[:12]})"
        ))
        fig_ef.add_trace(go.Scatter(
            x=[eq_vol * 100], y=[eq_ret * 100], mode='markers',
            marker=dict(color='#f43f5e', size=14, symbol='x'),
            name="1/N Equal-Weight Benchmark"
        ))
        fig_ef.update_layout(
            title="Risk-Return Efficient Frontier with Individual Asset Coordinates",
            xaxis_title="Annualized Volatility / Standard Deviation (%)",
            yaxis_title="Expected Annualized Return (%)",
            template="plotly_dark", height=500,
            paper_bgcolor='rgba(15,23,42,0.7)', plot_bgcolor='rgba(15,23,42,0.4)',
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_ef, use_container_width=True)
        
    with ef_c2:
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=mc_vols * 100, y=mc_cvars * 100, z=mc_rets * 100,
            mode='markers',
            marker=dict(size=3.5, color=mc_sharpes, colorscale='Viridis', opacity=0.8),
            name="Simulated Portfolios"
        )])
        fig_3d.add_trace(go.Scatter3d(
            x=[opt_vol * 100], y=[opt_cvar95 * np.sqrt(252) * 100], z=[opt_ret * 100],
            mode='markers',
            marker=dict(size=10, color='#00ff87', symbol='diamond'),
            name="Optimal Portfolio"
        ))
        fig_3d.update_layout(
            title="3D Risk-Return-CVaR Fitness Landscape",
            scene=dict(
                xaxis_title="Volatility (%)",
                yaxis_title="Ann. CVaR 95% (%)",
                zaxis_title="Return (%)"
            ),
            template="plotly_dark", height=500,
            margin=dict(l=0, r=0, b=0, t=40),
            paper_bgcolor='rgba(15,23,42,0.7)'
        )
        st.plotly_chart(fig_3d, use_container_width=True)

# ------------------------------------------
# TAB 3: LIVE US EQUITY ORDER BOOK
# ------------------------------------------
with tab3:
    st.subheader(f"Real-World US Equity Execution Sheet (Capital Mandate: ${capital_usd:,.0f})")
    st.write("Translating continuous mathematical weights into **exact integer share quantities** based on live US closing prices, plus **Marginal Contribution to Risk (MCR)**.")
    
    latest_prices = prices.iloc[-1].values
    target_usd = primary_w * capital_usd
    exact_shares = np.floor(target_usd / latest_prices).astype(int)
    actual_invested = exact_shares * latest_prices
    unallocated_cash = capital_usd - np.sum(actual_invested)
    
    residual_order = np.argsort(-(target_usd - actual_invested))
    for idx in residual_order:
        if unallocated_cash >= latest_prices[idx] and (actual_invested[idx] + latest_prices[idx]) / capital_usd <= max_weight_cap:
            add_qty = int(unallocated_cash // latest_prices[idx])
            if add_qty > 0:
                exact_shares[idx] += add_qty
                actual_invested[idx] += add_qty * latest_prices[idx]
                unallocated_cash = capital_usd - np.sum(actual_invested)

    marginal_risk = (cov_matrix @ primary_w) / (opt_vol + 1e-9)
    component_risk = primary_w * marginal_risk
    risk_pct = (component_risk / np.sum(component_risk)) * 100

    order_df = pd.DataFrame({
        "Ticker Symbol": clean_names,
        "Live LTP ($)": np.round(latest_prices, 2),
        "Model Weight (%)": np.round(primary_w * 100, 2),
        "Risk Contribution (%)": np.round(risk_pct, 2),
        "Discrete Shares to BUY": exact_shares,
        "Actual Capital Deployed ($)": np.round(actual_invested, 2),
        "Realized Weight (%)": np.round((actual_invested / capital_usd) * 100, 2)
    }).sort_values(by="Model Weight (%)", ascending=False)

    ob_c1, ob_c2 = st.columns([3, 2])
    with ob_c1:
        st.dataframe(order_df, use_container_width=True, hide_index=True)
        st.info(
            f"💰 **Total Deployed Capital:** `${np.sum(actual_invested):,.2f}` ({np.sum(actual_invested)/capital_usd*100:.2f}%) | "
            f"🏦 **Residual Cash Balance:** `${unallocated_cash:,.2f}` | "
            f"📦 **Active Positions:** `{np.sum(exact_shares > 0)} / {num_assets} Stocks`"
        )
        csv_bytes = order_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export Broker Execution CSV (Interactive Brokers / Institutional Basket)",
            data=csv_bytes,
            file_name="US_Optimal_Order_Book.csv",
            mime="text/csv"
        )
        
    with ob_c2:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=order_df["Ticker Symbol"], y=order_df["Model Weight (%)"],
            name="Capital Weight (%)", marker_color='#00f2fe'
        ))
        fig_bar.add_trace(go.Bar(
            x=order_df["Ticker Symbol"], y=order_df["Risk Contribution (%)"],
            name="Euler Risk Contribution (%)", marker_color='#f43f5e'
        ))
        fig_bar.update_layout(
            title="Capital Allocation vs. Euler Risk Contribution",
            barmode='group', template="plotly_dark", height=380,
            paper_bgcolor='rgba(15,23,42,0.7)', plot_bgcolor='rgba(15,23,42,0.4)',
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------------------------
# TAB 4: BACKTEST & DRAWDOWN ANALYSIS
# ------------------------------------------
with tab4:
    st.subheader("Historical Cumulative Wealth & Underwater Drawdown Backtest")
    
    opt_daily_ret = returns @ primary_w
    eq_daily_ret = returns @ eq_w
    sa_daily_ret = returns @ best_sa_w
    evo_daily_ret = returns @ best_evo_w
    
    wealth_opt = capital_usd * (1 + opt_daily_ret).cumprod()
    wealth_eq = capital_usd * (1 + eq_daily_ret).cumprod()
    wealth_sa = capital_usd * (1 + sa_daily_ret).cumprod()
    wealth_evo = capital_usd * (1 + evo_daily_ret).cumprod()
    
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=returns.index, y=wealth_evo, name="Evolutionary Strategy (μ+λ)", line=dict(color='#00ff87', width=2.5)))
    fig_bt.add_trace(go.Scatter(x=returns.index, y=wealth_sa, name="Simulated Annealing", line=dict(color='#00f2fe', width=2, dash='dot')))
    fig_bt.add_trace(go.Scatter(x=returns.index, y=wealth_eq, name="Equal-Weight (1/N) Benchmark", line=dict(color='#94a3b8', width=2)))
    fig_bt.update_layout(
        title=f"Cumulative Portfolio Growth of ${capital_usd:,.0f} ({lookback_years}-Year Backtest)",
        xaxis_title="Date", yaxis_title="Portfolio Value ($ USD)",
        template="plotly_dark", height=420,
        paper_bgcolor='rgba(15,23,42,0.7)', plot_bgcolor='rgba(15,23,42,0.4)',
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_bt, use_container_width=True)
    
    dd_opt = (wealth_opt / wealth_opt.cummax() - 1) * 100
    dd_eq = (wealth_eq / wealth_eq.cummax() - 1) * 100
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=returns.index, y=dd_opt, fill='tozeroy', name="Optimal Portfolio Drawdown (%)", line=dict(color='#f43f5e', width=1.8)))
    fig_dd.add_trace(go.Scatter(x=returns.index, y=dd_eq, name="Equal-Weight Drawdown (%)", line=dict(color='#94a3b8', width=1.2, dash='dash')))
    fig_dd.update_layout(
        title=f"Underwater Drawdown Curve (Max Drawdown: {dd_opt.min():.2f}% vs Benchmark {dd_eq.min():.2f}%)",
        xaxis_title="Date", yaxis_title="Drawdown from Peak (%)",
        template="plotly_dark", height=280,
        paper_bgcolor='rgba(15,23,42,0.7)', plot_bgcolor='rgba(15,23,42,0.4)',
        legend=dict(orientation="h", y=1.15)
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# ------------------------------------------
# TAB 5: CORRELATION & BLACK-SWAN STRESS TEST
# ------------------------------------------
with tab5:
    st.subheader("Cross-Asset Correlation Matrix & Macro Black-Swan Stress Simulator")
    st.c1, st.c2 = st.columns([1, 1])
    
    with st.c1:
        corr_matrix = returns.corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Empirical Return Correlation Matrix"
        )
        fig_corr.update_layout(
            template="plotly_dark", height=420,
            paper_bgcolor='rgba(15,23,42,0.7)'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with st.c2:
        st.markdown("#### 💥 Parametric Shock & Tail-Distribution Simulator")
        shock_Asset = st.selectbox("Select Constituent for Isolated Crash Simulation", clean_names)
        shock_pct = st.slider(f"Simulate Single-Day Crash in {shock_Asset} (%)", -40, -5, -15, step=5)
        
        asset_idx = clean_names.index(shock_Asset)
        asset_vols = np.sqrt(np.diag(cov_matrix))
        corr_vec = corr_matrix.iloc[:, asset_idx].values
        spillover_shocks = (shock_pct / 100.0) * corr_vec * (asset_vols / asset_vols[asset_idx])
        
        opt_Portfolio_Impact = np.sum(primary_w * spillover_shocks) * 100
        eq_Portfolio_Impact = np.sum(eq_w * spillover_shocks) * 100
        usd_loss = (opt_Portfolio_Impact / 100.0) * capital_usd
        
        sc1, sc2 = st.columns(2)
        sc1.metric("Contagion Portfolio Impact", f"{opt_Portfolio_Impact:.2f}%", f"${usd_loss:,.0f} PnL Shock", delta_color="inverse")
        sc2.metric("Equal-Weight Contagion Impact", f"{eq_Portfolio_Impact:.2f}%", f"{opt_Portfolio_Impact - eq_Portfolio_Impact:+.2f}% Relative Shield")
        
        port_daily_pct = (returns_matrix @ primary_w) * 100
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=port_daily_pct, nbinsx=60, marker_color='#38bdf8', opacity=0.75, name="Daily Returns"))
        fig_dist.add_vline(x=-opt_var95 * 100, line_dash="dash", line_color="#f59e0b", annotation_text=f"95% VaR (-{opt_var95*100:.2f}%)")
        fig_dist.add_vline(x=-opt_cvar95 * 100, line_dash="solid", line_color="#f43f5e", annotation_text=f"95% CVaR (-{opt_cvar95*100:.2f}%)")
        fig_dist.update_layout(
            title="Daily Portfolio Return Distribution & Tail-Risk Cutoffs",
            xaxis_title="Daily Return (%)", yaxis_title="Frequency (Days)",
            template="plotly_dark", height=260,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(15,23,42,0.7)', plot_bgcolor='rgba(15,23,42,0.4)'
        )
        st.plotly_chart(fig_dist, use_container_width=True)