import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Page config
st.set_page_config(page_title="P&L Simulator – Options Trading Tool", layout="wide")

# Title (harmonized with pricing app)
st.markdown("""
<h1 style='text-align: center; font-family:sans-serif;'>P&L Simulator – Options Trading Tool</h1>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📊 P&L Simulator ")
    st.markdown("**Created by:**")
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Loïc%20Chantoury-blue?logo=linkedin)](https://www.linkedin.com/in/loïcchantoury/)")

    option_type = st.selectbox("Option Type", ["Call", "Put"])
    K = st.number_input("Strike Price", min_value=0.01, value=100.0, step=0.1)
    T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.01)
    r = st.number_input("Risk-Free Interest Rate", min_value=0.0, value=0.01, step=0.01)
    purchase_price = st.number_input("Purchase Price of Option", min_value=0.0, value=10.0, step=0.1)

    st.markdown("---")
    st.subheader("Heatmap Parameters")
    spot_min = st.number_input("Min Spot Price", min_value=1.0, value=80.0)
    spot_max = st.number_input("Max Spot Price", min_value=1.0, value=120.0)
    vol_min = st.slider("Min Volatility", 0.01, 1.0, 0.1)
    vol_max = st.slider("Max Volatility", 0.01, 1.5, 0.5)

# Black-Scholes Function
def black_scholes(S, K, T, r, sigma, option="call"):
    d1 = (np.log(S/K) + (r + sigma**2 / 2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T)*norm.cdf(d2)
    elif option == "put":
        return K * np.exp(-r*T)*norm.cdf(-d2) - S * norm.cdf(-d1)

# Grid Values
spot_range = np.linspace(spot_min, spot_max, 10)
vol_range = np.linspace(vol_min, vol_max, 10)
pnl_matrix = np.zeros((10, 10))

for i, vol in enumerate(vol_range):
    for j, spot in enumerate(spot_range):
        option_value = black_scholes(spot, K, T, r, vol, option=option_type.lower())
        pnl_matrix[i, j] = option_value - purchase_price

# Recap Styled Block
st.markdown(f"""
<div style='margin: 10px auto 20px auto; text-align: center; color: #cccccc; font-family: sans-serif;'>
    <span style='font-size:16px;'>Selected Parameters:</span><br>
    <span style='font-size:14px;'>
        <b style='color:#4da6ff;'>Option Type:</b> {option_type.upper()} &nbsp; | &nbsp;
        <b style='color:#4da6ff;'>Strike Price:</b> {K:.2f} &nbsp; | &nbsp;
        <b style='color:#4da6ff;'>Time to Maturity:</b> {T:.2f} yrs &nbsp; | &nbsp;
        <b style='color:#4da6ff;'>Risk-Free Rate:</b> {r:.2%} &nbsp; | &nbsp;
        <b style='color:#4da6ff;'>Purchase Price:</b> {purchase_price:.2f}
    </span>
</div>
""", unsafe_allow_html=True)

# Header
title = "Options Price - Interactive Heatmap"
st.markdown(f"""
<h2 style='text-align: center; font-family:sans-serif;'>{title}</h2>
""", unsafe_allow_html=True)

# Info box
st.markdown("""
<div style='text-align: center; background-color: #0e2a47; color: white; padding: 20px; border-radius: 10px;'>
This heatmap displays the P&L (Profit & Loss) resulting from different combinations of spot price and volatility.<br>Green = Profit / Red = Loss.
</div>
""", unsafe_allow_html=True)

# Heatmap plot
fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.get_cmap("RdYlGn")
im = ax.imshow(pnl_matrix, cmap=cmap, aspect="auto")

for i in range(pnl_matrix.shape[0]):
    for j in range(pnl_matrix.shape[1]):
        value = pnl_matrix[i, j]
        color = "black" if abs(value) > 5 else "white"
        ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=color, fontsize=9, weight="bold")

ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
ax.set_xticklabels([f"{x:.2f}" for x in spot_range], rotation=45)
ax.set_yticklabels([f"{v:.2f}" for v in vol_range])
ax.set_xlabel("Spot Price", fontsize=12)
ax.set_ylabel("Volatility", fontsize=12)
fig.colorbar(im, ax=ax, shrink=0.8, label="P&L")
fig.patch.set_facecolor('#f4f6f7')

st.pyplot(fig)

# Link to pricing app
st.markdown("""
---
<p style='text-align: center;'>
    <a href="http://localhost:8501" target="_blank" style='color:#1f77b4; text-decoration: none;'>← Back to Black-Scholes Option Pricing</a>
</p>
""", unsafe_allow_html=True)

