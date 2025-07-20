
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set page config
st.set_page_config(layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>Black-Scholes Pricing Model</h1>", unsafe_allow_html=True)

# Sidebar - Paramètres
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.markdown("**Created by:**")
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Loïc%20Chantoury-blue?logo=linkedin)](https://www.linkedin.com/in/loïcchantoury/)")

    S = st.number_input("Current Asset Price", min_value=0.01, value=100.0, step=0.1)
    K = st.number_input("Strike Price", min_value=0.01, value=100.0, step=0.1)
    T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.01)
    sigma = st.number_input("Volatility (σ)", min_value=0.01, value=0.2, step=0.01)
    r = st.number_input("Risk-Free Interest Rate", min_value=0.0, value=0.01, step=0.01)

    st.markdown("---")
    st.subheader("Heatmap Parameters")
    spot_min = st.number_input("Min Spot Price", min_value=1.0, value=80.0)
    spot_max = st.number_input("Max Spot Price", min_value=1.0, value=120.0)
    vol_min = st.slider("Min Volatility for Heatmap", 0.01, 1.0, 0.1)
    vol_max = st.slider("Max Volatility for Heatmap", 0.01, 1.0, 0.5)

# Black-Scholes-Merton formulas
def black_scholes(S, K, T, r, sigma, option="call"):
    d1 = (np.log(S/K) + (r + sigma**2 / 2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T)*norm.cdf(d2)
    elif option == "put":
        return K * np.exp(-r*T)*norm.cdf(-d2) - S * norm.cdf(-d1)

call_price = black_scholes(S, K, T, r, sigma, option="call")
put_price = black_scholes(S, K, T, r, sigma, option="put")

# Prices
st.markdown(f"""
<div style='display: flex; justify-content: center; gap: 100px;'>
    <div style='text-align: center'>
        <b style='color: green;'>CALL Option Price</b><br>
        <span style='font-size: 36px; color: green;'>${call_price:.2f}</span>
    </div>
    <div style='text-align: center'>
        <b style='color: red;'>PUT Option Price</b><br>
        <span style='font-size: 36px; color: red;'>${put_price:.2f}</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Options Price - Interactive Heatmap</h2>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; background-color: #0e2a47; color: white; padding: 20px; border-radius: 10px;'>
Explore how option prices fluctuate with varying 'Spot Prices' and 'Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.
</div>
""", unsafe_allow_html=True)

# Grille des valeurs
spot_range = np.linspace(spot_min, spot_max, 10)
vol_range = np.linspace(vol_min, vol_max, 10)
call_matrix = np.zeros((10, 10))
put_matrix = np.zeros((10, 10))

for i, vol in enumerate(vol_range):
    for j, spot in enumerate(spot_range):
        call_matrix[i, j] = black_scholes(spot, K, T, r, vol, option="call")
        put_matrix[i, j] = black_scholes(spot, K, T, r, vol, option="put")
# Heatmaps
fig1, ax1 = plt.subplots()
im1 = ax1.imshow(call_matrix, cmap="viridis", aspect="auto")

for i in range(call_matrix.shape[0]):
    for j in range(call_matrix.shape[1]):
        ax1.text(j, i, f"{call_matrix[i, j]:.1f}", ha="center", va="center", color="white", fontsize=8)
ax1.set_xticks(np.arange(10))
ax1.set_yticks(np.arange(10))
ax1.set_xticklabels([f"{x:.2f}" for x in spot_range], rotation=45)
ax1.set_yticklabels([f"{v:.2f}" for v in vol_range])
ax1.set_title("CALL")
ax1.set_xlabel("Spot Price")
ax1.set_ylabel("Volatility")
fig1.colorbar(im1, ax=ax1)

fig2, ax2 = plt.subplots()
im2 = ax2.imshow(put_matrix, cmap="plasma", aspect="auto")

for i in range(put_matrix.shape[0]):
    for j in range(put_matrix.shape[1]):
        ax2.text(j, i, f"{put_matrix[i, j]:.1f}", ha="center", va="center", color="white", fontsize=8)
ax2.set_xticks(np.arange(10))
ax2.set_yticks(np.arange(10))
ax2.set_xticklabels([f"{x:.2f}" for x in spot_range], rotation=45)
ax2.set_yticklabels([f"{v:.2f}" for v in vol_range])
ax2.set_title("PUT")
ax2.set_xlabel("Spot Price")
ax2.set_ylabel("Volatility")
fig2.colorbar(im2, ax=ax2)

# Affichage côte à côte
col3, col4 = st.columns(2)
with col3:
    st.pyplot(fig1)
with col4:
    st.pyplot(fig2)

# Link to P&L Simulator
st.markdown("""
---
<p style='text-align: center;'>
    <a href="http://localhost:8502" target="_blank" style='color:#1f77b4; text-decoration: none;'>→ Go to P&L Simulator</a>
</p>
""", unsafe_allow_html=True)




