# app_tanh.py — Tanh (Hyperbolic Tangent) Activation Function Visualiser
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="Tanh Activation", page_icon="📊", layout="wide")
st.title("📊 Tanh — Hyperbolic Tangent")
st.markdown(r"""
**Definition:**  $\tanh(x) = \dfrac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$

**Derivative:**  $\tanh'(x) = 1 - \tanh^{2}(x)$

**Properties:** Zero-centred output in **(-1, 1)**; usually trains faster than
Sigmoid because the mean of its outputs is ≈ 0. Still suffers from
vanishing gradient at large $|x|$.
""")

# ---------- 1. Function + derivative plot ----------
st.header("1. Function and Derivative")
x = np.linspace(-6, 6, 500)

def fn(x):        return np.tanh(x)
def fn_prime(x):  return 1 - np.tanh(x) ** 2

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.plot(x, fn(x), lw=2, color="purple")
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.axhline( 1, color="lightgray", lw=0.8, ls="--")
    ax.axhline(-1, color="lightgray", lw=0.8, ls="--")
    ax.set_title("tanh(x)")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.set_ylim(-1.1, 1.1); ax.grid(True)
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots()
    ax.plot(x, fn_prime(x), lw=2, color="indigo")
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_title("tanh'(x) = 1 - tanh²(x)")
    ax.set_xlabel("x"); ax.set_ylabel("f'(x)")
    ax.set_ylim(-0.05, 1.1); ax.grid(True)
    st.pyplot(fig)

# ---------- 2. Interactive point ----------
st.header("2. Interactive Value")
xv = st.slider("Choose x", -6.0, 6.0, 1.0, 0.1)
st.write(f"tanh({xv}) = **{fn(np.array([xv]))[0]:.4f}**")
st.write(f"tanh'({xv}) = **{fn_prime(np.array([xv]))[0]:.4f}**")

# ---------- 3. Small NN demo on make_moons ----------
st.header("3. Effect in a Neural Network (make_moons)")
n_samples = st.slider("Samples", 100, 1000, 400, 50)
noise     = st.slider("Noise", 0.0, 0.5, 0.20, 0.01)
seed      = st.number_input("Random seed", 0, 9999, 42)

X, y = make_moons(n_samples=n_samples, noise=noise, random_state=int(seed))
Xs = StandardScaler().fit_transform(X)

clf = MLPClassifier(hidden_layer_sizes=(16, 16), activation="tanh",
                    max_iter=1500, random_state=int(seed)).fit(Xs, y)
acc = clf.score(Xs, y)

xx, yy = np.meshgrid(np.linspace(Xs[:,0].min()-1, Xs[:,0].max()+1, 250),
                     np.linspace(Xs[:,1].min()-1, Xs[:,1].max()+1, 250))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(6,5))
ax.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(["#EDDDFF", "#DDEEFF"]))
ax.scatter(Xs[:,0], Xs[:,1], c=y, s=20, edgecolor="k",
           cmap=ListedColormap(["#660099", "#0044AA"]))
ax.set_title(f"MLP with Tanh — accuracy = {acc:.3f}")
ax.set_xlabel("x1"); ax.set_ylabel("x2")
st.pyplot(fig)

st.success(f"Training accuracy: **{acc*100:.2f}%**")