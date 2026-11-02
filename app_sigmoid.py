# app_sigmoid.py — Sigmoid Activation Function Visualiser
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="Sigmoid Activation", page_icon="📉", layout="wide")
st.title("📉 Sigmoid — Logistic Activation")
st.markdown(r"""
**Definition:**  $\sigma(x) = \dfrac{1}{1 + e^{-x}}$

**Derivative:**  $\sigma'(x) = \sigma(x)\,(1 - \sigma(x))$

**Properties:** Squashes output to **(0, 1)** — useful for probability outputs.
Smooth and differentiable everywhere, but suffers from the **vanishing gradient**
problem for large $|x|$ (gradient → 0), which slows deep-network training.
""")

# ---------- 1. Function + derivative plot ----------
st.header("1. Function and Derivative")
x = np.linspace(-6, 6, 500)

def fn(x):        return 1 / (1 + np.exp(-x))
def fn_prime(x):
    s = fn(x)
    return s * (1 - s)

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.plot(x, fn(x), lw=2, color="seagreen")
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.axhline(0.5, color="lightgray", lw=0.8, ls="--")
    ax.set_title("Sigmoid(x) = 1 / (1 + e^-x)")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.set_ylim(-0.1, 1.1); ax.grid(True)
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots()
    ax.plot(x, fn_prime(x), lw=2, color="darkgreen")
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_title("Sigmoid'(x) = σ(x)(1-σ(x))")
    ax.set_xlabel("x"); ax.set_ylabel("f'(x)")
    ax.set_ylim(-0.02, 0.28); ax.grid(True)
    st.pyplot(fig)

# ---------- 2. Interactive point ----------
st.header("2. Interactive Value")
xv = st.slider("Choose x", -6.0, 6.0, 1.0, 0.1)
st.write(f"σ({xv}) = **{fn(np.array([xv]))[0]:.4f}**")
st.write(f"σ'({xv}) = **{fn_prime(np.array([xv]))[0]:.4f}**")

# ---------- 3. Small NN demo on make_moons ----------
st.header("3. Effect in a Neural Network (make_moons)")
n_samples = st.slider("Samples", 100, 1000, 400, 50)
noise     = st.slider("Noise", 0.0, 0.5, 0.20, 0.01)
seed      = st.number_input("Random seed", 0, 9999, 42)

X, y = make_moons(n_samples=n_samples, noise=noise, random_state=int(seed))
Xs = StandardScaler().fit_transform(X)

# sklearn calls sigmoid "logistic"
clf = MLPClassifier(hidden_layer_sizes=(16, 16), activation="logistic",
                    max_iter=1500, random_state=int(seed)).fit(Xs, y)
acc = clf.score(Xs, y)

xx, yy = np.meshgrid(np.linspace(Xs[:,0].min()-1, Xs[:,0].max()+1, 250),
                     np.linspace(Xs[:,1].min()-1, Xs[:,1].max()+1, 250))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(6,5))
ax.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(["#DDFFDD", "#DDEEFF"]))
ax.scatter(Xs[:,0], Xs[:,1], c=y, s=20, edgecolor="k",
           cmap=ListedColormap(["#006600", "#0044AA"]))
ax.set_title(f"MLP with Sigmoid — accuracy = {acc:.3f}")
ax.set_xlabel("x1"); ax.set_ylabel("x2")
st.pyplot(fig)

st.success(f"Training accuracy: **{acc*100:.2f}%**")