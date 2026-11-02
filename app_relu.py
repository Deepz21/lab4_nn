# app_relu.py — ReLU Activation Function Visualiser
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="ReLU Activation", page_icon="📈", layout="wide")
st.title("📈 ReLU — Rectified Linear Unit")
st.markdown(r"""
**Definition:**  $f(x) = \max(0, x)$

**Derivative:**  $f'(x) = 1$ if $x>0$, else $0$

**Properties:** Cheap to compute, avoids vanishing gradient for positive inputs, but
can suffer from the *dying ReLU* problem. Non-linear, so a network can learn complex patterns.
""")

# ---------- 1. Function + derivative plot ----------
st.header("1. Function and Derivative")
x = np.linspace(-6, 6, 500)

def relu(x):        return np.maximum(0, x)
def relu_prime(x):  return (x > 0).astype(float)

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.plot(x, relu(x), lw=2, color="crimson")
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_title("ReLU(x) = max(0, x)"); ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.grid(True)
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots()
    ax.plot(x, relu_prime(x), lw=2, color="navy")
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_title("ReLU'(x)"); ax.set_xlabel("x"); ax.set_ylabel("f'(x)")
    ax.set_ylim(-0.2, 1.2); ax.grid(True)
    st.pyplot(fig)

# ---------- 2. Interactive point ----------
st.header("2. Interactive Value")
xv = st.slider("Choose x", -6.0, 6.0, 1.0, 0.1)
st.write(f"f({xv}) = **{relu(np.array([xv]))[0]:.3f}**")
st.write(f"f'({xv}) = **{relu_prime(np.array([xv]))[0]:.3f}**")

# ---------- 3. Small NN demo on make_moons ----------
st.header("3. Effect in a Neural Network (make_moons)")
n_samples = st.slider("Samples", 100, 1000, 400, 50)
noise     = st.slider("Noise", 0.0, 0.5, 0.20, 0.01)
seed      = st.number_input("Random seed", 0, 9999, 42)

X, y = make_moons(n_samples=n_samples, noise=noise, random_state=int(seed))
Xs = StandardScaler().fit_transform(X)

clf = MLPClassifier(hidden_layer_sizes=(16, 16), activation="relu",
                    max_iter=1500, random_state=int(seed)).fit(Xs, y)
acc = clf.score(Xs, y)

# decision boundary
xx, yy = np.meshgrid(np.linspace(Xs[:,0].min()-1, Xs[:,0].max()+1, 250),
                     np.linspace(Xs[:,1].min()-1, Xs[:,1].max()+1, 250))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(6,5))
ax.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(["#FFDDDD", "#DDEEFF"]))
ax.scatter(Xs[:,0], Xs[:,1], c=y, s=20, edgecolor="k",
           cmap=ListedColormap(["#CC0000", "#0044AA"]))
ax.set_title(f"MLP with ReLU — accuracy = {acc:.3f}")
ax.set_xlabel("x1"); ax.set_ylabel("x2")
st.pyplot(fig)

st.success(f"Training accuracy: **{acc*100:.2f}%**")