# Neural Network Calculus Cheat Sheet

This cheat sheet unifies the three perspectives of neural network calculus: **Scalar** (scalar calculus per neuron), **Vector** (single sample, row-oriented), and **Batched Matrix** (batch size $N$, standard in NumPy/PyTorch).

---

### Variable Dimensions & Shapes

Let $D_{\text{in}}$ be the number of incoming features, $D_{\text{out}}$ be the number of neurons in the current layer, and $N$ be the batch size.

| Variable | Meaning | Scalar Index | Single Sample (Row Vector) | Batched Matrix (NumPy / PyTorch) |
| --- | --- | --- | --- | --- |
| **Input / Activation** | Inputs to layer | $x_k$ | $x \in \mathbb{R}^{1 \times D_{\text{in}}}$ | $X \in \mathbb{R}^{N \times D_{\text{in}}}$ |
| **Weights** | Layer parameters | $w_{k, j}$ | $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ | $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ |
| **Biases** | Layer parameters | $b_j$ | $b \in \mathbb{R}^{1 \times D_{\text{out}}}$ | $B \in \mathbb{R}^{1 \times D_{\text{out}}}$ |
| **Pre-activation** | Affine output | $z_j$ | $z \in \mathbb{R}^{1 \times D_{\text{out}}}$ | $Z \in \mathbb{R}^{N \times D_{\text{out}}}$ |
| **Post-activation** | After non-linearity | $a_j$ | $a \in \mathbb{R}^{1 \times D_{\text{out}}}$ | $A \in \mathbb{R}^{N \times D_{\text{out}}}$ |

---

### Forward Pass

| Step | Scalar Form (Neuron $j$) | Single Sample (Row Vector) | Batched Matrix ($N$ Samples) |
| --- | --- | --- | --- |
| **Linear Combination** | $z_j = \sum_{k=1}^{D_{\text{in}}} x_k w_{k, j} + b_j$ | $z = x W + b$ | $Z = XW + B$ |
| **Activation** | $a_j = \sigma(z_j)$ | $a = \sigma(z)$ | $A = \sigma(Z)$ |

---

### Backward Pass (Chain Rule Flow)

Let $\text{dout}$ be the incoming gradient from the layer to the right ($\frac{\partial L}{\partial A}$), and $\odot$ denote element-wise multiplication.

| Step | Target Gradient | Scalar Form | Single Sample (Row Vector) | Batched Matrix ($N$ Samples) |
| --- | --- | --- | --- | --- |
| **1. Pre-activation** | $\frac{\partial L}{\partial Z}$ | $dz_j = da_j \cdot \sigma'(z_j)$ | $dz = da \odot \sigma'(z)$ | $dZ = dA \odot \sigma'(Z)$ |
| **2. Weights** | $\frac{\partial L}{\partial W}$ | $dw_{k, j} = x_k \cdot dz_j$ | $dW = x^T dz$ | $dW = X^T dZ$ |
| **3. Biases** | $\frac{\partial L}{\partial B}$ | $db_j = dz_j$ | $db = dz$ | $dB = \sum_{i=1}^N dZ_{i, :}$ |
| **4. Pass to previous layer** | $\frac{\partial L}{\partial X}$ | $dx_k = \sum_{j=1}^{D_{\text{out}}} dz_j \cdot w_{k, j}$ | $dx = dz \, W^T$ | $dX = dZ \, W^T$ |

---

### Parameter Update Step (Gradient Descent)

For learning rate $\eta$:

| Parameter | Scalar Form | Single Sample / Batched Form |
| --- | --- | --- |
| **Weights** | $w_{k, j} \leftarrow w_{k, j} - \eta \cdot dw_{k, j}$ | $W \leftarrow W - \eta \cdot dW$ |
| **Biases** | $b_j \leftarrow b_j - \eta \cdot db_j$ | $B \leftarrow B - \eta \cdot dB$ |

---

### Appendix: Textbook Column-Vector Convention

Academic textbooks (such as Goodfellow et al. or Michael Nielsen) often represent a single sample as a column vector ($x \in \mathbb{R}^{D_{\text{in}} \times 1}$) and flip the weight matrix shape to $W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$.

| Step | Column-Vector Formula ($D \times 1$) | Row-Vector / Batched Formula (NumPy / PyTorch) |
| --- | --- | --- |
| **Linear Step** | $z = W x + b$ | $Z = XW + B$ |
| **Activation Step** | $a = \sigma(z)$ | $A = \sigma(Z)$ |
| **Pre-activation Grad** | $dz = da \odot \sigma'(z)$ | $dZ = dA \odot \sigma'(Z)$ |
| **Weight Grad** | $dW = dz \, x^T$ (Outer product) | $dW = X^T dZ$ (Sum of outer products) |
| **Bias Grad** | $db = dz$ | $dB = \sum_{\text{axis}=0} dZ$ |
| **Upstream Grad** | $dx = W^T dz$ | $dX = dZ \, W^T$ |
