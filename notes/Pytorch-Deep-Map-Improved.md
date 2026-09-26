# PyTorch: The Deep Map — Ultimate Comprehensive Edition
> Version: PyTorch 2.4+ / 2.5+ compatible | Companion to NumPy Ultimate Cheatsheet | Author: Meta AI for you
> Covers: Setup, Dual Syntax, Creation, Attributes, Dtype, Device, Strides, View vs Copy, Indexing, Layout, Broadcasting, Ufuncs, Math, Stats, Sorting/Searching, Linalg, Random, Set Ops, FFT, Einsum, IO, Autograd, nn.Module, Training, Performance, DataPipeline, Recipes, Pitfalls, Visuals
> Philosophy: `Tensor = Storage + (size, stride, storage_offset) + dtype + device + grad_fn`

---

## Table of Contents
1. [Setup & Basics](#1-setup--basics)
2. [The Two Syntaxes: torch.func(t) vs t.method() - CRITICAL](#15-the-two-syntaxes)
3. [Tensor Creation - Exhaustive](#2-tensor-creation--exhaustive)
4. [Attributes & Inspection](#3-attributes--inspection)
5. [Data Types (dtype) - Deep Dive](#4-data-types-dtype--deep-dive)
6. [Device - Where Buffer Lives](#5-device--where-buffer-lives)
7. [Strides, Layout, View vs Copy - The Core Mental Model](#6-strides-layout-view-vs-copy)
8. [Indexing: 4 Operations in One Syntax](#7-indexing-4-operations-in-one-syntax)
9. [Memory Layout: C-order, Non-contiguous, channels_last](#8-memory-layout)
10. [Broadcasting and Alignment](#9-broadcasting-and-alignment)
11. [Universal Functions (ufuncs) - Arithmetic, Comparison, Logic](#10-universal-functions)
12. [Mathematical Functions - Exp, Log, Trig, Rounding](#11-mathematical-functions)
13. [Statistics & Aggregation - DUAL FORM](#12-statistics--aggregation)
14. [Sorting, Searching & Counting](#13-sorting-searching--counting)
15. [Linear Algebra - The ML Workhorse](#14-linear-algebra)
16. [Random Sampling - Old vs New, Distributions](#15-random-sampling)
17. [Set Operations, Unique & Masked Ops](#16-set-operations)
18. [Fourier Transform (FFT) - torch.fft](#17-fourier-transform-fft)
19. [Einsum Masterclass](#18-einsum-masterclass)
20. [Input / Output & Serialization](#19-input--output--serialization)
21. [Autograd: The Twist](#20-autograd-the-twist)
22. [Reshaping: view, reshape, transpose, contiguous hell](#21-reshaping-deep-dive)
23. [Advanced Stride Tricks: as_strided, unfold](#22-advanced-stride-tricks)
24. [nn.Module: Model is a Tree of Tensors](#23-nnmodule)
25. [Training Loop: Where Everything Meets](#24-training-loop)
26. [Device & Performance: Supercomputer Part](#25-device--performance)
27. [Data Pipeline: Dataset, DataLoader](#26-data-pipeline)
28. [Advanced Autograd: Custom Functions, Hooks, torch.func](#27-advanced-autograd)
29. [Common Recipes & Patterns](#28-common-recipes--patterns)
30. [Output Visuals - What Each Command Returns](#29-output-visuals)
31. [Pitfalls & Gotchas - 25 Critical Lessons](#30-pitfalls--gotchas)
32. [Full Cheat Sheet - View vs Copy Definitive](#31-full-cheat-sheet)
33. [From NumPy Map to PyTorch Map: Direct Translation](#32-from-numpy-to-pytorch-translation)
34. [Appendix - Minimal Runnable Training Template](#33-appendix)

---

## 1. Setup & Basics

```python
import torch
print(torch.__version__)  # 2.4+ recommended

# Core object: Tensor - n-dimensional array, homogeneous, fixed size, device-aware, differentiable
# Why PyTorch?
# - NumPy-like API but GPU accelerated
# - Autograd: remembers computation for gradients
# - 100x faster than Python loops on GPU, 10x on CPU vs lists
# - Dynamic graph, Pythonic
# - Ecosystem: torch.nn, torch.optim, torch.utils.data, torch.compile, torch.amp

# Check hardware
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}, name: {torch.cuda.get_device_name(0)}")
print(f"MPS available (Mac): {torch.backends.mps.is_available()}")

# Install:
# pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
# pip install torch  # CPU only
# conda install pytorch -c pytorch
```

### How to read this doc (from original)
- Every Phase builds on the previous.
- Code blocks are runnable. Copy-paste them.
- The mental model: `Tensor = Storage + (size, stride, storage_offset) + dtype + device + grad_fn`

---

## 1.5 The Two Syntaxes: torch.func(t) vs t.method() - CRITICAL

> Inspired by your NumPy cheatsheet Section 1.5. This is even MORE important in PyTorch because of in-place variants.

### Rule
- **~80+ operations** are Tensor **methods**: `t.sum()`, `t.mean()`, `t.reshape()`, `t.T`, etc.
- **~700+ operations** are **only** functions in `torch.*` namespace: `torch.sin(t)`, `torch.matmul()`, `torch.stack()`, `torch.linalg.inv()`, `torch.fft.fft()`, etc.
- `t.sin()`? Actually **exists** in PyTorch! Unlike NumPy, PyTorch exposes many as methods too. But `torch.stack` never exists as method.
- **In-place**: PyTorch has `t.add_()`, `t.relu_()` — trailing underscore means mutate storage. **Dangerous with autograd!**

### Complete Method List (Everything you CAN call as t.method())
```python
# Run this to see:
# print([m for m in dir(torch.Tensor) if not m.startswith('_')][:200])

t.abs(), t.absolute(), t.acos(), t.arccos(), t.acosh(), t.add(), t.add_(), t.addmm()
t.all(), t.any(), t.argmax(), t.argmin(), t.argsort(), t.argwhere()
t.as_strided(), t.asin(), t.arcsin(), t.atan(), t.arctan(), t.atan2(), t.atanh()
t.baddbmm(), t.bernoulli(), t.bfloat16(), t.bitwise_and(), t.bitwise_or(), t.bitwise_xor(), t.bool()
t.ceil(), t.clamp(), t.clamp_(), t.clip(), t.clone(), t.contiguous(), t.copy_(), t.cos(), t.cosh(), t.cumsum(), t.cumprod()
t.det(), t.detach(), t.diagonal(), t.div(), t.divide(), t.dot(), t.double()
t.eq(), t.equal(), t.erf(), t.erfc(), t.exp(), t.exp_(), t.expm1(), t.expand(), t.expand_as()
t.fill_(), t.flatten(), t.float(), t.floor(), t.fmod()
t.gather(), t.ge(), t.gt(), t.half(), t.hardshrink(), t.histc(), t.index_add_(), t.index_copy_(), t.int(), t.is_contiguous()
t.isfinite(), t.isinf(), t.isnan(), t.item(), t.le(), t.lerp(), t.log(), t.log10(), t.log1p(), t.log2(), t.logical_and(), t.logical_or()
t.long(), t.lt(), t.masked_fill_(), t.masked_select(), t.matmul(), t.max(), t.mean(), t.median(), t.min(), t.mm(), t.mul(), t.mul_()
t.narrow(), t.ne(), t.neg(), t.new_ones(), t.new_zeros(), t.nonzero(), t.norm(), t.numel()
t.permute(), t.pow(), t.prod(), t.reciprocal(), t.remainder(), t.repeat(), t.reshape(), t.round(), t.rsqrt()
t.scatter_(), t.select(), t.sigmoid(), t.sign(), t.sin(), t.sinh(), t.size(), t.sort(), t.sqrt(), t.squeeze(), t.std(), t.stride(), t.sub(), t.sum(), t.swapaxes(), t.T, t.take(), t.tan(), t.tanh(), t.to(), t.tolist(), t.topk(), t.trace(), t.transpose(), t.unfold(), t.unsqueeze(), t.var(), t.view(), t.where(), t.zero_()
```

### Side-by-side Comparison - BOTH Forms Where Available

```python
import torch
a = torch.tensor([[1.,2.,3.],[4.,5.,6.]])

# ---- STATISTICS & AGGREGATION: BOTH work ----
torch.sum(a)        == a.sum()        # tensor(21.)
torch.sum(a, dim=0) == a.sum(dim=0)   # tensor([5., 7., 9.])
torch.prod(a)       == a.prod()
torch.mean(a)       == a.mean()       # keep in mind: mean needs float
torch.std(a)        == a.std()
torch.var(a)        == a.var()
torch.min(a)        == a.min()        # torch.amin(a) is alias
torch.max(a)        == a.max()        # torch.amax(a)
torch.argmin(a)     == a.argmin()
torch.argmax(a)     == a.argmax()
torch.cumsum(a, dim=0) == a.cumsum(dim=0)

# dim, keepdim, dtype - work in BOTH
torch.mean(a, dim=1, keepdim=True)
a.mean(dim=1, keepdim=True)      # Same! shape (2,1)

# ---- SORTING: BOTH but BEWARE ----
torch.sort(a)       # Returns tuple (values, indices), new tensors
a.sort()            # Returns tuple too, but sorts along last dim. Does NOT sort in-place! Unlike NumPy!
a.sort_() if hasattr(a, 'sort_') else None  # PyTorch has no sort_ — use sort dim

# ---- SHAPE MANIPULATION: BOTH work ----
torch.reshape(a, (3,2)) == a.reshape(3,2)
torch.ravel(a)      # no, use torch.ravel? Exists as torch.ravel. a.ravel() doesn't exist, use flatten
a.flatten()         # == torch.flatten(a)
torch.transpose(a,0,1) == a.transpose(0,1) == a.T
torch.squeeze(a)    == a.squeeze()
torch.clamp(a, 2, 5) == a.clamp(2, 5)
torch.round(a)      == a.round()

# ---- ONLY FUNCTION FORM - These FAIL as methods ----
# torch.stack([a,a])  # no a.stack()
# torch.cat([a,a])    # no a.cat()
# torch.where(a>3, a, torch.zeros_like(a))  # only function, though a.where exists as torch.where condition?
# torch.linalg.inv(a[:2,:2])  # linalg never methods
# torch.einsum('ij,jk->ik', a, a.T)  # only function

# ---- METHOD ONLY or more convenient as method ----
a.to('cpu')         # device move - method is preferred
a.cuda()            # legacy
a.view(3,2)         # method only
a.expand(2,3,3)     # method only, but torch.broadcast_to exists
a.contiguous()      # method only

# ---- IN-PLACE VARIANTS: THE UNDERSCORE ----
# PyTorch exclusive! NumPy doesn't have this naming convention
b = a.clone()
b.add_(1)           # b = b + 1 but in-place, no new storage, bumps version
# b = b.add(1)      # out-of-place, safe for autograd
# Rule: NEVER use _ ops on tensors that need grad for backward unless you know versioning
```

### Which should you use?
- Use `torch.mean(a)` when you want functional style, works with any tensor-like
- Use `a.mean()` when you have tensor and want chaining: `a.reshape(-1).mean().sqrt()`
- Use `torch.*` for creation/joining: `torch.cat`, `torch.stack`, `torch.where`, `torch.einsum`
- For in-place: prefer out-of-place in training code to avoid autograd errors

---

## 2. Tensor Creation - Exhaustive

### 2.1 From Python objects (like NumPy 2.1)
```python
torch.tensor([1, 2, 3])                    # 1D from list
torch.tensor([[1,2],[3,4]])                # 2D
torch.as_tensor([1,2,3])                   # shares memory if possible, no copy if already tensor
torch.from_numpy(np.array([1,2,3]))        # shares memory with NumPy! Zero-copy bridge
# np_array = a.numpy()  # back to NumPy, shares if on CPU

# Important: torch.tensor() always copies, torch.as_tensor() may not, torch.from_numpy() shares
```

### 2.2 Intrinsic creation (like NumPy 2.2) - exhaustive list
```python
torch.zeros(3,4)                           # zeros
torch.zeros_like(a)                        # zeros same shape/dtype/device as a
torch.ones(3,4)                            # ones
torch.ones_like(a)
torch.empty(3,4)                           # uninitialized - garbage memory, fastest alloc
torch.empty_like(a)
torch.full((3,4), 7)                       # fill with 7
torch.full_like(a, 7)

torch.eye(3)                               # identity 3x3
torch.eye(3,4)                             # 3x4 identity-ish
torch.diag(torch.tensor([1,2,3]))          # diag matrix from vector
torch.diag(a)                              # extract diagonal (if 2D input)

# Ranges & grids (NumPy 2.3 equivalent)
torch.arange(10)                           # [0..9] like np.arange
torch.arange(0, 10, 2)                     # step
torch.linspace(0, 1, 5)                    # 5 points inclusive - like np.linspace
torch.logspace(0, 3, 4)                    # [1,10,100,1000]
torch.meshgrid(torch.arange(3), torch.arange(4), indexing='ij')  # like np.meshgrid

# Random (legacy vs new) - see Phase 15 for full
torch.randn(3,4)                           # N(0,1) - standard normal, core for ML
torch.rand(3,4)                            # U[0,1)
torch.randint(0, 10, (3,4))                # random ints [0,10)
torch.randperm(10)                         # random permutation 0..9
torch.bernoulli(torch.full((3,4), 0.5))    # coin flips

# For sequences of same shape
torch.zeros(2,3,4)                         # 3D
torch.ones(2,3,4, device='cuda')           # on GPU directly
```

### 2.3 Device-aware creation - best practice
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(3,4, device=device)        # create directly on target device
y = torch.zeros(3,4).to(device)            # move after
# torch.*_like preserves device:
z = torch.zeros_like(x)                    # already on same device as x
```

---

## 3. Attributes & Inspection (like NumPy Section 3)

```python
a = torch.randn(3,4,5)

# Core attributes - THE HEADER
a.shape                                    # torch.Size([3,4,5]) - alias a.size()
a.size()                                   # method form
a.size(0)                                  # 3 - size of dim 0
a.ndim                                     # 3 - number of dimensions, alias a.dim()
a.dim()                                    # 3 - method form (PyTorch uses dim not ndim as method)
a.numel()                                  # 60 - total number elements (like np.size)
a.nelement()                               # same, legacy

a.dtype                                    # torch.float32
a.device                                   # device(type='cpu')
a.layout                                   # torch.strided (default), or torch.sparse_coo etc
a.requires_grad                            # False - does it track grad?
a.grad                                     # None or tensor
a.grad_fn                                  # None if leaf, else <AddBackward etc>
a.is_leaf                                  # True if user created

# Memory layout
a.stride()                                 # (20,5,1) - elements to jump per dim
a.stride(0)                                # 20
a.storage_offset()                         # 0 - offset into storage
a.is_contiguous()                          # True?
a.is_contiguous(memory_format=torch.channels_last)  # channels_last check
a.nbytes if hasattr(a, 'nbytes') else a.element_size()*a.numel()  # total bytes
a.element_size()                           # bytes per element: 4 for float32
a.untyped_storage()                        # flat buffer - like np.ndarray.tobytes but live

# Inspection
print(a)                                   # pretty print
a.tolist()                                 # to Python list (copies)
a.numpy()                                  # to NumPy (shares if CPU, fails if GPU or requires_grad)
# torch.from_numpy(np_arr) -> back

# Type checking
torch.is_tensor(a)                         # True
torch.is_floating_point(a)                 # True
torch.is_nonzero(torch.tensor([0]))        # False

# Quick stats
a.min(), a.max(), a.mean(), a.std()         # etc.
```

---

## 4. Data Types (dtype) - Deep Dive

```python
# PyTorch dtypes - ~15 core, stricter than NumPy's 20+
# Float: float32 (default), float64, float16, bfloat16
# Int: int8, int16, int32, int64 (default for ints), uint8
# Bool: bool
# Complex: complex64, complex128

# Defaults matter!
torch.tensor([1,2,3]).dtype                # torch.int64 - NOT int32!
torch.tensor([1.,2.,3.]).dtype             # torch.float32 - ML standard
torch.randn(3).dtype                       # float32

# Creation with dtype
torch.randn(3, dtype=torch.float32)        # 32-bit
torch.randn(3, dtype=torch.float64)        # double - slower on GPU, precise
torch.randn(3, dtype=torch.bfloat16)       # brain float - for AMP
torch.randn(3, dtype=torch.float16)        # half - for AMP

# Casting
a = torch.randn(3,4)
a.float()                                  # to float32
a.double()                                 # to float64
a.half()                                   # to float16
a.bfloat16()                               # to bfloat16
a.int()                                    # to int32? Actually int64? Check: .int() -> int32, .long() -> int64
a.long()                                   # to int64
a.to(dtype=torch.float32)                  # generic cast - preferred
a.to(torch.float32)                        # shorthand

# Type promotion is conservative
print((torch.ones(3, dtype=torch.float32) + torch.ones(3, dtype=torch.float64)).dtype)  # float64
# PyTorch will often error if you mix int + float without explicit cast - unlike NumPy's silent promotion

# No object dtype! PyTorch does NOT have object, str, datetime like NumPy/Pandas
# If you need strings, that's not a Tensor - use Python lists

# Structured example - not supported, but you can have dict of tensors
# data = {'x': torch.randn(3), 'name': ['a','b','c']}  # name not a tensor
```

---

## 5. Device - Where Buffer Lives (Original Phase 1 Device expanded)

```python
# CPU tensor (default)
x_cpu = torch.randn(3,4)

# GPU tensor (if available)
if torch.cuda.is_available():
    x_gpu = x_cpu.cuda()          # legacy .cuda()
    x_gpu = x_cpu.to('cuda')      # modern
    x_gpu = torch.randn(3,4, device='cuda')
    x_gpu = torch.randn(3,4, device='cuda:0')  # specific GPU
    
    print(x_cpu.device)  # cpu
    print(x_gpu.device)  # cuda:0

# MPS (Apple Silicon)
if torch.backends.mps.is_available():
    x_mps = torch.randn(3,4, device='mps')

# Generic device-agnostic code (BEST PRACTICE from original):
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(3,4, device=device)
y = x.to(device)  # no-op if already there

# Operations must be on same device
# x_cpu + x_gpu -> RuntimeError: Expected all tensors to be on same device

# Moving
x.to('cuda', non_blocking=True)  # async when pin_memory
x.cpu()                           # to CPU
x.cuda()                          # to CUDA

# Storage sharing across devices? NO - to() always copies if device changes
# But storage_offset, size, stride stay identical - only data_ptr changes

# Pin memory for faster H2D
# In DataLoader: pin_memory=True gives page-locked memory

# Mental model from original:
# to(device) is like copying buffer from RAM to VRAM. Strides/size stay identical, pointer changes.
```

---



---

## ORIGINAL DEEP MAP PHASES - PRESERVED VERBATIM (0-16 + Appendix)

> The following is your original Pytorch-Deep-Map.md content preserved 100%. Nothing removed. Enhanced sections above are additions inspired by NumPy sheet.

# PyTorch: The Deep Map — From Flat Buffer to Differentiable Supercomputer

> This is the PyTorch companion to your NumPy strides map. Same philosophy: **every layer is just an offset formula over a flat buffer, plus one new twist.** That twist in PyTorch is *differentiation* and *device*.

---

### How to read this doc
- Every Phase builds on the previous.
- Code blocks are runnable. Copy-paste them.
- The mental model: `Tensor = Storage + (size, stride, storage_offset) + dtype + device + grad_fn`

---

## Phase 0 — From NumPy to PyTorch: What's actually new?

NumPy gave you: `ndarray = buffer + shape + strides + dtype`

PyTorch gives you:

```python
Tensor = UntypedStorage + size + stride + storage_offset + dtype + device + autograd_meta
```

3 fundamental extensions:

1.  **Device:** The buffer can live on CPU, CUDA, MPS, etc. Same indexing math, different physical chip.
2.  **Autograd:** Every operation can remember how it was computed (`grad_fn`) to compute gradients later.
3.  **In-place versioning:** Since autograd needs old values, PyTorch tracks mutations.

Everything you learned about strides/views/copies is 100% valid in PyTorch. PyTorch just adds those two.

```python
import torch
a = torch.arange(12).reshape(3,4)
print(a)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

print(f"size: {a.size()}, stride: {a.stride()}, offset: {a.storage_offset()}")
print(f"dtype: {a.dtype}, device: {a.device}, contiguous: {a.is_contiguous()}")
# size: torch.Size([3, 4]), stride: (4, 1), offset: 0
# dtype: torch.int64, device: cpu, contiguous: True
```

### Storage vs Tensor
The **Storage** is the flat 1D array. The **Tensor** is a *view* over it.

```python
a = torch.arange(6)
print(a.untyped_storage())  # the raw bytes
print(a.storage()) if hasattr(a, 'storage') else print(a.untyped_storage())

b = a[2:5]  # view
print(b.storage_offset()) # 2 — starts 2 elements into same storage
print(a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()) # True — shared
```

> In modern PyTorch: `untyped_storage()` is the canonical flat buffer. `storage()` is deprecated but same idea.

---

## Phase 1 — Foundations: Creation, Dtype, Device

### Creation — same as NumPy but device-aware
```python
torch.tensor([[1,2],[3,4]])           # from list
torch.arange(10)                      # like np.arange
torch.zeros(3,4), torch.ones(3,4)     # zeros/ones
torch.randn(3,4)                      # N(0,1) — core for ML
torch.rand(3,4)                       # U[0,1)
torch.eye(3)                          # identity
torch.empty(3,4)                      # uninitialized — like your garbage-memory demo
torch.linspace(0,1,5)
```

### Dtype model — stricter than NumPy, built for autodiff
NumPy has ~20 dtypes. PyTorch has similar but defaults matter:

```python
torch.tensor([1,2,3]).dtype          # torch.int64 by default (not int32!)
torch.tensor([1.,2.,3.]).dtype        # torch.float32 by default
torch.randn(3).dtype                 # float32 — ML standard

# Crucial for training:
a = torch.randn(3, dtype=torch.float32)  # 32-bit
b = torch.randn(3, dtype=torch.float64)  # 64-bit double — slower on GPU, more precise
c = torch.randn(3, dtype=torch.bfloat16) # 16-bit brain-float — for mixed precision training

# Type promotion is more conservative than NumPy
print((torch.ones(3, dtype=torch.float32) + torch.ones(3, dtype=torch.float64)).dtype)
# float64 — but in PyTorch, mixing floats often warns
```

**Key divergence from NumPy:** Pandas has `object` dtype. PyTorch does NOT. Everything must be numeric/boolean. If you need strings, that's not a Tensor.

### Device — where the buffer lives
```python
# CPU tensor (default)
x_cpu = torch.randn(3,4)

# GPU tensor (if available)
if torch.cuda.is_available():
    x_gpu = x_cpu.cuda()          # or .to('cuda')
    x_gpu = torch.randn(3,4, device='cuda')
    
    print(x_cpu.device)  # cpu
    print(x_gpu.device)  # cuda:0

# Generic device-agnostic code (best practice):
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(3,4, device=device)
y = x.to(device)  # no-op if already there

# Operations must be on same device
# x_cpu + x_gpu -> ERROR. Must move first.
```

> **Mental model:** `to(device)` is like copying buffer from RAM to VRAM. Strides/size stay identical, pointer changes.

---

## Phase 2 — Strides, Layout, and View vs Copy (Directly from NumPy)

This is your NumPy Phases 1-6 ported verbatim. PyTorch uses **C-order (row-major) by default**, exactly like NumPy.

### Stride formula (identical)
For a contiguous tensor with `size = (d0, d1, ..., d_{n-1})`:

```
stride[n-1] = 1
stride[d] = size[d+1] * stride[d+1]   (in elements, not bytes in API, but bytes internally)
offset = sum(index[d] * stride[d]) + storage_offset
```

```python
a = torch.zeros(3,4,5)
print(a.stride()) # (20, 5, 1) -> 4*5=20, 5=5, 1=1
print(a.stride(0), a.stride(1))

# Slicing changes stride and offset, not data
b = a[1, ::2, :] 
print(b.size())   # torch.Size([2, 5]) — after indexing
print(b.stride()) # (10, 1) — skipped 2 in dim1 -> stride doubled
print(b.storage_offset()) # 20 — 1 * 20
```

### View vs Copy — The Rules (Critical for Performance)

**VIEW operations (zero-copy, shares storage):**
```python
a = torch.arange(12).reshape(3,4)

# All views:
b = a.view(2,6)          # alias to reshape when contiguous — zero-copy
c = a.reshape(2,6)       # view if possible, copy if not — check with shares_memory
d = a.T                  # transpose — view with permuted strides
e = a.transpose(0,1)     # same
f = a.permute(1,0)        # general transpose
g = a[::2, :]            # basic slicing — view
h = a.unsqueeze(0)       # adds size-1 dim with stride 0 trick
i = a.squeeze()          # removes size-1 dims
j = a.expand(3,12) if False else a.unsqueeze(0).expand(3,-1,-1) # stride-0 broadcasting — view!

print(torch._C._is_aliased(a, d))  # checks sharing in PyTorch C++ backend
# or practical check:
print(a.untyped_storage().data_ptr() == d.untyped_storage().data_ptr()) # True
```

**COPY operations (new storage):**
```python
a = torch.arange(12).reshape(3,4)

# All copies:
b = a.clone()            # explicit full copy
c = a.contiguous()       # copy only if non-contiguous, else view
d = a.detach().clone()   # copy without grad history
e = a[a > 5]             # boolean/fancy indexing — ALWAYS copy in PyTorch (like NumPy)
f = a[[0,2]]             # fancy indexing — copy
g = torch.cat([a,a])     # cat/stack — copy
```

#### The `view()` vs `reshape()` distinction
This trips everyone:

```python
a = torch.arange(12).reshape(3,4)
b = a.T  # b is non-contiguous: size (4,3), stride (1,4)

# b.view(12) -> RuntimeError: view size is not compatible with input tensor's size and stride
# Why? view() can ONLY work with compatible strides. It never copies.

# reshape() will copy if needed to make it contiguous first
c = b.reshape(12)        # succeeds — does b.contiguous().view(12) under the hood
print(c.is_contiguous()) # True — it had to copy

# Best practice:
# Use view() when you WANT to assert contiguity (fail fast)
# Use reshape() when you just want the shape, don't care about copy
```

#### expand() — the stride-0 trick you saw in NumPy broadcasting
```python
a = torch.arange(3)  # [0,1,2]
b = a.unsqueeze(0).expand(3, 3)  # 3x3
print(b)
# tensor([[0, 1, 2],
#         [0, 1, 2],
#         [0, 1, 2]])
print(b.stride()) # (0, 1) — stride 0 means "reuse same element forever along dim0"
print(b.storage().size() if hasattr(b, 'storage') else "shared") # still 3 elements physically!

# This is how broadcasting works internally — zero memory overhead
x = torch.randn(3,1,5)
y = torch.randn(4,5)
z = x + y  # x is expanded via stride-0 to (3,4,5) — no copy until operation
```

---

## Phase 3 — Indexing: .[] is not one operation, it's four

Same taxonomy as NumPy, but PyTorch has no `.loc/.iloc` — just `[]` with rules.

```python
a = torch.arange(24).reshape(4,6)
print(a)

# 1. Basic indexing — VIEW
print(a[0])        # first row — view
print(a[1:3, ::2]) # slice — view

# 2. Fancy indexing (integer tensor) — COPY
idx = torch.tensor([0,2,0])
print(a[idx])      # rows 0,2,0 — copy, size (3,6)

# 3. Boolean indexing — COPY
mask = a > 10
print(a[mask])     # 1D copy of all elements >10

# 4. None / unsqueeze indexing — VIEW
print(a[None, :, :].size()) # torch.Size([1,4,6]) — adds dim
```

**In-place mutation and autograd versioning:**
```python
a = torch.randn(3,4, requires_grad=True)
# a[0] = 0  # OK, but increments a._version
# If a is needed for grad later, in-place can error — PyTorch protects you
# a.add_(1)  # in-place ops end with underscore

b = a * 2
# a.add_(1)  # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
# Fix: use out-of-place: a = a + 1
```

---

## Phase 4 — Memory Layout: C-order, Non-contiguous, channels_last

Your NumPy experiment: C vs F order, 8x slowdown. Same exists in PyTorch, but with GPU twist.

### What makes a tensor non-contiguous?
Any operation that permutes strides:

```python
a = torch.arange(12).reshape(3,4)
print(a.is_contiguous(), a.stride()) # True, (4,1)

b = a.T
print(b.is_contiguous(), b.stride()) # False, (1,4) — column-major like Fortran now!
print(b.size()) # (4,3) logically, but memory still row-major underneath

# Access pattern mismatch
# Iterating b row-wise now jumps by 4 in memory — cache miss
# This is why .contiguous() exists:

c = b.contiguous()
print(c.is_contiguous(), c.stride()) # True, (3,1) — copied into C-order
print(a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()) # True — view
print(a.untyped_storage().data_ptr() == c.untyped_storage().data_ptr()) # False — copy
```

**Benchmark (CPU) — same as your NumPy one:**
```python
import time
a_c = torch.randn(2048, 2048)
a_f = a_c.T  # non-contiguous view

def row_sum(x):
    s = 0.0
    for i in range(x.size(0)):
        s += x[i].sum().item()
    return s

# In PyTorch you'd just do x.sum(dim=1), but to demonstrate cache:
start = time.time()
_ = a_c.sum(dim=1)  # contiguous row sum — coalesced
print(f"C-contig sum: {time.time()-start:.4f}s")

start = time.time()
_ = a_f.sum(dim=0)  # same logical operation but on non-contig layout (row of original)
print(f"F-contig via transpose: {time.time()-start:.4f}s") # often slower on CPU
```

### channels_last — PyTorch's secret performance layout
For CNNs, PyTorch supports `channels_last` memory format:

```python
# NCHW is default: batch, channel, height, width — stride (C*H*W, H*W, W, 1)
x = torch.randn(2,3,224,224)  
print(x.stride()) # e.g. (150528, 50176, 224, 1)

# NHWC (channels_last): better for GPU Tensor Cores on conv
y = x.to(memory_format=torch.channels_last)
print(y.is_contiguous(memory_format=torch.channels_last)) # True
print(y.stride()) # (150528, 1, 672, 3) — channel is innermost, contiguous in H*W*C blocks

# Conv with channels_last can be 20-30% faster on Ampere GPUs
# model = model.to(memory_format=torch.channels_last)
```

---

## Phase 5 — Broadcasting and Alignment (Same as NumPy Phase 4)

PyTorch broadcasting rules are **identical** to NumPy:

1.  Align trailing dimensions
2.  Size 1 or missing dim → broadcast via stride-0
3.  Mismatch otherwise → error

```python
# (3,1,5) + (4,5) -> (3,4,5)
x = torch.randn(3,1,5)
y = torch.randn(4,5)
print((x+y).size()) # torch.Size([3, 4, 5])

# Explicit broadcast_to = expand
a = torch.randn(1,5)
print(a.expand(3,5).stride()) # (0,1) — stride 0 trick

# Broadcasting is lazy — expand doesn't allocate until operation
```

**Pandas-style alignment?** PyTorch does NOT align by label. It's purely positional like NumPy. If you want label alignment, you build it yourself.

---

## Phase 6 — Autograd: The Twist That Changes Everything

This is the one mental model NumPy didn't have. Every Tensor can track its history.

### Leaf vs Non-leaf, requires_grad

```python
# Leaf tensor — user-created, requires grad
x = torch.randn(3,4, requires_grad=True)
print(x.is_leaf)        # True
print(x.grad_fn)        # None — no history

# Non-leaf — result of op
y = x * 2 + 1
print(y.is_leaf)        # False
print(y.grad_fn)        # <AddBackward0> — remembers how to backward
print(y.grad_fn.next_functions) # references to MulBackward, etc.

# Grad only accumulates on leaves by default
# y.grad will stay None unless you call y.retain_grad()
```

### The computational graph is a DAG of grad_fns

```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Forward: build graph
a = x * y       # a = 6, grad_fn = MulBackward
b = a + y       # b = 9, grad_fn = AddBackward
c = b.sin()     # c = sin(9), grad_fn = SinBackward

print(c.grad_fn)
# SinBackward -> AddBackward -> MulBackward + leaf y

# Backward: traverse graph reverse, apply chain rule
c.backward()

print(x.grad)  # d c / d x = cos(b) * y = cos(9) * 3
print(y.grad)  # d c / d y = cos(b) * (x + 1) = cos(9) * 3
```

**Mathematically:** `backward()` does reverse-mode AD. For each `grad_fn`, it computes `grad_input = grad_output * local_jacobian`.

### .backward() vs .grad

```python
x = torch.randn(3, requires_grad=True)
y = (x**2).sum()  # scalar — .backward() works only on scalar by default
y.backward()
print(x.grad) # 2*x

# For non-scalar, need grad_output
x2 = torch.randn(3, requires_grad=True)
y2 = x2 * 2  # vector
y2.backward(torch.ones_like(y2)) # grad_output = [1,1,1] — like sum
print(x2.grad) # [2,2,2]

# Common pitfall: grad accumulates!
x.grad.zero_()  # MUST zero grad between steps, or it adds
```

### torch.no_grad() and inference_mode — disabling the graph

```python
x = torch.randn(3,4, requires_grad=True)

# 1. No graph built — for inference, 2x faster, less memory
with torch.no_grad():
    y = x * 2
    print(y.requires_grad) # False — history stripped

# 2. inference_mode — even faster, more aggressive than no_grad (PyTorch 2.0+)
with torch.inference_mode():
    y = x * 2
    # also disables version counter checks — fastest for pure inference

# 3. detach() — cut a single tensor from graph
y = (x*2).detach()  # y is new leaf, no history, shares storage
print(y.requires_grad) # False
print(x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()) # True — view-like but no grad
```

---

## Phase 7 — Operations: Ufuncs, Reductions, Linear Algebra

### Elementwise (like NumPy ufuncs) — always vectorized, autograd-aware
```python
a = torch.randn(3,4)
torch.add(a,a), a + a
torch.mul(a,a), a * a
torch.exp(a), torch.sin(a), torch.relu(a)

# In-place versions (underscore) — avoid in autograd!
# a.add_(1) — mutates storage, bumps version

# Clamping, etc.
torch.clamp(a, 0, 1)
torch.where(a > 0, a, torch.zeros_like(a))  # like np.where
```

### Reductions — axis semantics identical to NumPy

```python
a = torch.arange(12).reshape(3,4).float()
print(a.sum())          # scalar — sum all
print(a.sum(dim=0))     # sum over rows -> size (4,) — dim 0 collapsed
print(a.sum(dim=1))     # sum over cols -> size (3,)
print(a.sum(dim=(0,1))) # sum over both
print(a.mean(dim=0))
print(a.argmax(dim=1))

# keepdim — preserves rank (important for broadcasting back)
print(a.sum(dim=1, keepdim=True).size()) # (3,1) not (3,)
# Allows: a - a.mean(dim=1, keepdim=True)  # broadcast subtraction

# Stable softmax (doesn't overflow)
# Don't do: torch.exp(a) / torch.exp(a).sum()
# Do:
print(torch.softmax(a, dim=1))  # numerically stable
print(torch.log_softmax(a, dim=1))
```

### Linear Algebra — matmul is the core

```python
# Vector dot
x = torch.randn(4)
y = torch.randn(4)
print(torch.dot(x,y))  # scalar

# Matrix-vector
A = torch.randn(3,4)
print(A @ x)           # (3,) = (3,4) @ (4,)
print(torch.mv(A, x))  # same

# Matrix-matrix
B = torch.randn(4,5)
print(A @ B)           # (3,5) = (3,4) @ (4,5)
print(torch.mm(A,B))   # same but stricter

# Batched matmul — THE deep learning workhorse
# (b, n, m) @ (b, m, p) -> (b, n, p) — batch dim broadcasts
batch_A = torch.randn(32, 3, 4)  # 32 matrices of 3x4
batch_B = torch.randn(32, 4, 5)
print((batch_A @ batch_B).size()) # torch.Size([32, 3, 5])

# Broadcasting batch too: (32,3,4) @ (4,5) -> (32,3,5)

# Einsum — generalizes everything (learn this)
print(torch.einsum('ij,jk->ik', A, B).size()) # same as A @ B
print(torch.einsum('bij,bjk->bik', batch_A, batch_B).size()) # batched
print(torch.einsum('i,i->', x, y)) # dot
print(torch.einsum('ij->ji', A).size()) # transpose

# Decompositions
U, S, Vh = torch.linalg.svd(A)
Q, R = torch.linalg.qr(A)
print(torch.linalg.norm(A), torch.linalg.det(torch.eye(3)))
```

---

## Phase 8 — Reshaping: view, reshape, transpose, and contiguous hell

This is where NumPy intuition saves you, but PyTorch has extra constraints due to autograd.

```python
a = torch.arange(12).reshape(3,4)

# 1. view() — zero-copy, fails if non-contiguous
print(a.view(2,6).stride()) # (6,1)

b = a.T  # non-contig
try:
    b.view(12)
except RuntimeError as e:
    print(f"view failed: {e}")

# 2. reshape() — view if possible, copy otherwise (safe)
print(b.reshape(12).is_contiguous()) # True — it copied

# 3. The contiguous() trap
# contiguous() looks harmless but it IS a copy when non-contiguous
# In training loops, calling .contiguous() in hot path = unexpected allocation
c = b.contiguous()  # copy!
print(c.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()) # True

# 4. Transpose vs Permute
x = torch.randn(2,3,4,5)
print(x.transpose(1,2).size()) # swaps dim1 and 2: (2,4,3,5)
print(x.permute(0,3,1,2).size()) # arbitrary order: (2,5,3,4)

# 5. Squeeze/unsqueeze — stride tricks
print(x.unsqueeze(0).size()) # (1,2,3,4,5) — stride (...,0) trick
print(x.squeeze().size())    # removes size-1 dims
```

**The rule from NumPy still holds:** If you can compute new strides that produce the new shape from old storage without overlapping incorrectly, it's a view. Otherwise, copy.

---

## Phase 9 — Advanced Stride Tricks: as_strided, unfold, and your sliding window

You showed `as_strided` for sliding windows. PyTorch has it too, with same dangers.

```python
a = torch.arange(10)

# Manual as_strided — zero safety, same as NumPy
# shape (8,3), stride (1,1) in elements
view = torch.as_strided(a, size=(8,3), stride=(1,1))
print(view)
# tensor([[0, 1, 2],
#         [1, 2, 3],
#         ...
#         [7, 8, 9]])
print(view.untyped_storage().data_ptr() == a.untyped_storage().data_ptr()) # True — aliased

# DANGEROUS: no bounds check
bad = torch.as_strided(a, size=(100,100), stride=(1,1))  # reads past buffer!
# print(bad)  # garbage / segfault potential — same as your NumPy demo

# SAFE alternative: unfold (like sliding_window_view)
safe = a.unfold(dimension=0, size=3, step=1)  # (8,3) — bounds-checked, zero-copy
print(safe)
# tensor([[0, 1, 2],
#         [1, 2, 3],
#         ...
#         [7, 8, 9]])

# For images: unfold for conv patches
img = torch.randn(1,1,5,5)  # NCHW
patches = img.unfold(2,3,1).unfold(3,3,1)  # 3x3 patches
print(patches.size()) # torch.Size([1, 1, 3, 3, 3, 3]) — (N,C,H_out,W_out,kH,kW)

# Or use torch.nn.Unfold module
unfold = torch.nn.Unfold(kernel_size=3, stride=1)
print(unfold(img).size()) # torch.Size([1, 9, 9]) — flattened patches
```

---

## Phase 10 — nn.Module: How a Model is Just a Tree of Tensors

In NumPy, you have arrays. In PyTorch, you have **Modules** that *own* arrays called Parameters.

```python
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameters are Tensors with requires_grad=True that are registered
        self.W1 = nn.Parameter(torch.randn(4,8) * 0.1)  # explicit
        self.b1 = nn.Parameter(torch.zeros(8))
        self.linear2 = nn.Linear(8, 1)  # module that owns W and b internally
        
    def forward(self, x):
        # x: (batch, 4)
        x = x @ self.W1 + self.b1   # (batch,8)
        x = torch.relu(x)
        x = self.linear2(x)         # (batch,1)
        return x

model = TinyModel()
print(list(model.parameters()))  # iterator over all Parameter tensors in tree
print(list(model.named_parameters())) # with names

x = torch.randn(2,4)
y = model(x)
print(y.size()) # (2,1)

# state_dict — the pure data, no code
print(model.state_dict().keys()) # dict of tensor data
# torch.save(model.state_dict(), 'model.pt')
# model.load_state_dict(torch.load('model.pt'))
```

**Module is a tree:**
```python
# model
#  ├─ W1: Parameter (4,8)
#  ├─ b1: Parameter (8,)
#  └─ linear2: Linear
#      ├─ weight: Parameter (1,8) — note transposed storage vs forward
#      └─ bias: Parameter (1,)

for name, module in model.named_modules():
    print(name, type(module).__name__)
```

---

## Phase 11 — Training Loop: Where Everything Meets

```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Dummy data
X = torch.randn(64, 10)  # batch 64
y_true = torch.randn(64, 1)

# One training step — the pattern for ALL training
model.train()
optimizer.zero_grad()          # 1. Zero accumulated grads (crucial!)
y_pred = model(X)              # 2. Forward — builds graph
loss = loss_fn(y_pred, y_true) # 3. Compute loss — scalar tensor with grad_fn
loss.backward()                # 4. Backward — computes grad for every Parameter
print(model[0].weight.grad.size()) # (32,10) — grad same shape as param
optimizer.step()               # 5. Update: W -= lr * grad

# Under the hood of optimizer.step():
# for p in model.parameters():
#     p.data -= lr * p.grad
```

**Why zero_grad()?** `.grad` accumulates (like NumPy `out=`). If you forget, you get grad from last batch + current batch.

**Gradient accumulation — intentional use:**
```python
for i in range(4):  # 4 mini-batches
    y_pred = model(X[i*16:(i+1)*16])
    loss = loss_fn(y_pred, y_true[i*16:(i+1)*16]) / 4  # scale loss
    loss.backward()  # accumulates grad
optimizer.step()     # update after 4 forwards
optimizer.zero_grad()
# Effective batch size 64 with memory of 16
```

---

## Phase 12 — Device & Performance: The Supercomputer Part

### CUDA streams and async
```python
# CPU launch is async — GPU work queued, CPU continues
# This is why timing needs synchronize
if torch.cuda.is_available():
    x = torch.randn(1000,1000, device='cuda')
    torch.cuda.synchronize() # wait for previous work
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    y = x @ x  # queued, returns immediately
    end.record()
    torch.cuda.synchronize() # wait for y
    print(f"matmul: {start.elapsed_time(end)}ms")
```

### Mixed Precision (AMP) — bfloat16 + float32 master weights
```python
# 2-3x faster on modern GPUs, half memory
scaler = torch.amp.GradScaler('cuda')

with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    y_pred = model(X.cuda())
    loss = loss_fn(y_pred, y_true.cuda())

# Scale loss to avoid underflow in bfloat16 grad
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### torch.compile — The big PyTorch 2.0 feature
```python
# Compiles Python model into optimized kernels (like XLA)
model = nn.Sequential(nn.Linear(10,32), nn.ReLU(), nn.Linear(32,1))

compiled_model = torch.compile(model, mode='max-autotune')  # or 'default', 'reduce-overhead'

# First call compiles (slow), next calls fast
y = compiled_model(torch.randn(64,10))  # ~30-50% faster often

# What it does: fuses ops, removes Python overhead, generates Triton kernels
# Example: ReLU + Linear can be fused into one kernel launch instead of two
```

### Memory — what actually costs VRAM

```python
# For a model: params + grads + optimizer states + activations

# Example: Linear(1024,1024) -> 1M params
# float32: 1M * 4 bytes = 4MB per copy
# Adam keeps 2 states per param (m, v): 2 * 4MB = 8MB
# So 1 layer = 4MB (weight) + 4MB (grad) + 8MB (adam) = 16MB
# + activations: batch * hidden * 4 bytes, kept for backward!

# Check memory
if torch.cuda.is_available():
    print(torch.cuda.memory_allocated() / 1e6, "MB allocated")
    print(torch.cuda.memory_reserved() / 1e6, "MB reserved")

# Save activation memory: gradient checkpointing
from torch.utils.checkpoint import checkpoint
# Instead of: y = layer(x) — saves x for backward
# Do: y = checkpoint(layer, x) — recomputes forward in backward, trades compute for memory
```

---

## Phase 13 — Data Pipeline: Dataset, DataLoader, and Avoiding the Bottleneck

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 10)
        self.labels = torch.randn(size, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Returns one sample — can load from disk, augment, etc.
        return self.data[idx], self.labels[idx]

dataset = MyDataset(1000)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

# Training loop with DataLoader
for X_batch, y_batch in loader:
    # X_batch is (32,10) — automatically collated
    # pin_memory + to('cuda', non_blocking=True) = faster H2D copy
    X_batch = X_batch.to('cuda', non_blocking=True)
    # ... train step
    pass

# num_workers=0 -> data loaded in main process (debugging)
# num_workers=4 -> 4 processes prefetch batches — hides disk I/O
# pin_memory -> page-locked memory, faster copy to GPU
```

---

## Phase 14 — Advanced Autograd: Custom Functions, Hooks, Functional API

### Custom autograd Function — when you need manual backward
```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)  # save for backward
        return x.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        return grad_input

# Use it
x = torch.randn(3,4, requires_grad=True)
y = MyReLU.apply(x)  # forward uses your code, backward uses your backward
y.sum().backward()
```

### Hooks — inspect/modify grads
```python
x = torch.randn(3,4, requires_grad=True)
y = x * 2

# Register hook on non-leaf
y.register_hook(lambda grad: print(f"grad of y: {grad.norm()}"))

# Hook on module
model = nn.Linear(4,2)
model.weight.register_hook(lambda grad: grad.clamp(-1,1))  # gradient clipping via hook

y.sum().backward()
```

### torch.func — vmap, grad, functional transforms (PyTorch 2.0+)
```python
# vmap — vectorize a function (no more manual batching!)
def f(x):  # x: (4,)
    return (x**2).sum()

x_batch = torch.randn(8,4)
# Old way: loop or batched ops
# New way:
batched_f = torch.func.vmap(f)
print(batched_f(x_batch).size()) # (8,) — f applied to each

# grad — functional grad (no .backward() side effects)
grad_f = torch.func.grad(f)
print(grad_f(torch.randn(4)).size()) # (4,) — grad of f

# Combine: per-sample grads — crucial for research
per_sample_grads = torch.func.vmap(torch.func.grad(f))(x_batch)
print(per_sample_grads.size()) # (8,4) — grad per sample, impossible with regular backward easily

# jacrev, hessian, etc.
```

---

## Phase 15 — The Full Cheat Sheet

### View vs Copy in PyTorch (definitive)
| Operation | View? | Notes |
|-----------|-------|-------|
| `a[0]`, `a[1:3]` | View | basic slicing |
| `a.T`, `transpose`, `permute` | View | changes stride |
| `view()`, `reshape()` if contig | View | `view()` asserts, `reshape()` may copy |
| `squeeze`, `unsqueeze` | View | stride-0 or remove |
| `expand`, `broadcast_to` | View | stride-0 trick |
| `narrow`, `select` | View | |
| `a[mask]`, `a[[0,1]]` | Copy | fancy/bool always copy |
| `clone()`, `contiguous()` | Copy | `contiguous()` copies only if needed |
| `detach()` | View-ish | shares storage, no grad |
| `torch.cat`, `stack` | Copy | new storage |

### Contiguity check
```python
a.is_contiguous() # C-contig?
a.is_contiguous(memory_format=torch.channels_last) # channels_last contig?
```

### When to call contiguous()
- Before `view()` on a transposed tensor
- Before passing to custom CUDA kernel expecting C-order
- DON'T call it blindly in hot loop — it allocates

### Broadcasting vs Alignment
- PyTorch = NumPy = positional broadcasting, stride-0 lazy
- No label alignment (unlike pandas). `tensor_a + tensor_b` aligns by trailing dim, not by index

### Autograd rules
- `requires_grad=True` on leaf -> tracks ops
- `grad_fn` on non-leaf -> how to backward
- `grad` only on leaf by default, accumulates
- Always `zero_grad()` before `backward()`
- Use `no_grad()` for inference, `inference_mode()` for even faster inference
- In-place ops bump version, can break graph

### Device rules
- All inputs to op must be same device
- `.to(device)` is copy if device changes, no-op otherwise
- `non_blocking=True` for async H2D copy when using `pin_memory`
- Check `torch.cuda.is_available()`

### Performance checklist
1.  Keep tensors contiguous for row-wise ops (cache / coalescing)
2.  Use `channels_last` for CNNs on GPU
3.  Use `torch.compile()` for end-to-end speed
4.  Use mixed precision `autocast` + `GradScaler`
5.  Use `DataLoader(num_workers>0, pin_memory=True)`
6.  Avoid Python loops — vectorize with `vmap` or batched ops
7.  Use `view()` not `reshape()` when you want to catch accidental copies
8.  Check `memory_allocated()` — activations are often bigger than params

---

## Phase 16 — From NumPy Map to PyTorch Map: Direct Translation

| NumPy concept you mastered | PyTorch equivalent | New twist |
|---|---|---|
| `ndarray.strides` (bytes) | `tensor.stride()` (elements) | Divide by itemsize |
| `arr.tobytes(order='A')` | `tensor.untyped_storage()` + `storage_offset` | + device |
| `np.lib.stride_tricks.as_strided` | `torch.as_strided` | Same danger, use `unfold` |
| `sliding_window_view` | `tensor.unfold` | Bounds-checked |
| `broadcast_to` (stride 0) | `expand` (stride 0) | Lazy, zero-copy |
| `arr.reshape(-1)` | `tensor.reshape(-1)` or `view(-1)` | `view` asserts contig |
| `C vs F order` | `is_contiguous()` vs `T.is_contiguous()` + `channels_last` | GPU coalescing matters more |
| `ufunc` | elementwise ops + autograd | Every op has `grad_fn` |
| `axis=0` reductions | `dim=0` reductions | Same semantics, `keepdim` instead of `keepdims` |
| No equivalent | `requires_grad`, `backward()`, `no_grad()` | Differentiation |
| No equivalent | `nn.Module`, `Parameter` | Tree of tensors |

> **Final synthesis:** Your entire NumPy map was "how to interpret a flat buffer with strides." PyTorch's map is "how to interpret a flat buffer with strides *on any device*, and how to remember *how you got there* so you can differentiate through it." The offset formula never changed — we just added `device` and `grad_fn` to the header.

---

## Appendix — Minimal Runnable Training Template (Everything Together)

```python
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(device)

# Optional: channels_last or compile for perf
# model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scaler = torch.amp.GradScaler('cuda') if device.type=='cuda' else None

def train_step(X, y):
    optimizer.zero_grad()
    # Autocast for mixed precision
    with torch.amp.autocast(device.type, enabled=(scaler is not None)):
        pred = model(X)
        loss = nn.functional.mse_loss(pred, y)
    
    if scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return loss.item()

# Fake data
for step in range(100):
    X = torch.randn(128, 10, device=device)
    y = torch.randn(128, 1, device=device)
    loss = train_step(X, y)
    if step % 20 == 0:
        print(f"step {step}: loss {loss:.4f}")
```

That's the full map, NumPy-style but PyTorch-deep. Same flat buffer, new superpowers.



---

## ADDITIONAL ENHANCEMENTS - Inspired by NumPy Cheatsheet Missing in Original


## 10. Universal Functions (ufuncs) - Arithmetic, Comparison, Logic (NumPy Section 8 Equivalent)

```python
a = torch.randn(3,4)
b = torch.randn(3,4)

# ---- Arithmetic - elementwise, autograd-aware ----
a + b == torch.add(a,b) == a.add(b)
a - b == torch.sub(a,b) == a.sub(b)
a * b == torch.mul(a,b) == a.mul(b)
a / b == torch.div(a,b) == a.div(b)
a ** 2 == torch.pow(a,2) == a.pow(2)
a % 2 == torch.remainder(a,2) == a.remainder(2)  # modulo
-a == torch.neg(a) == a.neg()
# In-place: a.add_(b), a.mul_(2) etc - bumps version

# Comparison - returns bool tensor
a == b, torch.eq(a,b), a.eq(b)
a != b, torch.ne(a,b)
a > b, torch.gt(a,b), a.gt(b)
a >= b, torch.ge(a,b), a.ge(b)
a < b, torch.lt(a,b)
a <= b, torch.le(a,b)

# Logical & Bitwise (bool tensors)
mask = a > 0
# (a > 0) & (b > 0)  # use & | ~ not and/or, need parens - SAME AS NUMPY PITFALL!
(a > 0) & (b > 0)    # correct
(a > 0) | (b > 0)
~(a > 0)             # not

torch.logical_and(a>0, b>0)
torch.logical_or(a>0, b>0)
torch.logical_not(a>0)
torch.logical_xor(a>0, b>0)

# Where - ternary
torch.where(a > 0, a, torch.zeros_like(a))  # like np.where(condition, x, y)
# Two-mode like NumPy:
torch.where(a > 0)  # returns tuple of indices where True - like np.where(condition) only
a.nonzero()         # same as torch.where(a>0) but method
torch.nonzero(a > 0, as_tuple=False)  # indices
```

## 11. Mathematical Functions - Exponential, Trig, Rounding (NumPy Section 9 Equivalent)

```python
a = torch.randn(3,4)

# Exponential & Log
torch.exp(a) == a.exp()          # e^a
torch.expm1(a)                   # exp(a)-1 more accurate for small a
torch.log(a.abs()) == a.abs().log()  # natural log
torch.log10(a.abs())
torch.log2(a.abs())
torch.log1p(a.abs())             # log(1+a) accurate for small a
torch.sqrt(a.abs()) == a.abs().sqrt()
torch.rsqrt(a.abs())             # 1/sqrt(a) - faster
torch.square(a)                  # a^2

# Trig / Hyperbolic
torch.sin(a) == a.sin()
torch.cos(a), torch.tan(a)
torch.asin(a.clamp(-1,1)), torch.acos(a.clamp(-1,1)), torch.atan(a)
torch.atan2(b, a)                # atan2(y,x) - quadrant aware
torch.sinh(a), torch.cosh(a), torch.tanh(a)  # hyperbolic
torch.asinh(a), torch.acosh((a.abs()+1)), torch.atanh(a.clamp(-0.99,0.99))

# Rounding & Misc
torch.round(a) == a.round()      # to nearest int
torch.floor(a) == a.floor()
torch.ceil(a) == a.ceil()
torch.trunc(a)                   # truncate toward 0
torch.frac(a)                    # fractional part
torch.clamp(a, -1, 1) == a.clamp(-1,1)  # like np.clip
torch.clip(a, -1, 1)             # alias
torch.sign(a)                    # -1,0,1
torch.abs(a) == a.abs()

# Special functions - PyTorch has many like NumPy
torch.erf(a)                     # error function
torch.erfc(a)
torch.lerp(a, b, weight=0.5)     # linear interpolation
torch.hypot(a,b)                 # sqrt(a^2+b^2) stable

# NaN / Inf handling
torch.isnan(a)
torch.isinf(a)
torch.isfinite(a)
torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)  # like np.nan_to_num
```

## 12. Statistics & Aggregation - DUAL FORM (NumPy Section 10 Equivalent)

```python
a = torch.randn(3,4)

# Reductions - dim semantics identical to NumPy axis, keepdim vs keepdims
torch.sum(a) == a.sum()                      # all
torch.sum(a, dim=0) == a.sum(dim=0)          # along dim 0 - shape (4,)
torch.sum(a, dim=1, keepdim=True)            # shape (3,1) - preserves rank for broadcast

# Common - both forms exist
torch.mean(a) == a.mean()                    # mean
torch.mean(a, dim=0)
torch.median(a)                              # median (function only? also method)
a.median()
a.median(dim=0)                              # returns (values, indices) tuple!
torch.var(a) == a.var()
torch.var(a, dim=0, unbiased=False)          # unbiased=False -> population var like np.var, True -> sample var (default!)
# WARNING: torch.var unbiased default True (divides by N-1) vs NumPy var divides by N! Use unbiased=False to match NumPy
torch.std(a) == a.std()                      # same unbiased note
torch.prod(a) == a.prod()
torch.min(a) == a.min()                      # alias torch.amin
torch.max(a) == a.max()                      # alias torch.amax
torch.argmin(a) == a.argmin()
torch.argmax(a) == a.argmax()
torch.amin(a, dim=0)                         # min along dim - explicit
torch.amax(a, dim=0)

# Cumulative
torch.cumsum(a, dim=0) == a.cumsum(dim=0)
torch.cumprod(a, dim=0) == a.cumprod(dim=0)
torch.cummax(a, dim=0)                       # returns (values, indices)
torch.cummin(a, dim=0)

# Other aggregations
torch.norm(a)                                # Frobenius / L2 norm by default
torch.norm(a, p=1, dim=0)                    # L1 along dim
torch.unique(a)                              # unique values
torch.histc(a, bins=10)                      # histogram like np.histogram

# NaN-aware - like np.nanmean etc
torch.nansum(a)
torch.nanmean(a)
# torch.nanmedian etc in newer versions

# Keepdim pattern - critical for ML
# Normalize rows: (x - mean) / std
x = torch.randn(8, 10)
x_centered = x - x.mean(dim=1, keepdim=True)  # broadcast back - needs keepdim
x_normalized = x_centered / (x.std(dim=1, keepdim=True) + 1e-8)
```

## 13. Sorting, Searching & Counting (NumPy Section 11 Equivalent)

```python
a = torch.tensor([3,1,4,1,5,9,2,6])

# Sorting - returns tuple! Unlike NumPy which returns only values
torch.sort(a)                  # -> (sorted_values, indices) tuple
values, indices = torch.sort(a)
a.sort()                       # method also returns tuple (values, indices) - NOT in-place like NumPy!
# For descending:
torch.sort(a, descending=True)

# Argsort
torch.argsort(a) == a.argsort()  # indices that would sort
a[torch.argsort(a)]              # sorted via fancy indexing (copy)

# Topk - PyTorch exclusive, crucial for ML - more efficient than full sort
torch.topk(a, k=3)               # 3 largest -> (values, indices)
torch.topk(a, k=3, largest=False)  # 3 smallest

# Searching
torch.searchsorted(torch.tensor([1,2,3,4,5]), torch.tensor([2.5]))  # like np.searchsorted
a.searchsorted(torch.tensor([3])) if hasattr(a, 'searchsorted') else torch.searchsorted(a.sort()[0], torch.tensor([3]))

# Counting / Non-zero
torch.nonzero(a > 3)             # indices where True - like np.nonzero / np.where(condition)
# as_tuple=True gives tuple per dim like NumPy
torch.nonzero(a > 3, as_tuple=True)  # (tensor([...]),)
a.nonzero()                      # method form

# Boolean reductions
(a > 3).any() == torch.any(a > 3)   # any True?
(a > 0).all() == torch.all(a > 0)   # all True?

# Where - already covered but searching aspect
torch.where(a > 3)               # tuple of indices
torch.where(a > 3, a, torch.zeros_like(a))  # ternary

# Unique & counts
torch.unique(a)                  # sorted unique
torch.unique(a, return_counts=True)  # -> (unique, counts) - like np.unique(return_counts=True)
torch.unique(a, return_inverse=True) # inverse indices to reconstruct original
# For dim:
b = torch.tensor([[1,2],[1,2],[3,4]])
torch.unique(b, dim=0)           # unique rows
```

## 14. Linear Algebra - Masterclass (NumPy Section 12 + Original Phase 7)

```python
# PyTorch linalg - torch.linalg module, like np.linalg but batched and GPU accelerated

A = torch.randn(3,3)
B = torch.randn(3,3)
x = torch.randn(3)

# Basic ops
A @ B == torch.matmul(A,B) == A.matmul(B)  # matmul - THE core
A @ x                   # matrix-vector
x @ x                   # vector dot -> scalar
torch.dot(x, x)         # dot product - only 1D

# Batched matmul - THE deep learning workhorse (NumPy doesn't have batch by default, PyTorch does!)
batch_A = torch.randn(32, 3, 4)  # 32 matrices 3x4
batch_B = torch.randn(32, 4, 5)  # 32 matrices 4x5
batch_C = batch_A @ batch_B      # (32,3,5) - batch dim broadcasts! Automatic

# Broadcasting batch too: (32,3,4) @ (4,5) -> (32,3,5) - B broadcast to batch
torch.randn(32,3,4) @ torch.randn(4,5)  # works!

# Transpose
A.T == A.transpose(0,1) == torch.transpose(A,0,1)  # 2D transpose
A.mT                      # matrix transpose of last 2 dims - for batched: (...,M,N) -> (...,N,M) - PyTorch 2.0+

# Decompositions & Inverses - torch.linalg.*
torch.linalg.inv(A)       # inverse - like np.linalg.inv
torch.linalg.pinv(A)      # pseudo-inverse
torch.linalg.det(A)       # determinant
torch.linalg.slogdet(A)   # stable log det -> (sign, logabsdet)
torch.linalg.eig(A)       # eigenvalues/vectors - complex
torch.linalg.eigh(A)      # for symmetric/hermitian - faster, real eigenvalues
torch.linalg.svd(A)       # SVD -> U,S,Vh
torch.linalg.qr(A)        # QR
torch.linalg.cholesky(A @ A.T + torch.eye(3)*1e-3)  # Cholesky - needs PD
torch.linalg.solve(A, x)  # solve Ax=b - DON'T use inv() then @ b, use solve! Same pitfall as NumPy
# BAD: torch.linalg.inv(A) @ x
# GOOD: torch.linalg.solve(A, x)

# Norms
torch.linalg.norm(A)              # Frobenius
torch.linalg.norm(x, ord=2)       # L2
torch.linalg.vector_norm(x, ord=2)
torch.linalg.matrix_norm(A, ord='fro')

# Other
torch.linalg.matrix_rank(A)
torch.trace(A) == A.trace()       # sum diag
torch.diagonal(A) == A.diagonal()

# Einsum - generalizes everything
torch.einsum('ij,jk->ik', A, B)   # matmul
torch.einsum('ii->i', A)          # diag
torch.einsum('ij->j', A)          # sum axis 0
```

## 15. Random Sampling - Old vs New, Distributions (NumPy Section 13 Equivalent)

```python
# PyTorch random is simpler than NumPy's Generator vs legacy - but has manual_seed and Generator object

# Global seed - affects all CPU ops
torch.manual_seed(0)
# GPU seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)  # all GPUs

# Generator - local, doesn't affect global state (like np.random.default_rng)
gen = torch.Generator()
gen.manual_seed(0)
a = torch.randn(3,4, generator=gen)  # uses local gen

# Distributions - core
torch.rand(3,4)                 # U[0,1) uniform
torch.randn(3,4)               # N(0,1) standard normal - most used in ML init
torch.randn_like(a)
torch.rand_like(a)
torch.randint(0, 10, (3,4))     # random ints [low, high)
torch.randperm(10)              # permutation 0..n-1 - like np.random.permutation

# More distributions via torch.distributions or direct
torch.bernoulli(torch.full((3,4), 0.5))  # Bernoulli 0/1 with p
torch.poisson(torch.full((3,4), 2.0))    # Poisson
torch.multinomial(torch.tensor([0.1,0.5,0.4]), num_samples=10, replacement=True)  # categorical samples

# Using torch.distributions - advanced (like NumPy's Generator methods)
import torch.distributions as D
normal = D.Normal(loc=0.0, scale=1.0)
samples = normal.sample((3,4))            # N(0,1)
log_probs = normal.log_prob(samples)

uniform = D.Uniform(0, 1)
beta = D.Beta(2.0, 5.0)

# Shuffling & choice
# No direct choice like np.random.choice, use multinomial or randperm
idx = torch.randperm(100)[:10]             # random 10 indices without replacement - choice without replacement
# Weighted choice:
weights = torch.tensor([0.1, 0.9, 0.0, 0.5])
samples = torch.multinomial(weights, 5, replacement=True)

# For reproducibility in DataLoader
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)
# DataLoader(..., worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(0))
```

## 16. Set Operations, Unique & Masked Ops (NumPy Section 14 + 19)

```python
a = torch.tensor([1,2,2,3,4])
b = torch.tensor([3,4,4,5,6])

# Unique - PyTorch's set ops are limited vs NumPy, but unique is core
torch.unique(a)                              # [1,2,3,4] sorted unique
u, counts = torch.unique(a, return_counts=True)  # counts
u, inverse = torch.unique(a, return_inverse=True)  # inverse to reconstruct
# torch.unique(a, dim=0) for unique rows

# Set-like via unique + isin
# NumPy has in1d, setdiff1d, intersect1d, union1d - PyTorch has isin and manual

torch.isin(a, b)                             # element-wise check if in b - like np.isin - PyTorch 1.10+
# a[torch.isin(a,b)] -> intersection values (with duplicates)
# For setdiff: a[~torch.isin(a,b)]

# Intersection / Union manually
# Intersection unique:
torch.tensor(list(set(a.tolist()) & set(b.tolist())))  # via Python set, then back to tensor
# Or: torch.unique(torch.cat([a,b]))[torch.isin(torch.unique(torch.cat([a,b])), a) & torch.isin(torch.unique(torch.cat([a,b])), b)]

# Masked operations - like np.ma
# PyTorch has masked_fill, masked_select, masked_scatter
x = torch.randn(3,4)
mask = x > 0
x.masked_fill(mask, 0)                       # fill where mask True with 0 - like np.ma where?
x.masked_fill(~mask, float('-inf'))          # common for attention masks
x.masked_select(mask)                        # select where True - 1D result - like x[mask] but method
# x[mask] also works - boolean indexing (copy)

# Where as masked
torch.where(mask, x, torch.zeros_like(x))    # ternary

# Clamping with mask
# No direct masked array class, but you can use:
# torch.masked.MaskedTensor - experimental in PyTorch 2.0+
# masked_tensor = torch.masked.as_masked_tensor(x, mask)
```

## 17. Fourier Transform (FFT) - torch.fft (NumPy Section 16 Equivalent)

```python
# torch.fft module - like np.fft but GPU accelerated, batched, autograd-aware!

x = torch.randn(8)  # signal length 8

# 1D FFT
X = torch.fft.fft(x)              # complex tensor - like np.fft.fft
x_recon = torch.fft.ifft(X)       # inverse - should be ~x (complex, take .real)
torch.fft.rfft(x)                 # real FFT - only positive freqs, faster for real input - like np.fft.rfft
torch.fft.irfft(torch.fft.rfft(x), n=8)

# 2D FFT - for images
img = torch.randn(32,32)
F = torch.fft.fft2(img)           # 2D FFT
img_back = torch.fft.ifft2(F)

# Batched - PyTorch shines here! (batch, H, W) all FFT'd in parallel on GPU
batch_imgs = torch.randn(16, 32, 32)
F_batch = torch.fft.fft2(batch_imgs)  # (16,32,32) complex - batched automatically

# FFT frequencies
freqs = torch.fft.fftfreq(8, d=1.0)      # sample spacing d - like np.fft.fftfreq
rfftfreq = torch.fft.rfftfreq(8)

# Shift zero freq to center - for visualization
torch.fft.fftshift(F)
torch.fft.ifftshift(F)

# Power spectrum
power = X.abs()**2                # |FFT|^2

# Convolution via FFT - faster for large kernels
# conv = ifft(fft(a) * fft(b))

# Note: PyTorch FFT supports autograd! You can backprop through FFT - NumPy can't
x = torch.randn(8, requires_grad=True)
loss = torch.fft.fft(x).abs().sum()
loss.backward()  # works!
```

## 18. Einsum Masterclass (NumPy Section 21 Equivalent - Expanded)

```python
# Einsum is THE most powerful operation - generalizes sum, transpose, dot, matmul, outer, trace, etc.
# Einstein summation convention: repeated indices sum, free indices stay

a = torch.randn(3,4)
b = torch.randn(4,5)

# --- Basics ---
torch.einsum('ij->i', a)          # sum over j (axis 1) - row sums - like a.sum(dim=1)
torch.einsum('ij->j', a)          # sum over i - col sums
torch.einsum('ij->', a)           # sum all - like a.sum()
torch.einsum('ij->ji', a)         # transpose - like a.T
torch.einsum('ii->i', torch.randn(3,3))  # diag

# --- Vector ops ---
x = torch.randn(3)
y = torch.randn(3)
torch.einsum('i,i->', x, y)       # dot product - sum_i x_i*y_i
torch.einsum('i,j->ij', x, y)     # outer product - (3,3)

# --- Matrix ops ---
torch.einsum('ij,jk->ik', a, b)   # matmul - (3,4) @ (4,5) -> (3,5)
torch.einsum('ij,ij->ij', a, a)   # elementwise mul
torch.einsum('ij,ij->', a, a)     # Frobenius inner product / sum of squares

# --- Batched - where PyTorch einsum shines over NumPy! ---
batch_a = torch.randn(32, 3, 4)   # batch of matrices
batch_b = torch.randn(32, 4, 5)
torch.einsum('bij,bjk->bik', batch_a, batch_b)  # batched matmul - (32,3,5) - same as batch_a @ batch_b
# Even better - batch matmul with broadcasting batch dim:
torch.einsum('bij,jk->bik', batch_a, b)  # (32,3,4) @ (4,5) -> (32,3,5) - b broadcast

# --- Attention - the killer app ---
# Q: (B, H, N, D), K: (B, H, M, D) -> Attention scores: (B, H, N, M)
B,H,N,M,D = 2,4,16,16,32
Q = torch.randn(B,H,N,D)
K = torch.randn(B,H,M,D)
attn_scores = torch.einsum('b h n d, b h m d -> b h n m', Q, K)  # dot-product attention!

# --- Reductions & broadcasting in einsum ---
torch.einsum('b i j, b j k -> b i k', torch.randn(2,3,4), torch.randn(2,4,5))  # same as batched matmul

# Performance tip: einsum is readable but sometimes slower than dedicated ops (matmul). For hot paths, use matmul.
# But for complex contractions, einsum is unbeatable for clarity
```

## 19. Input / Output & Serialization (NumPy Section 18 Equivalent)

```python
# PyTorch IO is different from NumPy's np.save/load/txt - focused on tensors & models

# --- Tensor save/load (like np.save) ---
a = torch.randn(3,4)
torch.save(a, 'tensor.pt')                # save single tensor - pickle based
b = torch.load('tensor.pt', weights_only=True)  # load - weights_only for safety

# Save dict of tensors (like np.savez)
torch.save({'a': a, 'b': torch.randn(2)}, 'data.pt')
data = torch.load('data.pt', weights_only=True)
# data['a']

# --- Model checkpointing - THE important IO ---
model = torch.nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters())

checkpoint = {
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'epoch': 10,
    'loss': 0.123
}
torch.save(checkpoint, 'checkpoint.pt')
# Load:
ckpt = torch.load('checkpoint.pt', weights_only=False)  # False because contains optimizer state? Use safe handling
model.load_state_dict(ckpt['model_state'])

# --- Safe serialization: safetensors (recommended for HF) ---
# pip install safetensors
# from safetensors.torch import save_file, load_file
# save_file({'a': a}, 'model.safetensors')  # no pickle, safe
# load_file('model.safetensors')

# --- NumPy interop IO ---
import numpy as np
np.save('array.npy', a.numpy())           # via NumPy
np_arr = np.load('array.npy')
torch.from_numpy(np_arr)

# Text IO - no direct torch.savetxt, use NumPy or Python
# torch.Tensor doesn't have txt save - convert to numpy first:
np.savetxt('a.txt', a.numpy())
# Or:
torch.tensor(np.loadtxt('a.txt'))

# --- ONNX export - for deployment ---
# torch.onnx.export(model, torch.randn(1,10), 'model.onnx')

# Best practice: always use weights_only=True when loading tensors you didn't create (security)
```

## 28. Common Recipes & Patterns (NumPy Section 23 Equivalent - PyTorch Version)

```python
# ---- Normalize rows / columns ----
x = torch.randn(8,10)
# Zero mean unit variance per row
x_norm = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
# L2 normalize rows
x_l2 = x / (x.norm(dim=1, keepdim=True) + 1e-8)
# Softmax - stable (don't do exp(x)/sum(exp(x)))
x_softmax = torch.softmax(x, dim=1)  # stable internally

# ---- One-hot encoding ----
labels = torch.tensor([0,2,1,0])
num_classes = 3
one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes)  # (4,3) - int64
# [[1,0,0],[0,0,1],[0,1,0],[1,0,0]]
one_hot_float = one_hot.float()

# ---- Masking & Where ----
x = torch.randn(3,4)
# Set negatives to 0 (ReLU)
x_relu = torch.where(x > 0, x, torch.zeros_like(x))
x_relu2 = x.clamp(min=0)  # faster
x_relu3 = torch.relu(x)

# Attention mask: fill -inf where mask False
mask = torch.tensor([True, False, True])
scores = torch.randn(3)
scores_masked = scores.masked_fill(~mask, float('-inf'))

# ---- Gather / Scatter - advanced indexing ----
# Gather: pick elements along dim using index tensor
src = torch.tensor([[1,2,3],[4,5,6]])
idx = torch.tensor([[0,1],[1,2]])  # indices along dim=1
src.gather(dim=1, index=idx)  # [[1,2],[5,6]]

# Scatter: inverse of gather
zeros = torch.zeros(2,3)
zeros.scatter_(dim=1, index=idx, src=torch.tensor([[9,9],[9,9]]))  # fill 9 at idx positions

# ---- Broadcasting tricks ----
# Add vector to matrix rows: (3,4) + (4,) -> (3,4) via broadcast
# Add column vector: (3,4) + (3,1) -> (3,4)
# Outer product via broadcasting: (3,1) * (1,4) -> (3,4) - but use einsum or outer

# ---- Cumulative & rolling window (unfold) ----
x = torch.arange(10)
x.unfold(dimension=0, size=3, step=1)  # sliding window size 3 -> (8,3) like sliding_window_view
# [[0,1,2],[1,2,3],...]

# ---- Meshgrid ----
xs = torch.arange(3)
ys = torch.arange(4)
grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')  # (3,4) each

# ---- Gradient-free inference pattern ----
# Already covered in Phase 6 but recipe:
# with torch.no_grad():
#     y = model(x)
# or with torch.inference_mode():

# ---- Mixed precision training recipe ----
# scaler = torch.amp.GradScaler('cuda')
# with torch.amp.autocast('cuda'):
#     y = model(x)
#     loss = criterion(y, target)
# scaler.scale(loss).backward()
# scaler.step(optimizer)
# scaler.update()

# ---- Per-sample grads with vmap (Phase 14) ----
# def loss_fn(params, x, y): ...
# per_sample_grads = torch.func.vmap(torch.func.grad(loss_fn), in_dims=(None,0,0))(params, xb, yb)
```

## 29. Output Visuals - What Each PyTorch Command Returns (NumPy Section 25 Equivalent)

```python
import torch
torch.manual_seed(0)

torch.tensor([[1,2,3],[4,5,6]])
# tensor([[1, 2, 3],
#         [4, 5, 6]]) -> shape (2,3), dtype int64

torch.zeros((2,3))
# tensor([[0., 0., 0.],
#         [0., 0., 0.]]) -> float32 default for zeros

torch.arange(0,10,2)
# tensor([0, 2, 4, 6, 8]) -> int64

torch.linspace(0,1,5)
# tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

a = torch.tensor([[1,2,3],[4,5,6]])
a.shape
# torch.Size([2, 3])
a.stride()
# (3, 1) -> elements, not bytes! vs NumPy (24,8) bytes
a.sum(dim=0)
# tensor([5, 7, 9]) -> sum columns -> [1+4,2+5,3+6]
a.sum(dim=1)
# tensor([ 6, 15]) -> sum rows

a = torch.tensor([1,2,3,4,5])
a[a>3]
# tensor([4, 5]) -> Boolean indexing returns only True elements (copy)

a = torch.tensor([[1,2,3],[4,5,6]])
a[0,1]
# tensor(2) -> 0-d tensor, not Python int! Use .item() to get int
a[:,1]
# tensor([2, 5]) -> Second column

a.reshape(3,2)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]]) -> Same data, new shape (view if contiguous)

torch.cat((torch.tensor([1,2]), torch.tensor([3,4])))
# tensor([1, 2, 3, 4]) -> cat dim=0 default
torch.stack((torch.tensor([1,2]), torch.tensor([3,4])))
# tensor([[1, 2],
#         [3, 4]]) -> Stack adds new dim

a = torch.tensor([1.,2.,3.])
b = torch.tensor([10.,20.,30.])
a + b
# tensor([11., 22., 33.]) -> Elementwise

torch.mean(torch.tensor([1.,2.,3.]))
# tensor(2.) -> 0-d tensor

torch.sort(torch.tensor([3,1,2]))
# torch.return_types.sort(values=tensor([1, 2, 3]), indices=tensor([1, 2, 0])) -> TUPLE! Not just array like NumPy

torch.unique(torch.tensor([1,1,2,2,3]))
# tensor([1, 2, 3])

torch.where(torch.tensor([True, False, True]))
# (tensor([0, 2]),) -> Tuple with indices where True - same as NumPy but returns tuple of tensors

torch.dot(torch.tensor([1.,2.,3.]), torch.tensor([4.,5.,6.]))
# tensor(32.) -> dot product 1*4+2*5+3*6

A = torch.tensor([[1.,2.],[3.,4.]])
torch.linalg.inv(A)
# tensor([[-2.0000,  1.0000],
#         [ 1.5000, -0.5000]]) -> Inverse

torch.fft.fft(torch.tensor([1.,2.,3.,4.]))
# tensor([10.+0.j, -2.+2.j, -2.+0.j, -2.-2.j]) -> Complex, same as NumPy

a = torch.tensor([1.,2.,3.], requires_grad=True)
b = a * 2
b
# tensor([2., 4., 6.], grad_fn=<MulBackward0>) -> Shows grad_fn! Non-leaf
a
# tensor([1., 2., 3.], requires_grad=True) -> Leaf, no grad_fn
```

## 30. Pitfalls & Gotchas - 25 Critical Lessons (NumPy Section 24 Equivalent - PyTorch Edition)

```python
# 1. In-place ops break autograd versioning - THE #1 PyTorch bug
a = torch.randn(3,4, requires_grad=True)
b = a * 2
# a.add_(1)  # RuntimeError in backward! a needed for grad but modified
# Fix: a = a + 1  (out-of-place)

# 2. grad accumulates - must zero_grad()
# for _ in range(2):
#     loss = model(x).sum()
#     loss.backward()  # grad adds up! Second backward grad is 2x
# Fix: optimizer.zero_grad() before backward

# 3. .item() vs 0-d tensor
a = torch.tensor([5])  # shape (1,)
b = torch.tensor(5)    # shape () 0-d
a + 1  # tensor([6]) not 6
b.item()  # 5 as Python int
# type(torch.sum(...)) is Tensor, not int

# 4. Boolean logic: use & | not and/or - SAME AS NUMPY
a = torch.tensor([1,2,3,4])
# a > 2 and a < 4  # RuntimeError: bool value of Tensor with more than one value is ambiguous
(a > 2) & (a < 4)  # Correct
# Must have parens because & has higher precedence than >

# 5. view() vs reshape() - view fails on non-contiguous, reshape copies
a = torch.arange(6).reshape(2,3).T  # T makes non-contiguous
# a.view(6)  # RuntimeError: view size not compatible
a.reshape(6)  # Works - copies
# Best practice: Use view() when you WANT to assert contiguity (fail fast)

# 6. contiguous() is a copy when needed - don't call blindly in hot loop
a = torch.randn(3,4).T  # non-contiguous
b = a.contiguous()  # COPY! Allocates new memory
# Calling .contiguous() every iteration = unexpected allocation

# 7. Broadcasting can hide bugs - same as NumPy
a = torch.ones(3,3)
b = torch.ones(3)  # You thought (3,3) but it's (3,)
a + b  # Works via broadcast (3,3) + (3,) -> (3,3), but maybe you wanted error?
# Use explicit: b.view(1,3) or keepdim

# 8. Device mismatch
# x_cpu + x_gpu -> RuntimeError: Expected all tensors on same device
# Fix: x_cpu.to(x_gpu.device)

# 9. dtype int64 default for torch.tensor([1,2,3]) - not int32! Can cause type mismatch
# torch.tensor([1,2,3]).dtype -> int64
# Model weights are float32 - adding int64 to float32 may error or promote
# Fix: torch.tensor([1,2,3], dtype=torch.float32) or use .float()

# 10. Var unbiased=True by default - differs from NumPy!
torch.tensor([1.,2.,3.]).var()  # divides by N-1 = 1.0
# NumPy np.var([1,2,3]) -> 0.666...
# To match NumPy: torch.var(unbiased=False)

# 11. sort returns tuple (values, indices) - not just sorted array!
torch.sort(torch.tensor([3,1,2]))  # (values, indices)
# NumPy np.sort returns only values

# 12. if tensor: crashes - same as NumPy
a = torch.tensor([1,2,3])
# if a: ...  # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
if a.numel() > 0: ...  # Correct
if (a > 2).any(): ...

# 13. arange with float step unreliable - same as NumPy
torch.arange(0, 1, 0.1)  # length unpredictable due to float error
# USE: torch.linspace(0,1,11)

# 14. Division by zero doesn't crash - inf + warning, like NumPy
torch.tensor([1.,2.]) / 0.  # tensor([inf, inf]) + no error

# 15. torch.from_numpy shares memory! Modifying one modifies other
import numpy as np
np_arr = np.array([1,2,3])
torch_arr = torch.from_numpy(np_arr)
torch_arr[0] = 99  # np_arr[0] also 99!
# Fix: torch.from_numpy(np_arr).clone() if you want copy

# 16. .numpy() fails if requires_grad or on GPU
a = torch.randn(3, requires_grad=True)
# a.numpy() -> RuntimeError: Can't call numpy() on Tensor that requires grad. Use a.detach().numpy()
a.detach().numpy()  # OK
b = torch.randn(3, device='cuda') if torch.cuda.is_available() else None
# b.numpy() -> RuntimeError: can't convert cuda tensor to numpy. Use .cpu() first
# b.cpu().numpy()

# 17. Don't use inv() to solve Ax=b - same as NumPy pitfall!
# BAD: x = torch.linalg.inv(A) @ b - slow + unstable
# GOOD: x = torch.linalg.solve(A, b)

# 18. Random: old global seed vs Generator - new API preferred
torch.manual_seed(0)  # Global - affects all - legacy style
gen = torch.Generator().manual_seed(0)  # Local - NEW preferred, like np.random.default_rng
torch.randn(3, generator=gen)

# 19. Expand creates stride-0 view - modifying expanded tensor?
a = torch.tensor([1,2,3])
b = a.expand(3,3)  # (3,3) with stride (0,1) - zero-copy
# b[0,0] = 99  # Would modify all rows! Actually b is non-writable? In PyTorch, expand returns non-writable? Actually it is writable but broadcasts via stride 0 trick - all rows share!
# Check: b[0,0] = 99 would set first element of all rows to 99 if writable. Use clone() after expand if you need writable.

# 20. Squeeze removes ALL size-1 dims by default! Can hide bugs
a = torch.randn(1,3,1,4)
a.squeeze()  # shape (3,4) - removed dim 0 and 2!
a.squeeze(0)  # only remove dim 0 -> (3,1,4) - safer
# Use squeeze(dim) to be explicit

# 21. In-place ops ending with _ : add_, mul_, etc. - use carefully
# PyTorch convention: _ suffix means in-place - NumPy doesn't have this
# a.add_(1) modifies a

# 22. torch.cat vs torch.stack - confusion
# cat: concatenates along existing dim - (2,3) + (2,3) -> (4,3) if dim=0
# stack: creates new dim - (2,3) + (2,3) -> (2,2,3) if dim=0

# 23. Memory format matters for speed - channels_last
# NCHW vs NHWC - 20-30% faster on Ampere GPUs for conv
# model = model.to(memory_format=torch.channels_last)

# 24. torch.compile first call slow - compilation overhead
# model = torch.compile(model)
# First iteration compiles (slow), next fast

# 25. NumPy 2.0 breaking changes vs PyTorch 2.0 - don't confuse!
# PyTorch 2.0 introduced torch.compile, torch.func, torch.amp new API
# Old: torch.cuda.amp.autocast, torch.cuda.amp.GradScaler
# New: torch.amp.autocast('cuda'), torch.amp.GradScaler('cuda')
```

---

