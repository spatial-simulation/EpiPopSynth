
# Optimization Reformulation in EpiPopSynth v2

This document describes the reformulation and acceleration of the optimization module introduced in EpiPopSynth v2. The objective is to improve computational efficiency and numerical stability while preserving the original modeling assumptions and feasibility constraints.

---

## Background and motivation

In the original implementation, population synthesis was formulated as a residual minimization problem solved using iterative nonlinear least-squares routines. Although flexible, this approach incurred substantial computational overhead for large-scale spatial instances and introduced unnecessary iterative complexity for a problem that is linear in the decision variables.

In v2, we reformulate the objective as a single constrained linear least-squares problem by constructing an augmented system that explicitly incorporates regularization terms. This reformulation enables the use of efficient linear solvers and sparse matrix operations, substantially improving scalability.

---

## Original optimization formulation

Let \( x \in \mathbb{R}^n \) denote the vector of decision variables representing the weights of population motifs or synthetic micro-units.

The original objective minimizes the weighted residuals between synthesized aggregates and observed marginal constraints:

\[
\min_x
\;\;
\| W_a^T x - y_a \|_2^2

+ \| W_g^T x - y_g \|_2^2
+ \| W_s^T x - y_s \|_2^2
+ \lambda \| x - x_{\text{init}} \|_2^2
\]

where:

- \( W_a, W_g, W_s \) are constraint mapping matrices associated with age, geographic, and socioeconomic marginals, respectively,
- \( y_a, y_g, y_s \) are corresponding observed marginal vectors,
- \( x_{\text{init}} \) is the initialization vector,
- \( \lambda \) controls the strength of the regularization term.

The last term stabilizes the solution by penalizing deviations from the initialization.

---

## Augmented least-squares reformulation

The objective above is quadratic and linear in \( x \). It can be rewritten as a standard least-squares problem by constructing an augmented system:

\[
A =
\begin{bmatrix}
W_a^T \\
W_g^T \\
W_s^T \\
\sqrt{\lambda} I
\end{bmatrix},
\quad
b =
\begin{bmatrix}
y_a \\
y_g \\
y_s \\
\sqrt{\lambda} x_{\text{init}}
\end{bmatrix}
\]

The optimization problem becomes:

\[
\min_x \| A x - b \|_2^2
\]

This formulation is mathematically equivalent to the original objective and eliminates the need for iterative nonlinear solvers. It allows the use of direct or constrained linear least-squares routines optimized for large-scale numerical computation.

---

## Constraint handling

The feasibility constraints in the original implementation are preserved in v2.

Specifically, we enforce:

\[
x \ge 0
\]

and

\[
x \le \alpha \cdot \max(x_{\text{init}})
\]

where \( \alpha \) is a user-defined scaling factor controlling the upper bound.

These constraints define the same feasible region as in the original model and ensure physical interpretability of the synthesized population weights.

In practice, the constrained least-squares problem is solved using:

```python
scipy.optimize.lsq_linear(A, b, bounds=(lower, upper))
```

which directly supports bound-constrained linear least squares.

---

## Sparse matrix construction

The augmented system matrix \( A \) exhibits a high degree of sparsity due to two structural properties:

1. The marginal mapping matrices \( W_a, W_g, W_s \) are typically one-hot or block-sparse by construction.
2. The regularization term introduces a diagonal identity matrix.

To exploit this structure, all matrix components are represented in compressed sparse row (CSR) format:

- \( W_a^T, W_g^T, W_s^T \) are converted to sparse matrices,
- the regularization block is constructed using `scipy.sparse.eye`,
- the augmented matrix is assembled using vertical sparse stacking.

This sparse representation significantly reduces memory footprint and improves solver performance for large problem instances.

---

## Computational impact

Compared to the original iterative solver, the v2 formulation provides several practical advantages:

- elimination of iterative convergence loops,
- improved numerical stability for large-scale systems,
- reduced memory usage through sparse matrix representation,
- faster execution for high-dimensional population synthesis tasks.

These improvements enable the framework to scale more effectively to large urban regions and fine-grained spatial resolutions.

---

## Design implications

The reformulated optimization module preserves the conceptual modeling assumptions of the original framework while separating numerical solution strategy from problem specification. This design choice improves maintainability and facilitates future extensions, such as alternative regularization schemes or solver backends.
