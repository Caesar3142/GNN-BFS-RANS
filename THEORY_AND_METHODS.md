# Theory and Methods

This document describes the theoretical background and methodological approaches used in this GNN-based RANS flow field surrogate simulator.

## Table of Contents

1. [Reynolds-Averaged Navier-Stokes (RANS) Equations](#reynolds-averaged-navier-stokes-rans-equations)
2. [Graph Neural Networks (GNNs)](#graph-neural-networks-gnns)
3. [Graph Construction from Unstructured Meshes](#graph-construction-from-unstructured-meshes)
4. [Message Passing in GNNs](#message-passing-in-gnns)
5. [GNN Architectures Used](#gnn-architectures-used)
6. [Training Methodology](#training-methodology)
7. [Normalization Strategies](#normalization-strategies)
8. [Loss Functions](#loss-functions)
9. [Surrogate Modeling Approach](#surrogate-modeling-approach)

---

## Reynolds-Averaged Navier-Stokes (RANS) Equations

### Background

The Navier-Stokes equations describe fluid motion, but solving them directly for turbulent flows is computationally expensive. RANS equations provide a time-averaged approach that models turbulence through additional terms.

### Governing Equations

The RANS equations consist of:

1. **Continuity Equation** (mass conservation):
   ```
   ∂(ρŪᵢ)/∂xᵢ = 0
   ```
   where ρ is density and Ūᵢ is the mean velocity component.

2. **Momentum Equation**:
   ```
   ∂(ρŪᵢ)/∂t + ∂(ρŪᵢŪⱼ)/∂xⱼ = -∂p̄/∂xᵢ + ∂/∂xⱼ[μ(∂Ūᵢ/∂xⱼ + ∂Ūⱼ/∂xᵢ)] - ∂(ρu'ᵢu'ⱼ)/∂xⱼ
   ```
   where p̄ is mean pressure, μ is dynamic viscosity, and -ρu'ᵢu'ⱼ is the Reynolds stress tensor.

3. **Turbulence Modeling** (k-ε model):
   - **Turbulent kinetic energy (k)**:
     ```
     ∂(ρk)/∂t + ∂(ρkŪᵢ)/∂xᵢ = Pₖ - ρε + ∂/∂xᵢ[(μ + μₜ/σₖ)∂k/∂xᵢ]
     ```
   - **Dissipation rate (ε)**:
     ```
     ∂(ρε)/∂t + ∂(ρεŪᵢ)/∂xᵢ = C₁ε(ε/k)Pₖ - C₂ερ(ε²/k) + ∂/∂xᵢ[(μ + μₜ/σₑ)∂ε/∂xᵢ]
     ```
   - **Turbulent viscosity (μₜ = ρνₜ)**:
     ```
     μₜ = ρCμ(k²/ε)
     ```

### What This Code Predicts

The GNN model learns to predict the following flow field quantities:
- **U** = [Ux, Uy, Uz]: Mean velocity vector (3 components)
- **p**: Mean pressure
- **k**: Turbulent kinetic energy
- **ε (epsilon)**: Turbulent dissipation rate
- **νₜ (nut)**: Turbulent kinematic viscosity

These are the key variables needed to fully characterize a RANS solution.

---

## Graph Neural Networks (GNNs)

### Overview

Graph Neural Networks are a class of neural networks designed to operate on graph-structured data. Unlike traditional neural networks that process fixed-size vectors or grids, GNNs can handle variable-sized, irregular data structures.

### Why GNNs for CFD?

1. **Unstructured Meshes**: CFD meshes are typically unstructured (irregular cell shapes and connectivity)
2. **Spatial Relationships**: Flow physics depend on local neighborhood interactions
3. **Variable Resolution**: Meshes can have varying cell sizes
4. **Topology Preservation**: GNNs naturally preserve mesh connectivity

### Graph Representation

A graph G = (V, E) consists of:
- **Nodes (V)**: Mesh cells (represented by cell centers)
- **Edges (E)**: Connections between adjacent cells (via shared faces)
- **Node Features**: Cell properties (coordinates, flow variables)
- **Edge Features**: Geometric relationships (distance, direction vector)

---

## Graph Construction from Unstructured Meshes

### Method

The graph is constructed from OpenFOAM mesh connectivity data:

1. **Nodes**: Each mesh cell becomes a graph node
   - Node features: Cell center coordinates [x, y, z]

2. **Edges**: Created from face connectivity
   - **Internal faces**: Connect owner and neighbour cells
   - **Bidirectional edges**: Each face creates two edges (owner→neighbour, neighbour→owner)
   - **Boundary faces**: Create self-loops for boundary cells (or can be excluded)

3. **Edge Attributes**: Computed geometric features
   - Direction vector: Normalized vector from source to target cell center
   - Distance: Euclidean distance between cell centers
   - Format: [dx, dy, dz, distance] (4D vector)

### Implementation Details

```python
# Pseudo-code for edge construction
for each internal_face:
    owner_cell = face.owner
    neighbour_cell = face.neighbour
    
    # Create bidirectional edges
    edges.append([owner_cell, neighbour_cell])
    edges.append([neighbour_cell, owner_cell])
    
    # Compute edge attributes
    direction = cell_centers[neighbour] - cell_centers[owner]
    distance = norm(direction)
    edge_attr = [direction/distance, distance]
```

### Robustness Features

- **Index validation**: Ensures all edge indices are within valid node range
- **Isolated node handling**: Automatically adds self-loops for nodes with no connections
- **Boundary handling**: Properly filters internal vs boundary cells when needed

---

## Message Passing in GNNs

### Concept

Message passing is the core operation in GNNs. Each node aggregates information from its neighbors to update its own representation.

### Mathematical Formulation

For a node i with neighbors N(i), the message passing operation is:

```
hᵢ^(l+1) = UPDATE(hᵢ^(l), AGGREGATE({MESSAGE(hⱼ^(l), eⱼᵢ) : j ∈ N(i)}))
```

where:
- `hᵢ^(l)`: Node i's features at layer l
- `eⱼᵢ`: Edge features between nodes j and i
- `MESSAGE`: Function that creates messages from neighbors
- `AGGREGATE`: Function that combines messages (sum, mean, max, etc.)
- `UPDATE`: Function that updates node features

### Implementation in This Code

The code uses PyTorch Geometric's message passing framework:

```python
# Example: GCN layer
x_new = GCNConv(hidden_dim, hidden_dim)(x, edge_index)

# Example: TransformerConv with edge attributes
x_new = TransformerConv(hidden_dim, hidden_dim)(x, edge_index, edge_attr=edge_attr)
```

### Robustness

- **Edge validation**: Checks edge indices before message passing
- **Empty graph handling**: Adds self-loops if no edges exist
- **Error handling**: Provides informative error messages for debugging

---

## GNN Architectures Used

### 1. Graph Convolutional Network (GCN)

**Formula**:
```
H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
```

where:
- A: Adjacency matrix
- D: Degree matrix
- H^(l): Node features at layer l
- W^(l): Learnable weight matrix
- σ: Activation function

**Characteristics**:
- Simple and efficient
- Good baseline for many tasks
- Uses normalized adjacency matrix

### 2. Graph Attention Network (GAT)

**Formula**:
```
hᵢ^(l+1) = σ(Σⱼ∈N(i) αᵢⱼ W^(l) hⱼ^(l))
```

where attention coefficients are:
```
αᵢⱼ = softmax(LeakyReLU(a^T [W hᵢ || W hⱼ]))
```

**Characteristics**:
- Learns attention weights for neighbors
- Can focus on important connections
- More expressive than GCN

### 3. Graph Isomorphism Network (GIN)

**Formula**:
```
hᵢ^(l+1) = MLP^((l+1))((1 + ε^(l+1)) · hᵢ^(l) + Σⱼ∈N(i) hⱼ^(l))
```

**Characteristics**:
- Theoretically most expressive
- Uses multi-layer perceptron (MLP) for updates
- Good for learning complex patterns

### 4. Transformer Convolution

**Formula**:
Similar to GAT but uses multi-head attention mechanism:
```
hᵢ^(l+1) = ||ₖ₌₁^K σ(Σⱼ∈N(i) αᵢⱼ^(k) W^(k) hⱼ^(l))
```

**Characteristics**:
- Can use edge attributes
- Multi-head attention for richer representations
- Good for complex relationships

### Architecture in This Code

The model uses:
- **Input projection**: Maps 3D coordinates → hidden dimension
- **Multiple GNN layers**: Stack of chosen GNN type (GCN/GAT/GIN/Transformer)
- **Residual connections**: x = x + x_new (helps with deep networks)
- **Batch normalization**: Stabilizes training
- **Output projection**: Maps hidden features → 7 outputs (U(3) + p(1) + k(1) + ε(1) + νₜ(1))

---

## Training Methodology

### Objective

Learn a mapping from mesh geometry (cell centers) to flow fields:
```
f: (x, y, z) → (U, p, k, ε, νₜ)
```

### Training Strategy

#### 1. Data Preparation
- Load multiple time snapshots from OpenFOAM simulation
- Normalize fields independently (see Normalization section)
- Construct graphs for each snapshot
- Use cell centers as input features

#### 2. Loss Function

**Field-wise Weighted MSE**:
```
L = w_U · L_U + w_p · L_p + w_k · L_k + w_ε · L_ε + w_νₜ · L_νₜ
```

where each field loss is:
```
L_field = mean((pred_field - target_field)²)
```

**Default weights**:
- w_U = 1.0 (velocity)
- w_p = 3.0 (pressure - critical for stability)
- w_k = 0.5 (turbulence)
- w_ε = 0.5
- w_νₜ = 0.5

**Pressure reference constraint** (prevents global drift):
```
L_pref = (mean(p_pred) - mean(p_ref))²
L_p_total = L_p + λ_pref · L_pref
```

#### 3. Curriculum Training (Optional)

**Phase 1** (epochs 1 to N):
- Train velocity and turbulence fields
- Freeze pressure output (zero gradients)
- Prevents early pressure oscillations

**Phase 2** (epochs N+1 to end):
- Unfreeze pressure
- Reduce learning rate by 50%
- Fine-tune all fields together

#### 4. Optimization

- **Optimizer**: Adam
- **Learning rate**: 3e-4 (optimized for CFD-GNNs)
- **Scheduler**: ReduceLROnPlateau (reduces LR when validation loss plateaus)
- **Regularization**:
  - Gradient clipping (max_norm=1.0)
  - Dropout (0.1)
  - Batch normalization

---

## Normalization Strategies

### Why Normalize?

Different flow fields have vastly different scales:
- Velocity: ~1-10 m/s
- Pressure: ~0-1000 Pa (or m²/s²)
- k: ~0.01-1 m²/s²
- ε: ~0.1-100 m²/s³
- νₜ: ~1e-6 to 1e-4 m²/s

Without normalization, the loss would be dominated by fields with larger magnitudes.

### Per-Component Velocity Normalization

**Problem**: Velocity components (Ux, Uy, Uz) may have different scales.

**Solution**: Normalize each component separately:
```
Ux_norm = (Ux - mean(Ux)) / std(Ux)
Uy_norm = (Uy - mean(Uy)) / std(Uy)
Uz_norm = (Uz - mean(Uz)) / std(Uz)
```

This ensures each component is normalized independently, improving training stability.

### Scalar Field Normalization

For scalar fields (p, k, ε, νₜ):
```
field_norm = (field - mean(field)) / std(field)
```

### Denormalization

During inference, predictions are denormalized:
```
field_pred = field_norm_pred · std(field) + mean(field)
```

---

## Loss Functions

### Field-wise Loss Computation

Instead of element-wise weighting, compute loss per field:

```python
# For each field
u_loss = mean((U_pred - U_target)²)
p_loss = mean((p_pred - p_target)²)
# ... etc

# Weighted combination
total_loss = w_U * u_loss + w_p * p_loss + w_k * k_loss + ...
```

**Advantages**:
- Balanced training across all fields
- Works well with independent field normalization
- Prevents one field from dominating

### Pressure Reference Constraint

Pressure is defined up to an arbitrary constant. To prevent global drift:

```python
p_mean_pred = mean(p_pred)
p_mean_target = mean(p_target)
p_ref_loss = (p_mean_pred - p_mean_target)²
p_total_loss = p_mse_loss + λ * p_ref_loss
```

This anchors the absolute pressure level.

---

## Surrogate Modeling Approach

### Problem Formulation

Given:
- Mesh geometry (cell centers, connectivity)
- Boundary conditions (implicit in geometry)

Predict:
- Steady-state RANS flow fields

### Key Assumptions

1. **Geometry-based prediction**: Flow fields are determined by geometry
2. **Steady-state**: Predicting converged solutions, not time evolution
3. **Single mesh**: Model trained on one geometry (can be extended)
4. **No explicit BCs**: Boundary conditions are learned from geometry

### Advantages of GNN Approach

1. **Mesh-agnostic**: Can handle unstructured meshes of any topology
2. **Local interactions**: Message passing naturally models local flow physics
3. **Efficient**: Single forward pass vs iterative CFD solver
4. **Differentiable**: Can be integrated into optimization loops

### Limitations

1. **Single geometry**: Current implementation for one mesh topology
2. **No time evolution**: Only steady-state predictions
3. **No explicit BCs**: Boundary conditions not explicitly encoded
4. **Training data**: Requires converged CFD solutions for training

---

## Mathematical Formulation Summary

### Forward Pass

For a graph G = (V, E) with node features X:

1. **Input projection**:
   ```
   H⁽⁰⁾ = X · W_in
   ```

2. **GNN layers** (for l = 1 to L):
   ```
   H⁽ˡ⁺¹⁾ = GNN_layer(H⁽ˡ⁾, E, A)
   H⁽ˡ⁺¹⁾ = H⁽ˡ⁾ + H⁽ˡ⁺¹⁾  (residual)
   H⁽ˡ⁺¹⁾ = BatchNorm(H⁽ˡ⁺¹⁾)
   H⁽ˡ⁺¹⁾ = ReLU(H⁽ˡ⁺¹⁾)
   H⁽ˡ⁺¹⁾ = Dropout(H⁽ˡ⁺¹⁾)
   ```

3. **Output projection**:
   ```
   Y = MLP(H⁽ᴸ⁾)
   ```
   where Y = [Ux, Uy, Uz, p, k, ε, νₜ]

### Loss Computation

```
L_total = Σ_field w_field · L_field + λ_pref · L_pref
```

where:
- `L_field = mean((Y_field - Y_target_field)²)`
- `L_pref = (mean(p_pred) - mean(p_target))²`

---

## References and Further Reading

### RANS Equations
- Pope, S. B. (2000). *Turbulent Flows*. Cambridge University Press.
- Wilcox, D. C. (2006). *Turbulence Modeling for CFD*. DCW Industries.

### Graph Neural Networks
- Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.
- Veličković, P., et al. (2018). Graph Attention Networks. *ICLR*.
- Xu, K., et al. (2019). How Powerful are Graph Neural Networks? *ICLR*.

### GNNs for CFD
- Pfaff, T., et al. (2021). Learning Mesh-Based Simulation with Graph Networks. *ICLR*.
- Lino, M., et al. (2021). Deep Convolutional Models for Learning Flow Field Representations. *JCP*.

### Message Passing
- Gilmer, J., et al. (2017). Neural Message Passing for Quantum Chemistry. *ICML*.

---

## Implementation Notes

### Key Design Decisions

1. **Bidirectional edges**: Ensures information flows both ways between cells
2. **Edge attributes**: Captures geometric relationships (distance, direction)
3. **Residual connections**: Helps with training deep networks
4. **Field-wise loss**: Ensures balanced learning across all fields
5. **Per-component normalization**: Critical for velocity accuracy
6. **Pressure weight (3.0)**: Prevents pressure under-training
7. **Curriculum training**: Stabilizes pressure learning

### Computational Complexity

- **Graph construction**: O(F) where F is number of faces
- **Message passing**: O(E · d) where E is edges, d is feature dimension
- **Forward pass**: O(L · E · d²) for L layers
- **Memory**: O(N · d) for N nodes with d-dimensional features

### Scalability

- Current implementation handles meshes with ~10k-100k cells efficiently
- Can be extended to larger meshes with:
  - Graph sampling/batching
  - Hierarchical graph structures
  - Multi-GPU training

---

## Future Extensions

1. **Multi-geometry training**: Learn across different mesh topologies
2. **Boundary condition encoding**: Explicitly encode BCs as node/edge features
3. **Time-dependent predictions**: Extend to unsteady flows
4. **Uncertainty quantification**: Add probabilistic outputs
5. **Transfer learning**: Pre-train on one geometry, fine-tune on another
6. **Physics-informed loss**: Add PDE residual terms to loss function
