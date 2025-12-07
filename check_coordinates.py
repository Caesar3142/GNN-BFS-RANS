"""
Diagnostic script to verify coordinate mapping from OpenFOAM.
"""

import numpy as np
import matplotlib.pyplot as plt
from openfoam_loader import OpenFOAMLoader

def main():
    loader = OpenFOAMLoader('OpenFOAM-data')
    mesh = loader.load_mesh()
    
    # Get internal cells only
    n_internal = 12225
    cell_centers = mesh['cell_centers'][:n_internal]
    
    print("=" * 60)
    print("Coordinate System Verification")
    print("=" * 60)
    print(f"\nTotal internal cells: {n_internal}")
    print(f"\nCell center coordinate ranges:")
    print(f"  X: {cell_centers[:, 0].min():.6f} to {cell_centers[:, 0].max():.6f}")
    print(f"  Y: {cell_centers[:, 1].min():.6f} to {cell_centers[:, 1].max():.6f}")
    print(f"  Z: {cell_centers[:, 2].min():.6f} to {cell_centers[:, 2].max():.6f}")
    
    print(f"\nExpected from blockMeshDict (after scale 0.001):")
    print(f"  X: -0.0206 to 0.29")
    print(f"  Y: -0.0254 to 0.0254")
    print(f"  Z: -0.0005 to 0.0005 (2D case)")
    
    print(f"\nGeometry check (BFS - Backward Facing Step):")
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    print(f"  Cells with X < 0 (inlet/step region): {np.sum(x < 0)}")
    print(f"  Cells with X = 0 (step location): {np.sum(np.abs(x) < 0.001)}")
    print(f"  Cells with X > 0 (downstream): {np.sum(x > 0)}")
    print(f"  Cells with Y < 0 (lower channel): {np.sum(y < 0)}")
    print(f"  Cells with Y > 0 (upper channel): {np.sum(y > 0)}")
    
    # Create a simple geometry plot
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(x, y, c='blue', s=0.5, alpha=0.5)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Step (X=0)')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('X [m] (Streamwise direction)', fontsize=12)
    ax.set_ylabel('Y [m] (Vertical direction)', fontsize=12)
    ax.set_title('BFS Mesh Geometry - Cell Centers', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(x.min() - 0.01, x.max() + 0.01)
    ax.set_ylim(y.min() - 0.01, y.max() + 0.01)
    
    plt.tight_layout()
    plt.savefig('mesh_geometry_check.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved geometry plot to: mesh_geometry_check.png")
    print("\nThis plot shows the cell center locations.")
    print("For BFS geometry, you should see:")
    print("  - Inlet region (X < 0) with step at X=0")
    print("  - Downstream expansion (X > 0)")
    print("  - Channel height in Y direction")
    
    # Check sample coordinates
    print(f"\nSample cell centers:")
    print(f"  First cell:  ({cell_centers[0][0]:.6f}, {cell_centers[0][1]:.6f}, {cell_centers[0][2]:.6f})")
    print(f"  Middle cell: ({cell_centers[n_internal//2][0]:.6f}, {cell_centers[n_internal//2][1]:.6f}, {cell_centers[n_internal//2][2]:.6f})")
    print(f"  Last cell:   ({cell_centers[-1][0]:.6f}, {cell_centers[-1][1]:.6f}, {cell_centers[-1][2]:.6f})")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()

