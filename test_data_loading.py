"""
Test script to verify OpenFOAM data loading and graph construction.
Run this to check if your data can be loaded correctly.
"""

import sys
from pathlib import Path

try:
    from openfoam_loader import OpenFOAMLoader
    from graph_constructor import GraphConstructor
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install requirements: pip3 install -r requirements.txt")
    sys.exit(1)

def test_data_loading(case_path='OpenFOAM-data'):
    """Test loading OpenFOAM data."""
    print(f"\nTesting data loading from: {case_path}")
    print("=" * 60)
    
    # Check if case path exists
    if not Path(case_path).exists():
        print(f"✗ Case path does not exist: {case_path}")
        return False
    
    try:
        # Load mesh
        print("\n1. Loading mesh...")
        loader = OpenFOAMLoader(case_path)
        mesh_data = loader.load_mesh()
        
        print(f"   ✓ Points: {len(mesh_data['points'])}")
        print(f"   ✓ Cells: {mesh_data['n_cells']}")
        print(f"   ✓ Faces: {len(mesh_data['owner'])}")
        print(f"   ✓ Internal faces: {len(mesh_data['neighbour'])}")
        print(f"   ✓ Boundaries: {list(mesh_data['boundaries'].keys())}")
        
        # Load fields
        print("\n2. Loading fields...")
        time_dirs = ['0', '100', '200', '282']
        available_fields = {}
        
        for time_dir in time_dirs:
            time_path = Path(case_path) / time_dir
            if time_path.exists():
                try:
                    fields = loader.load_fields(time_dir)
                    available_fields[time_dir] = list(fields.keys())
                    print(f"   ✓ Time {time_dir}: {list(fields.keys())}")
                    for field_name, field_data in fields.items():
                        print(f"      - {field_name}: shape {field_data.shape}")
                except Exception as e:
                    print(f"   ✗ Time {time_dir}: {e}")
        
        # Construct graph
        print("\n3. Constructing graph...")
        graph_constructor = GraphConstructor(mesh_data)
        graph = graph_constructor.build_graph(node_features=mesh_data['cell_centers'])
        
        print(f"   ✓ Nodes: {graph.num_nodes}")
        print(f"   ✓ Edges: {graph.edge_index.shape[1]}")
        print(f"   ✓ Node features: {graph.x.shape}")
        print(f"   ✓ Edge features: {graph.edge_attr.shape}")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! Data is ready for training.")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test OpenFOAM data loading')
    parser.add_argument('--case_path', type=str, default='OpenFOAM-data',
                       help='Path to OpenFOAM case directory')
    args = parser.parse_args()
    
    success = test_data_loading(args.case_path)
    sys.exit(0 if success else 1)

