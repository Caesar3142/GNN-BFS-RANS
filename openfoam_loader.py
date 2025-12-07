"""
OpenFOAM Data Loader
Parses OpenFOAM mesh and field files to extract graph structure and field data.
"""

import numpy as np
import re
from pathlib import Path
from typing import Dict, Tuple, Optional


class OpenFOAMLoader:
    """Loads OpenFOAM mesh and field data."""
    
    def __init__(self, case_path: str):
        """
        Initialize loader with OpenFOAM case directory.
        
        Args:
            case_path: Path to OpenFOAM case directory
        """
        self.case_path = Path(case_path)
        self.mesh_path = self.case_path / "constant" / "polyMesh"
        
    def read_points(self) -> np.ndarray:
        """Read mesh points (node coordinates)."""
        points_file = self.mesh_path / "points"
        with open(points_file, 'r') as f:
            content = f.read()
        
        # Extract number of points
        match = re.search(r'(\d+)\s*\(', content)
        if not match:
            raise ValueError("Could not find number of points")
        n_points = int(match.group(1))
        
        # Extract point coordinates
        points = []
        pattern = r'\(([-\d.eE+\s]+)\)'
        matches = re.findall(pattern, content)
        
        for match in matches:
            coords = [float(x) for x in match.split()]
            points.append(coords)
        
        return np.array(points)
    
    def read_owner_neighbour(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read owner and neighbour arrays (cell-face connectivity)."""
        owner_file = self.mesh_path / "owner"
        neighbour_file = self.mesh_path / "neighbour"
        
        def read_array(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            match = re.search(r'(\d+)\s*\(', content)
            if not match:
                raise ValueError(f"Could not find array size in {file_path}")
            n = int(match.group(1))
            
            # Extract numbers
            pattern = r'(\d+)'
            matches = re.findall(pattern, content)
            # Skip the first match (the count)
            return np.array([int(x) for x in matches[1:n+1]], dtype=np.int32)
        
        owner = read_array(owner_file)
        neighbour = read_array(neighbour_file)
        
        return owner, neighbour
    
    def read_faces(self) -> np.ndarray:
        """Read face definitions."""
        faces_file = self.mesh_path / "faces"
        with open(faces_file, 'r') as f:
            content = f.read()
        
        match = re.search(r'(\d+)\s*\(', content)
        if not match:
            raise ValueError("Could not find number of faces")
        n_faces = int(match.group(1))
        
        faces = []
        # Pattern to match face definitions: nPoints(point1 point2 ...)
        pattern = r'(\d+)\s*\(([\d\s]+)\)'
        matches = re.findall(pattern, content)
        
        for n_points_str, points_str in matches:
            point_indices = [int(x) for x in points_str.split()]
            faces.append(point_indices)
        
        return np.array(faces, dtype=object)
    
    def read_boundary(self) -> Dict:
        """Read boundary patch information."""
        boundary_file = self.mesh_path / "boundary"
        with open(boundary_file, 'r') as f:
            content = f.read()
        
        boundaries = {}
        # Extract boundary patches
        pattern = r'(\w+)\s*\{[^}]*type\s+(\w+);[^}]*nFaces\s+(\d+);[^}]*startFace\s+(\d+);'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for name, btype, n_faces, start_face in matches:
            boundaries[name] = {
                'type': btype,
                'nFaces': int(n_faces),
                'startFace': int(start_face)
            }
        
        return boundaries
    
    def read_scalar_field(self, time_dir: str, field_name: str) -> np.ndarray:
        """Read scalar field (e.g., pressure, k, epsilon)."""
        field_file = self.case_path / time_dir / field_name
        if not field_file.exists():
            raise FileNotFoundError(f"Field file not found: {field_file}")
        
        with open(field_file, 'r') as f:
            content = f.read()
        
        # Extract internal field
        match = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*(\d+)', content)
        if not match:
            raise ValueError(f"Could not find internal field in {field_name}")
        n_cells = int(match.group(1))
        
        # Extract values
        # Find the values section
        values_match = re.search(r'internalField[^\(]*\(([^)]+)\)', content, re.DOTALL)
        if not values_match:
            raise ValueError(f"Could not find values in {field_name}")
        
        values_str = values_match.group(1)
        # Extract all numbers
        pattern = r'([-\d.eE+]+)'
        matches = re.findall(pattern, values_str)
        
        # Take first n_cells values
        values = [float(x) for x in matches[:n_cells]]
        return np.array(values)
    
    def read_vector_field(self, time_dir: str, field_name: str) -> np.ndarray:
        """Read vector field (e.g., velocity U)."""
        field_file = self.case_path / time_dir / field_name
        if not field_file.exists():
            raise FileNotFoundError(f"Field file not found: {field_file}")
        
        with open(field_file, 'r') as f:
            lines = f.readlines()
        
        # Find internalField line
        internal_start = None
        n_cells = None
        for i, line in enumerate(lines):
            if 'internalField' in line and 'nonuniform' in line:
                # The number of cells is on the next line
                if i + 1 < len(lines):
                    match = re.search(r'(\d+)', lines[i + 1])
                    if match:
                        n_cells = int(match.group(1))
                # Find the opening parenthesis (should be on line i+2)
                for j in range(i + 1, min(i + 5, len(lines))):
                    if '(' in lines[j]:
                        internal_start = j + 1
                        break
                break
        
        if n_cells is None or internal_start is None:
            raise ValueError(f"Could not find internal field in {field_name}")
        
        # Extract vectors until we have n_cells
        vectors = []
        i = internal_start
        while i < len(lines) and len(vectors) < n_cells:
            line = lines[i].strip()
            # Match vector pattern: (x y z)
            match = re.search(r'\(([-\d.eE+\s]+)\)', line)
            if match:
                coords = [float(x) for x in match.group(1).split()]
                if len(coords) == 3:
                    vectors.append(coords)
            i += 1
        
        if len(vectors) != n_cells:
            raise ValueError(f"Expected {n_cells} vectors, found {len(vectors)} in {field_name}")
        
        return np.array(vectors)
    
    def get_cell_centers(self, points: np.ndarray, owner: np.ndarray, 
                         neighbour: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Calculate cell centers from mesh connectivity.
        Cell center is the centroid of all unique vertices belonging to the cell.
        """
        n_cells = max(np.max(owner), np.max(neighbour)) + 1
        cell_centers = np.zeros((n_cells, 3))
        
        # For each cell, collect all unique points from its faces
        from collections import defaultdict
        cell_points = defaultdict(set)
        
        # Collect points for each cell from owner faces
        for i in range(len(owner)):
            cell_idx = owner[i]
            face = faces[i]
            face_indices = np.array(face, dtype=np.int32)
            cell_points[cell_idx].update(face_indices)
        
        # Collect points for each cell from neighbour faces
        for i in range(len(neighbour)):
            cell_idx = neighbour[i]
            face = faces[i]
            face_indices = np.array(face, dtype=np.int32)
            cell_points[cell_idx].update(face_indices)
        
        # Calculate cell centers as average of all unique points
        for cell_idx in range(n_cells):
            if cell_idx in cell_points and len(cell_points[cell_idx]) > 0:
                point_indices = np.array(list(cell_points[cell_idx]), dtype=np.int32)
                cell_centers[cell_idx] = np.mean(points[point_indices], axis=0)
            else:
                # Fallback: use zero (shouldn't happen for valid cells)
                cell_centers[cell_idx] = np.zeros(3)
        
        return cell_centers
    
    def get_internal_cells(self, owner: np.ndarray, neighbour: np.ndarray) -> np.ndarray:
        """
        Identify internal cells (cells that appear in neighbour array).
        Internal cells are those that have at least one internal face.
        
        Returns:
            Boolean mask of shape [n_cells] where True indicates internal cell
        """
        n_cells = max(np.max(owner), np.max(neighbour)) + 1
        internal_mask = np.zeros(n_cells, dtype=bool)
        
        # Cells that appear in neighbour array are internal
        for cell_idx in neighbour:
            internal_mask[cell_idx] = True
        
        # Also include owner cells of internal faces (they're internal too)
        for i in range(len(neighbour)):
            internal_mask[owner[i]] = True
        
        return internal_mask
    
    def load_mesh(self) -> Dict:
        """Load complete mesh information."""
        points = self.read_points()
        owner, neighbour = self.read_owner_neighbour()
        faces = self.read_faces()
        boundaries = self.read_boundary()
        cell_centers = self.get_cell_centers(points, owner, neighbour, faces)
        internal_mask = self.get_internal_cells(owner, neighbour)
        
        return {
            'points': points,
            'owner': owner,
            'neighbour': neighbour,
            'faces': faces,
            'boundaries': boundaries,
            'cell_centers': cell_centers,
            'n_cells': len(cell_centers),
            'internal_mask': internal_mask,
            'n_internal_cells': np.sum(internal_mask)
        }
    
    def load_fields(self, time_dir: str, fields: list = None) -> Dict:
        """
        Load field data for a specific time directory.
        
        Args:
            time_dir: Time directory name (e.g., '0', '100', '282')
            fields: List of field names to load. If None, loads all available.
        
        Returns:
            Dictionary of field arrays
        """
        if fields is None:
            fields = ['U', 'p', 'k', 'epsilon', 'nut']
        
        field_data = {}
        
        for field in fields:
            try:
                if field == 'U':
                    field_data[field] = self.read_vector_field(time_dir, field)
                else:
                    field_data[field] = self.read_scalar_field(time_dir, field)
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not load field {field}: {e}")
        
        return field_data

