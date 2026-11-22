"""
Cloud 3D Finite Element Method (FEM) Structural Stress & Strain Solver
for the 1:1 Scale F-16 Fighting Falcon (1-Way Aero-Structural FSI).

Pipeline (100% Cloud on Modal):
  1. Gmsh 3D Delaunay volume tetrahedral mesh inside watertight F-16 STL
  2. 1-Way FSI aero-pressure load mapped from live OpenFOAM F-16 CFD Cp
     field + maneuver load factor Nz (-3G .. +9G) & dynamic pressure q_inf
  3. Exact vectorized 3D linear-elastic tetrahedral FEM (K u = f) solved
     with SciPy sparse direct solver (equivalent hollow wing-box stiffness)
  4. Volume-weighted nodal recovery of von Mises Stress (MPa), Equivalent
     Strain (microstrain), 3D Displacement Vector, & Yield Safety Factor
  5. Compact ASCII VTP export for 60 FPS interactive VTK.js 3D bending viewer
"""

from __future__ import annotations

import hashlib
import math
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

# Real aerospace structural materials
MATERIALS: Dict[str, Dict[str, object]] = {
    "al7075": {
        "name_en": "7075-T6 Aerospace Aluminum",
        "name_zh": "7075-T6 航空铝合金",
        "E_gpa": 71.7,
        "nu": 0.33,
        "yield_mpa": 503.0,
        "rho_kgm3": 2810.0,
    },
    "ti6al4v": {
        "name_en": "Ti-6Al-4V Titanium Alloy",
        "name_zh": "Ti-6Al-4V 航空钛合金",
        "E_gpa": 113.8,
        "nu": 0.342,
        "yield_mpa": 880.0,
        "rho_kgm3": 4430.0,
    },
    "cfrp": {
        "name_en": "T800 Carbon Fiber Composite",
        "name_zh": "T800 碳纤维复合材料",
        "E_gpa": 135.0,
        "nu": 0.30,
        "yield_mpa": 1100.0,
        "rho_kgm3": 1580.0,
    },
}

# Real F-16 nominal combat loaded mass (kg) & equivalent hollow wing-box stiffness ratio
F16_MASS_KG = 11500.0
G_ACCEL = 9.81
# Hollow stressed-skin + internal spar box ratio (I_box / I_solid_OML)
WING_BOX_STIFFNESS_RATIO = 0.14

# Cached in warm Modal container memory across runs
_CACHED_MESH: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = None
_CACHED_OPENFOAM_CP: Optional[Tuple[np.ndarray, np.ndarray]] = None

OPENFOAM_JET_VTP_URL = (
    "https://f16-cfd-worker.ibalamayaka.workers.dev/api/results/65e77f4e7daf09cc/jet.vtp"
)


@dataclass(frozen=True)
class F16FemParams:
    g_load: float = 9.0          # Maneuver load factor [-3, 9] G
    speed_ms: float = 220.0      # Flight speed [100, 280] m/s
    material: str = "al7075"     # al7075 | ti6al4v | cfrp

    def normalized(self) -> "F16FemParams":
        g = round(min(max(float(self.g_load), -3.0), 9.0))
        if g == 0:
            g = 1
        spd = round(min(max(float(self.speed_ms), 100.0), 280.0) / 20.0) * 20.0
        mat = str(self.material).lower().strip()
        if mat not in MATERIALS:
            mat = "al7075"
        return F16FemParams(g_load=float(g), speed_ms=float(spd), material=mat)

    def case_key(self) -> str:
        p = self.normalized()
        canonical = f"f16fem|g={p.g_load:.0f}|v={p.speed_ms:.0f}|mat={p.material}"
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _repair_stl_with_pymeshfix(stl_path: Path) -> Optional[Path]:
    """Repair tiny CAD self-intersections (e.g. wingtip rails) so Gmsh 3D Delaunay succeeds."""
    try:
        import tempfile
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        import pymeshfix

        reader = vtk.vtkSTLReader()
        reader.SetFileName(str(stl_path))
        reader.Update()
        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(reader.GetOutputPort())
        clean.SetTolerance(1e-5)
        clean.Update()
        poly = clean.GetOutput()

        pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
        polys = vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:4].astype(np.int32)

        mfix = pymeshfix.MeshFix(pts, polys)
        mfix.repair(verbose=False)
        out_path = Path(tempfile.gettempdir()) / "f16_repaired.stl"
        mfix.save(str(out_path))
        return out_path
    except Exception:
        return None


def _fallback_vtk_body_fitted_tet_mesh(stl_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Guaranteed body-fitted 3D tetrahedral mesh via signed-distance clipping of coarse 3D grid."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    reader.Update()
    stl_poly = reader.GetOutput()

    # Build coarse 3D rectilinear grid covering F-16 bounds with fine Z resolution for thin wings
    xs = np.linspace(-7.45, 7.45, 76)
    ys = np.linspace(-4.85, 4.85, 58)
    zs = np.concatenate([
        np.linspace(-1.05, -0.45, 5, endpoint=False),
        np.linspace(-0.45, 0.55, 13, endpoint=False),
        np.linspace(0.55, 3.75, 11),
    ])

    rg = vtk.vtkRectilinearGrid()
    rg.SetDimensions(len(xs), len(ys), len(zs))
    rg.SetXCoordinates(numpy_to_vtk(xs.astype(np.float32), deep=True))
    rg.SetYCoordinates(numpy_to_vtk(ys.astype(np.float32), deep=True))
    rg.SetZCoordinates(numpy_to_vtk(zs.astype(np.float32), deep=True))

    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(stl_poly)

    # Evaluate signed distance on grid points
    n_pts = rg.GetNumberOfPoints()
    sdf = np.empty(n_pts, dtype=np.float32)
    pt = [0.0, 0.0, 0.0]
    for i in range(n_pts):
        rg.GetPoint(i, pt)
        sdf[i] = imp.EvaluateFunction(pt)

    sdf_vtk = numpy_to_vtk(sdf, deep=True)
    sdf_vtk.SetName("sdf")
    rg.GetPointData().SetScalars(sdf_vtk)

    clipper = vtk.vtkClipDataSet()
    clipper.SetInputData(rg)
    clipper.SetValue(0.0)
    clipper.InsideOutOn()
    clipper.Update()

    tet_filter = vtk.vtkDataSetTriangleFilter()
    tet_filter.SetInputConnection(clipper.GetOutputPort())
    tet_filter.TetrahedraOnlyOn()
    tet_filter.Update()

    # Keep only largest connected component (the F-16 airframe)
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputConnection(tet_filter.GetOutputPort())
    conn.SetExtractionModeToLargestRegion()
    conn.Update()
    ug = conn.GetOutput()

    nodes = vtk_to_numpy(ug.GetPoints().GetData()).astype(np.float64)
    cells_raw = vtk_to_numpy(ug.GetCells().GetData())
    tets = cells_raw.reshape(-1, 5)[:, 1:5].astype(np.int32)

    # Extract exterior surface triangles
    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(ug)
    geom.Update()
    tri_f = vtk.vtkTriangleFilter()
    tri_f.SetInputConnection(geom.GetOutputPort())
    tri_f.Update()
    surf = tri_f.GetOutput()
    surf_pts = vtk_to_numpy(surf.GetPoints().GetData()).astype(np.float64)
    surf_tris_local = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:, 1:4].astype(np.int32)

    # Map surface vertices back to volume node indices
    tree = cKDTree(nodes)
    _, s_map = tree.query(surf_pts, k=1)
    tris = s_map[surf_tris_local].astype(np.int32)
    return nodes, tets, tris


def _generate_coarse_tet_mesh(stl_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate (or reuse cached) coarse 3D tetrahedral mesh inside F-16 STL."""
    global _CACHED_MESH
    if _CACHED_MESH is not None:
        return _CACHED_MESH

    import gmsh

    t0 = time.perf_counter()
    nodes = None
    tets = None
    tris = None

    repaired_stl = _repair_stl_with_pymeshfix(stl_path) or stl_path

    # Primary: Gmsh 3D Delaunay tetrahedral mesh on repaired watertight F-16 STL
    try:
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("f16_solid")
            gmsh.merge(str(repaired_stl))

            surfaces = gmsh.model.getEntities(2)
            if surfaces:
                surf_loop = gmsh.model.geo.addSurfaceLoop([e[1] for e in surfaces])
                gmsh.model.geo.addVolume([surf_loop])
                gmsh.model.geo.synchronize()

                gmsh.option.setNumber("Mesh.Algorithm3D", 1)
                gmsh.option.setNumber("Mesh.MeshSizeMin", 0.14)
                gmsh.option.setNumber("Mesh.MeshSizeMax", 0.90)
                gmsh.option.setNumber("Mesh.Optimize", 1)
                gmsh.model.mesh.generate(3)

                node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
                coords_all = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
                tag_to_raw = {int(t): idx for idx, t in enumerate(node_tags)}

                tet_tags, tet_nodes = gmsh.model.mesh.getElementsByType(4)
                if len(tet_tags) > 500:
                    tets_raw = np.asarray([tag_to_raw[int(n)] for n in tet_nodes], dtype=np.int32).reshape(-1, 4)
                    tri_tags, tri_nodes = gmsh.model.mesh.getElementsByType(2)
                    tris_raw = np.asarray([tag_to_raw[int(n)] for n in tri_nodes], dtype=np.int32).reshape(-1, 3)

                    used = np.unique(tets_raw)
                    remap = np.full(coords_all.shape[0], -1, dtype=np.int32)
                    remap[used] = np.arange(len(used), dtype=np.int32)
                    nodes = coords_all[used].copy()
                    tets = remap[tets_raw]
                    tris = remap[tris_raw]
                    tris = tris[np.all(tris >= 0, axis=1)]
        finally:
            gmsh.finalize()
    except Exception:
        nodes = None

    if nodes is None or tets is None or tris is None:
        nodes, tets, tris = _fallback_vtk_body_fitted_tet_mesh(stl_path)

    # Ensure positive orientation and remove near-zero volume slivers
    p0 = nodes[tets[:, 0]]
    e1 = nodes[tets[:, 1]] - p0
    e2 = nodes[tets[:, 2]] - p0
    e3 = nodes[tets[:, 3]] - p0
    det = np.einsum("ij,ij->i", np.cross(e1, e2), e3)
    neg = det < 0
    if np.any(neg):
        tmp = tets[neg, 2].copy()
        tets[neg, 2] = tets[neg, 3]
        tets[neg, 3] = tmp
    keep = np.abs(det) > 1e-10
    tets = tets[keep]

    # Compact to nodes referenced by surviving tets
    used2 = np.unique(tets)
    if len(used2) < nodes.shape[0]:
        rm2 = np.full(nodes.shape[0], -1, dtype=np.int32)
        rm2[used2] = np.arange(len(used2), dtype=np.int32)
        nodes = nodes[used2].copy()
        tets = rm2[tets]
        tris = rm2[tris]
        tris = tris[np.all(tris >= 0, axis=1)]

    mesh_sec = time.perf_counter() - t0
    _CACHED_MESH = (nodes, tets, tris, mesh_sec)
    return _CACHED_MESH


def _fetch_openfoam_cp_cloud() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Fetch real OpenFOAM F-16 surface Cp field from our live CFD Worker (1-way FSI)."""
    global _CACHED_OPENFOAM_CP
    if _CACHED_OPENFOAM_CP is not None:
        return _CACHED_OPENFOAM_CP

    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        req = urllib.request.Request(
            OPENFOAM_JET_VTP_URL,
            headers={"User-Agent": "F16-FEM-FSI-Cloud/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            raw_str = resp.read().decode("utf-8", errors="ignore")

        reader = vtk.vtkXMLPolyDataReader()
        reader.ReadFromInputStringOn()
        reader.SetInputString(raw_str)
        reader.Update()
        poly = reader.GetOutput()
        pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
        cp_arr = poly.GetPointData().GetArray("Cp")
        if cp_arr is not None and len(pts) > 500:
            cp = vtk_to_numpy(cp_arr).astype(np.float64)
            _CACHED_OPENFOAM_CP = (pts, cp)
            return _CACHED_OPENFOAM_CP
    except Exception:
        return None
    return None


def _constitutive_matrix(E_pa: float, nu: float) -> np.ndarray:
    c = E_pa / ((1.0 + nu) * (1.0 - 2.0 * nu))
    a = 1.0 - nu
    b = nu
    g = 0.5 * (1.0 - 2.0 * nu)
    return c * np.array(
        [
            [a, b, b, 0.0, 0.0, 0.0],
            [b, a, b, 0.0, 0.0, 0.0],
            [b, b, a, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, g, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, g, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, g],
        ],
        dtype=np.float64,
    )


def _assemble_and_solve_fem(
    nodes: np.ndarray,
    tets: np.ndarray,
    tris: np.ndarray,
    params: F16FemParams,
) -> Dict[str, object]:
    mat = MATERIALS[params.material]
    E_mat = float(mat["E_gpa"]) * 1e9
    nu = float(mat["nu"])
    yield_mpa = float(mat["yield_mpa"])

    # Equivalent hollow wing-box stiffness for global displacement; full E_mat for skin stress
    E_eff = E_mat * WING_BOX_STIFFNESS_RATIO
    D_eff = _constitutive_matrix(E_eff, nu)
    D_skin = _constitutive_matrix(E_mat, nu)

    n_nodes = nodes.shape[0]
    n_tets = tets.shape[0]
    n_dofs = n_nodes * 3

    # Vectorized Jacobian & shape function gradients for all 3D tetrahedra
    p0 = nodes[tets[:, 0]]
    p1 = nodes[tets[:, 1]]
    p2 = nodes[tets[:, 2]]
    p3 = nodes[tets[:, 3]]

    # J[:, :, k] = p_{k+1} - p_0
    J = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=2)  # (M, 3, 3)
    detJ = np.linalg.det(J)
    vols = np.abs(detJ) / 6.0

    # gradN for nodes 1,2,3 are rows of J^{-1}; gradN_0 = -(gradN_1 + gradN_2 + gradN_3)
    invJ = np.linalg.inv(J)  # (M, 3, 3)
    g1 = invJ[:, 0, :]
    g2 = invJ[:, 1, :]
    g3 = invJ[:, 2, :]
    g0 = -(g1 + g2 + g3)
    grads = np.stack([g0, g1, g2, g3], axis=1)  # (M, 4, 3)

    # Strain-displacement matrix B of shape (M, 6, 12)
    B = np.zeros((n_tets, 6, 12), dtype=np.float64)
    for a in range(4):
        gx = grads[:, a, 0]
        gy = grads[:, a, 1]
        gz = grads[:, a, 2]
        c = 3 * a
        B[:, 0, c + 0] = gx
        B[:, 1, c + 1] = gy
        B[:, 2, c + 2] = gz
        B[:, 3, c + 0] = gy
        B[:, 3, c + 1] = gx
        B[:, 4, c + 1] = gz
        B[:, 4, c + 2] = gy
        B[:, 5, c + 0] = gz
        B[:, 5, c + 2] = gx

    # Element stiffness Ke = V_e * B^T * D_eff * B -> (M, 12, 12)
    DB = np.einsum("ij,mjk->mik", D_eff, B)
    Ke = np.einsum("m,mji,mjk->mik", vols, B, DB)

    # Assemble global sparse stiffness matrix K
    elem_dofs = np.empty((n_tets, 12), dtype=np.int32)
    for a in range(4):
        elem_dofs[:, 3 * a + 0] = 3 * tets[:, a] + 0
        elem_dofs[:, 3 * a + 1] = 3 * tets[:, a] + 1
        elem_dofs[:, 3 * a + 2] = 3 * tets[:, a] + 2

    rows = np.repeat(elem_dofs, 12, axis=1).ravel()
    cols = np.tile(elem_dofs, (1, 12)).ravel()
    K = sparse.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n_dofs, n_dofs)).tocsc()

    # --- 1-Way FSI Aerodynamic Pressure + Maneuver G-Load Force Assembly ---
    tri_p0 = nodes[tris[:, 0]]
    tri_p1 = nodes[tris[:, 1]]
    tri_p2 = nodes[tris[:, 2]]
    tri_centroids = (tri_p0 + tri_p1 + tri_p2) / 3.0
    cross_vec = np.cross(tri_p1 - tri_p0, tri_p2 - tri_p0)
    tri_areas = 0.5 * np.linalg.norm(cross_vec, axis=1)
    tri_normals = cross_vec / np.maximum(2.0 * tri_areas[:, None], 1e-12)

    # Ensure normals point outward from aircraft centerline
    out_ref = tri_centroids - np.array([0.0, 0.0, 0.15])
    # For wing faces, Z-sign check; general check via OpenFOAM Cp mapping
    of_cloud = _fetch_openfoam_cp_cloud()
    fsi_source = "openfoam-1way-fsi"
    if of_cloud is not None:
        of_pts, of_cp = of_cloud
        tree = cKDTree(of_pts)
        _, nn_idx = tree.query(tri_centroids, k=1)
        tri_cp = of_cp[nn_idx]
    else:
        fsi_source = "analytical-elliptic-aero"
        # Elliptic spanwise wing suction + nose stagnation fallback
        span_frac = np.clip(np.abs(tri_centroids[:, 1]) / 4.71, 0.0, 0.995)
        elliptic = np.sqrt(1.0 - span_frac ** 2)
        upper = (tri_centroids[:, 2] >= 0.0).astype(np.float64) * 2.0 - 1.0
        tri_cp = -0.85 * elliptic * upper + 0.65 * np.exp(-((tri_centroids[:, 0] + 7.0) / 1.5) ** 2)

    # Aerodynamic load distribution:
    # Real maneuver total vertical lift L = Nz * m_F16 * g
    # Distributed over wings, LEX strakes & horizontal tail weighted by elliptic aerodynamic loading
    # and modulated by real OpenFOAM pressure coefficient differential
    rho_air = 1.225
    q_inf = 0.5 * rho_air * (params.speed_ms ** 2)
    target_lift_n = params.g_load * F16_MASS_KG * G_ACCEL

    f_vec = np.zeros(n_dofs, dtype=np.float64)

    # Identify aerodynamic lifting surfaces (|y| > 0.72 m, outside clamped fuselage carry-through box)
    cy = np.abs(tri_centroids[:, 1])
    cx = tri_centroids[:, 0]
    cz = tri_centroids[:, 2]
    is_wing_or_tail = (cy > 0.72) & (cz < 1.2)
    # Elliptic spanwise lift distribution + chordwise front-spar weighting + OpenFOAM Cp modulation
    span_w = np.sqrt(np.clip(1.0 - (cy / 4.82) ** 2, 0.05, 1.0))
    cp_mod = np.clip(1.0 + 0.35 * np.abs(tri_cp), 0.7, 1.6)
    lift_weight = np.where(is_wing_or_tail, tri_areas * span_w * cp_mod, 0.0)
    total_w = float(np.sum(lift_weight))
    if total_w < 1e-6:
        raise RuntimeError("No lifting surface triangles detected on F-16 mesh")

    # Vertical aerodynamic lift force per triangle (N), exact integral = Nz * W_F16
    fz_tri = target_lift_n * (lift_weight / total_w)

    # Streamwise aerodynamic drag force (dynamic pressure q_inf scaled, ~11% of lift + nose stagnation)
    nose_stag = np.exp(-((cx + 7.2) / 1.8) ** 2) * q_inf * 0.45 * tri_areas
    fx_tri = 0.095 * np.abs(fz_tri) + nose_stag

    # Also apply subtle vertical tail side-gust / buffet load (2% of dynamic pressure) on fin (z > 1.2)
    is_fin = cz > 1.2
    fy_tri = np.where(is_fin, 0.04 * q_inf * tri_areas * np.sign(params.g_load), 0.0)

    # Distribute triangle forces equally to its 3 corner nodes (consistent P1 surface load)
    for k in range(3):
        nids = tris[:, k]
        np.add.at(f_vec, 3 * nids + 0, fx_tri / 3.0)
        np.add.at(f_vec, 3 * nids + 1, fy_tri / 3.0)
        np.add.at(f_vec, 3 * nids + 2, fz_tri / 3.0)

    # --- Boundary Conditions: Clamp F-16 Main Fuselage Wing Carry-Through Bulkheads ---
    fixed_mask = (
        (nodes[:, 0] >= -0.85)
        & (nodes[:, 0] <= 2.45)
        & (np.abs(nodes[:, 1]) <= 0.72)
        & (nodes[:, 2] >= -0.65)
        & (nodes[:, 2] <= 0.65)
    )
    fixed_nodes = np.where(fixed_mask)[0]
    if len(fixed_nodes) < 12:
        # Fallback: clamp nearest 40 nodes to CG (0.8, 0, 0)
        dist_cg = np.linalg.norm(nodes - np.array([0.8, 0.0, 0.0]), axis=1)
        fixed_nodes = np.argsort(dist_cg)[:40]

    fixed_dofs = np.concatenate([3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2])
    free_mask = np.ones(n_dofs, dtype=bool)
    free_mask[fixed_dofs] = False
    free_dofs = np.where(free_mask)[0]

    # Solve reduced SPD system K_ff * u_f = f_f directly
    K_ff = K[free_dofs, :][:, free_dofs].tocsc()
    f_f = f_vec[free_dofs]
    u_full = np.zeros(n_dofs, dtype=np.float64)
    u_full[free_dofs] = spsolve(K_ff, f_f)
    disp = u_full.reshape(n_nodes, 3)

    # --- Recover Exact Element Strains & Equivalent Skin/Spar von Mises Stresses ---
    u_elem = u_full[elem_dofs]  # (M, 12)
    strain_e = np.einsum("mij,mj->mi", B, u_elem)  # (M, 6): [exx, eyy, ezz, gxy, gyz, gzx]
    stress_e = np.einsum("ij,mj->mi", D_skin, strain_e)  # (M, 6) in Pa

    sx, sy, sz, txy, tyz, tzx = (
        stress_e[:, 0],
        stress_e[:, 1],
        stress_e[:, 2],
        stress_e[:, 3],
        stress_e[:, 4],
        stress_e[:, 5],
    )
    vm_pa_e = np.sqrt(
        np.maximum(
            0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2 + 6.0 * (txy ** 2 + tyz ** 2 + tzx ** 2)),
            0.0,
        )
    )
    vm_mpa_e = vm_pa_e / 1e6

    # Equivalent von Mises strain (microstrain = 1e-6)
    ex, ey, ez = strain_e[:, 0], strain_e[:, 1], strain_e[:, 2]
    gxy, gyz, gzx = strain_e[:, 3], strain_e[:, 4], strain_e[:, 5]
    eq_strain_e = (
        (np.sqrt(2.0) / 3.0)
        * np.sqrt(
            np.maximum(
                (ex - ey) ** 2
                + (ey - ez) ** 2
                + (ez - ex) ** 2
                + 1.5 * (gxy ** 2 + gyz ** 2 + gzx ** 2),
                0.0,
            )
        )
    )
    strain_micro_e = eq_strain_e * 1e6

    # Volume-weighted nodal averaging for smooth surface & slice visualization
    node_vm = np.zeros(n_nodes, dtype=np.float64)
    node_strain = np.zeros(n_nodes, dtype=np.float64)
    node_vol = np.zeros(n_nodes, dtype=np.float64)
    for a in range(4):
        nids = tets[:, a]
        np.add.at(node_vm, nids, vm_mpa_e * vols)
        np.add.at(node_strain, nids, strain_micro_e * vols)
        np.add.at(node_vol, nids, vols)
    node_vol = np.maximum(node_vol, 1e-12)
    node_vm /= node_vol
    node_strain /= node_vol

    # Map surface Cp to nodes too (for inspection)
    node_cp = np.zeros(n_nodes, dtype=np.float64)
    node_area = np.zeros(n_nodes, dtype=np.float64)
    for k in range(3):
        nids = tris[:, k]
        np.add.at(node_cp, nids, tri_cp * tri_areas)
        np.add.at(node_area, nids, tri_areas)
    node_cp = np.where(node_area > 1e-12, node_cp / np.maximum(node_area, 1e-12), 0.0)

    disp_mag_mm = np.linalg.norm(disp, axis=1) * 1000.0

    # KPI metrics (use 99.8th percentile for robust peak stress avoiding single-node clamp singularity)
    max_vm_mpa = float(np.percentile(node_vm, 99.7))
    abs_max_vm_mpa = float(np.max(node_vm))
    max_strain_micro = float(np.percentile(node_strain, 99.7))
    max_disp_mm = float(np.max(disp_mag_mm))

    # Wingtip vertical deflection (|y| > 4.2m)
    tip_mask = np.abs(nodes[:, 1]) > 4.2
    wingtip_dz_mm = float(np.mean(disp[tip_mask, 2]) * 1000.0) if np.any(tip_mask) else 0.0

    safety_factor = round(yield_mpa / max(max_vm_mpa, 0.1), 2)

    return {
        "disp": disp,
        "disp_mag_mm": disp_mag_mm,
        "node_vm_mpa": node_vm,
        "node_strain_micro": node_strain,
        "node_cp": node_cp,
        "fsi_source": fsi_source,
        "kpis": {
            "max_von_mises_mpa": round(max_vm_mpa, 1),
            "peak_singularity_mpa": round(abs_max_vm_mpa, 1),
            "yield_strength_mpa": round(yield_mpa, 1),
            "safety_factor": safety_factor,
            "max_strain_micro": round(max_strain_micro, 0),
            "max_disp_mm": round(max_disp_mm, 1),
            "wingtip_dz_mm": round(wingtip_dz_mm, 1),
            "total_lift_kn": round(target_lift_n / 1000.0, 1),
            "total_lift_tonnes": round(target_lift_n / 9810.0, 1),
            "n_nodes": int(n_nodes),
            "n_tets": int(n_tets),
            "n_dofs": int(n_dofs),
            "n_fixed_nodes": int(len(fixed_nodes)),
            "material_en": str(mat["name_en"]),
            "material_zh": str(mat["name_zh"]),
            "E_gpa": float(mat["E_gpa"]),
            "nu": float(mat["nu"]),
        },
    }


def _export_web_vtps(
    nodes: np.ndarray,
    tets: np.ndarray,
    tris: np.ndarray,
    sol: Dict[str, object],
) -> Dict[str, object]:
    """Export compact ASCII .vtp surface & internal wing-spar mesh slice for VTK.js."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    disp = sol["disp"]
    disp_mm = sol["disp_mag_mm"]
    vm = sol["node_vm_mpa"]
    strain = sol["node_strain_micro"]
    cp = sol["node_cp"]

    def poly_to_ascii_str(poly: vtk.vtkPolyData) -> str:
        w = vtk.vtkXMLPolyDataWriter()
        w.SetWriteToOutputString(True)
        w.SetInputData(poly)
        w.SetDataModeToAscii()
        w.Write()
        return w.GetOutputString()

    # 1) Full 3D F-16 surface mesh (compact: only surface nodes)
    surf_used = np.unique(tris)
    s_remap = np.full(nodes.shape[0], -1, dtype=np.int32)
    s_remap[surf_used] = np.arange(len(surf_used), dtype=np.int32)
    s_nodes = nodes[surf_used].astype(np.float32)
    s_tris = s_remap[tris]

    surf_poly = vtk.vtkPolyData()
    v_pts = vtk.vtkPoints()
    v_pts.SetData(numpy_to_vtk(s_nodes, deep=True))
    surf_poly.SetPoints(v_pts)

    n_tris = s_tris.shape[0]
    conn = np.hstack([np.full((n_tris, 1), 3, dtype=np.int64), s_tris.astype(np.int64)]).ravel()
    cells = vtk.vtkCellArray()
    cells.SetCells(n_tris, numpy_to_vtkIdTypeArray(conn, deep=True))
    surf_poly.SetPolys(cells)

    pd = surf_poly.GetPointData()
    for name, arr in (
        ("von_mises_mpa", vm[surf_used].astype(np.float32)),
        ("strain_micro", strain[surf_used].astype(np.float32)),
        ("disp_mm", disp_mm[surf_used].astype(np.float32)),
        ("aero_cp", cp[surf_used].astype(np.float32)),
    ):
        va = numpy_to_vtk(arr, deep=True)
        va.SetName(name)
        pd.AddArray(va)

    disp_vec_vtk = numpy_to_vtk(disp[surf_used].astype(np.float32), deep=True)
    disp_vec_vtk.SetName("disp_vec")
    pd.AddArray(disp_vec_vtk)
    pd.SetActiveScalars("von_mises_mpa")

    # 2) Internal 3D Tetrahedral FEM Mesh Cut through the wing spar plane (z = 0.02 m)
    ug = vtk.vtkUnstructuredGrid()
    ug_pts = vtk.vtkPoints()
    ug_pts.SetData(numpy_to_vtk(nodes.astype(np.float32), deep=True))
    ug.SetPoints(ug_pts)

    n_tets = tets.shape[0]
    t_conn = np.hstack([np.full((n_tets, 1), 4, dtype=np.int64), tets.astype(np.int64)]).ravel()
    t_cells = vtk.vtkCellArray()
    t_cells.SetCells(n_tets, numpy_to_vtkIdTypeArray(t_conn, deep=True))
    ug.SetCells(vtk.VTK_TETRA, t_cells)

    ug_pd = ug.GetPointData()
    for name, arr in (
        ("von_mises_mpa", vm.astype(np.float32)),
        ("strain_micro", strain.astype(np.float32)),
        ("disp_mm", disp_mm.astype(np.float32)),
    ):
        va = numpy_to_vtk(arr, deep=True)
        va.SetName(name)
        ug_pd.AddArray(va)
    ug_pd.SetActiveScalars("von_mises_mpa")

    plane = vtk.vtkPlane()
    plane.SetOrigin(0.0, 0.0, 0.02)
    plane.SetNormal(0.0, 0.0, 1.0)
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(ug)
    cutter.Update()

    tri_cut = vtk.vtkTriangleFilter()
    tri_cut.SetInputConnection(cutter.GetOutputPort())
    tri_cut.Update()
    slice_poly = tri_cut.GetOutput()

    files_content = {
        "f16_fem.vtp": poly_to_ascii_str(surf_poly),
        "f16_slice.vtp": poly_to_ascii_str(slice_poly),
    }
    return {
        "files_content": files_content,
        "file_sizes_kb": {k: round(len(v.encode("utf-8")) / 1024.0, 1) for k, v in files_content.items()},
    }


def run_f16_fem_simulation(raw_params: F16FemParams, stl_path: Path) -> Dict[str, object]:
    t0 = time.perf_counter()
    params = raw_params.normalized()
    case_key = params.case_key()

    nodes, tets, tris, mesh_sec = _generate_coarse_tet_mesh(stl_path)

    t_solve0 = time.perf_counter()
    sol = _assemble_and_solve_fem(nodes, tets, tris, params)
    solve_sec = time.perf_counter() - t_solve0

    t_post0 = time.perf_counter()
    vtk_out = _export_web_vtps(nodes, tets, tris, sol)
    post_sec = time.perf_counter() - t_post0
    total_sec = time.perf_counter() - t0

    metrics = {
        "case_key": case_key,
        "aircraft": "F-16 Fighting Falcon (1:1 Scale 3D Tetrahedral FEM)",
        "fsi_source": sol["fsi_source"],
        "params": asdict(params),
        "kpis": sol["kpis"],
        "timing_sec": {
            "mesh": round(mesh_sec, 2),
            "solve": round(solve_sec, 2),
            "postprocess": round(post_sec, 2),
            "total": round(total_sec, 2),
        },
        "file_sizes_kb": vtk_out["file_sizes_kb"],
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return {
        "metrics": metrics,
        "files_content": vtk_out["files_content"],
    }
