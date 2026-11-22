"""
Modal Serverless Cloud Compute Endpoint for F-16 3D Structural FEM (Stress & Strain).
Scale-to-zero ($0/month idle) with Gmsh 3D tetrahedral meshing + SciPy sparse FEM.
"""

import os
from pathlib import Path

import modal

app = modal.App("f16-fighter-fem")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libglu1-mesa",
        "libxrender1",
        "libxcursor1",
        "libxft2",
        "libxinerama1",
        "libgl1",
        "libgomp1",
    )
    .pip_install("gmsh", "pymeshfix", "numpy", "scipy", "vtk", "fastapi[standard]")
    .add_local_file("compute/f16_fem_solver.py", "/app/f16_fem_solver.py")
    .add_local_file("compute/assets/f16_normalized.stl", "/app/assets/f16_normalized.stl")
)

STL_IN_CONTAINER = Path("/app/assets/f16_normalized.stl")


@app.function(
    image=image,
    cpu=4.0,
    memory=4096,
    timeout=240,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("f16-fem-secrets")],
)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, "/app")
    from fastapi import Body, FastAPI, Header, HTTPException
    from f16_fem_solver import F16FemParams, run_f16_fem_simulation

    web_app = FastAPI(title="F-16 3D Cloud FEM Solver")

    @web_app.get("/health")
    def health():
        return {"status": "ok", "solver": "gmsh-3d-tet-fem", "aircraft": "F-16"}

    @web_app.post("/simulate")
    def simulate(payload: dict = Body(...), authorization: str | None = Header(default=None)):
        expected = os.environ.get("COMPUTE_AUTH_TOKEN", "")
        if expected:
            if not authorization or authorization != f"Bearer {expected}":
                raise HTTPException(status_code=401, detail="Unauthorized")
        try:
            params = F16FemParams(
                g_load=float(payload.get("g_load", 9.0)),
                speed_ms=float(payload.get("speed_ms", 220.0)),
                material=str(payload.get("material", "al7075")),
            )
            return run_f16_fem_simulation(params, STL_IN_CONTAINER)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return web_app
