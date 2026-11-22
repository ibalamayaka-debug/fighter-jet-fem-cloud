# F-16 Fighter Jet 3D Structural FEM & Aero-Stress Cloud Lab

100% Cloud-computed **3D Finite Element Method (FEM)** stress, strain, and elastic wing-bending laboratory for a 1:1 scale watertight **F-16 Fighting Falcon** CAD model (`14.53m × 9.42m × 4.53m`), coupled via **1-way Aero-Structural FSI** with our live OpenFOAM F-16 CFD pressure field.

## Live Links
- **3D Interactive FEM App (中文)**: https://f16-fem-cloud.ibalamayaka.workers.dev/?lang=zh
- **3D Interactive FEM App (EN)**: https://f16-fem-cloud.ibalamayaka.workers.dev/?lang=en
- **Cloudflare API Gateway**: https://f16-fem-worker.ibalamayaka.workers.dev/api/cases

## Architecture ($0/month Serverless Stack)
- **Cloud 3D FEM Compute (`compute/`)**: Modal serverless Python (`Gmsh` 3D Delaunay tetrahedral volume mesher + exact vectorized 3D linear-elastic tetrahedral FEM `K u = f` via SciPy sparse direct solver + 1-way FSI OpenFOAM pressure mapping), scale-to-zero after 30s.
- **Edge Gateway & Cache (`worker/`)**: Cloudflare Worker + `F16_FEM_KV` edge caching + per-IP rate limiting.
- **3D WebGL Frontend (`frontend/`)**: Vite + React + TypeScript + Kitware `@kitware/vtk.js` with 60 FPS interactive 3D elastic wing-bending animation, undeformed ghost wireframe comparison, internal tetrahedral spar slice, and `[中文 | EN]` bilingual switching.
