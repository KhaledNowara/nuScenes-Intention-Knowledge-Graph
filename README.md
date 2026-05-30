# nuScenes Intention Knowledge Graph (NIKG)

Companion code and artifacts for the paper *A Semantic Knowledge Graph Construction Pipeline for Vehicle Intention Prediction in nuScenes*.

This repository contains:

- A deterministic pipeline that builds an **RDF/Turtle** knowledge graph from [nuScenes](https://www.nuscenes.org/) (mini or full).
- An **OWL TTL** ontology (`ontologies/ontology.ttl`) merged into the output graph, plus a compact diagram view (`ontologies/ontology_diagram.ttl`) and a stand-alone T-Box (`graphs/ontology_tbox.ttl`).
- Evaluation scripts that reproduce **coverage**, **semantic consistency**, and **temporal label coherence** statistics on the generated graph.
- A **scene study** generator (`evaluation/scene_study.py`) that renders annotated frames and the per-scene subgraph used in the paper.

## Prerequisites

- Python 3.11 (see `environment.yml` for a conda environment)
- nuScenes dataset root (directory containing `samples/`, `sweeps/`, and e.g. `v1.0-mini/`)

Install (conda):

```bash
conda env create -f environment.yml
conda activate exp006   # or the name in environment.yml
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Build the knowledge graph

From this directory:

```bash
python main.py \
  --dataroot /path/to/nuscenes \
  --version v1.0-mini \
  --ontology ontologies/ontology.ttl \
  --out-ttl graphs/multi_scene.ttl \
  --out-sparql graphs/multi_scene.sparql
```

### `main.py` arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataroot` | *(required)* | nuScenes root (contains `samples/`, `sweeps/`, version JSON). |
| `--version` | `v1.0-mini` | DB version folder name (e.g. `v1.0-mini`, `v1.0-trainval`). |
| `--scenes` | all | Comma-separated scene names (e.g. `scene-0061,scene-1094`). Omit for all scenes. |
| `--max-keyframes` | all | Max CAM_FRONT keyframes **per scene** (for smoke tests). |
| `--ontology` | `ontologies/ontology.ttl` | OWL T-Box merged into the output graph. |
| `--out-ttl` | `graphs/multi_scene.ttl` | Output Turtle file (schema + data). |
| `--out-sparql` | `graphs/multi_scene.sparql` | SPARQL `INSERT DATA` wrapper (optional; for bulk load use **TTL** in triple stores). |
| `--tbox-out` | off | If set, re-serializes the ontology-only **T-Box** (no instance triples) to the given path. |
| `--save-artifacts` | off | If set, writes annotated frames under `traces/annotated_frames/`. |

The default output **`graphs/multi_scene.ttl`** is the artifact referenced in the paper: self-describing (ontology + instance triples).

## Evaluation (reproduce paper statistics)

**Requires** an existing `graphs/multi_scene.ttl` (run `main.py` first).  
Some scripts also need `--dataroot` because they re-load nuScenes to cross-check against raw geometry.

### All checks (stats + optional nuScenes-dependent steps)

```bash
python evaluation/run_all.py \
  --ttl graphs/multi_scene.ttl \
  --dataroot /path/to/nuscenes \
  --version v1.0-mini \
  --sample 20
```

If `--dataroot` is omitted, only **TTL-based** steps run (`kg_stats`, `semantic_consistency`).

### Individual scripts

| Script | Needs dataroot? | Output |
|--------|-----------------|--------|
| `evaluation/kg_stats.py` | no | `evaluation/results/kg_stats.{json,md}` — triple count, class counts, relation value distributions. |
| `evaluation/semantic_consistency.py` | no | `evaluation/results/semantic_consistency.{json,md}` — cross-relation logical checks. |
| `evaluation/motion_forward_check.py` | **yes** | `evaluation/results/motion_forward_check.json` — non-circular temporal coherence of `motionTowardEgo`. |
| `evaluation/spot_check_render.py` | **yes** | Renders sample CAM_FRONT images + CSV (optional spot-checks). |

Examples:

```bash
# Only TTL (no nuScenes re-load)
python evaluation/kg_stats.py --ttl graphs/multi_scene.ttl
python evaluation/semantic_consistency.py --ttl graphs/multi_scene.ttl

# Full pipeline alignment with nuScenes
python evaluation/motion_forward_check.py --dataroot /path/to/nuscenes --version v1.0-mini
```

### Scene study (paper case study)

`evaluation/scene_study.py` reproduces the focused case study in the paper:
annotated `CAM_FRONT` frames for a chosen scene/frame range, the per-scene
subgraph (`scene_kg.ttl`), and the raw numerical detail behind each label
(`scene_study_details.json`). It **requires** `--dataroot` because it re-renders
frames from nuScenes images.

```bash
python evaluation/scene_study.py \
  --dataroot /path/to/nuscenes \
  --version v1.0-mini \
  --scene scene-1100 \
  --frames 19-22 \
  --out-dir evaluation/results/scene_study \
  --ontology ontologies/ontology.ttl
```

A pre-generated example for `scene-1100`, frames 19–22 (the one used in the
paper) is committed under `evaluation/results/scene_study/`.

## Artifact layout (paper)

- **`graphs/multi_scene.ttl`** — main release file: self-describing graph (ontology + instance triples).
- **`graphs/ontology_tbox.ttl`** — ontology-only T-Box (no instance triples).
- **`ontologies/ontology.ttl`** — the OWL ontology; `ontology_diagram.ttl` is a compact view for rendering the schema figure.
- **`evaluation/results/*.json` / `*.md`** — numbers reported in the paper (regenerate to match your run).
- **`evaluation/results/scene_study/`** — annotated frames, subgraph, and raw label detail for the paper's scene study.

## Data and licensing

The **code** in this repository is released under the MIT License (see `LICENSE`).

The generated knowledge graph (`graphs/*.ttl`) and the annotated images under
`evaluation/results/scene_study/` are **derived from the nuScenes dataset** and
are therefore subject to the [nuScenes terms of use](https://www.nuscenes.org/terms-of-use)
(CC BY-NC-SA 4.0, non-commercial). To regenerate them you must download nuScenes
yourself and agree to its license; the dataset is **not** redistributed here.

If you use this work, please cite the accompanying paper (see `CITATION.cff`).
