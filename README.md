# PRANA-G AI

PRANA-G AI is a two-phase physics-guided screening pipeline. The current codebase first trains fast surrogate models from Physics-Informed Neural Networks (PINNs), then uses those surrogates to score candidate entities from a universal parquet index against a prompt-derived specification.

This README describes the current implementation only.

## Current Architecture

The active system has two phases:

1. **Training phase**
   - Entry point: `unified_pipeline.py`
   - Batch launcher: `train_all_domains.py`
   - Code: `training/`
   - Outputs: `unified_pipeline_new_output/`

2. **Inference and validation phase**
   - Entry point: `run_complete_project.py`
   - Inference code: `Pipeline_New/`
   - Validation and dashboard code: `Validator_New/`
   - Inputs: `datasrc/spec_*.json` and `datasrc/universal_index_final.parquet`
   - Outputs: `Pipeline_New/inference_handoff.csv`, `Validator_New/*.json`, and `pranag_dashboard.html`

## Setup

Create and activate a virtual environment, then install dependencies from `requirements.txt`.

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

The downstream router can call Gemini to convert a spec into a surrogate execution DAG. Create a `.env` file in the project root and add:

```env
GEMINI_API_KEY=your_api_key_here
```

If `GEMINI_API_KEY` is missing or the Gemini call fails, `Pipeline_New/dag_router.py` falls back to a local mock router. The pipeline can still run, but routing quality will be simpler and keyword based.

## Phase 1: Training Surrogate Models

Run training for all currently configured domains:

```powershell
python train_all_domains.py
```

Or run the unified trainer directly:

```powershell
python unified_pipeline.py
```

`unified_pipeline.py` trains one domain at a time through `PranagPipeline`. The current batch launcher trains:

- `heat`
- `darcy`
- `stress`
- `arrhenius`
- `biology`
- `logistic`

The output folder also currently contains a trained `reaction_diffusion` surrogate.

### What Training Does

For each domain, the training phase:

1. Uses `training/simulation_generator.py` to resolve the domain into a physics configuration.
2. Uses `training/pinn_factory.py` to create the right PINN model.
3. Uses `training/pinn_trainer.py` to train the PINN with collocation points, boundary constraints, and initial-condition constraints.
4. Generates synthetic parametric data from the trained PINN using Latin Hypercube Sampling.
5. Trains a fast `RandomForestRegressor` surrogate on the generated PINN outputs.
6. Evaluates surrogate accuracy with R2.
7. Computes PRANA-G diagnostic loss components.
8. Saves checkpoints, surrogate models, plots, and metrics.

### Training Outputs

All training artifacts are written to:

```text
unified_pipeline_new_output/
  pinn/        PINN checkpoints and training loss plots
  surrogate/   Surrogate_<domain>.joblib files used by inference
  plots/       Surrogate R2 performance plots
  results/     surrogate_metrics.json
```

The important runtime artifacts for Phase 2 are the `.joblib` files in:

```text
unified_pipeline_new_output/surrogate/
```

`Pipeline_New/inference_pipeline.py` loads these files by name, for example:

```text
Surrogate_heat.joblib
Surrogate_darcy.joblib
Surrogate_stress.joblib
Surrogate_arrhenius.joblib
Surrogate_biology.joblib
Surrogate_logistic.joblib
```

## Phase 2: Inference, Scoring, Validation

Run the full downstream pipeline:

```powershell
python run_complete_project.py
```

To use a specific spec:

```powershell
python run_complete_project.py datasrc\spec_20260604_062219_5ab7e575.json
```

### Phase 2 Inputs

The downstream phase uses two kinds of input from `datasrc/`.

**Prompt-analysis specs**

Files like:

```text
datasrc/spec_20260604_062219_5ab7e575.json
```

These are structured analyses of a user prompt. They include fields such as crop/material target, location, temperature, stress conditions, target traits, retrieved scientific traits, confidence, and supporting basis.

**Universal candidate database**

```text
datasrc/universal_index_final.parquet
```

This parquet file is treated as the universal candidate/entity index. The inference pipeline streams it in batches and expects fields such as:

- `entity_id`
- `name`
- `domain`
- `description`
- optional `tags`
- optional `intrinsic_property_1`
- optional `intrinsic_property_2`
- optional `key_prop_1`, `key_prop_2`, `key_prop_3`

## How Scoring Works

Scoring happens in `Pipeline_New/inference_pipeline.py`.

The high-level process is:

1. `run_complete_project.py` creates an `InferenceEngine`.
2. The engine loads the selected `spec_*.json`.
3. `Pipeline_New/dag_router.py` builds an execution DAG from the spec.
4. The DAG chooses which trained surrogates to run, in what order, with what weights.
5. The inference engine streams `datasrc/universal_index_final.parquet` in batches.
6. Each batch is semantically filtered to focus on relevant candidates.
7. Each selected surrogate scores every candidate that survives filtering.
8. Per-domain scores are combined into a final `viability_score`.
9. Candidates below the threshold are removed, with a top-percent fallback to avoid empty batches.
10. Survivors are written to `Pipeline_New/inference_handoff.csv`.

### DAG Routing

The router converts the spec into a model execution chain. The Gemini prompt in `Pipeline_New/dag_router.py` currently exposes these surrogate schemas:

```text
heat       -> time, space, intrinsic properties, boundary/initial temperature
darcy      -> space, intrinsic properties, boundary/initial pressure
stress     -> time, intrinsic properties, boundary/initial stress
biology    -> time, intrinsic properties, boundary/initial biomass
logistic   -> time, intrinsic properties, boundary/initial growth
arrhenius  -> temperature, intrinsic properties, boundary/initial rate
```

Each DAG node includes:

- `model`: the surrogate to run
- `inputs`: expected input names
- `output_maps_to`: where the model output is written for possible downstream chaining
- `target_value`: the target extracted from the spec
- `initial_value`: the starting condition
- `optimization_goal`: `maximize`, `minimize`, or `target`

The router also emits model weights, target entities, and semantic keywords.

### Semantic Filtering

Before scoring, each parquet batch is filtered using:

1. **Strict target entities** from the DAG, such as `maize`, `corn`, or `zea mays`.
2. **Broader semantic keywords** if strict matching finds nothing.
3. The full batch if no useful semantic filters are available.

Filtering checks candidate `name`, `description`, and optional `tags`.

### Model Input Construction

For each DAG node, `_build_inputs()` creates the exact numeric feature matrix expected by the loaded surrogate.

It handles:

- normalized time and space coordinates
- boundary and initial values from the spec
- intrinsic parquet properties when available
- deterministic hash-based fallback values when data is missing
- regex extraction from descriptions for values like temperature, pH, mass, or yield
- feature padding so the input shape matches `model.n_features_in_`

This is where prompt-level conditions become surrogate-ready numeric inputs.

### Viability Score Calculation

Each surrogate produces a raw prediction. `_calculate_score()` turns that prediction into a `0.0` to `1.0` score using the DAG node's optimization goal.

- `maximize`: higher prediction is better, up to the target.
- `minimize`: lower prediction is better, with score decreasing toward the target.
- `target`: score decays as prediction moves away from the target.

The pipeline uses fixed scalers for consistent target normalization:

```text
temperature / heat -> 1500.0
pressure / stress  -> 1000.0
biomass / growth   -> 10000.0
pH                 -> 14.0
default            -> 100.0
```

The final `viability_score` is a weighted sum of all domain scores, plus a missing-data penalty when key properties are absent.

### Inference Handoff Output

The main scoring output is:

```text
Pipeline_New/inference_handoff.csv
```

It contains columns like:

```text
entity_id,name,domain,heat_score,darcy_score,stress_score,arrhenius_score,biology_score,logistic_score,viability_score
```

Only candidates that pass the physics viability filter, or are included by the top-percent fallback, are written to this handoff file.

## Cross-Domain Validation

After inference, `run_complete_project.py` runs:

```text
Validator_New/cross_domain_validator.py
```

This validator reads:

```text
Pipeline_New/inference_handoff.csv
```

Then it applies final safety checks across whichever domain scores exist. Current hard gates include:

- reject candidates with `arrhenius_score < 0.2`
- reject candidates with `heat_score < 0.2`
- reject candidates with `stress_score < 0.2`

The validator sorts safe candidates by `viability_score` and writes the top 100.

### Validation Outputs

```text
Validator_New/Top_100_Validated_Designs.json
Validator_New/Failed_Designs_Log.json
```

`Top_100_Validated_Designs.json` contains:

- `total_evaluated`
- `total_safe`
- `top_100`
- per-candidate `entity_id`, `name`, `domain`, `viability_score`, and domain scores

`Failed_Designs_Log.json` records rejected candidates and reasons.

## Dashboard

The final HTML dashboard is generated by:

```text
Validator_New/dashboard_generator.py
```

Output:

```text
pranag_dashboard.html
```

The dashboard summarizes:

- total entities scanned
- number of physics survivors
- number removed below threshold
- pass rate
- execution DAG sequence
- top validated designs
- cross-domain failure log

## Project Structure

```text
.
|-- datasrc/
|   |-- spec_*.json                    Prompt-analysis specifications
|   |-- universal_index_final.parquet   Universal candidate/entity index
|   |-- real_data_combined.parquet      Supporting data artifact
|   |-- openlandmap_soil_india.parquet  Supporting soil data artifact
|   `-- data_loader.py                  Supporting/legacy data utility
|
|-- training/
|   |-- simulation_generator.py         Domain/equation to PINN configuration
|   |-- pinn_factory.py                 PINN registry and model factory
|   |-- pinn_trainer.py                 Two-stage PINN trainer
|   |-- surrogate_trainer.py            General surrogate trainer utility
|   `-- sympy_loss_generator.py         Math expression to loss/PDE helpers
|
|-- Pipeline_New/
|   |-- dag_router.py                   Spec to surrogate execution DAG
|   |-- inference_pipeline.py           Batch scoring and handoff generation
|   `-- inference_handoff.csv           Current scored survivor handoff
|
|-- Validator_New/
|   |-- cross_domain_validator.py       Final safety/top-100 validator
|   |-- dashboard_generator.py          HTML dashboard generator
|   |-- Top_100_Validated_Designs.json  Current final top designs
|   `-- Failed_Designs_Log.json         Current rejection log
|
|-- unified_pipeline_new_output/
|   |-- pinn/                           PINN checkpoints and loss plots
|   |-- surrogate/                      Trained surrogate joblib files
|   |-- plots/                          Surrogate R2 plots
|   `-- results/                        Surrogate metrics
|
|-- unified_pipeline.py                 Phase 1 training pipeline
|-- train_all_domains.py                Trains the configured domain set
|-- run_complete_project.py             Phase 2 full inference/validation run
|-- requirements.txt                    Python dependencies
`-- pranag_dashboard.html               Current dashboard output
```

## Common Commands

Install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Train all configured surrogates:

```powershell
python train_all_domains.py
```

Run downstream inference and validation with the default spec:

```powershell
python run_complete_project.py
```

Run downstream inference and validation with a specific spec:

```powershell
python run_complete_project.py datasrc\spec_20260604_062219_5ab7e575.json
```

Run only the inference pipeline:

```powershell
python Pipeline_New\inference_pipeline.py
```

Run only the validator:

```powershell
python Validator_New\cross_domain_validator.py
```

## Notes and Current Behavior

- The inference phase depends on trained surrogate files in `unified_pipeline_new_output/surrogate/`.
- If Gemini is unavailable, routing falls back to local mock rules.
- If the router requests a missing surrogate, the current inference code remaps that model to `logistic`.
- The universal parquet is streamed in batches, so scoring can scale beyond memory.
- The current validation stage is intentionally lightweight: it is a final safety filter and top-100 extractor, not a full retraining or lab-feedback system.
- `datasrc/data_loader.py` exists, but the current primary training path in `unified_pipeline.py` generates PINN-labelled training samples in memory rather than using it as the main training driver.
