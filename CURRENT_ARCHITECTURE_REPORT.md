# PRANA-G Current Architecture Report

This report explains the current PRANA-G codebase in detail for teammates who need to understand how the system is set up, how to run it, what each active file does, how training works, and how downstream scoring/filtering/validation happens.

The important mental model is simple:

```text
Phase 1: Train physics surrogates
    unified_pipeline.py
    training/
    -> unified_pipeline_new_output/

Phase 2: Score candidates from specs
    run_complete_project.py
    Pipeline_New/
    Validator_New/
    datasrc/
    -> inference_handoff.csv
    -> Top_100_Validated_Designs.json
    -> pranag_dashboard.html
```

`Model/`, `Pipelines/`, and `Validators/` are old architecture folders. They are not the active architecture anymore. The current implementation uses `training/`, `Pipeline_New/`, `Validator_New/`, `datasrc/`, `unified_pipeline.py`, and `run_complete_project.py`.

## 1. Getting Started

### 1.1 Prerequisites

You need:

- Python installed.
- A working virtual environment.
- Dependencies installed from `requirements.txt`.
- A Gemini API key if you want real LLM DAG routing.
- The parquet and spec files inside `datasrc/`.
- Trained surrogate `.joblib` files inside `unified_pipeline_new_output/surrogate/` before running downstream inference.

The pipeline can still run without Gemini because `Pipeline_New/dag_router.py` has a mock fallback router. But Gemini routing is better because it can inspect the full spec and choose a more relevant execution DAG.

### 1.2 Create Virtual Environment

From the project root:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks script activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\activate
```

### 1.3 Add Gemini API Key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

`Pipeline_New/dag_router.py` loads this key with `python-dotenv` and uses it to call Gemini. If the key is missing or the API call fails, the code prints a warning and falls back to local mock routing.

### 1.4 Run Training

To train all configured domains:

```powershell
python train_all_domains.py
```

To run the unified pipeline directly:

```powershell
python unified_pipeline.py
```

The current `train_all_domains.py` trains:

```text
heat
darcy
stress
arrhenius
biology
logistic
```

The output directory also currently contains artifacts for:

```text
reaction_diffusion
```

### 1.5 Run Full Downstream Project

To run the full inference plus validation plus dashboard flow:

```powershell
python run_complete_project.py
```

By default, it uses:

```text
datasrc/spec_20260604_062219_5ab7e575.json
```

To pass a specific spec:

```powershell
python run_complete_project.py datasrc\spec_20260604_062219_5ab7e575.json
```

### 1.6 Main Outputs

After training:

```text
unified_pipeline_new_output/
  pinn/
  surrogate/
  plots/
  results/
```

After downstream inference and validation:

```text
Pipeline_New/inference_handoff.csv
Validator_New/Top_100_Validated_Designs.json
Validator_New/Failed_Designs_Log.json
pranag_dashboard.html
```

## 2. What This Project Does

PRANA-G is currently a physics-guided candidate screening system.

The system starts from a structured user specification in `datasrc/spec_*.json`. That spec describes what the user wants, such as a crop, stress condition, temperature, target traits, scientific basis, and confidence score.

The system then looks inside a universal candidate database:

```text
datasrc/universal_index_final.parquet
```

That parquet file acts like a broad generic database of entities. These entities may include biological items, materials, genes, proteins, interactions, or other indexed candidates depending on the dataset. The pipeline filters and scores candidates against the user spec using trained surrogate models.

The final result is a ranked list of designs/candidates that survived physics scoring and cross-domain validation.

## 3. Old Architecture vs Current Architecture

There are older folders in the repo:

```text
Model/
Pipelines/
Validators/
```

These were part of an older architecture. They are not the active architecture described by this report.

The current active architecture is:

```text
training/
Pipeline_New/
Validator_New/
datasrc/
unified_pipeline.py
train_all_domains.py
run_complete_project.py
unified_pipeline_new_output/
```

This distinction matters because the old README described things like old batch simulators, older validators, old model locations, and wider claims that do not match what the current code actually runs.

## 4. High-Level Flow

### 4.1 Complete System Flow

```text
User prompt already converted into spec_*.json
              |
              v
datasrc/spec_*.json
              |
              v
Pipeline_New/dag_router.py
Builds execution DAG:
which surrogates, what order, what weights, what targets
              |
              v
Pipeline_New/inference_pipeline.py
Streams universal_index_final.parquet in batches
Filters candidates semantically
Builds numeric model inputs
Runs trained surrogate models
Computes per-domain scores
Combines into viability_score
              |
              v
Pipeline_New/inference_handoff.csv
              |
              v
Validator_New/cross_domain_validator.py
Applies final hard safety gates
Sorts by viability_score
Extracts top 100
              |
              v
Validator_New/Top_100_Validated_Designs.json
Validator_New/Failed_Designs_Log.json
              |
              v
Validator_New/dashboard_generator.py
              |
              v
pranag_dashboard.html
```

### 4.2 Two Phases

The project has two major phases:

1. **Training phase**

   This phase creates the trained surrogate models. It is run when models need to be trained or refreshed.

2. **Inference and validation phase**

   This phase uses already-trained surrogate models to score candidates from the universal parquet database against a spec.

## 5. Folder and File Responsibilities

### 5.1 Root Files

#### `unified_pipeline.py`

This is the main Phase 1 training pipeline.

It defines:

```python
class PranagPipeline
```

The main method is:

```python
run_end_to_end()
```

For one domain, this method:

1. Builds a PINN.
2. Trains the PINN.
3. Generates synthetic data from the PINN.
4. Trains a Random Forest surrogate.
5. Evaluates the surrogate.
6. Computes PRANA-G diagnostics.
7. Saves model artifacts and metrics.

Important output folders:

```text
unified_pipeline_new_output/pinn/
unified_pipeline_new_output/surrogate/
unified_pipeline_new_output/plots/
unified_pipeline_new_output/results/
```

#### `train_all_domains.py`

This is a small launcher that runs `PranagPipeline` over multiple domains:

```python
domains = ["heat", "darcy", "stress", "arrhenius", "biology", "logistic"]
```

Use this when the team wants to train the current core surrogate set.

#### `run_complete_project.py`

This is the main Phase 2 orchestrator.

It:

1. Adds `Pipeline_New/` and `Validator_New/` to Python import paths.
2. Imports:
   - `InferenceEngine`
   - `CrossDomainValidator`
   - `generate_dashboard`
3. Runs inference using the selected spec.
4. Runs cross-domain validation on the inference handoff.
5. Generates the HTML dashboard.

The core function is:

```python
run_project(spec_path: str)
```

Default spec:

```text
datasrc/spec_20260604_062219_5ab7e575.json
```

#### `requirements.txt`

Dependency list for setting up the Python environment.

Install with:

```powershell
pip install -r requirements.txt
```

#### `pranag_dashboard.html`

Generated output dashboard from the latest downstream run.

### 5.2 `training/`

This folder contains the Phase 1 training code.

#### `training/simulation_generator.py`

This file maps domain names, hints, or equation strings into PINN configurations.

It contains:

- `EquationInfo`
- `SimulationConfig`
- `EQUATION_PATTERNS`
- `LOSS_TEMPLATES`
- `SimulationGenerator`

The generator can create configs from:

```python
from_hint(...)
from_domain(...)
from_equation(...)
```

It knows many scientific equation families, including examples like:

- heat/diffusion
- wave
- Navier-Stokes
- Burgers
- Poisson
- Schrodinger
- reaction diffusion
- Allen-Cahn
- Darcy
- SIR/SEIR
- logistic growth
- Arrhenius

In the current active training flow, `unified_pipeline.py` uses it to determine the input dimension and physics setup for a domain before building/training a PINN.

#### `training/pinn_factory.py`

This file creates PINN models.

It contains:

- `_GenericPINN`
- static PINN classes like `HeatPINN`, `DarcyPINN`, `StressPINN`, `BiologyPINN`, `ArrheniusPINN`, `LogisticPINN`, etc.
- `_BUILTIN_REGISTRY`
- `PINNFactory`

`PINNFactory.create(domain, ...)` is the main method used by `unified_pipeline.py`.

The registry maps a domain string to a model class and default parameters. For example:

```text
heat       -> HeatPINN
darcy      -> DarcyPINN
stress     -> StressPINN
biology    -> BiologyPINN
arrhenius  -> ArrheniusPINN
logistic   -> LogisticPINN
```

The factory also supports dynamic model generation if a requested domain is not directly registered. It tries to use `SimulationGenerator` and `SymPyLossGenerator` for this.

#### `training/pinn_trainer.py`

This file trains a PINN using PyTorch Lightning.

Important class:

```python
PINNLightningModule
```

Important functions:

```python
train_pinn_model(...)
validate_pinn(...)
```

Training uses:

1. Collocation points.
2. Boundary points.
3. Initial-condition points.
4. Physics residual loss from the PINN.
5. Parametric boundary loss.
6. Parametric initial-condition loss.
7. Adam optimizer first.
8. L-BFGS optimizer second.

The key idea is that the PINN learns a physics-constrained function, not just a data fit.

#### `training/surrogate_trainer.py`

This is a general utility for training fast surrogates from PINN-generated data.

It supports:

- generating synthetic data from a trained PINN
- fitting a scikit-learn surrogate
- saving/loading `.joblib` artifacts
- measuring R2 and prediction speed

The current `unified_pipeline.py` trains its surrogate inline with `RandomForestRegressor`, but this file remains useful as a supporting trainer utility.

#### `training/sympy_loss_generator.py`

This file converts mathematical expressions into PyTorch loss functions and can compile PDE-like strings for DeepXDE-style functions.

It supports examples like:

```text
P_escape * impact
max(0, cost - budget)
toxicity + pathogenicity + allergenicity
```

The factory can use this in dynamic model-generation paths.

### 5.3 `datasrc/`

This folder contains specs and data artifacts used by Phase 2.

#### `datasrc/spec_*.json`

These files are prompt-analysis specs. They are not trained models. They are structured descriptions of the user request.

Example fields:

```json
{
  "crop": "maize",
  "location": "unknown",
  "temperature": 45.0,
  "stress_conditions": ["heat stress", "drought stress"],
  "target_traits": ["drought resistance", "high yield"],
  "retrieved_traits": [
    "maize drought tolerance stay-green trait delayed senescence under water stress"
  ],
  "scientific_basis": [
    "HSP70 and HSP90 expression is strongly correlated with survival rates above 42C in cereal crops."
  ],
  "confidence": 0.91
}
```

These specs are used by the DAG router.

#### `datasrc/universal_index_final.parquet`

This is the universal candidate/entity database.

The inference pipeline streams it in batches. It expects columns such as:

```text
entity_id
name
domain
description
tags
intrinsic_property_1
intrinsic_property_2
key_prop_1
key_prop_2
key_prop_3
```

Some columns are optional. If properties are missing, the inference pipeline uses deterministic fallback values and applies missing-data penalties when relevant.

#### `datasrc/data_loader.py`

This is a supporting data utility. It can parse prompt JSON and build feature matrices.

Current note:

The active training flow in `unified_pipeline.py` mostly bypasses this loader and generates PINN-labelled samples in memory. So this file should be considered supporting/legacy utility code unless the team intentionally reconnects it to the main training path.

### 5.4 `Pipeline_New/`

This is the active Phase 2 inference code.

#### `Pipeline_New/dag_router.py`

This file converts a spec into a surrogate execution DAG.

The main class:

```python
class DAGRouter
```

Main method:

```python
build_dag(spec_path: str) -> dict
```

It has two routing modes:

1. **Real LLM routing**

   Uses Gemini with the system prompt defined in `LLM_SYSTEM_PROMPT`.

2. **Mock routing**

   Uses local keyword rules when Gemini is not available.

The router decides:

- which surrogate models to run
- the order to run them in
- the model inputs
- target values
- initial values
- optimization goals
- per-model weights
- target entities
- semantic keywords

#### `Pipeline_New/inference_pipeline.py`

This is where the main scoring happens.

The main class:

```python
class InferenceEngine
```

Important constants:

```python
SURROGATE_DIR = ../unified_pipeline_new_output/surrogate
PARQUET_PATH = ../datasrc/universal_index_final.parquet
OUTPUT_PATH = Pipeline_New/inference_handoff.csv
```

Important methods:

```python
_load_surrogate(domain)
_build_inputs(df, spec, n_features, node)
_calculate_score(pred, spec, target_value, optimization_goal, scaler)
run(spec_path)
```

`run(spec_path)` is the main inference function.

#### `Pipeline_New/inference_handoff.csv`

This is the scored survivor file created by `InferenceEngine`.

Example columns:

```text
entity_id
name
domain
heat_score
darcy_score
stress_score
arrhenius_score
biology_score
logistic_score
viability_score
```

This file is then consumed by `Validator_New/cross_domain_validator.py`.

### 5.5 `Validator_New/`

This is the active final validation and dashboard code.

#### `Validator_New/cross_domain_validator.py`

This file reads `Pipeline_New/inference_handoff.csv` and applies final safety gates.

Main class:

```python
class CrossDomainValidator
```

Main method:

```python
validate()
```

The current hard safety checks are:

```text
arrhenius_score < 0.2 -> reject
heat_score      < 0.2 -> reject
stress_score    < 0.2 -> reject
```

It writes:

```text
Validator_New/Top_100_Validated_Designs.json
Validator_New/Failed_Designs_Log.json
```

#### `Validator_New/dashboard_generator.py`

This creates:

```text
pranag_dashboard.html
```

It reads metrics from the run plus:

```text
Validator_New/Top_100_Validated_Designs.json
Validator_New/Failed_Designs_Log.json
```

The dashboard shows:

- total scanned entities
- number of physics survivors
- number deleted below threshold
- pass rate
- execution DAG sequence
- top validated designs
- failure log

## 6. Phase 1 Deep Dive: Training

### 6.1 Training Goal

The goal of Phase 1 is to train fast surrogate models that approximate physics-aware PINN outputs.

PINNs are more physically meaningful but slower. Surrogates are much faster and are used in Phase 2 to score many candidate entities from parquet.

In other words:

```text
PINN = teacher
surrogate = fast student
```

### 6.2 Domain Setup

For a domain like `heat`, `darcy`, or `arrhenius`, `unified_pipeline.py` does:

```python
cfg = SimulationGenerator().from_hint(self.domain)
actual_input_dim = len(cfg.equation_info.independent) + 2
```

The `+ 2` is important.

The model input includes:

```text
physical dimensions + 2 parametric targets
```

Those two parametric targets are usually interpreted as:

```text
boundary target
initial-condition target
```

This lets the model learn a parametric response rather than one fixed simulation.

### 6.3 PINN Creation

The pipeline creates a PINN:

```python
pinn_model = self.factory.create(
    self.domain,
    input_dim=actual_input_dim,
    hidden_dim=128,
    num_layers=6,
    dynamic=True
)
```

Then it sets:

```python
pinn_model.base_dim = len(cfg.equation_info.independent)
```

This matters because the trainer needs to know which input columns are physical dimensions and which input columns are parametric target values.

### 6.4 PINN Training Data

`training/pinn_trainer.py` generates:

1. **Interior collocation points**

   These are random points across the simulation domain where the physics residual is enforced.

2. **Boundary points**

   These are points forced to physical boundaries, such as spatial edge `x = -1` or `x = 1`.

3. **Initial-condition points**

   These are points forced to the start of the simulation, usually `t = 0`.

These are concatenated into a training dataset.

### 6.5 PINN Losses

The trainer computes:

#### Physics loss

Each PINN implements:

```python
physics_loss(x)
```

This uses autograd derivatives to calculate equation residuals. For example:

- Heat PINN enforces heat equation behavior.
- Darcy PINN enforces porous-flow behavior.
- Logistic PINN enforces growth equation behavior.
- Arrhenius PINN enforces reaction-rate behavior.

#### Boundary loss

If the input contains parametric boundary targets, the trainer enforces that predictions at physical boundaries match the boundary parameter.

Conceptually:

```text
when x is at the boundary:
    predicted output should match boundary_target
```

#### Initial-condition loss

If the input contains an initial-condition parameter, the trainer enforces that predictions at `t = 0` match that initial target.

Conceptually:

```text
when t = 0:
    predicted output should match initial_value
```

#### Adaptive constraint weighting

The trainer dynamically computes weights for boundary and initial-condition losses from the ratio between physics loss and constraint loss. This avoids hardcoding one fixed weight for all domains.

### 6.6 Two-Stage Optimization

Training uses two optimizers:

1. **Adam**

   Used first as a broad exploration optimizer.

2. **L-BFGS**

   Used second as a more precise optimizer over the full dataset.

The training code calls this:

```text
Stage 1: Adam
Stage 2: L-BFGS
```

### 6.7 PINN Validation

After training, `validate_pinn()` samples unseen points and computes physics residual. It prints whether the model passed a simple physics-residual threshold.

### 6.8 Synthetic Data Generation

After the PINN is trained, `unified_pipeline.py` generates synthetic data in memory.

It uses Latin Hypercube Sampling:

```python
sampler = qmc.LatinHypercube(d=actual_input_dim)
sample = sampler.random(n=target_total_points)
```

The data is scaled:

```text
spatial/parametric dimensions -> [-1, 1]
time dimension                -> [0, 1]
```

Then the trained PINN predicts outputs for these sampled inputs.

So the training dataset for the surrogate is:

```text
X = sampled parametric inputs
y = trained PINN outputs
```

### 6.9 Surrogate Training

The surrogate is:

```python
RandomForestRegressor
```

The pipeline splits PINN-generated data into train/test sets:

```python
X_train, X_test, y_train, y_test = train_test_split(...)
```

Then fits:

```python
surrogate.fit(X_train, y_train)
```

Evaluation metric:

```python
r2_score(y_test, y_pred)
```

The current metrics file shows high R2 for the trained surrogates.

### 6.10 PRANA-G Diagnostics During Training

After surrogate evaluation, `unified_pipeline.py` computes diagnostic losses using:

```python
create_cross_domain_loss_generator()
```

It builds an `inputs_dict` with components such as:

```text
physics
data
boundary
biology
ecology
economics
safety
```

Then it writes a breakdown to:

```text
unified_pipeline_new_output/results/surrogate_metrics.json
```

These diagnostics are not the same thing as Phase 2 candidate scoring. They are training-time quality/constraint diagnostics for the surrogate.

### 6.11 Training Artifacts

For each trained domain, the pipeline writes:

```text
unified_pipeline_new_output/pinn/Parametric_<Domain>PINN.ckpt
unified_pipeline_new_output/pinn/Parametric_<Domain>PINN_loss_history.png
unified_pipeline_new_output/surrogate/Surrogate_<domain>.joblib
unified_pipeline_new_output/plots/surrogate_<domain>_r2.png
unified_pipeline_new_output/results/surrogate_metrics.json
```

## 7. Phase 2 Deep Dive: DAG Routing

### 7.1 Why a DAG?

Different specs need different physics checks.

Example:

```text
"maize under heat and drought stress with high yield"
```

This may require:

- heat model for thermal viability
- darcy model for water/pressure/retention behavior
- logistic or biology model for growth/yield

The DAG lets the system decide:

```text
which models should run
in what order
how strongly each should influence final viability
what output from one model should feed into the next
```

### 7.2 DAG Output Format

The router returns something like:

```json
{
  "execution_chain": [
    {
      "model": "heat",
      "inputs": [
        "time",
        "space",
        "intrinsic_property_1",
        "intrinsic_property_2",
        "boundary_temperature",
        "initial_temperature"
      ],
      "output_maps_to": "temperature",
      "target_value": 45.0,
      "initial_value": 25.0,
      "optimization_goal": "target"
    },
    {
      "model": "darcy",
      "inputs": [
        "space_x",
        "space_y",
        "intrinsic_property_1",
        "intrinsic_property_2",
        "boundary_pressure",
        "initial_pressure"
      ],
      "output_maps_to": "water_retention",
      "target_value": 0.0,
      "initial_value": -999.0,
      "optimization_goal": "minimize"
    },
    {
      "model": "logistic",
      "inputs": [
        "time",
        "intrinsic_property_1",
        "intrinsic_property_2",
        "boundary_growth",
        "initial_growth"
      ],
      "output_maps_to": "biomass",
      "target_value": 1000.0,
      "initial_value": 10.0,
      "optimization_goal": "maximize"
    }
  ],
  "weights": {
    "heat": 0.4,
    "darcy": 0.3,
    "logistic": 0.3
  },
  "target_entities": ["maize", "corn", "zea mays"],
  "semantic_keywords": ["crop", "plant", "agriculture", "yield", "botany"],
  "context": "Generated via Gemini API."
}
```

### 7.3 Available Surrogate Schemas

The Gemini prompt currently exposes this model schema:

```text
heat:
  time, space, intrinsic_property_1, intrinsic_property_2,
  boundary_temperature, initial_temperature

darcy:
  space_x, space_y, intrinsic_property_1, intrinsic_property_2,
  boundary_pressure, initial_pressure

stress:
  time, intrinsic_property_1, intrinsic_property_2,
  boundary_stress, initial_stress

biology:
  time, intrinsic_property_1, intrinsic_property_2,
  boundary_biomass, initial_biomass

logistic:
  time, intrinsic_property_1, intrinsic_property_2,
  boundary_growth, initial_growth

arrhenius:
  temperature, intrinsic_property_1, intrinsic_property_2,
  boundary_rate, initial_rate
```

The prompt explicitly tells Gemini not to invent unsupported model names.

### 7.4 Mock Router

If Gemini cannot be used, `_mock_llm_routing()` parses text from:

```text
stress_conditions
target_traits
```

It uses simple keyword rules:

- heat/temperature -> add `heat`
- drought/flood/disease -> add `darcy`
- yield/growth -> add `logistic`
- salinity/pH/soil -> add `gray_scott`

If nothing matches, it falls back to:

```text
heat + logistic
```

### 7.5 Anti-Hallucination Mapping

The real Gemini path includes a guard. If Gemini returns an unsupported model name, the router maps it to something supported:

- stress-like names -> `stress`
- biology/growth/yield-like names -> `biology`
- otherwise -> `logistic`

Then `Pipeline_New/inference_pipeline.py` also checks whether the requested surrogate `.joblib` exists. If it does not exist, current behavior remaps that node to:

```text
logistic
```

## 8. Phase 2 Deep Dive: Candidate Filtering

### 8.1 Why Filtering Exists

The universal parquet may contain many candidates. Not all are relevant to the user spec.

For example, if the spec is about maize, the pipeline should focus on maize-related entities before scoring.

### 8.2 Batch Streaming

The inference engine opens:

```python
pq.ParquetFile(self.PARQUET_PATH)
```

Then streams:

```python
parquet_file.iter_batches(batch_size=50000)
```

This means the system does not need to load the entire parquet into memory at once.

### 8.3 Strict Target Entity Filtering

The DAG can include:

```json
"target_entities": ["maize", "corn", "zea mays"]
```

The engine builds a regex and searches:

```text
name
description
tags
```

This is strict filtering. It tries to isolate candidates specifically related to the target entity.

### 8.4 Semantic Keyword Fallback

If strict filtering returns zero rows, the engine uses:

```json
"semantic_keywords": ["crop", "plant", "seed", "botany", "agriculture"]
```

This broader filter helps avoid empty results when the target entity names are not present.

### 8.5 Full-Batch Fallback

If there are no useful target entities or semantic keywords, the code falls back to scoring the full batch.

## 9. Phase 2 Deep Dive: Model Input Construction

### 9.1 Why Input Construction Is Needed

Each surrogate was trained with a specific number of numeric input features.

The parquet rows and specs are not directly in that numeric shape. `_build_inputs()` transforms each candidate row plus the spec into the exact matrix the surrogate expects.

### 9.2 Feature Count

The engine gets expected feature count from:

```python
n_features = getattr(model, "n_features_in_", 3)
```

Then `_build_inputs()` creates a matrix:

```text
number_of_candidates x n_features
```

### 9.3 Time and Space Inputs

For requested inputs like:

```text
time
t
time_days
```

The engine uses a normalized value:

```text
0.5
```

For space-like inputs:

```text
space
x
space_x
space_y
depth
```

The engine uses:

```text
0.0
```

This matches the normalized coordinate style used during training.

### 9.4 Boundary and Initial Inputs

For inputs like:

```text
boundary_temperature
initial_temperature
boundary_pressure
initial_pressure
boundary_growth
initial_growth
```

The engine reads:

```text
target_value
initial_value
```

from the DAG node.

Then it scales values into normalized ranges.

Examples:

```text
temperature -> divide by 1500.0
pressure    -> divide by 1000.0
biomass     -> divide by 10000.0
pH          -> divide by 14.0
default     -> divide by 100.0
```

If an initial value is missing, the code uses domain-specific defaults.

### 9.5 Intrinsic Properties

For:

```text
intrinsic_property_1
intrinsic_property_2
```

The engine first tries to use actual parquet columns.

If those columns are absent, it generates deterministic fallback values using a hash of:

```text
entity_id + feature_name
```

This is important: fallback values are deterministic, not random every run.

### 9.6 Regex Extraction From Description

For some requested features, the engine tries to extract numeric values from `description`.

Examples:

```text
temperature -> looks for temp/temperature followed by a number
pH          -> looks for pH followed by a number
mass/yield  -> looks for mass/weight/yield followed by a number
```

If extraction fails, the deterministic hash fallback is used.

### 9.7 Chained Outputs

If a previous model wrote a value into the dataframe, `_build_inputs()` can use that column for a later model.

Example:

```text
heat output_maps_to -> temperature
later arrhenius input -> temperature
```

This is how one model's output can become another model's input.

### 9.8 Padding

If fewer columns are built than the surrogate expects, the engine pads the remaining inputs with zeros:

```python
while len(cols) < n_features:
    cols.append(np.zeros(n, dtype=np.float32))
```

This prevents shape crashes.

## 10. Phase 2 Deep Dive: Scoring

### 10.1 Raw Prediction

Each surrogate receives its input matrix:

```python
raw_pred = model.predict(X)
```

If the prediction has multiple columns, the engine uses the first one:

```python
raw_pred = raw_pred[:, 0]
```

### 10.2 Chaining

After prediction, the engine writes:

```python
df[maps_to] = raw_pred
```

So if a node says:

```json
"output_maps_to": "temperature"
```

then later nodes can read `temperature` as an input feature.

### 10.3 Score Calculation

The method:

```python
_calculate_score(pred, spec, target_value, optimization_goal, scaler)
```

turns raw surrogate predictions into a normalized score from `0.0` to `1.0`.

The scoring behavior depends on:

```text
optimization_goal
```

#### Maximize

Used when higher is better.

Example:

```text
yield
biomass
growth
```

The score approaches `1.0` when prediction reaches or exceeds the target.

#### Minimize

Used when lower is better.

Example:

```text
stress
hazard
failure
pressure if pressure is harmful
```

The score is high when prediction is low and falls as prediction approaches/exceeds the target.

#### Target

Used when the best result is near a specific value.

Example:

```text
temperature should be around 45C
pH should be around 7
```

The code uses exponential decay:

```text
score = exp(-abs(prediction - target) / target)
```

after normalization.

### 10.4 Scalers

The engine chooses scalers based on the output name:

```text
temp / heat       -> 1500.0
pressure / stress -> 1000.0
biomass / growth  -> 10000.0
pH                -> 14.0
default           -> 100.0
```

This keeps targets and predictions in a comparable normalized range.

### 10.5 Per-Domain Scores

Each domain score is written into `scores_df`:

```text
heat_score
darcy_score
stress_score
arrhenius_score
biology_score
logistic_score
```

The exact columns depend on which DAG nodes ran.

### 10.6 Final Viability Score

The final score is:

```text
weighted sum of domain scores + missing data penalty
```

The DAG provides weights:

```json
"weights": {
  "heat": 0.4,
  "darcy": 0.3,
  "logistic": 0.3
}
```

The engine normalizes the weights so they sum to `1.0`.

Then:

```python
viability_score += y_score * normalized_weights.get(domain, 0.0)
```

Finally:

```python
scores_df["viability_score"] = np.clip(viability_score, 0.0, 1.0)
```

### 10.7 Missing Data Penalty

The code checks optional columns:

```text
key_prop_1
key_prop_2
key_prop_3
```

If two or more are missing, it applies:

```text
-0.20
```

to the final viability score.

### 10.8 Survivor Selection

The primary threshold is:

```text
viability_score >= 0.70
```

These candidates survive.

If too few survive, the pipeline keeps the top 5 percent of that batch:

```python
min_survivors = int(n * 0.05)
```

This prevents a batch from producing zero useful candidates.

### 10.9 Handoff CSV

Survivors are appended to:

```text
Pipeline_New/inference_handoff.csv
```

This is the handoff between scoring and validation.

## 11. Phase 2 Deep Dive: Validation

### 11.1 Validator Purpose

`Validator_New/cross_domain_validator.py` is a final safety and ranking pass.

It does not retrain models. It does not rerun the surrogates. It reads the already-scored candidates from:

```text
Pipeline_New/inference_handoff.csv
```

### 11.2 Safety Gates

Current hard gates:

```text
arrhenius_score < 0.2 -> critical chemistry hazard
heat_score      < 0.2 -> critical thermal hazard
stress_score    < 0.2 -> critical stress/structural failure
```

Only columns that exist are checked.

So if a DAG did not run `arrhenius`, then `arrhenius_score` will not exist and that check is skipped.

### 11.3 Failed Designs Log

Rejected candidates are written to:

```text
Validator_New/Failed_Designs_Log.json
```

Each record includes:

```text
entity_id
name
reason
```

### 11.4 Top 100

Safe candidates are sorted by:

```text
viability_score descending
```

The validator keeps:

```text
top 100
```

and writes:

```text
Validator_New/Top_100_Validated_Designs.json
```

Each top candidate includes:

```text
entity_id
name
domain
viability_score
domain_scores
```

## 12. Dashboard

`Validator_New/dashboard_generator.py` creates a static HTML dashboard:

```text
pranag_dashboard.html
```

It uses metrics from `run_complete_project.py`, plus:

```text
Validator_New/Top_100_Validated_Designs.json
Validator_New/Failed_Designs_Log.json
```

The dashboard displays:

- total entities scanned
- physics survivors
- deleted count
- pass rate
- execution DAG sequence
- top validated designs
- cross-domain failure log

This is intended as a human-readable summary of a run.

## 13. Example Walkthrough: Maize Heat and Drought

Suppose the spec says:

```json
{
  "crop": "maize",
  "temperature": 45.0,
  "stress_conditions": ["heat stress", "drought stress"],
  "target_traits": ["drought resistance", "high yield"]
}
```

### 13.1 Routing

The router may build a DAG like:

```text
heat -> darcy -> logistic
```

Reasoning:

- heat stress requires thermal scoring
- drought stress requires water/pressure/retention-related scoring
- high yield requires growth/biomass scoring

### 13.2 Filtering

The engine first searches candidates for:

```text
maize
corn
zea mays
```

If nothing is found, it searches broader keywords:

```text
crop
plant
seed
botany
agriculture
```

### 13.3 Input Building

For each candidate:

```text
heat input:
  time = 0.5
  space = 0.0
  intrinsic properties from parquet or hash fallback
  boundary_temperature = 45 / 1500
  initial_temperature = 25 / 1500

darcy input:
  space_x = 0.0
  space_y = 0.0
  intrinsic properties
  pressure fields from DAG/defaults

logistic input:
  time = 0.5
  intrinsic properties
  boundary_growth and initial_growth
```

### 13.4 Scoring

Each model predicts a raw value.

Then:

```text
heat prediction -> heat_score
darcy prediction -> darcy_score
logistic prediction -> logistic_score
```

The final viability might be:

```text
viability_score =
  0.4 * heat_score
  + 0.3 * darcy_score
  + 0.3 * logistic_score
  - missing_data_penalty
```

### 13.5 Validation

If `heat_score < 0.2`, the candidate is rejected as a thermal hazard.

If it passes all active hard gates, it can appear in the top 100.

## 14. Example Walkthrough: Rice Flood Stress

Spec:

```json
{
  "crop": "rice",
  "location": "Assam",
  "temperature": 25.0,
  "stress_conditions": ["flood stress"],
  "target_traits": ["high yield", "disease resistance"]
}
```

Expected routing may include:

```text
darcy -> logistic
```

Because:

- flood stress maps to water/pressure/retention behavior
- high yield maps to growth/biomass behavior

Candidate filtering will first look for:

```text
rice
```

Then score rice-related rows from the parquet.

## 15. Example Walkthrough: Rice Salinity Stress

Spec:

```json
{
  "crop": "rice",
  "location": "Gujarat",
  "stress_conditions": ["salinity stress"],
  "target_traits": ["disease resistance"]
}
```

The mock router may map salinity/pH/soil language to `gray_scott`, but the active trained surrogates currently include `reaction_diffusion`, not necessarily `gray_scott` as a `.joblib` under that exact name. If the requested surrogate file is missing, the current inference code remaps to:

```text
logistic
```

This is a current behavior to be aware of. For better salinity chemistry routing, the team may want to align router model names with the exact trained surrogate filenames.

## 16. Data Contracts

### 16.1 Spec Contract

The router benefits from specs containing:

```text
crop or material
location
temperature
humidity
rainfall
soil_type
stress_conditions
target_traits
retrieved_traits
scientific_basis
constraints
confidence
```

Not every field is mandatory, but richer specs give better routing.

### 16.2 Parquet Contract

The inference pipeline expects:

```text
entity_id
name
domain
description
```

It can also use:

```text
tags
intrinsic_property_1
intrinsic_property_2
key_prop_1
key_prop_2
key_prop_3
```

Missing optional columns do not necessarily crash the pipeline. The code has fallbacks. But missing optional columns can reduce score confidence and apply penalties.

### 16.3 Surrogate Contract

Surrogate files must be named:

```text
Surrogate_<domain>.joblib
```

and stored in:

```text
unified_pipeline_new_output/surrogate/
```

For example:

```text
Surrogate_heat.joblib
Surrogate_darcy.joblib
Surrogate_stress.joblib
Surrogate_arrhenius.joblib
Surrogate_biology.joblib
Surrogate_logistic.joblib
```

The router's model names must match these domain names, otherwise inference may remap to `logistic`.

## 17. Current Strengths

The current system has several good properties:

- Clear separation between training and inference.
- Fast downstream scoring through `.joblib` surrogates.
- Batch streaming for large parquet files.
- DAG-based model routing.
- Semantic pre-filtering before scoring.
- Deterministic fallback feature generation.
- Chained model outputs.
- Weighted multi-domain viability scoring.
- Final safety gates and top-100 extraction.
- Dashboard output for teammates and reviewers.

## 18. Current Limitations and Caveats

### 18.1 Router and Surrogate Name Alignment

The router exposes `gray_scott`, but the trained file currently visible is:

```text
Surrogate_reaction_diffusion.joblib
```

If `gray_scott` is requested and no matching file exists, the inference engine remaps it to `logistic`.

### 18.2 Missing Surrogate Behavior

The code currently remaps missing requested models to `logistic`. This keeps the pipeline running, but it may hide coverage gaps.

A stricter future behavior could be:

```text
fail clearly if required surrogate is missing
```

or:

```text
record missing surrogate in dashboard/report
```

### 18.3 Simplified Validator

The validator currently checks only low-score hazards for:

```text
arrhenius
heat
stress
```

It is a useful safety filter, but not a complete scientific validation layer.

### 18.4 Data Loader Is Not Main Training Driver

`datasrc/data_loader.py` contains useful feature-building logic, but the current main training path generates synthetic PINN-labelled samples in memory.

### 18.5 Scoring Depends on Normalization Choices

The fixed scalers are important:

```text
temperature / heat -> 1500.0
pressure / stress  -> 1000.0
biomass / growth   -> 10000.0
pH                 -> 14.0
default            -> 100.0
```

If these are not aligned with how surrogates were trained, scores may be skewed.

## 19. Recommended Team Workflow

### 19.1 When Adding a New Domain

1. Add or confirm domain support in `training/simulation_generator.py`.
2. Add a PINN or registry entry in `training/pinn_factory.py`.
3. Train it through `unified_pipeline.py`.
4. Confirm `Surrogate_<domain>.joblib` is created.
5. Add the domain schema to `Pipeline_New/dag_router.py`.
6. Make sure router model name exactly matches the surrogate filename domain.
7. Add scoring/validation rules if needed.
8. Run `run_complete_project.py` on representative specs.

### 19.2 When Adding a New Spec

1. Put the new spec in `datasrc/`.
2. Make sure it includes useful fields such as target crop/material, stress conditions, and target traits.
3. Run:

   ```powershell
   python run_complete_project.py datasrc\your_spec.json
   ```

4. Inspect:

   ```text
   Pipeline_New/inference_handoff.csv
   Validator_New/Top_100_Validated_Designs.json
   pranag_dashboard.html
   ```

### 19.3 When Retraining

1. Activate venv.
2. Run:

   ```powershell
   python train_all_domains.py
   ```

3. Check:

   ```text
   unified_pipeline_new_output/results/surrogate_metrics.json
   unified_pipeline_new_output/plots/
   unified_pipeline_new_output/surrogate/
   ```

4. Then rerun downstream inference.

## 20. Quick Reference

### Main Commands

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

python train_all_domains.py
python run_complete_project.py
python run_complete_project.py datasrc\spec_20260604_062219_5ab7e575.json
```

### Main Inputs

```text
datasrc/spec_*.json
datasrc/universal_index_final.parquet
unified_pipeline_new_output/surrogate/Surrogate_<domain>.joblib
```

### Main Outputs

```text
unified_pipeline_new_output/results/surrogate_metrics.json
Pipeline_New/inference_handoff.csv
Validator_New/Top_100_Validated_Designs.json
Validator_New/Failed_Designs_Log.json
pranag_dashboard.html
```

### Active Code

```text
training/
Pipeline_New/
Validator_New/
datasrc/
unified_pipeline.py
train_all_domains.py
run_complete_project.py
```

### Old Architecture

```text
Model/
Pipelines/
Validators/
```

These folders are from the old architecture and are not used by the current flow documented in this report.

## 21. Final Summary

The current PRANA-G system is a two-phase physics-guided screening pipeline.

First, `unified_pipeline.py` trains domain-specific PINNs, uses them to generate synthetic parametric data, and fits fast Random Forest surrogates. These surrogate models are saved under `unified_pipeline_new_output/surrogate/`.

Second, `run_complete_project.py` uses a prompt-analysis spec from `datasrc/`, routes it through `Pipeline_New/dag_router.py`, streams candidate entities from `datasrc/universal_index_final.parquet`, scores candidates with trained surrogates in `Pipeline_New/inference_pipeline.py`, validates survivors with `Validator_New/cross_domain_validator.py`, and generates a final dashboard with `Validator_New/dashboard_generator.py`.

The most important operational idea for teammates is:

```text
Train surrogates first.
Then use specs plus universal parquet to route, score, validate, and dashboard candidates.
```

