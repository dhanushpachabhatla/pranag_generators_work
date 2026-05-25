"""
batch_simulator.py  —  ARYAN  (Task 2 + 3)
Runs traits through Srikar's PINN models in batches.
Output: trait_id + viability_score for every trait.
Target: 100K traits < 30 min, 1M traits < 4 hours.

HOW TO PLUG IN SRIKAR'S MODELS:
  from batch_simulator import BatchSimulator
  sim = BatchSimulator(model_dir="path/to/srikar/models/")
  results = sim.run_all()
"""

import time
import os
import json
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass, field, asdict
from datetime import datetime
from data_loader import DataLoader

try:
    import joblib
except Exception:
    joblib = None


CHECKPOINT_FILE = "checkpoints/batch_checkpoint.json"
RESULTS_FILE    = "results/simulation_results.parquet"
SCORE_THRESHOLD = 0.70   # below this → filtered out before validation


@dataclass
class TraitResult:
    trait_id:        str
    entity_type:     str
    viability_score: float
    biology_score:   float
    physics_score:   float
    material_score:  float
    chemistry_score: float
    passed_filter:   bool
    source:          str = ""
    simulated_at:    str = field(default_factory=lambda: datetime.now().isoformat())


# ── SRIKAR MODEL INTERFACE ────────────────────────────────────────────────────

class SrikarModelInterface:
    """
    Interface to Srikar's PINN models.
    When Srikar hands over his models, replace the _mock_* methods
    with real model.predict() calls.

    HOW TO PLUG IN:
        1. Replace `self.models` loading with:
               import torch
               self.heat_model    = torch.load(model_dir + "heat_pinn.pt")
               self.stress_model  = torch.load(model_dir + "stress_pinn.pt")
               self.growth_model  = torch.load(model_dir + "growth_pinn.pt")
               self.biology_model = torch.load(model_dir + "biology_pinn.pt")
               self.chem_model    = torch.load(model_dir + "chemistry_pinn.pt")
        2. Replace each _predict_* method with real model inference.
    """

    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or self._default_model_dir()
        self.surrogate_dir = self._default_surrogate_dir()
        self.models_loaded = False
        self.surrogates_loaded = False
        self.heat_model = None
        self.stress_model = None
        self.growth_model = None
        self.biology_model = None
        self.chem_model = None
        self.surrogates = {}
        self._try_load_models()

    def _default_model_dir(self) -> str:
        here = os.path.dirname(__file__)
        return os.path.normpath(os.path.join(here, "..", "Model", "outputs", "models"))

    def _default_surrogate_dir(self) -> str:
        here = os.path.dirname(__file__)
        return os.path.normpath(os.path.join(here, "..", "Model", "outputs", "surrogates"))

    def _try_load_models(self):
        """Load Srikar models/surrogates from Model/outputs folders."""
        self._try_load_surrogates()
        try:
            from pathlib import Path
            import sys

            model_root = Path(__file__).resolve().parents[1] / "Model"
            if str(model_root) not in sys.path:
                sys.path.insert(0, str(model_root))
            from models.physics_models import HeatPINN, StressPINN, GrowthPINN, BiologyPINN, ChemistryPINN

            required = {
                "heat": os.path.join(self.model_dir, "heat_pinn.pt"),
                "stress": os.path.join(self.model_dir, "stress_pinn.pt"),
                "growth": os.path.join(self.model_dir, "growth_pinn.pt"),
                "biology": os.path.join(self.model_dir, "biology_pinn.pt"),
                "chemistry": os.path.join(self.model_dir, "chemistry_pinn.pt"),
            }
            if not all(os.path.exists(p) for p in required.values()):
                print(f"⚠️  Missing one or more PINN checkpoints in: {self.model_dir}")
                return

            self.heat_model = HeatPINN()
            self.stress_model = StressPINN()
            self.growth_model = GrowthPINN()
            self.biology_model = BiologyPINN()
            self.chem_model = ChemistryPINN()

            self._load_checkpoint(self.heat_model, required["heat"])
            self._load_checkpoint(self.stress_model, required["stress"])
            self._load_checkpoint(self.growth_model, required["growth"])
            self._load_checkpoint(self.biology_model, required["biology"])
            self._load_checkpoint(self.chem_model, required["chemistry"])

            for m in [self.heat_model, self.stress_model, self.growth_model, self.biology_model, self.chem_model]:
                m.eval()
            self.models_loaded = True
            print(f"Loaded real PINN checkpoints from: {self.model_dir}")
        except Exception as e:
            print(f"Could not load PINN checkpoints ({e})")

    def _load_checkpoint(self, model: torch.nn.Module, ckpt_path: str):
        data = torch.load(ckpt_path, map_location="cpu")
        state = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            # Old checkpoints (HeatFNO, DeepONet, Transformer) have different key layouts.
            # Load what matches; missing keys stay at xavier-initialised values.
            incompatible = model.load_state_dict(state, strict=False)
            if incompatible.missing_keys:
                print(f"  ⚠️  Partial load for {type(model).__name__}: "
                      f"{len(incompatible.missing_keys)} missing / "
                      f"{len(incompatible.unexpected_keys)} unexpected keys "
                      f"(old checkpoint format — surrogates preferred)")

    def _try_load_surrogates(self):
        if joblib is None:
            return
        if not os.path.isdir(self.surrogate_dir):
            return
        names = ["heat", "stress", "growth", "biology", "chemistry"]
        loaded = 0
        for n in names:
            p = os.path.join(self.surrogate_dir, f"{n}.joblib")
            if not os.path.exists(p):
                continue
            try:
                payload = joblib.load(p)
                self.surrogates[n] = payload.get("surrogate")
                loaded += 1
            except Exception:
                continue
        self.surrogates_loaded = loaded >= 5
        if self.surrogates_loaded:
            print(f"Loaded 5 surrogate models from: {self.surrogate_dir}")

    # ── Data enrichment ───────────────────────────────────────────────────────

    @staticmethod
    def _kp(row: dict, key: str, default: float = 0.0) -> float:
        """Parse a key_prop_x string safely, returning default on failure/NaN."""
        try:
            v = float(str(row.get(key, default)).strip())
            return default if (not np.isfinite(v)) else v
        except Exception:
            return default

    def _enrich_row(self, row: dict) -> dict:
        """
        Derive physics features from universal_index_final.parquet key_props.

        Per-domain extraction rules (from real data structure):
          chemistry   : key_prop_1=MW(Da), key_prop_2=LogP, key_prop_3=rot_bonds
          materials   : key_prop_1=band_gap(eV), key_prop_2=formation_energy(eV/atom),
                        key_prop_3=density(g/cm3)
          biology     : key_prop_1=chromosome/gene_len, key_prop_2=organism,
                        key_prop_3=(empty)
          physics     : key_prop_1=material_id, key_prop_2=category, key_prop_3=db
          environment : key_prop_1=pH, key_prop_2=sand%, key_prop_3=organic_carbon(g/kg)

        Falls back to deterministic RNG (entity_id hash) for any value not
        extractable from key_props.
        """
        import hashlib
        entity_id = str(row.get("entity_id", row.get("trait_id", row.get("design_id", "default"))))
        seed = int(hashlib.md5(entity_id.encode()).hexdigest()[:8], 16)
        rng  = np.random.default_rng(seed)

        domain = str(row.get("domain", "general")).lower()
        tags   = str(row.get("tags",   "")).lower()
        src    = str(row.get("source", "")).lower()
        desc   = str(row.get("description", "")).lower()
        enriched = dict(row)

        def _set(k, lo, hi):
            """Write key only when absent or NaN. Guards NaN bounds."""
            v = enriched.get(k)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                    lo, hi = 0.2, 0.8
                enriched[k] = float(rng.uniform(lo, hi))

        def _put(k, val, lo_clip=None, hi_clip=None):
            """Write a derived value (overrides if it's NaN/missing)."""
            v = enriched.get(k)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                if lo_clip is not None:
                    val = max(lo_clip, val)
                if hi_clip is not None:
                    val = min(hi_clip, val)
                enriched[k] = float(val)

        # ── Chemistry (PubChem / ChEMBL)  ────────────────────────────────────
        # key_prop_1 = MW (Da)  key_prop_2 = LogP  key_prop_3 = rot_bonds
        if "chem" in domain or "compound" in tags or src in ("pubchem", "chembl"):
            mw       = self._kp(row, "key_prop_1", 300.0)   # MW in Da
            logp     = self._kp(row, "key_prop_2",   2.0)   # LogP
            rot_bnds = self._kp(row, "key_prop_3",   3.0)   # rotatable bonds

            # MW → approximate boiling/reaction temperature proxy
            # Lipinski-style: higher MW → higher operating temperature
            t_base = np.clip(25.0 + (mw - 100.0) * 0.06, 20.0, 150.0)
            _put("temperature_max", t_base, 20.0, 150.0)
            _put("temperature_k",   t_base + 273.15, 293.0, 423.0)

            # LogP → polarity / pH proxy
            # Acidic compounds (low LogP) → lower effective pH
            ph_est = np.clip(7.0 + logp * 0.3, 2.0, 12.0)
            _put("ph", ph_est, 2.0, 12.0)

            # MW → concentration (heavier molecules → lower molar conc at 1g/L)
            conc_est = np.clip(1.0 / (1.0 + mw / 200.0), 0.05, 0.95)
            _put("concentration", conc_est, 0.05, 0.95)

            # rot_bonds → flexibility → strain proxy
            strain = np.clip(rot_bnds / 20.0, 0.05, 0.90)
            _put("strain_x", strain, 0.05, 0.90)
            _set("strain_y",          0.05,   0.90)
            _set("water",             0.30,   0.80)
            _set("time",              0.0,   24.0)

        # ── Materials (Materials Project / AFLOW)  ────────────────────────────
        # key_prop_1 = band_gap (eV)  key_prop_2 = formation_energy (eV/atom)
        # key_prop_3 = density (g/cm³)
        elif "material" in domain or "crystal" in tags or src in ("materials_project", "aflow"):
            band_gap  = self._kp(row, "key_prop_1",  1.0)   # eV
            form_en   = self._kp(row, "key_prop_2",  0.0)   # eV/atom (can be neg)
            density   = self._kp(row, "key_prop_3",  5.0)   # g/cm³ (may be "N/A")
            if density <= 0 or density > 25:
                density = 5.0  # fallback for 'N/A' or out-of-range

            # Metallic (band_gap≈0) vs insulator (band_gap>3) → temperature range
            if band_gap < 0.1:       # metal/conductor
                t_max = np.clip(200.0 + density * 50.0, 100.0, 1500.0)
            elif band_gap < 2.0:     # semiconductor
                t_max = np.clip(100.0 + density * 20.0,  50.0,  600.0)
            else:                     # insulator / ceramic
                t_max = np.clip( 60.0 + density * 15.0,  20.0,  400.0)

            _put("temperature_max", t_max, 20.0, 1500.0)

            # Density → mass-based strength proxy (denser → stronger)
            strength_est = np.clip(density * 150.0, 100.0, 2500.0)
            _put("strength", strength_est, 100.0, 2500.0)

            # Formation energy → stability → strain tolerance
            # More negative = more stable = lower strain to failure
            strain_est = np.clip(0.5 - form_en * 0.05, 0.05, 0.95)
            _put("strain_x", strain_est, 0.05, 0.95)
            _set("strain_y",          0.05,   0.95)

            # Band gap → conductivity (inverse relationship)
            conductivity_est = np.clip(200.0 * np.exp(-band_gap * 0.8), 1.0, 200.0)
            _put("conductivity", conductivity_est, 1.0, 200.0)
            _set("ph",   6.5,  8.0)
            _set("water", 0.0,  0.2)   # metals have very low water content
            _set("time",  0.0, 24.0)

        # ── Environment (OpenLandMap / SoilGrids / NASA POWER / open-meteo) ────
        # key_prop_1 = pH  key_prop_2 = sand%  key_prop_3 = organic_carbon(g/kg)
        elif "environment" in domain or "soil" in tags or "climate" in tags:
            ph_soil    = self._kp(row, "key_prop_1", 6.5)   # pH
            sand_pct   = self._kp(row, "key_prop_2", 40.0)  # sand %
            org_carbon = self._kp(row, "key_prop_3",  1.0)  # g/kg

            _put("ph",    ph_soil, 3.0, 10.0)

            # Sand% → water retention (loamy=low sand → high water hold)
            water_est = np.clip(1.0 - sand_pct / 130.0, 0.20, 0.90)
            _put("water", water_est, 0.20, 0.90)

            # Organic carbon → nitrogen proxy (SOM decomposition)
            nitrogen_est = np.clip(org_carbon / 15.0, 0.02, 0.80)
            _put("nitrogen", nitrogen_est, 0.02, 0.80)

            # India-relevant temperature range (10-45°C)
            _set("temperature_max", 10.0, 45.0)
            _set("light_intensity",  0.3,  0.9)
            _set("time",             0.0, 24.0)
            _set("concentration",    0.1,  0.5)
            _set("strength",       100.0, 800.0)

        # ── Physics / NASA TPSX (engineering materials)  ─────────────────────
        # key_prop_1 = material_id  key_prop_2 = category  key_prop_3 = database
        elif "physics" in domain or "nasa" in tags or "tpsx" in tags:
            category = str(row.get("key_prop_2", "")).lower()
            mat_id   = self._kp(row, "key_prop_1", 500.0)  # numeric id ≈ temp proxy

            # Category → operating temperature range
            if "metal" in category or "steel" in category or "alloy" in category:
                t_lo, t_hi = 100.0, 1200.0
            elif "rubber" in category or "polymer" in category or "plastic" in category:
                t_lo, t_hi =  20.0,  200.0
            elif "ceramic" in category or "brick" in category or "oxide" in category:
                t_lo, t_hi =  50.0,  800.0
            elif "composite" in category or "carbon" in category:
                t_lo, t_hi =  50.0,  600.0
            elif "foam" in category or "insul" in category:
                t_lo, t_hi = -50.0,  150.0
            else:
                t_lo, t_hi =  20.0,  500.0

            # mat_id modulates within the category range (deterministic variation)
            t_frac = np.clip((mat_id % 100) / 100.0, 0.0, 1.0)
            t_val  = t_lo + t_frac * (t_hi - t_lo)
            _put("temperature_max", t_val, t_lo, t_hi)

            # Strength proxy from temperature class
            str_est = np.clip(t_hi * 0.8, 100.0, 2000.0)
            _put("strength", str_est, 100.0, 2000.0)
            _set("strain_x",     0.01,  0.50)
            _set("strain_y",     0.01,  0.50)
            _set("conductivity",  1.0, 200.0)
            _set("ph",            6.0,   8.0)
            _set("water",         0.0,   0.1)
            _set("time",          0.0,  24.0)

        # ── Biology (UniProt / PDB proteins, NCBI genes)  ─────────────────────
        # key_prop_1 = chromosome or gene_length  key_prop_2 = organism
        # key_prop_3 = empty
        elif "bio" in domain or "protein" in tags or "gene" in tags:
            gene_len = self._kp(row, "key_prop_1", 10.0)   # chromosome/gene_id
            organism = str(row.get("key_prop_2", "")).lower()

            # Organism → temperature range
            if "homo sapiens" in organism or "human" in organism:
                t_lo, t_hi = 36.0, 38.0   # human body temperature
            elif "e. coli" in organism or "escherichia" in organism:
                t_lo, t_hi = 30.0, 42.0   # lab strain range
            elif "thermophil" in organism or "pyro" in organism:
                t_lo, t_hi = 60.0, 80.0   # thermophile
            elif "plant" in organism or "arabidopsis" in organism:
                t_lo, t_hi = 20.0, 35.0
            elif "mus musculus" in organism or "mouse" in organism:
                t_lo, t_hi = 36.0, 38.0
            else:
                t_lo, t_hi = 25.0, 45.0   # general biological range

            _set("temperature_max", t_lo, t_hi)

            # Gene length → complexity proxy → water / metabolic demand
            len_norm = np.clip(gene_len / 30.0, 0.0, 1.0)
            _put("water",    0.4 + len_norm * 0.3, 0.40, 0.90)
            _put("nitrogen", 0.3 + len_norm * 0.4, 0.30, 0.80)
            _set("ph",               7.0,   7.8)
            _set("light_intensity",  0.2,   0.9)
            _set("concentration",    0.2,   0.8)
            _set("strength",       400.0, 1600.0)
            _set("time",             4.0,  20.0)

        # ── Generic fallback  ─────────────────────────────────────────────────
        else:
            kp1 = self._kp(row, "key_prop_1", 40.0)
            base_t = float(np.clip(kp1 * 10.0 if kp1 < 6.0 else kp1, 20.0, 80.0))
            _set("temperature_max", max(20.0, base_t - 5), min(80.0, base_t + 5))
            _set("ph",               4.0,  10.0)
            _set("water",            0.3,   0.9)
            _set("nitrogen",         0.2,   0.7)
            _set("light_intensity",  0.2,   0.9)
            _set("time",             0.0,  24.0)
            _set("concentration",    0.1,   0.9)
            _set("strength",       200.0, 1800.0)
            _set("conductivity",    10.0,  200.0)
            _set("strain_x",         0.1,   0.8)
            _set("strain_y",         0.1,   0.8)

        return enriched

    def _enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply _enrich_row to an entire DataFrame when expected columns are missing."""
        physics_cols = {"temperature_max", "ph", "water", "nitrogen", "light_intensity",
                        "strain_x", "strength", "concentration"}
        present = set(df.columns) & physics_cols
        if len(present) >= 3:
            return df  # already has enough physics columns
        enriched_rows = [self._enrich_row(row) for row in df.to_dict("records")]
        return pd.DataFrame(enriched_rows)

    def _normalise(self, val, lo, hi):
        if hi == lo:
            return 0.5
        if np.isnan(val) or np.isinf(val):
            return 0.5
        return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))

    def _as_tensor(self, values):
        return torch.tensor([values], dtype=torch.float32)

    def _safe_float(self, row: dict, key: str, default: float = 0.0):
        try:
            v = float(row.get(key, default))
            # NaN/Inf propagation guard — return default instead
            if np.isnan(v) or np.isinf(v):
                return float(default) if not np.isnan(float(default)) else 0.0
            return v
        except Exception:
            return default

    # ── VECTORIZED HELPERS (batch path) ───────────────────────────────────────

    def _col(self, df: pd.DataFrame, col: str, default: float) -> np.ndarray:
        """Extract a DataFrame column as float32 array, filling missing with default."""
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default).values.astype(np.float32)
        return np.full(len(df), default, dtype=np.float32)

    def _norm_arr(self, arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Vectorised normalise: entire array at once, output clipped to [0, 1]."""
        if hi == lo:
            return np.full(len(arr), 0.5, dtype=np.float32)
        out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        # Replace any NaN/Inf that survived (e.g. from NaN inputs) with 0.5
        out = np.where(np.isfinite(out), out, 0.5)
        return out.astype(np.float32)

    def _features_heat(self, row: dict):
        x_position = self._normalise(self._safe_float(row, "x_position", 0.5), 0, 1)
        depth = self._normalise(self._safe_float(row, "depth", 0.2), 0, 2)
        time_v = self._normalise(self._safe_float(row, "time", 12), 0, 24)
        return [x_position, depth, time_v]

    def _features_stress(self, row: dict):
        strain_x = self._normalise(self._safe_float(row, "strain_x", self._safe_float(row, "strength", 1000) / 2000), 0, 1)
        strain_y = self._normalise(self._safe_float(row, "strain_y", 0.5 * strain_x), 0, 1)
        temp = self._normalise(self._safe_float(row, "temperature_max", 50), 0, 100)  # Adjusted: 0-100°C range, default 50
        return [strain_x, strain_y, temp]

    def _features_growth(self, row: dict):
        time_v = self._normalise(self._safe_float(row, "time", 12), 0, 24)
        temp = self._normalise(self._safe_float(row, "temperature_max", 50), 0, 100)  # Adjusted: 0-100°C range, default 50
        water = self._normalise(self._safe_float(row, "water", 0.5), 0, 1)
        return [time_v, temp, water]

    def _features_biology(self, row: dict):
        temp = self._normalise(self._safe_float(row, "temperature_max", 40), 0, 100)
        water = self._normalise(self._safe_float(row, "water", 0.5), 0, 1)
        nitrogen = self._normalise(self._safe_float(row, "nitrogen", 0.4), 0, 1)
        light = self._normalise(self._safe_float(row, "light_intensity", 0.6), 0, 1)
        time_v = self._normalise(self._safe_float(row, "time", 12), 0, 24)
        return [temp, water, nitrogen, light, time_v]

    def _features_chemistry(self, row: dict):
        temp_k = self._normalise(self._safe_float(row, "temperature_k", self._safe_float(row, "temperature_max", 40) + 273.15), 250, 330)
        concentration = self._normalise(self._safe_float(row, "concentration", 0.5), 0, 1)
        ph = self._normalise(self._safe_float(row, "ph", 7), 2, 12)
        time_v = self._normalise(self._safe_float(row, "time", 12), 0, 24)
        return [temp_k, concentration, ph, time_v]

    # ── VECTORIZED FEATURE EXTRACTORS (batch path) ────────────────────────────
    # Each method mirrors its row-level counterpart but operates on a whole
    # DataFrame and returns a float32 ndarray of shape (N, n_features).

    def _features_biology_batch(self, df: pd.DataFrame) -> np.ndarray:
        """(N, 5): [temperature, water, nitrogen, light_intensity, time]"""
        temp  = self._norm_arr(self._col(df, "temperature_max",   40.0), 0,  100)
        water = self._norm_arr(self._col(df, "water",              0.5),  0,  1)
        nitro = self._norm_arr(self._col(df, "nitrogen",           0.4),  0,  1)
        light = self._norm_arr(self._col(df, "light_intensity",    0.6),  0,  1)
        time  = self._norm_arr(self._col(df, "time",              12.0),  0,  24)
        return np.column_stack([temp, water, nitro, light, time])

    def _features_heat_batch(self, df: pd.DataFrame) -> np.ndarray:
        """(N, 3): [x_position, depth, time]"""
        x_pos = self._norm_arr(self._col(df, "x_position",  0.5),  0, 1)
        depth = self._norm_arr(self._col(df, "depth",        0.2),  0, 2)
        time  = self._norm_arr(self._col(df, "time",        12.0),  0, 24)
        return np.column_stack([x_pos, depth, time])

    def _features_stress_batch(self, df: pd.DataFrame) -> np.ndarray:
        """(N, 3): [strain_x, strain_y, temperature]
        strain_x falls back to strength/2000; strain_y falls back to 0.5*strain_x.
        """
        # strain_x: prefer explicit column, fall back to strength/2000
        strength = self._col(df, "strength", 1000.0)
        fallback_sx = strength / 2000.0
        if "strain_x" in df.columns:
            sx_raw = pd.to_numeric(df["strain_x"], errors="coerce").values.astype(np.float32)
            strain_x_raw = np.where(np.isnan(sx_raw), fallback_sx, sx_raw)
        else:
            strain_x_raw = fallback_sx
        strain_x = self._norm_arr(strain_x_raw, 0, 1)

        # strain_y: prefer explicit column, fall back to 0.5 * strain_x
        if "strain_y" in df.columns:
            sy_raw = pd.to_numeric(df["strain_y"], errors="coerce").values.astype(np.float32)
            strain_y_raw = np.where(np.isnan(sy_raw), 0.5 * strain_x, sy_raw)
        else:
            strain_y_raw = 0.5 * strain_x
        strain_y = self._norm_arr(strain_y_raw.astype(np.float32), 0, 1)

        temp = self._norm_arr(self._col(df, "temperature_max", 50.0), 0, 100)  # Adjusted: 0-100°C range, default 50
        return np.column_stack([strain_x, strain_y, temp])

    def _features_chemistry_batch(self, df: pd.DataFrame) -> np.ndarray:
        """(N, 4): [temperature_k, concentration, ph, time]
        temperature_k falls back to temperature_max + 273.15.
        """
        temp_max = self._col(df, "temperature_max", 40.0)
        fallback_tk = temp_max + 273.15
        if "temperature_k" in df.columns:
            tk_raw = pd.to_numeric(df["temperature_k"], errors="coerce").values.astype(np.float32)
            temp_k_raw = np.where(np.isnan(tk_raw), fallback_tk, tk_raw)
        else:
            temp_k_raw = fallback_tk
        temp_k = self._norm_arr(temp_k_raw.astype(np.float32), 250, 330)

        conc = self._norm_arr(self._col(df, "concentration", 0.5),  0, 1)
        ph   = self._norm_arr(self._col(df, "ph",            7.0),  2, 12)
        time = self._norm_arr(self._col(df, "time",         12.0),  0, 24)
        return np.column_stack([temp_k, conc, ph, time])

    def _features_growth_batch(self, df: pd.DataFrame) -> np.ndarray:
        """(N, 3): [time, temperature, water]"""
        time  = self._norm_arr(self._col(df, "time",           12.0), 0,  24)
        temp  = self._norm_arr(self._col(df, "temperature_max", 50.0), 0, 100)  # Adjusted: 0-100°C range, default 50
        water = self._norm_arr(self._col(df, "water",           0.5),  0, 1)
        return np.column_stack([time, temp, water])

    def _predict_from_surrogate(self, name: str, features: list) -> float:
        model = self.surrogates.get(name)
        if model is None:
            return 0.5
        # Final NaN/Inf guard before surrogate prediction
        x = np.array([features], dtype=np.float32)
        x = np.where(np.isfinite(x), x, 0.5)
        y = model.predict(x)
        y = float(np.array(y).reshape(-1)[0])
        return float(np.clip(y, 0.0, 1.0))

    def predict_biology(self, row: dict) -> float:
        if self.surrogates_loaded:
            return self._predict_from_surrogate("biology", self._features_biology(row))
        if self.models_loaded:
            with torch.no_grad():
                out = self.biology_model(self._as_tensor(self._features_biology(row)))
                # output dim=2 -> biomass/stress proxy; use biomass head.
                return float(np.clip(out[0, 0].item(), 0.0, 1.0))
        ph = self._safe_float(row, "ph", 7.0)
        salinity = self._safe_float(row, "salinity", 0.0)
        return self._normalise(ph, 2, 12) * 0.6 + (1 - self._normalise(salinity, 0, 50)) * 0.4

    def predict_physics(self, row: dict) -> float:
        if self.surrogates_loaded:
            heat = self._predict_from_surrogate("heat", self._features_heat(row))
            stress = self._predict_from_surrogate("stress", self._features_stress(row))
            return float(np.clip(0.5 * heat + 0.5 * stress, 0.0, 1.0))
        if self.models_loaded:
            with torch.no_grad():
                heat_out = self.heat_model(self._as_tensor(self._features_heat(row)))
                stress_out = self.stress_model(self._as_tensor(self._features_stress(row)))
                heat = float(np.clip(heat_out[0, 0].item(), 0.0, 1.0))
                stress = float(np.clip(float(stress_out[0].mean().item()), 0.0, 1.0))
                return float(np.clip(0.5 * heat + 0.5 * stress, 0.0, 1.0))
        temp = self._safe_float(row, "temperature_max", 50.0)
        strength = self._safe_float(row, "strength", 1000.0)
        physics = self._normalise(temp, 0, 100) * 0.4 + self._normalise(strength, 0, 2000) * 0.6
        return float(np.clip(physics, 0.0, 1.0))

    def predict_material(self, row: dict) -> float:
        if self.surrogates_loaded:
            stress = self._predict_from_surrogate("stress", self._features_stress(row))
            return float(np.clip(stress, 0.0, 1.0))
        if self.models_loaded:
            with torch.no_grad():
                stress_out = self.stress_model(self._as_tensor(self._features_stress(row)))
                material = float(np.clip(float(stress_out[0].mean().item()), 0.0, 1.0))
                return float(np.clip(material, 0.0, 1.0))
        strength = self._safe_float(row, "strength", 1000.0)
        conductivity = self._safe_float(row, "conductivity", 100.0)
        material = self._normalise(strength, 0, 2000) * 0.5 + self._normalise(conductivity, 0, 200) * 0.5
        return float(np.clip(material, 0.0, 1.0))

    def predict_chemistry(self, row: dict) -> float:
        if self.surrogates_loaded:
            chemistry = self._predict_from_surrogate("chemistry", self._features_chemistry(row))
            return float(np.clip(chemistry, 0.0, 1.0))
        if self.models_loaded:
            with torch.no_grad():
                out = self.chem_model(self._as_tensor(self._features_chemistry(row)))
                chemistry = float(np.clip(out[0, 0].item(), 0.0, 1.0))
                return float(np.clip(chemistry, 0.0, 1.0))
        ph = self._safe_float(row, "ph", 7.0)
        conductivity = self._safe_float(row, "conductivity", 100.0)
        chemistry = self._normalise(ph, 2, 12) * 0.5 + self._normalise(conductivity, 0, 200) * 0.5
        return float(np.clip(chemistry, 0.0, 1.0))

    def predict_growth(self, row: dict) -> float:
        if self.surrogates_loaded:
            return self._predict_from_surrogate("growth", self._features_growth(row))
        if self.models_loaded:
            with torch.no_grad():
                out = self.growth_model(self._as_tensor(self._features_growth(row)))
                return float(np.clip(out[0, 0].item(), 0.0, 1.0))
        temp = self._safe_float(row, "temperature_max", 0.0)
        ph = self._safe_float(row, "ph", 7.0)
        return self._normalise(temp, 0, 1500) * 0.5 + self._normalise(ph, 2, 12) * 0.5

    def predict_all(self, row: dict) -> dict:
        """Run all 5 PINNs on a single trait row."""
        bio  = self.predict_biology(row)
        phy  = self.predict_physics(row)
        mat  = self.predict_material(row)
        chem = self.predict_chemistry(row)
        grow = self.predict_growth(row)

        overall = (bio * 0.20 + phy * 0.30 + mat * 0.25 + chem * 0.15 + grow * 0.10)

        return {
            "biology_score":  round(bio,  6),
            "physics_score":  round(phy,  6),
            "material_score": round(mat,  6),
            "chemistry_score":round(chem, 6),
            "growth_score":   round(grow, 6),
            "viability_score":round(overall, 6),
        }

    def predict_batch_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized batch prediction — replaces the row-by-row predict_batch loop.

        Calls each surrogate / PINN / fallback once with the full (N, features)
        matrix instead of N separate single-sample calls.

        Returns a DataFrame with columns:
            biology_score, physics_score, material_score,
            chemistry_score, growth_score, viability_score
        """
        if self.surrogates_loaded:
            # Path A: 5 GBM batch calls — C-level tree traversal for all N rows
            # BiologyPINN has output_dim=2 → surrogate returns (N,2); take col 0 (biomass head)
            bio_raw = self.surrogates["biology"].predict(
                          self._features_biology_batch(df)).astype(np.float32)
            bio    = np.clip(bio_raw[:, 0] if bio_raw.ndim > 1 else bio_raw, 0, 1)

            heat   = np.clip(self.surrogates["heat"].predict(
                        self._features_heat_batch(df)).astype(np.float32), 0, 1)

            # StressPINN has output_dim=2 → surrogate returns (N,2); take mean (stress_x, stress_y)
            stress_raw = self.surrogates["stress"].predict(
                             self._features_stress_batch(df)).astype(np.float32)
            stress = np.clip(stress_raw.mean(axis=1) if stress_raw.ndim > 1 else stress_raw, 0, 1)

            chem   = np.clip(self.surrogates["chemistry"].predict(
                        self._features_chemistry_batch(df)).astype(np.float32), 0, 1)
            grow   = np.clip(self.surrogates["growth"].predict(
                        self._features_growth_batch(df)).astype(np.float32), 0, 1)

        elif self.models_loaded:
            # Path B: 5 PyTorch batched forward passes
            with torch.no_grad():
                bio = np.clip(
                    self.biology_model(
                        torch.tensor(self._features_biology_batch(df))
                    )[:, 0].numpy(), 0, 1)

                heat_t = self.heat_model(
                    torch.tensor(self._features_heat_batch(df)))
                heat = np.clip(heat_t[:, 0].numpy(), 0, 1)

                stress_t = self.stress_model(
                    torch.tensor(self._features_stress_batch(df)))
                stress = np.clip(stress_t.mean(dim=1).numpy(), 0, 1)

                chem = np.clip(
                    self.chem_model(
                        torch.tensor(self._features_chemistry_batch(df))
                    )[:, 0].numpy(), 0, 1)

                grow = np.clip(
                    self.growth_model(
                        torch.tensor(self._features_growth_batch(df))
                    )[:, 0].numpy(), 0, 1)

        else:
            # Path C: vectorized fallback — pure numpy, no models loaded
            ph         = self._col(df, "ph",            7.0)
            salinity   = self._col(df, "salinity",       0.0)
            temp       = self._col(df, "temperature_max", 0.0)
            strength   = self._col(df, "strength",       0.0)
            conductivity = self._col(df, "conductivity", 0.0)

            bio    = (self._norm_arr(ph, 2, 12) * 0.6
                      + (1.0 - self._norm_arr(salinity, 0, 50)) * 0.4)

            heat   = self._norm_arr(temp, 0, 1500) * 0.4 + self._norm_arr(strength, 0, 2000) * 0.6
            stress = (self._norm_arr(strength, 0, 2000) * 0.5
                      + self._norm_arr(conductivity, 0, 200) * 0.5)

            chem   = (self._norm_arr(ph, 2, 12) * 0.5
                      + self._norm_arr(conductivity, 0, 200) * 0.5)

            grow   = (self._norm_arr(temp, 0, 1500) * 0.5
                      + self._norm_arr(ph, 2, 12) * 0.5)

        # Physics is the average of heat and stress (same as row-level logic)
        physics = np.clip(0.5 * heat + 0.5 * stress, 0.0, 1.0).astype(np.float32)

        # Weighted overall score  (matches predict_all weights exactly)
        overall = (bio * 0.20 + physics * 0.30 + stress * 0.25
                   + chem * 0.15 + grow * 0.10).astype(np.float32)

        return pd.DataFrame({
            "biology_score":   bio,
            "physics_score":   physics,
            "material_score":  stress,
            "chemistry_score": chem,
            "growth_score":    grow,
            "viability_score": overall,
        })

    def predict_batch(self, df: pd.DataFrame) -> list:
        """Run predictions on a full DataFrame batch. Returns list of score dicts."""
        return [self.predict_all(row) for row in df.to_dict("records")]


# ── BATCH SIMULATOR ───────────────────────────────────────────────────────────

class BatchSimulator:
    """
    Runs all traits through Srikar's models in batches.
    Supports checkpointing so it can resume after a crash.
    """

    def __init__(self,
                 model_dir: str = None,
                 parquet_path: str = None,
                 batch_size: int = 5000):
        self.model      = SrikarModelInterface(model_dir)
        self.loader     = DataLoader(parquet_path)
        self.batch_size = batch_size
        self.results    = []
        self.stats      = {
            "total_processed": 0,
            "total_passed":    0,
            "total_filtered":  0,
            "batches_done":    0,
            "start_time":      None,
            "elapsed_sec":     0,
        }
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results",     exist_ok=True)

    def _load_checkpoint(self) -> int:
        """Returns last completed batch number (0 if none)."""
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE) as f:
                data = json.load(f)
            print(f"♻️  Resuming from batch {data['last_batch']}")
            return data["last_batch"]
        return 0

    def _save_checkpoint(self, batch_num: int):
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({
                "last_batch":  batch_num,
                "timestamp":   datetime.now().isoformat(),
                "processed":   self.stats["total_processed"],
            }, f, indent=2)

    def _process_batch(self, batch_num: int, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run one batch through all models.

        Returns a pd.DataFrame — no TraitResult construction, no iterrows(),
        no asdict(). All heavy lifting is done by predict_batch_vectorized.
        """
        df = self.model._enrich_dataframe(df)
        scores = self.model.predict_batch_vectorized(df)
        n = len(df)
        now = datetime.now().isoformat()

        # Pull identity columns as plain arrays (fast, no Python loop)
        def _str_col(col, fallback_prefix):
            if col in df.columns:
                return df[col].fillna("").astype(str).values
            return np.array([f"{fallback_prefix}{i}" for i in range(n)])

        # Use entity_id (new universal_index) or trait_id (legacy) as identifier
        if "entity_id" in df.columns and "trait_id" not in df.columns:
            trait_ids = _str_col("entity_id", "T")
        else:
            trait_ids = _str_col("trait_id", "T")
            if all(t.startswith("T") and t[1:].isdigit() for t in trait_ids[:5]):
                # Auto-generated fallback — prefer entity_id if available
                entity_ids = _str_col("entity_id", "")
                if any(e for e in entity_ids[:5]):
                    trait_ids = entity_ids

        # entity_type: use domain column if entity_type not present
        if "entity_type" not in df.columns and "domain" in df.columns:
            entity_types = _str_col("domain", "")
        else:
            entity_types = _str_col("entity_type", "")

        return pd.DataFrame({
            "trait_id":        trait_ids,
            "entity_type":     entity_types,
            "source":          _str_col("source",      ""),
            "viability_score": scores["viability_score"].values,
            "biology_score":   scores["biology_score"].values,
            "physics_score":   scores["physics_score"].values,
            "material_score":  scores["material_score"].values,
            "chemistry_score": scores["chemistry_score"].values,
            "passed_filter":   scores["viability_score"].values >= SCORE_THRESHOLD,
            "simulated_at":    now,
        })

    def run_all(self, resume: bool = True) -> pd.DataFrame:
        """
        Main entry point. Runs all traits through models.
        Set resume=True to pick up from last checkpoint after a crash.
        """
        t_start = time.perf_counter()
        self.stats["start_time"] = datetime.now().isoformat()

        start_batch = self._load_checkpoint() if resume else 0
        total       = self.loader.count()

        print(f"\n🚀 Starting batch simulation")
        print(f"   Total traits  : {total:,}")
        print(f"   Batch size    : {self.batch_size:,}")
        print(f"   Score filter  : >{SCORE_THRESHOLD}")
        print(f"   Starting from : batch {start_batch + 1}")
        print(f"{'─'*52}")

        all_dfs: list = []

        for batch_num, df in self.loader.get_batches(self.batch_size):
            if batch_num <= start_batch:
                continue

            t0 = time.perf_counter()
            batch_df = self._process_batch(batch_num, df)
            elapsed = time.perf_counter() - t0

            passed   = int(batch_df["passed_filter"].sum())
            filtered = len(batch_df) - passed

            all_dfs.append(batch_df)
            self.stats["total_processed"] += len(batch_df)
            self.stats["total_passed"]    += passed
            self.stats["total_filtered"]  += filtered
            self.stats["batches_done"]    += 1

            self._save_checkpoint(batch_num)

            # Progress log
            pct = self.stats["total_processed"] / total * 100
            traits_sec = len(batch_df) / elapsed if elapsed > 0 else 0
            print(f"  Batch {batch_num:04d} | "
                  f"{len(df):,} traits | "
                  f"passed {passed:,} | "
                  f"filtered {filtered:,} | "
                  f"{elapsed*1000:.0f}ms | "
                  f"{traits_sec:,.0f} t/s | "
                  f"{pct:.1f}% done")

        # Save results — pd.concat instead of [asdict(r) for r in all_results]
        total_elapsed = time.perf_counter() - t_start
        self.stats["elapsed_sec"] = round(total_elapsed, 2)

        results_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        results_df.to_parquet(RESULTS_FILE, index=False)

        self._print_summary(total_elapsed)
        return results_df

    def _print_summary(self, elapsed: float):
        s = self.stats
        traits_per_hr = s["total_processed"] / elapsed * 3600 if elapsed > 0 else 0
        print(f"\n{'='*52}")
        print(f"  SIMULATION COMPLETE")
        print(f"{'='*52}")
        print(f"  Total processed : {s['total_processed']:,}")
        print(f"  Passed filter   : {s['total_passed']:,}  ({s['total_passed']/max(s['total_processed'],1)*100:.1f}%)")
        print(f"  Filtered out    : {s['total_filtered']:,}")
        print(f"  Total time      : {elapsed:.1f}s")
        print(f"  Speed           : {traits_per_hr:,.0f} traits/hour")
        print(f"  1M trait est.   : {1_000_000/max(traits_per_hr,1):.1f} hours")
        print(f"  Results saved   : {RESULTS_FILE}")
        target_ok = traits_per_hr >= 250000  # 1M in 4 hours
        print(f"  4hr target      : {'✅ MET' if target_ok else '⚠️ NEEDS GPU'}")


if __name__ == "__main__":
    sim = BatchSimulator(batch_size=5000)
    results = sim.run_all()
    print(f"\nSample output:")
    print(results[["trait_id","viability_score","biology_score",
                   "physics_score","passed_filter"]].head(5).to_string(index=False))
