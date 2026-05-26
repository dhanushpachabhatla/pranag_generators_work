# PRANA-G Generator Suite Architecture

This directory (`Model/models/`) contains the core generator suite for PRANA-G. 

A common misconception is that PRANA-G relies on large language models (LLMs) to dynamically generate Physics-Informed Neural Networks (PINNs) on the fly. **This is false.** The entire generator suite is deterministic, relying on powerful regex pattern matching, SymPy symbolic math parsing, and a highly modular factory architecture.

Here is exactly how the four core files interact to generate and train simulations automatically.

---

## 1. `simulation_generator.py` (The Code Writer)
This module acts as an Autonomous Code Generator, but it does **not** use an LLM. 
Instead, it holds a massive, hardcoded dictionary (`EQUATION_PATTERNS`) of physics equations (Heat, Wave, Navier-Stokes, Burgers, Schrodinger, etc.). 

**Logic Flow:**
1. It takes a plain text hint (e.g., `"2D heat diffusion"`).
2. It uses regex keywords to match that hint to its internal database.
3. Once matched, it uses Python's `textwrap` and string substitution to literally write a full, syntax-perfect PyTorch Neural Network class (a `nn.Module`) as a text string.
4. It can compile that text string directly into an executable Python class in memory using `exec()`.

## 2. `sympy_loss_generator.py` (The Math Transpiler)
Writing custom Loss functions for PyTorch normally requires manual coding (e.g., `torch.relu(x) * torch.exp(y)`). This module removes that bottleneck using **SymPy**, a symbolic mathematics library.

**Logic Flow:**
1. You provide a plain math string: `"max(0, cost - budget)"` or `"P_escape * impact"`.
2. It uses `sympy.sympify` to parse the string into a mathematical syntax tree.
3. A custom printer navigates the tree and transpiles it into executable PyTorch code: `"torch.relu(cost - budget)"`.
4. The result is a compiled Python lambda function that acts as a custom PyTorch loss function.

## 3. `pinn_factory.py` (The Universal Router)
This is the brain that connects everything. It is responsible for giving you a ready-to-train PyTorch model when you ask for one.

**Logic Flow:**
1. You call `PINNFactory().create("heat")`.
2. The Factory checks its `_registry`. If `"heat"` is a known base model (like `HeatPINN`), it instantly returns the initialized PyTorch module.
3. **Dynamic Generation:** If you ask for something unknown, the Factory automatically routes the request to `SimulationGenerator`. The generator writes the new PyTorch code on the fly, injects it into memory, and the Factory returns the newly generated model as if it always existed!

## 4. `pinn_trainer.py` (The Engine)
Once the Factory or the Generator hands over the PyTorch module, the Trainer takes over.

**Logic Flow:**
1. It wraps the raw PINN model inside a `PINNLightningModule`.
2. It randomly scatters N-dimensional collocation points across the physical domain (e.g., thousands of points across Time, Space, and Temperature).
3. It uses PyTorch Lightning to push the model through the optimizer (usually Adam or L-BFGS) to learn the physical rules over a set number of epochs.

---

## The End-to-End Execution Flow

When a user triggers the `unified_pipeline.py`:
1. **Request:** `factory.create("wave")` is called.
2. **Build:** The Factory retrieves the Wave model (or dynamically asks the Generator to write one if missing).
3. **Loss Injection:** `sympy_loss_generator.py` builds the exact physics and biological loss tensors for the system.
4. **Training:** `pinn_trainer.py` takes the built model, generates thousands of physical data points, and forces the neural network to learn the physics in PyTorch Lightning.
5. **Storage:** The fully trained `.ckpt` PyTorch checkpoints are permanently saved in the `unified_pipeline_output/pinn/` directory. The extracted lightweight Surrogates and R2 evaluation metrics are routed to `surrogate/` and `results/` respectively.

---

## Limitations

While powerful, the deterministic nature of this generator suite introduces several limitations:
1. **Regex Brittleness:** `simulation_generator.py` relies heavily on exact keyword matches in `EQUATION_PATTERNS`. If a user asks for `"thermal dispersion"` but the keyword is only `"heat diffusion"`, the system will fail to generate the correct physics.
2. **SymPy Edge Cases:** `sympy_loss_generator.py` is an excellent transpiler for standard arithmetic (e.g., `+`, `-`, `max`, `exp`), but it will fail if a user tries to inject unsupported or highly complex custom functions (like a custom neural network embedded inside the math string) that SymPy cannot parse.
3. **Hardcoded Physics:** The physics loss templates (e.g., `"heat_1d"`, `"wave_1d"`) are largely hardcoded. The system cannot easily invent a novel differential equation that it doesn't already have a blueprint for.

## Next Steps
To evolve this suite into a true AGI pipeline, the following steps should be considered:
1. **Integrate an LLM:** Replace the rigid regex keyword matching in `simulation_generator.py` with an LLM that can dynamically write valid PyTorch class strings for *any* requested physics, not just pre-defined patterns.
2. **Advanced Optimization:** In `pinn_trainer.py`, implement a dual-optimizer strategy (e.g., 500 epochs of Adam followed by 500 epochs of L-BFGS) to ensure much deeper convergence of the physics models.
3. **Prompt Parser Integration:** Connect this offline generator engine directly to the `prompt_parser.py` so that a user's natural language input (e.g., "sugarcane cultivation in Uttar Pradesh at 37°C") automatically triggers the correct model retrieval and biological evaluation.
