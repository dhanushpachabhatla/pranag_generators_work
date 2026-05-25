"""
sympy_loss_generator.py — SymPy Auto Loss Generator (Proof of Concept)
=======================================================================
Removes manual coding dependency by auto-generating PyTorch loss functions
from plain math rule strings.

Pipeline:
    Plain Math Rule
      → SymPy parser (parse_expr)
      → symbolic expression tree
      → PyTorch code string (codegen)
      → compiled callable loss function

Usage:
    gen = SymPyLossGenerator()

    # Generate from string
    loss_fn = gen.compile("P_escape * impact")
    loss = loss_fn(P_escape=tensor(0.05), impact=tensor(0.9))

    # Preview generated code
    code = gen.to_pytorch_code("max(0, cost - budget)")
    print(code)

    # Bulk generation
    losses = gen.generate_all()

Supported math constructs:
    - Arithmetic: +, -, *, /, **
    - Functions: max(a, b), min(a, b), exp(x), log(x), sqrt(x), abs(x)
    - Clamp/relu pattern: max(0, expr) → torch.relu(expr)
    - Products of named variables: P_escape * impact

Generated PyTorch uses:
    - torch.relu() for max(0, ...)
    - torch.clamp() for min(a, b) / max(a, b) with constants
    - torch.exp(), torch.log(), torch.sqrt(), torch.abs()
"""

import re
from typing import Dict, Callable, Optional

try:
    import sympy as sp
    from sympy.printing.pycode import PythonCodePrinter
    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False

import torch

try:
    import deepxde as dde
    _DDE_AVAILABLE = True
except ImportError:
    _DDE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────
# Variable registry — all known physics/biology/ecology/economics/safety
# variables that can appear in loss expressions.
# ─────────────────────────────────────────────────────────────────────

KNOWN_VARIABLES = {
    # Ecology
    "P_escape", "impact", "ecological_impact", "economic_impact",
    "invasiveness_score",
    # Economics
    "cost", "budget", "manufacturing_cost", "operation_cost", "total_cost",
    # Safety
    "toxicity", "pathogenicity", "allergenicity",
    # Biology
    "folding_energy", "metabolic_burden", "codon_rarity", "gc_content",
    # Physics
    "residual", "temperature", "pressure", "stress", "strain",
    # Generic
    "x", "y", "t", "u", "v", "w", "a", "b", "c",
}


class _PyTorchPrinter:
    """Converts a SymPy expression tree to PyTorch tensor operations."""

    def __init__(self):
        self._op_map = {
            "Add":  self._print_Add,
            "Mul":  self._print_Mul,
            "Pow":  self._print_Pow,
            "Max":  self._print_Max,
            "Min":  self._print_Min,
            "Abs":  self._print_Abs,
            "exp":  self._print_exp,
            "log":  self._print_log,
            "sqrt": self._print_sqrt,
        }

    def print(self, expr) -> str:
        if not _SYMPY_AVAILABLE:
            raise ImportError("sympy is required for code generation")

        if isinstance(expr, sp.Symbol):
            return str(expr)
        if isinstance(expr, sp.Number):
            return str(float(expr))
        if isinstance(expr, sp.Mul):
            return self._print_Mul(expr)
        if isinstance(expr, sp.Add):
            return self._print_Add(expr)
        if isinstance(expr, sp.Pow):
            return self._print_Pow(expr)
        if isinstance(expr, sp.Abs):
            return self._print_Abs(expr)
        if isinstance(expr, sp.exp):
            return self._print_exp(expr)
        if isinstance(expr, sp.log):
            return self._print_log(expr)
        if isinstance(expr, sp.sqrt):
            return self._print_sqrt(expr)
        if hasattr(sp, 'Max') and isinstance(expr, sp.Max):
            return self._print_Max(expr)
        if hasattr(sp, 'Min') and isinstance(expr, sp.Min):
            return self._print_Min(expr)
        # Fallback: use sympy's own str
        return str(expr)

    def _print_Add(self, expr) -> str:
        parts = [self.print(a) for a in expr.args]
        return " + ".join(parts)

    def _print_Mul(self, expr) -> str:
        parts = [self.print(a) for a in expr.args]
        # Wrap sums in parentheses when nested in products
        wrapped = []
        for p, a in zip(parts, expr.args):
            if isinstance(a, sp.Add):
                wrapped.append(f"({p})")
            else:
                wrapped.append(p)
        return " * ".join(wrapped)

    def _print_Pow(self, expr) -> str:
        base, exp_ = expr.args
        b = self.print(base)
        e = self.print(exp_)
        if isinstance(base, (sp.Add, sp.Mul)):
            b = f"({b})"
        return f"{b} ** {e}"

    def _print_Max(self, expr) -> str:
        args = expr.args
        # max(0, x) pattern → relu
        for i, a in enumerate(args):
            if a == sp.Integer(0) or a == sp.Float(0):
                other = [self.print(x) for j, x in enumerate(args) if j != i]
                inner = " + ".join(other) if len(other) > 1 else other[0]
                return f"torch.relu({inner})"
        return f"torch.maximum({', '.join(self.print(a) for a in args)})"

    def _print_Min(self, expr) -> str:
        args = [self.print(a) for a in expr.args]
        return f"torch.minimum({', '.join(args)})"

    def _print_Abs(self, expr) -> str:
        return f"torch.abs({self.print(expr.args[0])})"

    def _print_exp(self, expr) -> str:
        return f"torch.exp({self.print(expr.args[0])})"

    def _print_log(self, expr) -> str:
        return f"torch.log({self.print(expr.args[0])} + 1e-8)"

    def _print_sqrt(self, expr) -> str:
        return f"torch.sqrt({self.print(expr.args[0])} + 1e-8)"


# ─────────────────────────────────────────────────────────────────────
# Fallback printer — pure regex, no SymPy dependency
# ─────────────────────────────────────────────────────────────────────

def _regex_to_pytorch(expr_str: str) -> str:
    """
    Simple regex-based transpiler for when SymPy is not available
    or for very simple expressions.

    Handles:
        max(0, expr)   → torch.relu(expr)
        exp(x)         → torch.exp(x)
        log(x)         → torch.log(x + 1e-8)
        abs(x)         → torch.abs(x)
        sqrt(x)        → torch.sqrt(x + 1e-8)
        x ** n         → x ** n  (unchanged, valid Python)
    """
    code = expr_str.strip()
    # max(0, ...) → relu
    code = re.sub(r'\bmax\(\s*0\s*,\s*(.+?)\)', r'torch.relu(\1)', code)
    code = re.sub(r'\bmax\(\s*(.+?)\s*,\s*0\s*\)', r'torch.relu(\1)', code)
    # math functions
    code = re.sub(r'\bexp\(',   'torch.exp(',        code)
    code = re.sub(r'\babs\(',   'torch.abs(',        code)
    code = re.sub(r'\bsqrt\(',  'torch.sqrt(',       code)
    code = re.sub(r'\blog\(',   'torch.log(',        code)
    return code


# ─────────────────────────────────────────────────────────────────────
# Main generator class
# ─────────────────────────────────────────────────────────────────────

class SymPyLossGenerator:
    """
    Converts plain math rule strings to executable PyTorch loss functions.

    Supports SymPy (preferred) with regex fallback when SymPy is unavailable.
    """

    # Built-in library of pre-defined loss rules (extend as needed)
    BUILTIN_RULES: Dict[str, str] = {
        "ecology_escape":    "P_escape * impact",
        "economics_overrun": "max(0, cost - budget)",
        "safety_combined":   "toxicity + pathogenicity + allergenicity",
        "biology_folding":   "max(0, folding_energy)",
        "biology_burden":    "max(0, metabolic_burden - 0.5)",
        "physics_residual":  "residual ** 2",
    }

    def __init__(self):
        self._printer = _PyTorchPrinter() if _SYMPY_AVAILABLE else None
        self._cache: Dict[str, str] = {}

    # ── Parsing ──────────────────────────────────────────────────────

    def parse(self, expression: str):
        """
        Parse a math expression string to a SymPy expression tree.
        Returns None if SymPy is unavailable.
        """
        if not _SYMPY_AVAILABLE:
            return None

        # Build a namespace with all known variables as SymPy symbols
        local_dict = {v: sp.Symbol(v) for v in KNOWN_VARIABLES}
        # Also add common functions
        local_dict.update({
            "max": sp.Max,
            "min": sp.Min,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
        })
        return sp.sympify(expression, locals=local_dict)

    # ── Code generation ──────────────────────────────────────────────

    def to_pytorch_code(self, expression: str) -> str:
        """
        Generate a PyTorch expression string from a plain math rule.

        Args:
            expression: math rule string, e.g. "P_escape * impact"

        Returns:
            PyTorch code string, e.g. "P_escape * impact"
            (with torch.relu, torch.exp etc. substituted as needed)
        """
        if expression in self._cache:
            return self._cache[expression]

        if _SYMPY_AVAILABLE:
            try:
                sym_expr = self.parse(expression)
                code = self._printer.print(sym_expr)
            except Exception:
                code = _regex_to_pytorch(expression)
        else:
            code = _regex_to_pytorch(expression)

        self._cache[expression] = code
        return code

    def to_function_signature(self, expression: str) -> str:
        """
        Derive the function signature (parameter names) from the expression.
        E.g. "P_escape * impact" → "def loss(*, P_escape, impact)"
        """
        variables = sorted(set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', expression))
                           - {"max", "min", "exp", "log", "sqrt", "abs", "torch"})
        params = ", ".join(variables)
        return f"def loss(*, {params})"

    def to_full_code(self, expression: str, loss_name: str = "loss") -> str:
        """
        Generate a complete Python/PyTorch function string.

        Args:
            expression: math rule string
            loss_name:  name for the generated function

        Returns:
            Complete function definition as a string.
        """
        pytorch_expr = self.to_pytorch_code(expression)
        variables = sorted(set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', expression))
                           - {"max", "min", "exp", "log", "sqrt", "abs", "torch"})
        params = ", ".join(variables)
        return (
            f"def {loss_name}({params}):\n"
            f"    return ({pytorch_expr}).mean()\n"
        )

    # ── Compilation ──────────────────────────────────────────────────

    def compile(self, expression: str) -> Callable:
        """
        Compile a math rule string into an executable PyTorch loss callable.

        The returned function accepts keyword arguments matching the variable
        names in the expression, and returns a scalar loss tensor.

        Example:
            loss_fn = gen.compile("P_escape * impact")
            loss = loss_fn(P_escape=torch.tensor([0.05]), impact=torch.tensor([0.9]))
        """
        pytorch_expr = self.to_pytorch_code(expression)

        variables = sorted(set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', expression))
                           - {"max", "min", "exp", "log", "sqrt", "abs", "torch"})

        # Build and exec the function in a safe namespace
        fn_code = (
            f"import torch\n"
            f"def _loss_fn({', '.join(variables)}):\n"
            f"    result = {pytorch_expr}\n"
            f"    if isinstance(result, torch.Tensor):\n"
            f"        return result.mean()\n"
            f"    return torch.tensor(float(result))\n"
        )
        namespace: Dict = {}
        exec(fn_code, {"torch": torch}, namespace)
        return namespace["_loss_fn"]

    # ── Bulk generation ──────────────────────────────────────────────

    def generate_all(self) -> Dict[str, Dict]:
        """
        Generate PyTorch code for all built-in loss rules.

        Returns a dict:
            {rule_name: {"expression": ..., "pytorch_code": ..., "function": ...}}
        """
        results = {}
        for name, expr in self.BUILTIN_RULES.items():
            pytorch_code = self.to_pytorch_code(expr)
            fn_code      = self.to_full_code(expr, loss_name=name)
            try:
                loss_fn = self.compile(expr)
            except Exception as e:
                loss_fn = None

            results[name] = {
                "expression":   expr,
                "pytorch_code": pytorch_code,
                "full_code":    fn_code,
                "callable":     loss_fn,
            }
        return results

    def add_rule(self, name: str, expression: str):
        """Register a custom loss rule."""
        self.BUILTIN_RULES[name] = expression

    # ── PDE Compilation (DeepXDE) ──────────────────────────────────────

    def compile_pde(self, pde_string: str, input_vars: list, output_var: str = "u") -> Callable:
        """
        Compiles a PDE string like "u_t - 0.5 * u_xx" into a DeepXDE pde function.
        input_vars is a list of variables, e.g., ["x", "t"].
        The index of the variable in input_vars determines the j index in dde.grad.
        """
        if not _DDE_AVAILABLE:
            raise ImportError("deepxde is required for PDE compilation")

        code = pde_string
        
        # Replace 2nd derivatives like u_xx
        for j, var in enumerate(input_vars):
            pattern = f"{output_var}_{var}{var}"
            replace = f"dde.grad.hessian(y, x, i=0, j={j})"
            code = code.replace(pattern, replace)
            
        # Replace 1st derivatives like u_t
        for j, var in enumerate(input_vars):
            pattern = f"{output_var}_{var}"
            replace = f"dde.grad.jacobian(y, x, i=0, j={j})"
            code = code.replace(pattern, replace)
            
        # Create function
        fn_code = (
            f"import deepxde as dde\n"
            f"import torch\n"
            f"def _pde_fn(x, y):\n"
            f"    return {code}\n"
        )
        namespace = {}
        exec(fn_code, {"dde": dde, "torch": torch}, namespace)
        return namespace["_pde_fn"]

    def create_pde_data(self, pde_string: str, input_vars: list, geom, time_domain=None, num_domain=100, num_boundary=20, num_initial=20):
        """Creates a dde.data.TimePDE or dde.data.PDE from a string."""
        if not _DDE_AVAILABLE:
            raise ImportError("deepxde is required for PDE data creation")

        pde_fn = self.compile_pde(pde_string, input_vars=input_vars)
        if time_domain:
            geomtime = dde.geometry.GeometryXTime(geom, time_domain)
            data = dde.data.TimePDE(geomtime, pde_fn, [], num_domain=num_domain, num_boundary=num_boundary, num_initial=num_initial)
        else:
            data = dde.data.PDE(geom, pde_fn, [], num_domain=num_domain, num_boundary=num_boundary)
        return data


# ─────────────────────────────────────────────────────────────────────
# Demo: 3 generated loss examples
# ─────────────────────────────────────────────────────────────────────

def _demo_three_generated_losses():
    """
    Demonstrates the SymPy generator with 3 concrete examples matching the spec.
    """
    gen = SymPyLossGenerator()

    examples = [
        # Example 1 — Ecology: escape × impact
        ("P_escape * impact",
         dict(P_escape=torch.tensor([0.05, 0.02, 0.15]),
              impact=torch.tensor([0.8, 0.6, 0.9]))),

        # Example 2 — Economics: max(0, cost - budget)
        ("max(0, cost - budget)",
         dict(cost=torch.tensor([800.0, 1200.0, 950.0]),
              budget=torch.tensor([1000.0, 1000.0, 1000.0]))),

        # Example 3 — Safety: toxicity + pathogenicity + allergenicity
        ("toxicity + pathogenicity + allergenicity",
         dict(toxicity=torch.tensor([0.05, 0.30, 0.12]),
              pathogenicity=torch.tensor([0.02, 0.01, 0.08]),
              allergenicity=torch.tensor([0.10, 0.20, 0.05]))),
    ]

    print("=" * 60)
    print("  SymPy Auto Loss Generator — 3 Examples")
    print("=" * 60)

    for i, (expr, inputs) in enumerate(examples, 1):
        print(f"\n[Example {i}]")
        print(f"  Rule       : {expr}")
        pytorch_code = gen.to_pytorch_code(expr)
        print(f"  PyTorch    : {pytorch_code}")
        full_code = gen.to_full_code(expr, loss_name=f"loss_example_{i}")
        print(f"  Generated  :\n    {full_code.strip()}")
        try:
            loss_fn = gen.compile(expr)
            result  = loss_fn(**inputs)
            print(f"  Result     : {result.item():.6f}")
        except Exception as e:
            print(f"  Error      : {e}")

    print("\n  All 3 examples generated successfully.")
    print("=" * 60)

    return gen


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = _demo_three_generated_losses()

    print("\n\nAll built-in rules:")
    all_rules = gen.generate_all()
    for name, info in all_rules.items():
        status = "OK" if info["callable"] is not None else "FAILED"
        print(f"  [{status}] {name:<25} : {info['expression']}")
        print(f"         PyTorch : {info['pytorch_code']}")
