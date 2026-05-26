import json
import re
import os
from datetime import datetime
from pathlib import Path

def update_markdown_report(report_md_path: str, report_json_path: str):
    if not os.path.exists(report_md_path):
        print(f"Warning: {report_md_path} not found.")
        return
    
    if not os.path.exists(report_json_path):
        print(f"Warning: {report_json_path} not found. Cannot update report.")
        return
        
    with open(report_json_path, 'r') as f:
        data = json.load(f)
        
    # Get the generated date from the JSON or use current date
    generated_at_str = data.get("generated_at", datetime.now().isoformat())
    try:
        # Parse and format the date string to YYYY-MM-DD
        dt = datetime.fromisoformat(generated_at_str)
        date_short = dt.strftime("%Y-%m-%d")
        date_long = dt.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        date_short = generated_at_str[:10]
        date_long = generated_at_str[:19]
        
    with open(report_md_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Replace the top generated date
    content = re.sub(r"Generated:\s*\d{4}-\d{2}-\d{2}", f"Generated: {date_short}", content, count=1)
    
    # Replace the bottom generated date
    content = re.sub(r"Generated:\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", f"Generated: {date_long}", content, count=1)
    
    # Optional: We could also dynamically update accuracy and pass rates here
    # For now, we mainly ensure the date updates to reflect the recent run.
    exec_summary = data.get("executive_summary", {})
    if "pass_rate_pct" in exec_summary:
        pass_rate = exec_summary["pass_rate_pct"]
        content = re.sub(r"\| Core Validation \| ✅ PASS \| .*? \|", f"| Core Validation | ✅ PASS | {pass_rate}% pass rate |", content)
        
    if "surrogate_accuracy" in exec_summary:
        acc = exec_summary["surrogate_accuracy"]
        content = re.sub(r"\| Accuracy Validation \| ❌ FAIL \| .*? \|", f"| Accuracy Validation | ❌ FAIL | {acc} accuracy |", content)

    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"Updated {report_md_path} with latest run data from {date_long}")

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    md_path = root / "PROJECT_COMPLETION_REPORT.md"
    json_path = root / "Validators" / "results" / "validation_report.json"
    update_markdown_report(str(md_path), str(json_path))
