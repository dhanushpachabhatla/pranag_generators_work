import json
import os
import datetime

def generate_dashboard(metrics: dict) -> str:
    """
    Generates a beautiful HTML dashboard summarizing the run.
    """
    top_designs_path = metrics.get("top_designs_json", "")
    failed_designs_path = metrics.get("failed_designs_json", "")
    
    # Load data
    top_designs = []
    if os.path.exists(top_designs_path):
        with open(top_designs_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                top_designs = data.get("top_100", []) if isinstance(data, dict) else data
            except:
                pass
                
    failed_designs = []
    if os.path.exists(failed_designs_path):
        with open(failed_designs_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                failed_designs = data.get("failed_designs", []) if isinstance(data, dict) else data
            except:
                pass
                
    execution_time = metrics.get("total_time_seconds", 0)
    total_processed = metrics.get("total_processed", 0)
    survivors = metrics.get("total_survivors", 0)
    chain = metrics.get("execution_chain", [])
    execution_chain_html = " &rarr; ".join([f"<span style='background:#e2e8f0; padding:2px 8px; border-radius:12px; font-size:14px;'>{c}</span>" for c in chain])
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate failure rates
    deleted_count = total_processed - survivors
    pass_rate = (survivors / total_processed * 100) if total_processed > 0 else 0.0
    
    # Build Top Designs Table Rows
    designs_rows = ""
    for i, d in enumerate(top_designs):
        scores_html = ""
        domain_scores = d.get("domain_scores", {})
        # Iterate to find domain scores flexibly
        for k, v in domain_scores.items():
            if "_score" in k:
                scores_html += f"<div><small>{k.replace('_score', '').title()}:</small> <strong>{v:.2f}</strong></div>"
                
        designs_rows += f"""
        <tr>
            <td><strong>#{i+1}</strong></td>
            <td>{d.get('name', d.get('entity_id', 'Unknown'))}</td>
            <td><strong style="color: #059669;">{d.get('viability_score', 0):.3f}</strong></td>
            <td>{scores_html}</td>
        </tr>
        """
        
    # Build Failed Designs Rows
    failed_rows = ""
    for f in failed_designs:
        failed_rows += f"""
        <tr>
            <td>{f.get('name', f.get('entity_id', 'Unknown'))}</td>
            <td><strong style="color: #ef4444;">{f.get('viability_score', 0):.3f}</strong></td>
            <td><span style="color:#b91c1c;">{f.get('failure_reason', 'Failed Physics Thresholds')}</span></td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRANA-G Execution Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 0; background: #f8fafc; color: #1e293b; }}
        .header {{ background: linear-gradient(135deg, #0f172a 0%, #334155 100%); color: white; padding: 30px 40px; }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 28px; font-weight: 600; letter-spacing: 1px; }}
        .header p {{ margin: 0; color: #cbd5e1; font-size: 14px; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border-top: 4px solid #3b82f6; }}
        .metric-card.success {{ border-top-color: #10b981; }}
        .metric-card.danger {{ border-top-color: #ef4444; }}
        .metric-card.neutral {{ border-top-color: #8b5cf6; }}
        .metric-card h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
        .metric-card .value {{ font-size: 32px; font-weight: 700; color: #0f172a; margin: 0; }}
        
        .section-box {{ background: white; border-radius: 12px; padding: 25px; margin-bottom: 30px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
        .section-box h2 {{ margin-top: 0; border-bottom: 2px solid #f1f5f9; padding-bottom: 15px; color: #0f172a; font-size: 20px; }}
        
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f8fafc; font-weight: 600; color: #475569; text-transform: uppercase; font-size: 13px; }}
        tr:hover {{ background-color: #f1f5f9; }}
        
        .tabs {{ display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
        .tab-btn {{ background: none; border: none; padding: 10px 20px; font-size: 16px; font-weight: 600; color: #64748b; cursor: pointer; border-radius: 6px; }}
        .tab-btn.active {{ background: #eff6ff; color: #2563eb; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>PRANA-G Execution Dashboard</h1>
        <p>Execution Completed: {timestamp} | Runtime: {execution_time} seconds</p>
    </div>

    <div class="container">
        <!-- Overview Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Entities Scanned</h3>
                <p class="value">{total_processed:,}</p>
            </div>
            <div class="metric-card success">
                <h3>Physics Survivors</h3>
                <p class="value">{survivors:,}</p>
            </div>
            <div class="metric-card danger">
                <h3>Entities Deleted (< 0.7)</h3>
                <p class="value">{deleted_count:,}</p>
            </div>
            <div class="metric-card neutral">
                <h3>Overall Pass Rate</h3>
                <p class="value">{pass_rate:.2f}%</p>
            </div>
        </div>

        <div class="section-box">
            <h2>DAG Execution Sequence</h2>
            <div style="padding: 15px; background: #f8fafc; border-radius: 8px; border: 1px dashed #cbd5e1;">
                {execution_chain_html}
            </div>
        </div>

        <!-- TABS -->
        <div class="section-box">
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('top-designs')">Top Validated Designs</button>
                <button class="tab-btn" onclick="switchTab('failed-designs')">Cross-Domain Failures Log</button>
            </div>

            <!-- Top Designs Tab -->
            <div id="top-designs" class="tab-content active">
                {'''<table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Entity Name</th>
                            <th>Viability Score</th>
                            <th>Domain Specific Scores</th>
                        </tr>
                    </thead>
                    <tbody>
                        ''' + designs_rows + '''
                    </tbody>
                </table>''' if len(top_designs) > 0 else "<p style='color:#64748b;'>No designs survived the physics simulation.</p>"}
            </div>

            <!-- Failed Designs Tab -->
            <div id="failed-designs" class="tab-content">
                {'''<table>
                    <thead>
                        <tr>
                            <th>Entity Name</th>
                            <th>Viability Score</th>
                            <th>Hazard Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                        ''' + failed_rows + '''
                    </tbody>
                </table>''' if len(failed_designs) > 0 else "<p style='color:#10b981; font-weight:600;'>Excellent! No candidates failed the Cross-Domain Hazard Validation.</p>"}
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabId) {{
            // Hide all contents
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            // Remove active from all buttons
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            
            // Show target
            document.getElementById(tabId).classList.add('active');
            event.currentTarget.classList.add('active');
        }}
    </script>
</body>
</html>
"""

    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pranag_dashboard.html'))
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
        
    return output_file
