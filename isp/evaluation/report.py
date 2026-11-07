"""HTML, CSV-ready and dependency-free PDF experiment report export."""
from __future__ import annotations
import html,json

def build_html_report(payload):
    methods=payload.get("methods",{})
    rows=[]
    for name,metrics in methods.items():
        for metric,value in metrics.items():
            if isinstance(value,dict) and "mean" in value:
                rows.append(f"<tr><td>{html.escape(name)}</td><td>{html.escape(metric)}</td><td>{value['mean']:.5f}</td><td>{value['std']:.5f}</td><td>{value['n']}</td></tr>")
    table="".join(rows) or "<tr><td colspan='5'>No measured benchmark available yet.</td></tr>"
    metadata=html.escape(json.dumps({k:v for k,v in payload.items() if k not in {"per_image","methods"}},indent=2))
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>HybridISP experiment report</title>
<style>body{{font:16px Arial;margin:2rem;color:#182630}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd;padding:.6rem;text-align:left}}th{{background:#e9f2f3}}pre{{background:#f1f4f5;padding:1rem;overflow:auto}}</style></head><body>
<h1>HybridISP experiment report</h1><p>Dataset: {html.escape(str(payload.get('data_root','unknown')))} · images: {payload.get('test_images',0)} · synthetic: {payload.get('synthetic_dataset',False)}</p>
<table><thead><tr><th>Pipeline</th><th>Metric</th><th>Mean</th><th>Std</th><th>N</th></tr></thead><tbody>{table}</tbody></table><h2>Run metadata</h2><pre>{metadata}</pre></body></html>"""

def build_pdf_report(lines):
    """Create a small one-page PDF without requiring a document-rendering library."""
    safe=[]
    for line in lines[:48]:
        text=str(line).encode("ascii","replace").decode("ascii").replace("\\","\\\\").replace("(","\\(").replace(")","\\)")
        safe.append(f"({text[:110]}) Tj 0 -14 Td")
    stream=("BT /F1 10 Tf 42 800 Td "+" ".join(safe)+" ET").encode("ascii")
    objects=[b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>",
        b"<< /Length "+str(len(stream)).encode()+b" >>\nstream\n"+stream+b"\nendstream"]
    output=bytearray(b"%PDF-1.4\n"); offsets=[0]
    for i,obj in enumerate(objects,1): offsets.append(len(output)); output+=f"{i} 0 obj\n".encode()+obj+b"\nendobj\n"
    xref=len(output); output+=f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode()
    for off in offsets[1:]: output+=f"{off:010d} 00000 n \n".encode()
    output+=f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(output)
