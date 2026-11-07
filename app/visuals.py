"""Pandas-free HTML tables and SVG charts for the Streamlit dashboards."""
from __future__ import annotations
import html
import numpy as np

def process_memory_mb():
    """Process working-set measurements, including the Windows OS peak RSS."""
    try:
        import psutil
        memory=psutil.Process().memory_info()
        peak=getattr(memory,"peak_wset",None)
        return {"rss_mb":memory.rss/1024**2,"peak_rss_mb":peak/1024**2 if peak else None}
    except ImportError:
        pass
    if __import__("sys").platform=="win32":
        try:
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD),
                    ("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),("QuotaPagedPoolUsage",ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),("QuotaNonPagedPoolUsage",ctypes.c_size_t),
                    ("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t),("PrivateUsage",ctypes.c_size_t)]
            counters=Counters(); counters.cb=ctypes.sizeof(counters)
            kernel=ctypes.WinDLL("kernel32",use_last_error=True); psapi=ctypes.WinDLL("psapi",use_last_error=True)
            kernel.GetCurrentProcess.restype=wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
                return {"rss_mb":counters.WorkingSetSize/1024**2,"peak_rss_mb":counters.PeakWorkingSetSize/1024**2}
        except (AttributeError,OSError):
            pass
    return {"rss_mb":None,"peak_rss_mb":None}

def render_html_table(rows, title=None):
    if not rows: return "<p>No measurements available.</p>"
    columns=list(rows[0]); head="".join(f"<th>{html.escape(str(col))}</th>" for col in columns)
    body=[]
    for row in rows:
        cells=[]
        for col in columns:
            value=row.get(col)
            if isinstance(value,(float,np.floating)): text=f"{float(value):.4f}" if np.isfinite(value) else "—"
            else: text="—" if value is None else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>"+"".join(cells)+"</tr>")
    caption=f"<h4>{html.escape(title)}</h4>" if title else ""
    table=f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>"
    return caption+"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;font-size:14px'>"+table+"</table></div>"

def render_svg_chart(series, title="", x_values=None, x_label="", y_label=""):
    width,height=900,390; left,right,top,bottom=72,24,42,76
    flat=[float(v) for values in series.values() for v in values if v is not None and np.isfinite(v)]
    if not flat: return "<p>No finite measurements to plot.</p>"
    ymin,ymax=min(flat),max(flat)
    if ymax-ymin<1e-9: ymin-=.5; ymax+=.5
    count=max(len(values) for values in series.values()); xs=list(x_values if x_values is not None else range(count))
    numeric_x=[float(v) for v in xs]; xmin,xmax=min(numeric_x),max(numeric_x)
    if xmax-xmin<1e-9: xmin-=.5; xmax+=.5
    colors=["#20c7a0","#49a7ff","#ffbd59","#e879f9"]
    sx=lambda x:left+(float(x)-xmin)/(xmax-xmin)*(width-left-right)
    sy=lambda y:top+(ymax-float(y))/(ymax-ymin)*(height-top-bottom)
    parts=[f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}' style='width:100%;height:auto;background:#121e27;border-radius:10px'>"]
    for step in range(5):
        value=ymin+(ymax-ymin)*step/4; y=sy(value)
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='#34434d'/><text x='{left-8}' y='{y+4:.1f}' fill='#c7d2da' text-anchor='end' font-size='12'>{value:.3g}</text>")
    parts.append(f"<text x='{width/2}' y='24' text-anchor='middle' fill='#edf5f6' font-size='17'>{html.escape(title)}</text>")
    for i,(name,values) in enumerate(series.items()):
        color=colors[i%len(colors)]; points=[f"{sx(x):.1f},{sy(v):.1f}" for x,v in zip(xs,values) if v is not None and np.isfinite(v)]
        if points: parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{' '.join(points)}'/>")
        lx=left+i*165
        parts.append(f"<line x1='{lx}' y1='{height-20}' x2='{lx+24}' y2='{height-20}' stroke='{color}' stroke-width='3'/><text x='{lx+30}' y='{height-16}' fill='#edf5f6' font-size='12'>{html.escape(str(name))}</text>")
    parts.append(f"<text x='{width/2}' y='{height-2}' text-anchor='middle' fill='#c7d2da' font-size='12'>{html.escape(x_label)}</text><text transform='translate(18 {height/2}) rotate(-90)' text-anchor='middle' fill='#c7d2da' font-size='12'>{html.escape(y_label)}</text></svg>")
    return "".join(parts)

def render_svg_bars(values, title="", y_label=""):
    valid=[(str(k),float(v)) for k,v in values.items() if v is not None and np.isfinite(v)]
    if not valid: return "<p>No finite measurements to plot.</p>"
    width,height=900,max(220,90+len(valid)*44); left,right,top=220,35,42
    maximum=max(max(v for _,v in valid),1e-8); plot=width-left-right
    parts=[f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}' style='width:100%;height:auto;background:#121e27;border-radius:10px'>",
        f"<text x='{width/2}' y='25' text-anchor='middle' fill='#edf5f6' font-size='17'>{html.escape(title)}</text>"]
    for i,(label,value) in enumerate(valid):
        y=top+i*44; bar=max(1,value/maximum*plot)
        parts.append(f"<text x='{left-10}' y='{y+21}' text-anchor='end' fill='#edf5f6' font-size='12'>{html.escape(label)}</text><rect x='{left}' y='{y}' width='{bar:.1f}' height='26' rx='4' fill='#20c7a0'/><text x='{min(left+bar+8,width-45):.1f}' y='{y+19}' fill='#edf5f6' font-size='12'>{value:.4g}</text>")
    parts.append(f"<text x='{width/2}' y='{height-4}' text-anchor='middle' fill='#c7d2da' font-size='12'>{html.escape(y_label)}</text></svg>")
    return "".join(parts)
