"""Shared image processing and display helpers for the multipage dashboard."""
from __future__ import annotations
import html
import numpy as np
from PIL import Image
import streamlit as st
from app.state import active_models
from isp.classical.pipeline import ClassicalISP
from isp.hybrid.hybrid_pipeline import HybridISPPipeline
from isp.classical.tone import tone_map
from isp.classical.sharpen import unsharp_mask
from isp.metrics.image_metrics import delta_e_76,delta_e_2000,ms_ssim,lpips_alex
from src.isp.metrics import psnr,ssim

def get_image():
    from app.state import get_input
    return get_input()

def make_pipeline():
    models,meta,device=active_models()
    config={"tile_size":int(st.session_state.get("tile_size",512)),"tile_overlap":64,
            "fusion_alpha":.85,"tone_method":st.session_state.get("tone_method","aces"),
            "demosaic_method":st.session_state.get("demosaic_method","bilinear"),
            "denoise_method":st.session_state.get("denoise_method","bilateral"),
            "wb_gains":st.session_state.get("wb_gains",(1.5,1.,1.8)),
            "fusion_alpha":st.session_state.get("fusion_alpha",.85),
            "exposure":st.session_state.get("exposure",1.),"sharpen_amount":st.session_state.get("sharpen_amount",.2),
            "enabled_stages":st.session_state.get("enabled_stages",{})}
    return HybridISPPipeline(ClassicalISP(),models.get("hybrid_refiner"),models.get("mobile_unet"),device,config),models,meta,device

def process_modes(image,kind,pattern=None,black=None,white=None,adaptive=None):
    pattern=pattern or st.session_state.get("bayer_pattern","RGGB")
    black=st.session_state.get("black_level",0.) if black is None else black
    white=st.session_state.get("white_level",65535.) if white is None else white
    adaptive=st.session_state.get("adaptive",True) if adaptive is None else adaptive
    pipeline,models,meta,device=make_pipeline()
    if kind=="raw":
        traditional=pipeline.process_raw(image,pattern,black,white,adaptive,"traditional")
        hybrid=pipeline.process_raw(image,pattern,black,white,adaptive,"hybrid")
        learned=None
        if models.get("mobile_unet") is not None:
            learned=pipeline.process_raw(image,pattern,black,white,adaptive,"learned")
        return {"traditional":traditional.traditional,"learned":learned.learned if learned else None,
                "hybrid":hybrid.hybrid,"policy":hybrid.policy,"timings":hybrid.timings_ms,"meta":meta,"device":device}
    classical=ClassicalISP().run(image,"rgb",wb_method="manual",wb_gains=st.session_state.get("wb_gains",(1,1,1)),
        denoise_method=st.session_state.get("denoise_method","bilateral"),noise_sigma=.02,
        tone_method=st.session_state.get("tone_method","aces"),exposure=st.session_state.get("exposure",1.),
        sharpen_amount=st.session_state.get("sharpen_amount",.2),enabled=st.session_state.get("enabled_stages",{}))
    return {"traditional":classical.image,"learned":None,"hybrid":classical.image,"policy":{},"timings":classical.timings_ms,"meta":meta,"device":device}

def visualized_raw(raw,pattern="RGGB",black=0,white=None):
    x=np.asarray(raw,np.float32); x=np.clip((x-black)/max((white or float(np.max(x)))-black,1),0,1)
    if x.ndim==3: return np.clip(x[...,:3],0,1)
    maps={"RGGB":((0,0,0),(0,1,1),(1,0,1),(1,1,2)),"BGGR":((0,0,2),(0,1,1),(1,0,1),(1,1,0)),
          "GRBG":((0,0,1),(0,1,0),(1,0,2),(1,1,1)),"GBRG":((0,0,1),(0,1,2),(1,0,0),(1,1,1))}
    out=np.zeros((*x.shape,3),np.float32)
    for y,xx,c in maps.get(pattern,maps["RGGB"]): out[y::2,xx::2,c]=x[y::2,xx::2]
    for c in range(3):
        mask=np.zeros(x.shape,np.float32)
        for y,xx,ch in maps.get(pattern,maps["RGGB"]):
            if ch==c: mask[y::2,xx::2]=1
        # Nearest same-colour interpolation purely for visualization.
        for _ in range(2):
            p=np.pad(out[...,c],1,mode="edge"); m=np.pad(mask,1,mode="edge")
            num=np.zeros_like(mask); den=np.zeros_like(mask)
            for dy,dx in ((0,1),(1,0),(1,2),(2,1)):
                num+=p[dy:dy+x.shape[0],dx:dx+x.shape[1]]*m[dy:dy+x.shape[0],dx:dx+x.shape[1]]
                den+=m[dy:dy+x.shape[0],dx:dx+x.shape[1]]
            fill=(mask==0)&(den>0); out[...,c][fill]=num[fill]/den[fill]; mask[fill]=1
    return np.clip(out,0,1)

def to_image(array): return np.uint8(np.clip(array,0,1)*255+.5)

def camera_style_preview(rgb, profile="huawei"):
    """Create a deterministic camera-style rendering preview, not a real device capture."""
    x=np.clip(np.asarray(rgb,np.float32),0,1)
    if x.ndim!=3 or x.shape[-1]<3:
        raise ValueError(f"Camera-style preview expects H×W×3 RGB, got {x.shape}")
    x=x[...,:3]
    luma=np.sum(x*np.array([.2126,.7152,.0722],np.float32),axis=-1,keepdims=True)
    if profile.lower() in {"huawei","p20"}:
        # Bright, saturated, locally crisp phone-style preview.
        y=np.clip((x-.46)*1.10+.46,0,1)
        lum=np.sum(y*np.array([.2126,.7152,.0722],np.float32),axis=-1,keepdims=True)
        y=np.clip(lum+(y-lum)*1.10,0,1)*np.array([1.018,1.0,.986],np.float32)
        pad=np.pad(y,((1,1),(1,1),(0,0)),mode="reflect")
        blur=sum(pad[dy:dy+y.shape[0],dx:dx+y.shape[1]] for dy in range(3) for dx in range(3))/9
        return np.clip(y+.16*(y-blur),0,1).astype(np.float32)
    if profile.lower() in {"canon","dslr"}:
        # Neutral, softer contrast with a gentle highlight roll-off.
        y=np.clip((x-.5)*.94+.5,0,1)
        y=y/(1+.035*y)
        lum=np.sum(y*np.array([.2126,.7152,.0722],np.float32),axis=-1,keepdims=True)
        return np.clip(lum+(y-lum)*.97,0,1).astype(np.float32)
    raise ValueError("profile must be 'huawei' or 'canon'")

def score_images(pred,target):
    return {"PSNR dB":psnr(pred,target),"SSIM":ssim(pred,target),"MS-SSIM":ms_ssim(pred,target),
            "ΔE76":delta_e_76(pred,target),"CIEDE2000":delta_e_2000(pred,target),"LPIPS AlexNet":lpips_alex(pred,target)}

def upload_reference(key="ref_image"):
    file=st.file_uploader("Aligned RGB ground truth (optional)",type=["png","jpg","jpeg","tif","tiff"],key=key)
    if file is None: return None
    return np.asarray(Image.open(file).convert("RGB"),np.float32)/255

def render_html_table(rows, title=None):
    """Render small result tables without importing pandas or native dataframe DLLs."""
    if not rows: return "<p>No measurements available.</p>"
    columns=list(rows[0])
    head="".join(f"<th>{html.escape(str(col))}</th>" for col in columns)
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
    return caption+"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;font-size:14px'>"+f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"

def render_svg_chart(series, title="", x_values=None, x_label="", y_label=""):
    """Create a responsive SVG line chart from lists of measured numeric values."""
    width,height=900,390; left,right,top,bottom=72,24,42,76
    flat=[float(value) for values in series.values() for value in values if value is not None and np.isfinite(value)]
    if not flat: return "<p>No finite measurements to plot.</p>"
    ymin,ymax=min(flat),max(flat)
    if ymax-ymin<1e-9: ymin-=.5; ymax+=.5
    count=max(len(values) for values in series.values()); xs=list(x_values if x_values is not None else range(count))
    numeric_x=[float(value) for value in xs]
    xmin,xmax=min(numeric_x),max(numeric_x)
    if xmax-xmin<1e-9: xmin-=.5; xmax+=.5
    colors=["#20c7a0","#49a7ff","#ffbd59","#e879f9"]
    sx=lambda x:left+(float(x)-xmin)/(xmax-xmin)*(width-left-right)
    sy=lambda y:top+(ymax-float(y))/(ymax-ymin)*(height-top-bottom)
    parts=[f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}' style='width:100%;height:auto;background:#121e27;border-radius:10px'>"]
    for step in range(5):
        value=ymin+(ymax-ymin)*step/4; y=sy(value)
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='#34434d'/><text x='{left-8}' y='{y+4:.1f}' fill='#c7d2da' text-anchor='end' font-size='12'>{value:.3g}</text>")
    parts.append(f"<text x='{width/2}' y='24' text-anchor='middle' fill='#edf5f6' font-size='17'>{html.escape(title)}</text>")
    for index,(name,values) in enumerate(series.items()):
        color=colors[index%len(colors)]; points=[]
        for x,value in zip(xs,values):
            if value is not None and np.isfinite(value): points.append(f"{sx(x):.1f},{sy(value):.1f}")
        if points: parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{' '.join(points)}'/>")
        lx=left+index*165
        parts.append(f"<line x1='{lx}' y1='{height-20}' x2='{lx+24}' y2='{height-20}' stroke='{color}' stroke-width='3'/><text x='{lx+30}' y='{height-16}' fill='#edf5f6' font-size='12'>{html.escape(str(name))}</text>")
    parts.append(f"<text x='{width/2}' y='{height-2}' text-anchor='middle' fill='#c7d2da' font-size='12'>{html.escape(x_label)}</text><text transform='translate(18 {height/2}) rotate(-90)' text-anchor='middle' fill='#c7d2da' font-size='12'>{html.escape(y_label)}</text></svg>")
    return "".join(parts)

def render_svg_bars(values, title="", y_label=""):
    """Render a compact bar chart without charting/dataframe dependencies."""
    valid=[(str(key),float(value)) for key,value in values.items() if value is not None and np.isfinite(value)]
    if not valid: return "<p>No finite measurements to plot.</p>"
    width,height=900,max(220,90+len(valid)*44); left,right,top,bottom=220,35,42,24
    maximum=max(max(value for _,value in valid),1e-8); plot=width-left-right
    parts=[f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}' style='width:100%;height:auto;background:#121e27;border-radius:10px'>",
        f"<text x='{width/2}' y='25' text-anchor='middle' fill='#edf5f6' font-size='17'>{html.escape(title)}</text>"]
    for index,(label,value) in enumerate(valid):
        y=top+index*44; bar=max(1,value/maximum*plot)
        parts.append(f"<text x='{left-10}' y='{y+21}' text-anchor='end' fill='#edf5f6' font-size='12'>{html.escape(label)}</text><rect x='{left}' y='{y}' width='{bar:.1f}' height='26' rx='4' fill='#20c7a0'/><text x='{min(left+bar+8,width-45):.1f}' y='{y+19}' fill='#edf5f6' font-size='12'>{value:.4g}</text>")
    parts.append(f"<text x='{width/2}' y='{height-4}' text-anchor='middle' fill='#c7d2da' font-size='12'>{html.escape(y_label)}</text></svg>")
    return "".join(parts)
