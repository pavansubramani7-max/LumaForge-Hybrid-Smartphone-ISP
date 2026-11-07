"""RAW loading, black/white normalization, Bayer detection and RGGB packing."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

BAYER_PATTERNS={
    "RGGB": (("R","G"),("G","B")),
    "BGGR": (("B","G"),("G","R")),
    "GRBG": (("G","R"),("B","G")),
    "GBRG": (("G","B"),("R","G")),
}

def load_raw(path: str | Path):
    path=Path(path); suffix=path.suffix.lower(); metadata={"path":str(path)}
    if suffix==".npy": raw=np.load(path,allow_pickle=False)
    elif suffix==".dng":
        try: import rawpy
        except ImportError as exc: raise RuntimeError("Install rawpy to decode DNG files") from exc
        with rawpy.imread(str(path)) as sensor:
            raw=sensor.raw_image_visible.copy(); metadata.update({"pattern_indices":sensor.raw_pattern.tolist() if sensor.raw_pattern is not None else None,
                "black_levels":list(sensor.black_level_per_channel),"white_level":int(sensor.white_level)})
            metadata["bayer_pattern"]=pattern_from_indices(sensor.raw_pattern)
    elif suffix in {".png",".tif",".tiff"}: raw=np.asarray(Image.open(path))
    elif suffix==".raw": raise ValueError("Headerless .raw requires width/height/bit-depth in the camera config")
    else: raise ValueError(f"Unsupported RAW format: {suffix}")
    metadata.update({"shape":list(raw.shape),"dtype":str(raw.dtype)})
    return raw,metadata

def pattern_from_indices(indices):
    if indices is None: return "unknown"
    color=np.asarray(indices).reshape(2,2)
    lookup={"RGGB":np.array([[0,1],[1,2]]),"BGGR":np.array([[2,1],[1,0]]),
            "GRBG":np.array([[1,0],[2,1]]),"GBRG":np.array([[1,2],[0,1]])}
    for name,expected in lookup.items():
        if np.array_equal(color,expected): return name
    return "unknown"

def normalize_raw(raw, black_level=0, white_level=None):
    raw=np.asarray(raw)
    x=raw.astype(np.float32)
    if white_level is None: white_level=float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype,np.integer) else 1.
    if np.issubdtype(raw.dtype,np.floating) and np.nanmax(x)<=1.5 and white_level>1: white_level=1.
    return np.clip((x-black_level)/max(float(white_level)-float(black_level),1e-8),0,1)

def guess_bayer_pattern(mosaic):
    """Low-confidence signal-statistics guess; camera metadata is authoritative."""
    x=np.asarray(mosaic,np.float32)
    if x.ndim!=2 or min(x.shape)<8: return {"pattern":"unknown","confidence":0.0,"method":"insufficient mosaic"}
    p=[x[0::2,0::2].ravel(),x[0::2,1::2].ravel(),x[1::2,0::2].ravel(),x[1::2,1::2].ravel()]
    def corr(a,b):
        n=min(len(a),len(b),20000); a=a[:n]; b=b[:n]
        if np.std(a)<1e-8 or np.std(b)<1e-8: return 0.
        return float(np.corrcoef(a,b)[0,1])
    g=(1,2) if corr(p[1],p[2])>=corr(p[0],p[3]) else (0,3)
    remaining=[i for i in range(4) if i not in g]
    r,b=(remaining if np.mean(p[remaining[0]])>=np.mean(p[remaining[1]]) else remaining[::-1])
    symbols=[None]*4; symbols[r]="R"; symbols[b]="B"; symbols[g[0]]=symbols[g[1]]="G"
    name="".join(symbols)
    gap=abs(corr(p[1],p[2])-corr(p[0],p[3]))
    return {"pattern":name if name in BAYER_PATTERNS else "unknown","confidence":float(np.clip(.3+.45*gap,.3,.75)),"method":"heuristic; verify metadata"}

def pack_bayer(mosaic, pattern="RGGB"):
    x=np.asarray(mosaic)
    if x.ndim!=2: raise ValueError("Bayer packing requires a 2D mosaic")
    if pattern not in BAYER_PATTERNS: raise ValueError(f"Unknown Bayer pattern {pattern}")
    h,w=x.shape; x=x[:h-h%2,:w-w%2]
    grid=BAYER_PATTERNS[pattern]
    coords={"R":[],"G":[],"B":[]}
    for y in range(2):
        for xx in range(2): coords[grid[y][xx]].append((y,xx))
    r=coords["R"][0]; g1,g2=coords["G"]; b=coords["B"][0]
    return np.stack([x[r[0]::2,r[1]::2],x[g1[0]::2,g1[1]::2],x[g2[0]::2,g2[1]::2],x[b[0]::2,b[1]::2]],axis=-1)

def unpack_bayer(packed, pattern="RGGB"):
    p=np.asarray(packed)
    if p.ndim!=3 or p.shape[-1]!=4: raise ValueError("Packed Bayer input must be H/2 x W/2 x 4")
    if pattern not in BAYER_PATTERNS: raise ValueError(f"Unknown Bayer pattern {pattern}")
    h,w=p.shape[:2]; mosaic=np.empty((h*2,w*2),p.dtype); grid=BAYER_PATTERNS[pattern]
    coords={"R":[],"G":[],"B":[]}
    for y in range(2):
        for x in range(2): coords[grid[y][x]].append((y,x))
    r=coords["R"][0]; g1,g2=coords["G"]; b=coords["B"][0]
    for channel,(y,x) in enumerate((r,g1,g2,b)): mosaic[y::2,x::2]=p[...,channel]
    return mosaic

def augment_packed(packed,pattern="RGGB",operation="identity"):
    """Apply a geometric transform while repacking into semantic R/G1/G2/B order."""
    if operation=="identity": return np.asarray(packed),pattern
    mosaic=unpack_bayer(packed,pattern); grid=np.asarray(BAYER_PATTERNS[pattern])
    transforms={"hflip":(lambda x:np.fliplr(x),lambda x:np.fliplr(x)),
                "vflip":(lambda x:np.flipud(x),lambda x:np.flipud(x)),
                "rot90":(lambda x:np.rot90(x,1),lambda x:np.rot90(x,1)),
                "rot180":(lambda x:np.rot90(x,2),lambda x:np.rot90(x,2)),
                "rot270":(lambda x:np.rot90(x,3),lambda x:np.rot90(x,3))}
    if operation not in transforms: raise ValueError(f"Unknown Bayer-preserving transform {operation}")
    mosaic_fn,rgb_fn=transforms[operation]
    transformed=mosaic_fn(mosaic); new_grid=mosaic_fn(grid)
    new_pattern="".join(str(v) for row in new_grid for v in row)
    result=pack_bayer(transformed,new_pattern)
    return result,new_pattern
