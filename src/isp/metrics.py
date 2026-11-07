"""Image quality metrics; optional scikit-image and colour-science backends."""
import numpy as np


def psnr(pred, target):
    mse = float(np.mean((np.asarray(pred, np.float64) - np.asarray(target, np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(pred, target):
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(np.asarray(pred), np.asarray(target), channel_axis=2, data_range=1.0))
    except ImportError:
        # Dependency-light, local 7x7 luminance SSIM fallback.
        x=np.asarray(pred,np.float32) @ np.array([.2126,.7152,.0722],np.float32)
        y=np.asarray(target,np.float32) @ np.array([.2126,.7152,.0722],np.float32)
        def box_mean(a, radius=3):
            h,w=a.shape; size=radius*2+1
            vpad=np.pad(a,((radius,radius),(0,0)),mode="reflect")
            vertical=np.zeros((h,w),np.float32)
            for i in range(size): vertical += vpad[i:i+h]
            hpad=np.pad(vertical,((0,0),(radius,radius)),mode="reflect")
            result=np.zeros((h,w),np.float32)
            for i in range(size): result += hpad[:,i:i+w]
            return result/(size*size)
        ux,uy=box_mean(x),box_mean(y)
        vx=np.maximum(box_mean(x*x)-ux*ux,0); vy=np.maximum(box_mean(y*y)-uy*uy,0)
        cov=box_mean(x*y)-ux*uy; c1=.01**2; c2=.03**2
        score=((2*ux*uy+c1)*(2*cov+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2))
        return float(np.mean(score))


def delta_e_76(pred, target):
    """Mean CIE76 Delta E after sRGB to CIELAB conversion."""
    def lab(rgb):
        x = np.asarray(rgb, np.float64)
        x = np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
        xyz = x @ np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]]).T
        xyz /= np.array([.95047, 1.0, 1.08883])
        f = np.where(xyz > (6/29)**3, np.cbrt(xyz), xyz/(3*(6/29)**2)+4/29)
        return np.stack([116*f[...,1]-16, 500*(f[...,0]-f[...,1]), 200*(f[...,1]-f[...,2])], -1)
    return float(np.mean(np.linalg.norm(lab(pred) - lab(target), axis=-1)))
