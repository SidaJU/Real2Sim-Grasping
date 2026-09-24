"""Optional: scripted lighting (alternative to setting lights in the editor).

04_render_orbit.py deliberately does not touch lighting. Run this once if you
want a reproducible, 3DGS-friendly setup instead of clicking in the GUI:

  * DomeLight 1000  - uniform ambient, so every orbit view is lit the same way.
                      Uniform lighting is what 3DGS wants; inconsistent shading
                      across views degrades reconstruction.
  * DistantLight 0  - no directional key light on purpose: it makes brightness
                      vary per view for the same reason.

If renders come out pure white while the viewport looks fine, that is exposure:
the viewport auto-exposes but writers have no tone mapping. Fix by lowering the
lights here, or by setting a negative EV on the camera in 04 (GetExposureAttr).
"""

import omni.usd
from pxr import UsdLux

stage = omni.usd.get_context().get_stage()

dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.GetIntensityAttr().Set(1000.0)
print("DomeLight /World/DomeLight intensity = 1000")

sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
sun.GetIntensityAttr().Set(0.0)
print("DistantLight /World/SunLight intensity = 0 (off)")
