"""Select an initialized software EGL device without changing robosuite."""
from pathlib import Path
from OpenGL import EGL
from mujoco.egl import egl_ext as E
root=Path(__file__).resolve().parent.parent
for i,dev in enumerate(E.eglQueryDevicesEXT()):
    display=E.eglGetPlatformDisplayEXT(E.EGL_PLATFORM_DEVICE_EXT,dev,None)
    try:
        ok=EGL.eglInitialize(display,None,None)
        vendor=EGL.eglQueryString(display,EGL.EGL_VENDOR)
    except Exception:
        continue
    if ok and vendor==b'Mesa Project':
        dest=root/'.config/libero/egl_device_id'
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(str(i)+'\n')
        print('Selected local Mesa EGL device',i)
        EGL.eglTerminate(display)
        break
else:
    raise RuntimeError('No usable local Mesa EGL device')
