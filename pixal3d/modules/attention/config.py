from typing import *

BACKEND = 'flash_attn' 
DEBUG = False
SAGE = False        # PIXAL3D_SAGE: the dense full self-attention through SageAttention (modules/sage.py)

def __from_env():
    import os
    
    global BACKEND
    global DEBUG
    global SAGE
    
    env_attn_backend = os.environ.get('ATTN_BACKEND')
    env_attn_debug = os.environ.get('ATTN_DEBUG')
    
    if env_attn_backend is not None and env_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'flash_attn_4', 'sdpa', 'naive']:
        BACKEND = env_attn_backend
    if env_attn_debug is not None:
        DEBUG = env_attn_debug == '1'

    from .. import sage
    SAGE = sage.from_env()

    print(f"[ATTENTION] Using backend: {BACKEND}"
          + ("; full self-attention: SageAttention (PIXAL3D_SAGE)" if SAGE else ""))
        

__from_env()
    

def set_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'flash_attn_4', 'sdpa', 'naive']):
    global BACKEND
    BACKEND = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_sage(on: bool):
    global SAGE
    SAGE = bool(on)
