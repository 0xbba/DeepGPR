import ctypes
import os
import subprocess
import platform
from pathlib import Path


system_name = platform.system()
lib_dir = Path(__file__).parent / 'lib'
cu_file = lib_dir / 'deepgpr.cu'

if system_name == "Windows":
    lib_extension = ".dll"
    lib_filename = f'deepgpr{lib_extension}'
    lib_path_obj = lib_dir / lib_filename
    
    nvcc_cmd = [
        'nvcc', '-shared', 
        '-o', str(lib_path_obj), 
        str(cu_file)
    ]
else:
    lib_extension = ".so"
    lib_filename = f'deepgpr{lib_extension}'
    lib_path_obj = lib_dir / lib_filename
    
    nvcc_cmd = [
        'nvcc', '-shared', '-Xcompiler', '-fPIC', 
        '-D_GLIBCXX_USE_CXX11_ABI=0', 
        '-o', str(lib_path_obj), 
        str(cu_file)
    ]

if not lib_path_obj.is_file():
    print(f'Compiling CUDA extension for {system_name} directly via nvcc...')
    try:
        subprocess.run(nvcc_cmd, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "Compilation failed: 'nvcc' command not found. "
            "Please ensure NVIDIA CUDA Toolkit is installed and 'nvcc' is added to your system PATH."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Compilation failed with error code {e.returncode}.")

lib_path = str(lib_path_obj)
c_lib = ctypes.cdll.LoadLibrary(lib_path)


c_lib.forward.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),

    ctypes.c_int, ctypes.c_int, ctypes.c_int, 
    ctypes.c_int, ctypes.c_int, ctypes.c_int, 

    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    
    ctypes.c_float,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int 
]
c_lib.forward.restype = None  

c_lib.backward.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),#16
    
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),#40

    ctypes.c_int, ctypes.c_int, ctypes.c_int, 
    ctypes.c_int, ctypes.c_int, ctypes.c_int, 

    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), #58
    
    ctypes.c_float,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int ,ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int 
]
c_lib.backward.restype = None 
__all__ = ['c_lib']


from .common import *
from .compute2 import *
from .multiscale import *
from .wavelet import *