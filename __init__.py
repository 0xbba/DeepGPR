import ctypes
import os
import subprocess
import platform
from pathlib import Path

# 1. 检测当前操作系统，并配置对应的扩展名和 nvcc 编译命令
system_name = platform.system()
lib_dir = Path(__file__).parent / 'lib'
cu_file = lib_dir / 'deepgpr.cu'

if system_name == "Windows":
    lib_extension = ".dll"
    lib_filename = f'deepgpr{lib_extension}'
    lib_path_obj = lib_dir / lib_filename
    
    # Windows 下直接调用 nvcc 的命令参数
    nvcc_cmd = [
        'nvcc', '-shared', 
        '-o', str(lib_path_obj), 
        str(cu_file)
    ]
else:
    lib_extension = ".so"
    lib_filename = f'deepgpr{lib_extension}'
    lib_path_obj = lib_dir / lib_filename
    
    # Linux 下直接调用 nvcc 的命令参数 (-fPIC 和 ABI 设置)
    nvcc_cmd = [
        'nvcc', '-shared', '-Xcompiler', '-fPIC', 
        '-D_GLIBCXX_USE_CXX11_ABI=0', 
        '-o', str(lib_path_obj), 
        str(cu_file)
    ]

# 2. 检查动态库是否存在，不存在则直接使用 Python 调用 nvcc 进行编译
if not lib_path_obj.is_file():
    print(f'Compiling CUDA extension for {system_name} directly via nvcc...')
    try:
        # 直接执行 nvcc 命令，完全摆脱 make 依赖
        subprocess.run(nvcc_cmd, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "Compilation failed: 'nvcc' command not found. "
            "Please ensure NVIDIA CUDA Toolkit is installed and 'nvcc' is added to your system PATH."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Compilation failed with error code {e.returncode}.")

# 3. 加载编译好的动态链接库
lib_path = str(lib_path_obj)
lib = ctypes.cdll.LoadLibrary(lib_path)

# 4. 定义 C 函数的参数和返回值类型
lib.forward.argtypes = [
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
    ctypes.c_int ]

lib.restype = None

lib.backward.argtypes = [
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
    ctypes.c_int, ctypes.c_int ]

lib.restype = None

__all__ = ['lib']