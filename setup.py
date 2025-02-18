from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='hello_cuda',
      ext_modules=[cpp_extension.CppExtension('hello_cuda', ['cuda.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})