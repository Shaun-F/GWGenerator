from setuptools import setup
from distutils.extension import Extension


cpu_extensions=dict(
    libraries=["gsl", "gslcblas"],
    language="c++",
    runtime_library_dirs=[],
    extra_compile_args=["-O3", "-shared", "-std=c++11", "-fPIC"],
    include_dirs=["/usr/lib64",
                    "/mnt/Data_Volume/Documents/software/Anaconda/envs/few_env/include/python3.7m",
                    "/home/shaunf/Documents/Computer/Code/projects/Massive_Vector_Field_Dynamical_Friction/ProcaAroundKerr/GWGenerator/include",
                    "include"
                    ],
    library_dirs=None
)


frequency_ext = Extension(
            "pyUtility",
            sources=["src/Utility.cc", "src/Utility.pyx"],
            **cpu_extensions
)

flux_ext = Extension(
            "pyAnalyticFluxes",
            sources=["src/dIdt8H_5PNe10.cc", "src/dIdt.pyx"],
            **cpu_extensions
)
print(cpu_extensions)

extensions = [flux_ext, frequency_ext]


setup(
    name="GWGen",
    version="0.1",
    description="Calculating modified waveform due to precense of Proca cloud around spinning black hole",
    author="Shaun Fell",
    packages=["GWGen"],
    ext_modules=extensions,

)
