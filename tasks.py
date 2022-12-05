import invoke


def print_banner(msg):
    print("==================================================")
    print("= {} ".format(msg))
    
src_filenames=["src/dIdt8H_5PNe10.cc", "src/Utility.cc"]
pyx_filenames=["src/dIdt.pyx", "src/utility_functions.pyx"]
so_filenames=["build/dIdt.so", "build/utility.so"]

assert len(so_filenames)==len(src_filenames), "error: missing library file names"
assert len(pyx_filenames)==len(src_filenames), "error: missing pyx files"

@invoke.task()
def build_cppmult(c, src_name=src_filenames[0], so_name=so_filenames[0]):
    """Build the shared library for the sample C++ code"""
    print_banner("Building C++ Library")
    invoke.run(
        "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC " + src_name
        "-o " + so_name
    )
    print("* Complete")


def compile_python_module(cpp_name, extension_name, so_name=so_filenames[0]):

	trimmed_so_name=so_name[:-3]
	ex_strig = 
    invoke.run(

        "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC "

        "`python3 -m pybind11 --includes` "

        "-I /mnt/Data_Volume/Computer_Programs/Anaconda/envs/few_env/include/python3.7m -I .  "

        "{0} "

        "-o {1}`python3.7-config --extension-suffix` "

        "-L. -l{2} -Wl,-rpath,.".format(cpp_name, extension_name, trimmed_so_name)

    )
    
@invoke.task(build_cppmult)
def build_cython(c):

    """ Build the cython extension module """

    print_banner("Building Cython Module")

    # Run cython on the pyx file to create a .cpp file

    invoke.run("cython --cplus -3 dIdt.pyx -o dIdt.cpp")


    # Compile and link the cython wrapper library

    compile_python_module("dIdt.cpp", "dIdt")

    print("* Complete")
