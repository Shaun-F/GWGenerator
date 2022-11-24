import invoke


def print_banner(msg):
    print("==================================================")
    print("= {} ".format(msg))

@invoke.task()
def build_cppmult(c):
    """Build the shared library for the sample C++ code"""
    print_banner("Building C++ Library")
    invoke.run(
        "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC Utility.cc "
        "-o libUtility.so "
    )
    print("* Complete")


def compile_python_module(cpp_name, extension_name):

    invoke.run(

        "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC "

        "`python3 -m pybind11 --includes` "

        "-I /usr/include/python3.10 -I .  "

        "{0} "

        "-o {1}`python3.10-config --extension-suffix` "

        "-L. -lUtility -Wl,-rpath,.".format(cpp_name, extension_name)

    )
    
@invoke.task(build_cppmult)
def build_cython(c):

    """ Build the cython extension module """

    print_banner("Building Cython Module")

    # Run cython on the pyx file to create a .cpp file

    invoke.run("cython --cplus -3 utility_functions.pyx -o Utility.cpp")


    # Compile and link the cython wrapper library

    compile_python_module("Utility.cpp", "Utility")

    print("* Complete")
