import invoke
import shutil
import os,glob


def print_banner(msg):
    print("\n========================== {0} =======================\n".format(msg))

src_filenames=["src/dIdt8H_5PNe10.cc", "src/Utility.cc"]
pyx_filenames=["src/dIdt.pyx", "src/Utility.pyx"]

so_filenames=["dIdt.so", "Utility.so"]
build_directory = "build/"
root_dir = "/home/shaunf/Documents/Computer/Code/projects/Massive_Vector_Field_Dynamical_Friction/ProcaAroundKerr/GWGenerator/"
include_dir = root_dir+"include"

assert len(so_filenames)==len(src_filenames), "error: missing library file names"
assert len(pyx_filenames)==len(src_filenames), "error: missing pyx files"


@invoke.task()
def build_all(c):
	for i in [1]:
		module_name=so_filenames[i][:-3]
		print("Building module {0}".format(module_name))

		print_banner("Building C++ Library")
		build_cppmult(src_name=src_filenames[i], lib_name=so_filenames[i])

		print_banner("Building cython Module")
		invoke.run("cython --cplus -3 {0} -o {1}".format(pyx_filenames[i], module_name+".cpp"))
		print("* Complete")

		print_banner("Compiling Python Module")
		compile_python_module(module_name+".cpp", module_name)

		generated_files = [os.path.basename(i) for i in glob.glob(os.getcwd()+"/*{0}*".format(module_name))]
		for j in generated_files:
			move_to_build(j)


def build_cppmult(src_name="", lib_name=""):
    """Build the shared library for the C++ code"""

	#errors: -Wall -Werror
    invoke.run(
        "g++ -O3 -shared -std=c++11 -fPIC {0} "
		"-I/usr/include/python3.10 -lpython3.10 "
		"-I {2} "
        "-o lib{1} ".format(src_name, lib_name, include_dir)
    )
    print("* Complete")


def compile_python_module(cpp_name, extension_name):

    invoke.run(

        "g++ -O3 -shared -std=c++11 -fPIC "

        "`python3 -m pybind11 --includes` "

        "-I /mnt/Data_Volume/Documents/software/Anaconda/envs/few_env/include/python3.7m -I . "

		"-I {2} "

        "{0} "

        "-o {1}`python3.7-config --extension-suffix` "

        "-L. -l{1} -Wl,-rpath,. "
		"-L/usr/lib64 -lgsl -lgslcblas -lm".format(cpp_name, extension_name, include_dir)

    )

def move_to_build(module_name):
	old_path = root_dir+module_name
	new_path=root_dir+build_directory+module_name
	shutil.move(old_path, new_path)



"""
def build_cython(c, pyx_name="", src_name=""):

    # Build the cython extension module

    print_banner("Building Cython Module")

    # Run cython on the pyx file to create a .cpp file

    invoke.run("cython --cplus -3 {0} -o {1}".format(pyx_name, src_name))


    # Compile and link the cython wrapper library

	module_name=src_name[:-3]
    compile_python_module(src_name, moduel_name)

    print("* Complete")
"""
