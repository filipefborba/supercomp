# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/filipefborba/Insper/supercomp/12-tarefas

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/filipefborba/Insper/supercomp/12-tarefas/build

# Include any dependencies generated for this target.
include CMakeFiles/fib_par.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fib_par.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fib_par.dir/flags.make

CMakeFiles/fib_par.dir/fib_par.cpp.o: CMakeFiles/fib_par.dir/flags.make
CMakeFiles/fib_par.dir/fib_par.cpp.o: ../fib_par.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/filipefborba/Insper/supercomp/12-tarefas/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fib_par.dir/fib_par.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fib_par.dir/fib_par.cpp.o -c /home/filipefborba/Insper/supercomp/12-tarefas/fib_par.cpp

CMakeFiles/fib_par.dir/fib_par.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fib_par.dir/fib_par.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/filipefborba/Insper/supercomp/12-tarefas/fib_par.cpp > CMakeFiles/fib_par.dir/fib_par.cpp.i

CMakeFiles/fib_par.dir/fib_par.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fib_par.dir/fib_par.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/filipefborba/Insper/supercomp/12-tarefas/fib_par.cpp -o CMakeFiles/fib_par.dir/fib_par.cpp.s

CMakeFiles/fib_par.dir/fib_par.cpp.o.requires:

.PHONY : CMakeFiles/fib_par.dir/fib_par.cpp.o.requires

CMakeFiles/fib_par.dir/fib_par.cpp.o.provides: CMakeFiles/fib_par.dir/fib_par.cpp.o.requires
	$(MAKE) -f CMakeFiles/fib_par.dir/build.make CMakeFiles/fib_par.dir/fib_par.cpp.o.provides.build
.PHONY : CMakeFiles/fib_par.dir/fib_par.cpp.o.provides

CMakeFiles/fib_par.dir/fib_par.cpp.o.provides.build: CMakeFiles/fib_par.dir/fib_par.cpp.o


# Object files for target fib_par
fib_par_OBJECTS = \
"CMakeFiles/fib_par.dir/fib_par.cpp.o"

# External object files for target fib_par
fib_par_EXTERNAL_OBJECTS =

fib_par: CMakeFiles/fib_par.dir/fib_par.cpp.o
fib_par: CMakeFiles/fib_par.dir/build.make
fib_par: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
fib_par: /usr/lib/x86_64-linux-gnu/libpthread.so
fib_par: CMakeFiles/fib_par.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/filipefborba/Insper/supercomp/12-tarefas/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable fib_par"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fib_par.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fib_par.dir/build: fib_par

.PHONY : CMakeFiles/fib_par.dir/build

CMakeFiles/fib_par.dir/requires: CMakeFiles/fib_par.dir/fib_par.cpp.o.requires

.PHONY : CMakeFiles/fib_par.dir/requires

CMakeFiles/fib_par.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fib_par.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fib_par.dir/clean

CMakeFiles/fib_par.dir/depend:
	cd /home/filipefborba/Insper/supercomp/12-tarefas/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/filipefborba/Insper/supercomp/12-tarefas /home/filipefborba/Insper/supercomp/12-tarefas /home/filipefborba/Insper/supercomp/12-tarefas/build /home/filipefborba/Insper/supercomp/12-tarefas/build /home/filipefborba/Insper/supercomp/12-tarefas/build/CMakeFiles/fib_par.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fib_par.dir/depend

