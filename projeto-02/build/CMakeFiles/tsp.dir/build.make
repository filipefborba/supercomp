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
CMAKE_SOURCE_DIR = /home/filipefborba/Insper/supercomp/projeto-02

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/filipefborba/Insper/supercomp/projeto-02/build

# Include any dependencies generated for this target.
include CMakeFiles/tsp.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tsp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tsp.dir/flags.make

CMakeFiles/tsp.dir/tsp.cpp.o: CMakeFiles/tsp.dir/flags.make
CMakeFiles/tsp.dir/tsp.cpp.o: ../tsp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/filipefborba/Insper/supercomp/projeto-02/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tsp.dir/tsp.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tsp.dir/tsp.cpp.o -c /home/filipefborba/Insper/supercomp/projeto-02/tsp.cpp

CMakeFiles/tsp.dir/tsp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tsp.dir/tsp.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/filipefborba/Insper/supercomp/projeto-02/tsp.cpp > CMakeFiles/tsp.dir/tsp.cpp.i

CMakeFiles/tsp.dir/tsp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tsp.dir/tsp.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/filipefborba/Insper/supercomp/projeto-02/tsp.cpp -o CMakeFiles/tsp.dir/tsp.cpp.s

CMakeFiles/tsp.dir/tsp.cpp.o.requires:

.PHONY : CMakeFiles/tsp.dir/tsp.cpp.o.requires

CMakeFiles/tsp.dir/tsp.cpp.o.provides: CMakeFiles/tsp.dir/tsp.cpp.o.requires
	$(MAKE) -f CMakeFiles/tsp.dir/build.make CMakeFiles/tsp.dir/tsp.cpp.o.provides.build
.PHONY : CMakeFiles/tsp.dir/tsp.cpp.o.provides

CMakeFiles/tsp.dir/tsp.cpp.o.provides.build: CMakeFiles/tsp.dir/tsp.cpp.o


# Object files for target tsp
tsp_OBJECTS = \
"CMakeFiles/tsp.dir/tsp.cpp.o"

# External object files for target tsp
tsp_EXTERNAL_OBJECTS =

tsp: CMakeFiles/tsp.dir/tsp.cpp.o
tsp: CMakeFiles/tsp.dir/build.make
tsp: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
tsp: /usr/lib/x86_64-linux-gnu/libpthread.so
tsp: CMakeFiles/tsp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/filipefborba/Insper/supercomp/projeto-02/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tsp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tsp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tsp.dir/build: tsp

.PHONY : CMakeFiles/tsp.dir/build

CMakeFiles/tsp.dir/requires: CMakeFiles/tsp.dir/tsp.cpp.o.requires

.PHONY : CMakeFiles/tsp.dir/requires

CMakeFiles/tsp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tsp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tsp.dir/clean

CMakeFiles/tsp.dir/depend:
	cd /home/filipefborba/Insper/supercomp/projeto-02/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/filipefborba/Insper/supercomp/projeto-02 /home/filipefborba/Insper/supercomp/projeto-02 /home/filipefborba/Insper/supercomp/projeto-02/build /home/filipefborba/Insper/supercomp/projeto-02/build /home/filipefborba/Insper/supercomp/projeto-02/build/CMakeFiles/tsp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tsp.dir/depend

