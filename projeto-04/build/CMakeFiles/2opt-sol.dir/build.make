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
CMAKE_SOURCE_DIR = /home/filipefborba/Insper/supercomp/projeto-04

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/filipefborba/Insper/supercomp/projeto-04/build

# Include any dependencies generated for this target.
include CMakeFiles/2opt-sol.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/2opt-sol.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/2opt-sol.dir/flags.make

CMakeFiles/2opt-sol.dir/2opt-sol.cpp.o: CMakeFiles/2opt-sol.dir/flags.make
CMakeFiles/2opt-sol.dir/2opt-sol.cpp.o: ../2opt-sol.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/filipefborba/Insper/supercomp/projeto-04/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/2opt-sol.dir/2opt-sol.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/2opt-sol.dir/2opt-sol.cpp.o -c /home/filipefborba/Insper/supercomp/projeto-04/2opt-sol.cpp

CMakeFiles/2opt-sol.dir/2opt-sol.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/2opt-sol.dir/2opt-sol.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/filipefborba/Insper/supercomp/projeto-04/2opt-sol.cpp > CMakeFiles/2opt-sol.dir/2opt-sol.cpp.i

CMakeFiles/2opt-sol.dir/2opt-sol.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/2opt-sol.dir/2opt-sol.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/filipefborba/Insper/supercomp/projeto-04/2opt-sol.cpp -o CMakeFiles/2opt-sol.dir/2opt-sol.cpp.s

CMakeFiles/2opt-sol.dir/2opt-sol.cpp.o.requires:

.PHONY : CMakeFiles/2opt-sol.dir/2opt-sol.cpp.o.requires

CMakeFiles/2opt-sol.dir/2opt-sol.cpp.o.provides: CMakeFiles/2opt-sol.dir/2opt-sol.cpp.o.requires
	$(MAKE) -f CMakeFiles/2opt-sol.dir/build.make CMakeFiles/2opt-sol.dir/2opt-sol.cpp.o.provides.build
.PHONY : CMakeFiles/2opt-sol.dir/2opt-sol.cpp.o.provides

CMakeFiles/2opt-sol.dir/2opt-sol.cpp.o.provides.build: CMakeFiles/2opt-sol.dir/2opt-sol.cpp.o


# Object files for target 2opt-sol
2opt__sol_OBJECTS = \
"CMakeFiles/2opt-sol.dir/2opt-sol.cpp.o"

# External object files for target 2opt-sol
2opt__sol_EXTERNAL_OBJECTS =

2opt-sol: CMakeFiles/2opt-sol.dir/2opt-sol.cpp.o
2opt-sol: CMakeFiles/2opt-sol.dir/build.make
2opt-sol: /usr/lib/x86_64-linux-gnu/libboost_mpi.so
2opt-sol: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
2opt-sol: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
2opt-sol: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
2opt-sol: CMakeFiles/2opt-sol.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/filipefborba/Insper/supercomp/projeto-04/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable 2opt-sol"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/2opt-sol.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/2opt-sol.dir/build: 2opt-sol

.PHONY : CMakeFiles/2opt-sol.dir/build

CMakeFiles/2opt-sol.dir/requires: CMakeFiles/2opt-sol.dir/2opt-sol.cpp.o.requires

.PHONY : CMakeFiles/2opt-sol.dir/requires

CMakeFiles/2opt-sol.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/2opt-sol.dir/cmake_clean.cmake
.PHONY : CMakeFiles/2opt-sol.dir/clean

CMakeFiles/2opt-sol.dir/depend:
	cd /home/filipefborba/Insper/supercomp/projeto-04/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/filipefborba/Insper/supercomp/projeto-04 /home/filipefborba/Insper/supercomp/projeto-04 /home/filipefborba/Insper/supercomp/projeto-04/build /home/filipefborba/Insper/supercomp/projeto-04/build /home/filipefborba/Insper/supercomp/projeto-04/build/CMakeFiles/2opt-sol.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/2opt-sol.dir/depend

