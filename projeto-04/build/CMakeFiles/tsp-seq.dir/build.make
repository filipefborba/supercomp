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
include CMakeFiles/tsp-seq.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tsp-seq.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tsp-seq.dir/flags.make

CMakeFiles/tsp-seq.dir/tsp-seq.cpp.o: CMakeFiles/tsp-seq.dir/flags.make
CMakeFiles/tsp-seq.dir/tsp-seq.cpp.o: ../tsp-seq.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/filipefborba/Insper/supercomp/projeto-04/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tsp-seq.dir/tsp-seq.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tsp-seq.dir/tsp-seq.cpp.o -c /home/filipefborba/Insper/supercomp/projeto-04/tsp-seq.cpp

CMakeFiles/tsp-seq.dir/tsp-seq.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tsp-seq.dir/tsp-seq.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/filipefborba/Insper/supercomp/projeto-04/tsp-seq.cpp > CMakeFiles/tsp-seq.dir/tsp-seq.cpp.i

CMakeFiles/tsp-seq.dir/tsp-seq.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tsp-seq.dir/tsp-seq.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/filipefborba/Insper/supercomp/projeto-04/tsp-seq.cpp -o CMakeFiles/tsp-seq.dir/tsp-seq.cpp.s

CMakeFiles/tsp-seq.dir/tsp-seq.cpp.o.requires:

.PHONY : CMakeFiles/tsp-seq.dir/tsp-seq.cpp.o.requires

CMakeFiles/tsp-seq.dir/tsp-seq.cpp.o.provides: CMakeFiles/tsp-seq.dir/tsp-seq.cpp.o.requires
	$(MAKE) -f CMakeFiles/tsp-seq.dir/build.make CMakeFiles/tsp-seq.dir/tsp-seq.cpp.o.provides.build
.PHONY : CMakeFiles/tsp-seq.dir/tsp-seq.cpp.o.provides

CMakeFiles/tsp-seq.dir/tsp-seq.cpp.o.provides.build: CMakeFiles/tsp-seq.dir/tsp-seq.cpp.o


# Object files for target tsp-seq
tsp__seq_OBJECTS = \
"CMakeFiles/tsp-seq.dir/tsp-seq.cpp.o"

# External object files for target tsp-seq
tsp__seq_EXTERNAL_OBJECTS =

tsp-seq: CMakeFiles/tsp-seq.dir/tsp-seq.cpp.o
tsp-seq: CMakeFiles/tsp-seq.dir/build.make
tsp-seq: CMakeFiles/tsp-seq.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/filipefborba/Insper/supercomp/projeto-04/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tsp-seq"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tsp-seq.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tsp-seq.dir/build: tsp-seq

.PHONY : CMakeFiles/tsp-seq.dir/build

CMakeFiles/tsp-seq.dir/requires: CMakeFiles/tsp-seq.dir/tsp-seq.cpp.o.requires

.PHONY : CMakeFiles/tsp-seq.dir/requires

CMakeFiles/tsp-seq.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tsp-seq.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tsp-seq.dir/clean

CMakeFiles/tsp-seq.dir/depend:
	cd /home/filipefborba/Insper/supercomp/projeto-04/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/filipefborba/Insper/supercomp/projeto-04 /home/filipefborba/Insper/supercomp/projeto-04 /home/filipefborba/Insper/supercomp/projeto-04/build /home/filipefborba/Insper/supercomp/projeto-04/build /home/filipefborba/Insper/supercomp/projeto-04/build/CMakeFiles/tsp-seq.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tsp-seq.dir/depend

