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
include CMakeFiles/tsp_bb.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tsp_bb.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tsp_bb.dir/flags.make

CMakeFiles/tsp_bb.dir/tsp_bb.cpp.o: CMakeFiles/tsp_bb.dir/flags.make
CMakeFiles/tsp_bb.dir/tsp_bb.cpp.o: ../tsp_bb.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/filipefborba/Insper/supercomp/projeto-02/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tsp_bb.dir/tsp_bb.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tsp_bb.dir/tsp_bb.cpp.o -c /home/filipefborba/Insper/supercomp/projeto-02/tsp_bb.cpp

CMakeFiles/tsp_bb.dir/tsp_bb.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tsp_bb.dir/tsp_bb.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/filipefborba/Insper/supercomp/projeto-02/tsp_bb.cpp > CMakeFiles/tsp_bb.dir/tsp_bb.cpp.i

CMakeFiles/tsp_bb.dir/tsp_bb.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tsp_bb.dir/tsp_bb.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/filipefborba/Insper/supercomp/projeto-02/tsp_bb.cpp -o CMakeFiles/tsp_bb.dir/tsp_bb.cpp.s

CMakeFiles/tsp_bb.dir/tsp_bb.cpp.o.requires:

.PHONY : CMakeFiles/tsp_bb.dir/tsp_bb.cpp.o.requires

CMakeFiles/tsp_bb.dir/tsp_bb.cpp.o.provides: CMakeFiles/tsp_bb.dir/tsp_bb.cpp.o.requires
	$(MAKE) -f CMakeFiles/tsp_bb.dir/build.make CMakeFiles/tsp_bb.dir/tsp_bb.cpp.o.provides.build
.PHONY : CMakeFiles/tsp_bb.dir/tsp_bb.cpp.o.provides

CMakeFiles/tsp_bb.dir/tsp_bb.cpp.o.provides.build: CMakeFiles/tsp_bb.dir/tsp_bb.cpp.o


# Object files for target tsp_bb
tsp_bb_OBJECTS = \
"CMakeFiles/tsp_bb.dir/tsp_bb.cpp.o"

# External object files for target tsp_bb
tsp_bb_EXTERNAL_OBJECTS =

tsp_bb: CMakeFiles/tsp_bb.dir/tsp_bb.cpp.o
tsp_bb: CMakeFiles/tsp_bb.dir/build.make
tsp_bb: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
tsp_bb: /usr/lib/x86_64-linux-gnu/libpthread.so
tsp_bb: CMakeFiles/tsp_bb.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/filipefborba/Insper/supercomp/projeto-02/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tsp_bb"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tsp_bb.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tsp_bb.dir/build: tsp_bb

.PHONY : CMakeFiles/tsp_bb.dir/build

CMakeFiles/tsp_bb.dir/requires: CMakeFiles/tsp_bb.dir/tsp_bb.cpp.o.requires

.PHONY : CMakeFiles/tsp_bb.dir/requires

CMakeFiles/tsp_bb.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tsp_bb.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tsp_bb.dir/clean

CMakeFiles/tsp_bb.dir/depend:
	cd /home/filipefborba/Insper/supercomp/projeto-02/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/filipefborba/Insper/supercomp/projeto-02 /home/filipefborba/Insper/supercomp/projeto-02 /home/filipefborba/Insper/supercomp/projeto-02/build /home/filipefborba/Insper/supercomp/projeto-02/build /home/filipefborba/Insper/supercomp/projeto-02/build/CMakeFiles/tsp_bb.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tsp_bb.dir/depend

