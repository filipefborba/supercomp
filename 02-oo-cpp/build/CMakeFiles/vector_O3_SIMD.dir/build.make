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
CMAKE_SOURCE_DIR = /home/filipefborba/Insper/supercomp/02-oo-cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/filipefborba/Insper/supercomp/02-oo-cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/vector_O3_SIMD.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/vector_O3_SIMD.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/vector_O3_SIMD.dir/flags.make

CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.o: CMakeFiles/vector_O3_SIMD.dir/flags.make
CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.o: ../experiment.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/filipefborba/Insper/supercomp/02-oo-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.o -c /home/filipefborba/Insper/supercomp/02-oo-cpp/experiment.cpp

CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/filipefborba/Insper/supercomp/02-oo-cpp/experiment.cpp > CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.i

CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/filipefborba/Insper/supercomp/02-oo-cpp/experiment.cpp -o CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.s

CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.o.requires:

.PHONY : CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.o.requires

CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.o.provides: CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.o.requires
	$(MAKE) -f CMakeFiles/vector_O3_SIMD.dir/build.make CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.o.provides.build
.PHONY : CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.o.provides

CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.o.provides.build: CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.o


CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.o: CMakeFiles/vector_O3_SIMD.dir/flags.make
CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.o: ../experimentLog.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/filipefborba/Insper/supercomp/02-oo-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.o -c /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentLog.cpp

CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentLog.cpp > CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.i

CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentLog.cpp -o CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.s

CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.o.requires:

.PHONY : CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.o.requires

CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.o.provides: CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.o.requires
	$(MAKE) -f CMakeFiles/vector_O3_SIMD.dir/build.make CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.o.provides.build
.PHONY : CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.o.provides

CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.o.provides.build: CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.o


CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.o: CMakeFiles/vector_O3_SIMD.dir/flags.make
CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.o: ../experimentPow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/filipefborba/Insper/supercomp/02-oo-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.o -c /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentPow.cpp

CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentPow.cpp > CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.i

CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentPow.cpp -o CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.s

CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.o.requires:

.PHONY : CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.o.requires

CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.o.provides: CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.o.requires
	$(MAKE) -f CMakeFiles/vector_O3_SIMD.dir/build.make CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.o.provides.build
.PHONY : CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.o.provides

CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.o.provides.build: CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.o


CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.o: CMakeFiles/vector_O3_SIMD.dir/flags.make
CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.o: ../experimentPow3.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/filipefborba/Insper/supercomp/02-oo-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.o -c /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentPow3.cpp

CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentPow3.cpp > CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.i

CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentPow3.cpp -o CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.s

CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.o.requires:

.PHONY : CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.o.requires

CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.o.provides: CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.o.requires
	$(MAKE) -f CMakeFiles/vector_O3_SIMD.dir/build.make CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.o.provides.build
.PHONY : CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.o.provides

CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.o.provides.build: CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.o


CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.o: CMakeFiles/vector_O3_SIMD.dir/flags.make
CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.o: ../experimentPow3Mult.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/filipefborba/Insper/supercomp/02-oo-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.o -c /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentPow3Mult.cpp

CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentPow3Mult.cpp > CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.i

CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentPow3Mult.cpp -o CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.s

CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.o.requires:

.PHONY : CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.o.requires

CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.o.provides: CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.o.requires
	$(MAKE) -f CMakeFiles/vector_O3_SIMD.dir/build.make CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.o.provides.build
.PHONY : CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.o.provides

CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.o.provides.build: CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.o


CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.o: CMakeFiles/vector_O3_SIMD.dir/flags.make
CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.o: ../experimentSum.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/filipefborba/Insper/supercomp/02-oo-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.o -c /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentSum.cpp

CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentSum.cpp > CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.i

CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentSum.cpp -o CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.s

CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.o.requires:

.PHONY : CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.o.requires

CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.o.provides: CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.o.requires
	$(MAKE) -f CMakeFiles/vector_O3_SIMD.dir/build.make CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.o.provides.build
.PHONY : CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.o.provides

CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.o.provides.build: CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.o


CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.o: CMakeFiles/vector_O3_SIMD.dir/flags.make
CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.o: ../experimentSumPositives.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/filipefborba/Insper/supercomp/02-oo-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.o -c /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentSumPositives.cpp

CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentSumPositives.cpp > CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.i

CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/filipefborba/Insper/supercomp/02-oo-cpp/experimentSumPositives.cpp -o CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.s

CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.o.requires:

.PHONY : CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.o.requires

CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.o.provides: CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.o.requires
	$(MAKE) -f CMakeFiles/vector_O3_SIMD.dir/build.make CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.o.provides.build
.PHONY : CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.o.provides

CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.o.provides.build: CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.o


CMakeFiles/vector_O3_SIMD.dir/main.cpp.o: CMakeFiles/vector_O3_SIMD.dir/flags.make
CMakeFiles/vector_O3_SIMD.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/filipefborba/Insper/supercomp/02-oo-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/vector_O3_SIMD.dir/main.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vector_O3_SIMD.dir/main.cpp.o -c /home/filipefborba/Insper/supercomp/02-oo-cpp/main.cpp

CMakeFiles/vector_O3_SIMD.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vector_O3_SIMD.dir/main.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/filipefborba/Insper/supercomp/02-oo-cpp/main.cpp > CMakeFiles/vector_O3_SIMD.dir/main.cpp.i

CMakeFiles/vector_O3_SIMD.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vector_O3_SIMD.dir/main.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/filipefborba/Insper/supercomp/02-oo-cpp/main.cpp -o CMakeFiles/vector_O3_SIMD.dir/main.cpp.s

CMakeFiles/vector_O3_SIMD.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/vector_O3_SIMD.dir/main.cpp.o.requires

CMakeFiles/vector_O3_SIMD.dir/main.cpp.o.provides: CMakeFiles/vector_O3_SIMD.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/vector_O3_SIMD.dir/build.make CMakeFiles/vector_O3_SIMD.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/vector_O3_SIMD.dir/main.cpp.o.provides

CMakeFiles/vector_O3_SIMD.dir/main.cpp.o.provides.build: CMakeFiles/vector_O3_SIMD.dir/main.cpp.o


# Object files for target vector_O3_SIMD
vector_O3_SIMD_OBJECTS = \
"CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.o" \
"CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.o" \
"CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.o" \
"CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.o" \
"CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.o" \
"CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.o" \
"CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.o" \
"CMakeFiles/vector_O3_SIMD.dir/main.cpp.o"

# External object files for target vector_O3_SIMD
vector_O3_SIMD_EXTERNAL_OBJECTS =

vector_O3_SIMD: CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.o
vector_O3_SIMD: CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.o
vector_O3_SIMD: CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.o
vector_O3_SIMD: CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.o
vector_O3_SIMD: CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.o
vector_O3_SIMD: CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.o
vector_O3_SIMD: CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.o
vector_O3_SIMD: CMakeFiles/vector_O3_SIMD.dir/main.cpp.o
vector_O3_SIMD: CMakeFiles/vector_O3_SIMD.dir/build.make
vector_O3_SIMD: CMakeFiles/vector_O3_SIMD.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/filipefborba/Insper/supercomp/02-oo-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX executable vector_O3_SIMD"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vector_O3_SIMD.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/vector_O3_SIMD.dir/build: vector_O3_SIMD

.PHONY : CMakeFiles/vector_O3_SIMD.dir/build

CMakeFiles/vector_O3_SIMD.dir/requires: CMakeFiles/vector_O3_SIMD.dir/experiment.cpp.o.requires
CMakeFiles/vector_O3_SIMD.dir/requires: CMakeFiles/vector_O3_SIMD.dir/experimentLog.cpp.o.requires
CMakeFiles/vector_O3_SIMD.dir/requires: CMakeFiles/vector_O3_SIMD.dir/experimentPow.cpp.o.requires
CMakeFiles/vector_O3_SIMD.dir/requires: CMakeFiles/vector_O3_SIMD.dir/experimentPow3.cpp.o.requires
CMakeFiles/vector_O3_SIMD.dir/requires: CMakeFiles/vector_O3_SIMD.dir/experimentPow3Mult.cpp.o.requires
CMakeFiles/vector_O3_SIMD.dir/requires: CMakeFiles/vector_O3_SIMD.dir/experimentSum.cpp.o.requires
CMakeFiles/vector_O3_SIMD.dir/requires: CMakeFiles/vector_O3_SIMD.dir/experimentSumPositives.cpp.o.requires
CMakeFiles/vector_O3_SIMD.dir/requires: CMakeFiles/vector_O3_SIMD.dir/main.cpp.o.requires

.PHONY : CMakeFiles/vector_O3_SIMD.dir/requires

CMakeFiles/vector_O3_SIMD.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/vector_O3_SIMD.dir/cmake_clean.cmake
.PHONY : CMakeFiles/vector_O3_SIMD.dir/clean

CMakeFiles/vector_O3_SIMD.dir/depend:
	cd /home/filipefborba/Insper/supercomp/02-oo-cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/filipefborba/Insper/supercomp/02-oo-cpp /home/filipefborba/Insper/supercomp/02-oo-cpp /home/filipefborba/Insper/supercomp/02-oo-cpp/build /home/filipefborba/Insper/supercomp/02-oo-cpp/build /home/filipefborba/Insper/supercomp/02-oo-cpp/build/CMakeFiles/vector_O3_SIMD.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/vector_O3_SIMD.dir/depend

