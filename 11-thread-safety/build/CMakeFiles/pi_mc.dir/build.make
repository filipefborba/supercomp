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
CMAKE_SOURCE_DIR = /home/filipefborba/Insper/supercomp/11-thread-safety

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/filipefborba/Insper/supercomp/11-thread-safety/build

# Include any dependencies generated for this target.
include CMakeFiles/pi_mc.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pi_mc.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pi_mc.dir/flags.make

CMakeFiles/pi_mc.dir/pi_mc.c.o: CMakeFiles/pi_mc.dir/flags.make
CMakeFiles/pi_mc.dir/pi_mc.c.o: ../pi_mc.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/filipefborba/Insper/supercomp/11-thread-safety/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/pi_mc.dir/pi_mc.c.o"
	/usr/bin/gcc-7 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/pi_mc.dir/pi_mc.c.o   -c /home/filipefborba/Insper/supercomp/11-thread-safety/pi_mc.c

CMakeFiles/pi_mc.dir/pi_mc.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pi_mc.dir/pi_mc.c.i"
	/usr/bin/gcc-7 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/filipefborba/Insper/supercomp/11-thread-safety/pi_mc.c > CMakeFiles/pi_mc.dir/pi_mc.c.i

CMakeFiles/pi_mc.dir/pi_mc.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pi_mc.dir/pi_mc.c.s"
	/usr/bin/gcc-7 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/filipefborba/Insper/supercomp/11-thread-safety/pi_mc.c -o CMakeFiles/pi_mc.dir/pi_mc.c.s

CMakeFiles/pi_mc.dir/pi_mc.c.o.requires:

.PHONY : CMakeFiles/pi_mc.dir/pi_mc.c.o.requires

CMakeFiles/pi_mc.dir/pi_mc.c.o.provides: CMakeFiles/pi_mc.dir/pi_mc.c.o.requires
	$(MAKE) -f CMakeFiles/pi_mc.dir/build.make CMakeFiles/pi_mc.dir/pi_mc.c.o.provides.build
.PHONY : CMakeFiles/pi_mc.dir/pi_mc.c.o.provides

CMakeFiles/pi_mc.dir/pi_mc.c.o.provides.build: CMakeFiles/pi_mc.dir/pi_mc.c.o


CMakeFiles/pi_mc.dir/random.c.o: CMakeFiles/pi_mc.dir/flags.make
CMakeFiles/pi_mc.dir/random.c.o: ../random.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/filipefborba/Insper/supercomp/11-thread-safety/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/pi_mc.dir/random.c.o"
	/usr/bin/gcc-7 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/pi_mc.dir/random.c.o   -c /home/filipefborba/Insper/supercomp/11-thread-safety/random.c

CMakeFiles/pi_mc.dir/random.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pi_mc.dir/random.c.i"
	/usr/bin/gcc-7 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/filipefborba/Insper/supercomp/11-thread-safety/random.c > CMakeFiles/pi_mc.dir/random.c.i

CMakeFiles/pi_mc.dir/random.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pi_mc.dir/random.c.s"
	/usr/bin/gcc-7 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/filipefborba/Insper/supercomp/11-thread-safety/random.c -o CMakeFiles/pi_mc.dir/random.c.s

CMakeFiles/pi_mc.dir/random.c.o.requires:

.PHONY : CMakeFiles/pi_mc.dir/random.c.o.requires

CMakeFiles/pi_mc.dir/random.c.o.provides: CMakeFiles/pi_mc.dir/random.c.o.requires
	$(MAKE) -f CMakeFiles/pi_mc.dir/build.make CMakeFiles/pi_mc.dir/random.c.o.provides.build
.PHONY : CMakeFiles/pi_mc.dir/random.c.o.provides

CMakeFiles/pi_mc.dir/random.c.o.provides.build: CMakeFiles/pi_mc.dir/random.c.o


# Object files for target pi_mc
pi_mc_OBJECTS = \
"CMakeFiles/pi_mc.dir/pi_mc.c.o" \
"CMakeFiles/pi_mc.dir/random.c.o"

# External object files for target pi_mc
pi_mc_EXTERNAL_OBJECTS =

pi_mc: CMakeFiles/pi_mc.dir/pi_mc.c.o
pi_mc: CMakeFiles/pi_mc.dir/random.c.o
pi_mc: CMakeFiles/pi_mc.dir/build.make
pi_mc: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
pi_mc: /usr/lib/x86_64-linux-gnu/libpthread.so
pi_mc: CMakeFiles/pi_mc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/filipefborba/Insper/supercomp/11-thread-safety/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable pi_mc"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pi_mc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pi_mc.dir/build: pi_mc

.PHONY : CMakeFiles/pi_mc.dir/build

CMakeFiles/pi_mc.dir/requires: CMakeFiles/pi_mc.dir/pi_mc.c.o.requires
CMakeFiles/pi_mc.dir/requires: CMakeFiles/pi_mc.dir/random.c.o.requires

.PHONY : CMakeFiles/pi_mc.dir/requires

CMakeFiles/pi_mc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pi_mc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pi_mc.dir/clean

CMakeFiles/pi_mc.dir/depend:
	cd /home/filipefborba/Insper/supercomp/11-thread-safety/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/filipefborba/Insper/supercomp/11-thread-safety /home/filipefborba/Insper/supercomp/11-thread-safety /home/filipefborba/Insper/supercomp/11-thread-safety/build /home/filipefborba/Insper/supercomp/11-thread-safety/build /home/filipefborba/Insper/supercomp/11-thread-safety/build/CMakeFiles/pi_mc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pi_mc.dir/depend

