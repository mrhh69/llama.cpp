# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.26.3/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.26.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/max/Downloads/llama/my_llama.cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/max/Downloads/llama/my_llama.cpp

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/opt/homebrew/Cellar/cmake/3.26.1/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/opt/homebrew/Cellar/cmake/3.26.3/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/max/Downloads/llama/my_llama.cpp/CMakeFiles /Users/max/Downloads/llama/my_llama.cpp//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/max/Downloads/llama/my_llama.cpp/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named BUILD_INFO

# Build rule for target.
BUILD_INFO: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 BUILD_INFO
.PHONY : BUILD_INFO

# fast build rule for target.
BUILD_INFO/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/BUILD_INFO.dir/build.make CMakeFiles/BUILD_INFO.dir/build
.PHONY : BUILD_INFO/fast

#=============================================================================
# Target rules for targets named ggml

# Build rule for target.
ggml: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 ggml
.PHONY : ggml

# fast build rule for target.
ggml/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ggml.dir/build.make CMakeFiles/ggml.dir/build
.PHONY : ggml/fast

#=============================================================================
# Target rules for targets named llama

# Build rule for target.
llama: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 llama
.PHONY : llama

# fast build rule for target.
llama/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/build
.PHONY : llama/fast

#=============================================================================
# Target rules for targets named common

# Build rule for target.
common: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 common
.PHONY : common

# fast build rule for target.
common/fast:
	$(MAKE) $(MAKESILENT) -f examples/CMakeFiles/common.dir/build.make examples/CMakeFiles/common.dir/build
.PHONY : common/fast

#=============================================================================
# Target rules for targets named main

# Build rule for target.
main: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 main
.PHONY : main

# fast build rule for target.
main/fast:
	$(MAKE) $(MAKESILENT) -f examples/main/CMakeFiles/main.dir/build.make examples/main/CMakeFiles/main.dir/build
.PHONY : main/fast

#=============================================================================
# Target rules for targets named quantize

# Build rule for target.
quantize: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 quantize
.PHONY : quantize

# fast build rule for target.
quantize/fast:
	$(MAKE) $(MAKESILENT) -f examples/quantize/CMakeFiles/quantize.dir/build.make examples/quantize/CMakeFiles/quantize.dir/build
.PHONY : quantize/fast

#=============================================================================
# Target rules for targets named quantize-stats

# Build rule for target.
quantize-stats: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 quantize-stats
.PHONY : quantize-stats

# fast build rule for target.
quantize-stats/fast:
	$(MAKE) $(MAKESILENT) -f examples/quantize-stats/CMakeFiles/quantize-stats.dir/build.make examples/quantize-stats/CMakeFiles/quantize-stats.dir/build
.PHONY : quantize-stats/fast

#=============================================================================
# Target rules for targets named vdot

# Build rule for target.
vdot: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 vdot
.PHONY : vdot

# fast build rule for target.
vdot/fast:
	$(MAKE) $(MAKESILENT) -f pocs/vdot/CMakeFiles/vdot.dir/build.make pocs/vdot/CMakeFiles/vdot.dir/build
.PHONY : vdot/fast

#=============================================================================
# Target rules for targets named q8dot

# Build rule for target.
q8dot: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 q8dot
.PHONY : q8dot

# fast build rule for target.
q8dot/fast:
	$(MAKE) $(MAKESILENT) -f pocs/vdot/CMakeFiles/q8dot.dir/build.make pocs/vdot/CMakeFiles/q8dot.dir/build
.PHONY : q8dot/fast

ggml.o: ggml.c.o
.PHONY : ggml.o

# target to build an object file
ggml.c.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ggml.dir/build.make CMakeFiles/ggml.dir/ggml.c.o
.PHONY : ggml.c.o

ggml.i: ggml.c.i
.PHONY : ggml.i

# target to preprocess a source file
ggml.c.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ggml.dir/build.make CMakeFiles/ggml.dir/ggml.c.i
.PHONY : ggml.c.i

ggml.s: ggml.c.s
.PHONY : ggml.s

# target to generate assembly for a file
ggml.c.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ggml.dir/build.make CMakeFiles/ggml.dir/ggml.c.s
.PHONY : ggml.c.s

llama.o: llama.cpp.o
.PHONY : llama.o

# target to build an object file
llama.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama.cpp.o
.PHONY : llama.cpp.o

llama.i: llama.cpp.i
.PHONY : llama.i

# target to preprocess a source file
llama.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama.cpp.i
.PHONY : llama.cpp.i

llama.s: llama.cpp.s
.PHONY : llama.s

# target to generate assembly for a file
llama.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama.cpp.s
.PHONY : llama.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... BUILD_INFO"
	@echo "... common"
	@echo "... ggml"
	@echo "... llama"
	@echo "... main"
	@echo "... q8dot"
	@echo "... quantize"
	@echo "... quantize-stats"
	@echo "... vdot"
	@echo "... ggml.o"
	@echo "... ggml.i"
	@echo "... ggml.s"
	@echo "... llama.o"
	@echo "... llama.i"
	@echo "... llama.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

