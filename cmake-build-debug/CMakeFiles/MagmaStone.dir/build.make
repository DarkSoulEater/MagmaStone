# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2020.3.1\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2020.3.1\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\eleno\CLionProjects\MagmaStone

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\eleno\CLionProjects\MagmaStone\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/MagmaStone.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/MagmaStone.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MagmaStone.dir/flags.make

CMakeFiles/MagmaStone.dir/main.cpp.obj: CMakeFiles/MagmaStone.dir/flags.make
CMakeFiles/MagmaStone.dir/main.cpp.obj: CMakeFiles/MagmaStone.dir/includes_CXX.rsp
CMakeFiles/MagmaStone.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\eleno\CLionProjects\MagmaStone\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MagmaStone.dir/main.cpp.obj"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\MagmaStone.dir\main.cpp.obj -c C:\Users\eleno\CLionProjects\MagmaStone\main.cpp

CMakeFiles/MagmaStone.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MagmaStone.dir/main.cpp.i"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\eleno\CLionProjects\MagmaStone\main.cpp > CMakeFiles\MagmaStone.dir\main.cpp.i

CMakeFiles/MagmaStone.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MagmaStone.dir/main.cpp.s"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\eleno\CLionProjects\MagmaStone\main.cpp -o CMakeFiles\MagmaStone.dir\main.cpp.s

# Object files for target MagmaStone
MagmaStone_OBJECTS = \
"CMakeFiles/MagmaStone.dir/main.cpp.obj"

# External object files for target MagmaStone
MagmaStone_EXTERNAL_OBJECTS =

MagmaStone.exe: CMakeFiles/MagmaStone.dir/main.cpp.obj
MagmaStone.exe: CMakeFiles/MagmaStone.dir/build.make
MagmaStone.exe: C:/VulkanSDK/1.2.154.1/Lib/vulkan-1.lib
MagmaStone.exe: CMakeFiles/MagmaStone.dir/linklibs.rsp
MagmaStone.exe: CMakeFiles/MagmaStone.dir/objects1.rsp
MagmaStone.exe: CMakeFiles/MagmaStone.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\eleno\CLionProjects\MagmaStone\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable MagmaStone.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\MagmaStone.dir\link.txt --verbose=$(VERBOSE)
	C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -noprofile -executionpolicy Bypass -file C:/scr/vcpkg/scripts/buildsystems/msbuild/applocal.ps1 -targetBinary C:/Users/eleno/CLionProjects/MagmaStone/cmake-build-debug/MagmaStone.exe -installedDir C:/scr/vcpkg/installed/x64-windows/debug/bin -OutVariable out

# Rule to build all files generated by this target.
CMakeFiles/MagmaStone.dir/build: MagmaStone.exe

.PHONY : CMakeFiles/MagmaStone.dir/build

CMakeFiles/MagmaStone.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\MagmaStone.dir\cmake_clean.cmake
.PHONY : CMakeFiles/MagmaStone.dir/clean

CMakeFiles/MagmaStone.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\eleno\CLionProjects\MagmaStone C:\Users\eleno\CLionProjects\MagmaStone C:\Users\eleno\CLionProjects\MagmaStone\cmake-build-debug C:\Users\eleno\CLionProjects\MagmaStone\cmake-build-debug C:\Users\eleno\CLionProjects\MagmaStone\cmake-build-debug\CMakeFiles\MagmaStone.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MagmaStone.dir/depend

