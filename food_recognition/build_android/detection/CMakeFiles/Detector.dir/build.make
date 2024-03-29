# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

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
CMAKE_COMMAND = /opt/cmake-3.12.2/bin/cmake

# The command to remove a file.
RM = /opt/cmake-3.12.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/haishi/FoodRecognition/FoodRecognition

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/haishi/FoodRecognition/FoodRecognition/build_android

# Include any dependencies generated for this target.
include detection/CMakeFiles/Detector.dir/depend.make

# Include the progress variables for this target.
include detection/CMakeFiles/Detector.dir/progress.make

# Include the compile flags for this target's objects.
include detection/CMakeFiles/Detector.dir/flags.make

detection/CMakeFiles/Detector.dir/detector.cpp.o: detection/CMakeFiles/Detector.dir/flags.make
detection/CMakeFiles/Detector.dir/detector.cpp.o: ../detection/detector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/haishi/FoodRecognition/FoodRecognition/build_android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object detection/CMakeFiles/Detector.dir/detector.cpp.o"
	cd /home/haishi/FoodRecognition/FoodRecognition/build_android/detection && /home/haishi/android-ndk-r16b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android --gcc-toolchain=/home/haishi/android-ndk-r16b/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64 --sysroot=/home/haishi/android-ndk-r16b/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Detector.dir/detector.cpp.o -c /home/haishi/FoodRecognition/FoodRecognition/detection/detector.cpp

detection/CMakeFiles/Detector.dir/detector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Detector.dir/detector.cpp.i"
	cd /home/haishi/FoodRecognition/FoodRecognition/build_android/detection && /home/haishi/android-ndk-r16b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android --gcc-toolchain=/home/haishi/android-ndk-r16b/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64 --sysroot=/home/haishi/android-ndk-r16b/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/haishi/FoodRecognition/FoodRecognition/detection/detector.cpp > CMakeFiles/Detector.dir/detector.cpp.i

detection/CMakeFiles/Detector.dir/detector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Detector.dir/detector.cpp.s"
	cd /home/haishi/FoodRecognition/FoodRecognition/build_android/detection && /home/haishi/android-ndk-r16b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android --gcc-toolchain=/home/haishi/android-ndk-r16b/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64 --sysroot=/home/haishi/android-ndk-r16b/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/haishi/FoodRecognition/FoodRecognition/detection/detector.cpp -o CMakeFiles/Detector.dir/detector.cpp.s

# Object files for target Detector
Detector_OBJECTS = \
"CMakeFiles/Detector.dir/detector.cpp.o"

# External object files for target Detector
Detector_EXTERNAL_OBJECTS =

detection/libDetector.so: detection/CMakeFiles/Detector.dir/detector.cpp.o
detection/libDetector.so: detection/CMakeFiles/Detector.dir/build.make
detection/libDetector.so: ../gpu_lib/r2-4/libtensorflowlite.so
detection/libDetector.so: ../gpu_lib/r2-4/libtensorflowlite_gpu_delegate.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_core.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_highgui.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_calib3d.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_features2d.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_flann.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_imgcodecs.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_imgproc.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_ml.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_objdetect.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_optflow.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_photo.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_shape.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_stitching.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_superres.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_video.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_videoio.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_videostab.so
detection/libDetector.so: ../gpu_lib/r2-4/libopencv_ximgproc.so
detection/libDetector.so: detection/CMakeFiles/Detector.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/haishi/FoodRecognition/FoodRecognition/build_android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libDetector.so"
	cd /home/haishi/FoodRecognition/FoodRecognition/build_android/detection && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Detector.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
detection/CMakeFiles/Detector.dir/build: detection/libDetector.so

.PHONY : detection/CMakeFiles/Detector.dir/build

detection/CMakeFiles/Detector.dir/clean:
	cd /home/haishi/FoodRecognition/FoodRecognition/build_android/detection && $(CMAKE_COMMAND) -P CMakeFiles/Detector.dir/cmake_clean.cmake
.PHONY : detection/CMakeFiles/Detector.dir/clean

detection/CMakeFiles/Detector.dir/depend:
	cd /home/haishi/FoodRecognition/FoodRecognition/build_android && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/haishi/FoodRecognition/FoodRecognition /home/haishi/FoodRecognition/FoodRecognition/detection /home/haishi/FoodRecognition/FoodRecognition/build_android /home/haishi/FoodRecognition/FoodRecognition/build_android/detection /home/haishi/FoodRecognition/FoodRecognition/build_android/detection/CMakeFiles/Detector.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : detection/CMakeFiles/Detector.dir/depend

