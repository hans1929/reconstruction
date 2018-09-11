# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/hans/Documents/opencv_workspace/reconstruction_sift

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hans/Documents/opencv_workspace/reconstruction_sift/build

# Include any dependencies generated for this target.
include CMakeFiles/reconstruction_sift.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/reconstruction_sift.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/reconstruction_sift.dir/flags.make

CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.o: CMakeFiles/reconstruction_sift.dir/flags.make
CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.o: ../reconstruction_sift.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hans/Documents/opencv_workspace/reconstruction_sift/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.o -c /home/hans/Documents/opencv_workspace/reconstruction_sift/reconstruction_sift.cpp

CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hans/Documents/opencv_workspace/reconstruction_sift/reconstruction_sift.cpp > CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.i

CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hans/Documents/opencv_workspace/reconstruction_sift/reconstruction_sift.cpp -o CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.s

CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.o.requires:

.PHONY : CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.o.requires

CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.o.provides: CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.o.requires
	$(MAKE) -f CMakeFiles/reconstruction_sift.dir/build.make CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.o.provides.build
.PHONY : CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.o.provides

CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.o.provides.build: CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.o


# Object files for target reconstruction_sift
reconstruction_sift_OBJECTS = \
"CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.o"

# External object files for target reconstruction_sift
reconstruction_sift_EXTERNAL_OBJECTS =

reconstruction_sift: CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.o
reconstruction_sift: CMakeFiles/reconstruction_sift.dir/build.make
reconstruction_sift: /usr/local/lib/libopencv_videostab.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_ts.a
reconstruction_sift: /usr/local/lib/libopencv_superres.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_stitching.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_contrib.so.2.4.13
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_system.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_thread.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_regex.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libpthread.so
reconstruction_sift: /usr/local/lib/libpcl_common.so
reconstruction_sift: /usr/local/lib/libpcl_octree.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
reconstruction_sift: /usr/local/lib/libpcl_kdtree.so
reconstruction_sift: /usr/local/lib/libpcl_search.so
reconstruction_sift: /usr/local/lib/libpcl_io.so
reconstruction_sift: /usr/local/lib/libpcl_sample_consensus.so
reconstruction_sift: /usr/local/lib/libpcl_filters.so
reconstruction_sift: /usr/local/lib/libpcl_visualization.so
reconstruction_sift: /usr/local/lib/libpcl_outofcore.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libqhull.so
reconstruction_sift: /usr/local/lib/libpcl_surface.so
reconstruction_sift: /usr/local/lib/libpcl_features.so
reconstruction_sift: /usr/local/lib/libpcl_keypoints.so
reconstruction_sift: /usr/local/lib/libpcl_segmentation.so
reconstruction_sift: /usr/local/lib/libpcl_registration.so
reconstruction_sift: /usr/local/lib/libpcl_recognition.so
reconstruction_sift: /usr/local/lib/libpcl_tracking.so
reconstruction_sift: /usr/local/lib/libpcl_people.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_system.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_thread.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libboost_regex.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libpthread.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libqhull.so
reconstruction_sift: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
reconstruction_sift: /usr/local/lib/vtk-5.8/libvtkGenericFiltering.so.5.8.0
reconstruction_sift: /usr/local/lib/vtk-5.8/libvtkGeovis.so.5.8.0
reconstruction_sift: /usr/local/lib/vtk-5.8/libvtkCharts.so.5.8.0
reconstruction_sift: /usr/local/lib/libopencv_nonfree.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_ocl.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_gpu.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_photo.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_objdetect.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_legacy.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_video.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_ml.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_calib3d.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_features2d.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_highgui.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_imgproc.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_flann.so.2.4.13
reconstruction_sift: /usr/local/lib/libopencv_core.so.2.4.13
reconstruction_sift: /usr/local/lib/libpcl_common.so
reconstruction_sift: /usr/local/lib/libpcl_octree.so
reconstruction_sift: /usr/local/lib/libpcl_kdtree.so
reconstruction_sift: /usr/local/lib/libpcl_search.so
reconstruction_sift: /usr/local/lib/libpcl_io.so
reconstruction_sift: /usr/local/lib/libpcl_sample_consensus.so
reconstruction_sift: /usr/local/lib/libpcl_filters.so
reconstruction_sift: /usr/local/lib/libpcl_visualization.so
reconstruction_sift: /usr/local/lib/libpcl_outofcore.so
reconstruction_sift: /usr/local/lib/libpcl_surface.so
reconstruction_sift: /usr/local/lib/libpcl_features.so
reconstruction_sift: /usr/local/lib/libpcl_keypoints.so
reconstruction_sift: /usr/local/lib/libpcl_segmentation.so
reconstruction_sift: /usr/local/lib/libpcl_registration.so
reconstruction_sift: /usr/local/lib/libpcl_recognition.so
reconstruction_sift: /usr/local/lib/libpcl_tracking.so
reconstruction_sift: /usr/local/lib/libpcl_people.so
reconstruction_sift: /usr/local/lib/vtk-5.8/libvtkViews.so.5.8.0
reconstruction_sift: /usr/local/lib/vtk-5.8/libvtkInfovis.so.5.8.0
reconstruction_sift: /usr/local/lib/vtk-5.8/libvtkWidgets.so.5.8.0
reconstruction_sift: /usr/local/lib/vtk-5.8/libvtkVolumeRendering.so.5.8.0
reconstruction_sift: /usr/local/lib/vtk-5.8/libvtkHybrid.so.5.8.0
reconstruction_sift: /usr/local/lib/vtk-5.8/libvtkRendering.so.5.8.0
reconstruction_sift: /usr/local/lib/vtk-5.8/libvtkImaging.so.5.8.0
reconstruction_sift: /usr/local/lib/vtk-5.8/libvtkGraphics.so.5.8.0
reconstruction_sift: /usr/local/lib/vtk-5.8/libvtkIO.so.5.8.0
reconstruction_sift: /usr/local/lib/vtk-5.8/libvtkFiltering.so.5.8.0
reconstruction_sift: /usr/local/lib/vtk-5.8/libvtkCommon.so.5.8.0
reconstruction_sift: /usr/local/lib/vtk-5.8/libvtksys.so.5.8.0
reconstruction_sift: CMakeFiles/reconstruction_sift.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hans/Documents/opencv_workspace/reconstruction_sift/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable reconstruction_sift"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reconstruction_sift.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/reconstruction_sift.dir/build: reconstruction_sift

.PHONY : CMakeFiles/reconstruction_sift.dir/build

CMakeFiles/reconstruction_sift.dir/requires: CMakeFiles/reconstruction_sift.dir/reconstruction_sift.cpp.o.requires

.PHONY : CMakeFiles/reconstruction_sift.dir/requires

CMakeFiles/reconstruction_sift.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/reconstruction_sift.dir/cmake_clean.cmake
.PHONY : CMakeFiles/reconstruction_sift.dir/clean

CMakeFiles/reconstruction_sift.dir/depend:
	cd /home/hans/Documents/opencv_workspace/reconstruction_sift/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hans/Documents/opencv_workspace/reconstruction_sift /home/hans/Documents/opencv_workspace/reconstruction_sift /home/hans/Documents/opencv_workspace/reconstruction_sift/build /home/hans/Documents/opencv_workspace/reconstruction_sift/build /home/hans/Documents/opencv_workspace/reconstruction_sift/build/CMakeFiles/reconstruction_sift.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/reconstruction_sift.dir/depend

