cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 14)

project(FoodRecognition CXX)

option(TENSORFLOW_WITH_CUDA "use cuda tensorflow" OFF)

if (ANDROID)
    set(PLATFORM "android")
    set(ABI ${ANDROID_ABI})
elseif (UNIX AND NOT APPLE)
    set(PLATFORM "linux")
    set(ABI "x86_64")
endif()

if (ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -llog -landroid -lz -Werror,-Wunused-command-line-argument")
endif()

if ("${ANDROID_ABI}" STREQUAL "x86_64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
endif()

#set(Tensorflow_DIR third_party/tensorflow_prebuilt/prebuilt)
#find_package(Tensorflow REQUIRED)

#if (ANDROID)
#    set(OpenCV_DIR third_party/opencv/prebuilt/sdk/native/jni)
#else ()
#    set(OpenCV_DIR third_party/opencv/prebuilt/share/OpenCV)
#endif()
#find_package(OpenCV 3.4 EXACT REQUIRED)

set(CP_TARGET_LIBS
    #tensorflowlite_gpu_gl
     tensorflowlite
    tensorflowlite_gpu_delegate
    opencv_core
    opencv_highgui
    opencv_calib3d
opencv_features2d
opencv_flann
opencv_imgcodecs
opencv_imgproc
opencv_ml
opencv_objdetect
opencv_optflow
opencv_photo
opencv_shape
opencv_stitching
opencv_superres
opencv_video
opencv_videoio
opencv_videostab
opencv_ximgproc
)
    #gl_delegate.pic
foreach(__component ${CP_TARGET_LIBS})
    add_library(share_${__component} SHARED IMPORTED)
    set_target_properties(
        share_${__component}
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_lib/r2-4/lib${__component}.so
    )
    #message(STATUS "mylog   ${CMAKE_CURRENT_SOURCE_DIR}/gpu_lib/r2-4/lib${__component}.so")
    set(LINK_LIBRARIES_LIST ${LINK_LIBRARIES_LIST} share_${__component})
endforeach()

#set(OpenCV_INCLUDE_DIRS include/opencv)

add_subdirectory(detection)

add_executable(FoodRecognition src/demo.cpp src/undistort.cpp)

target_include_directories(FoodRecognition
    PRIVATE ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    #${Tensorflow_INCLUDE_DIRS}/lite_include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(FoodRecognition
    PRIVATE
    #${OpenCV_LIBS}
    ${LINK_LIBRARIES_LIST}
    Detector
)
