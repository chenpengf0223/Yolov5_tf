add_library(Detector SHARED ${CMAKE_CURRENT_SOURCE_DIR}/detector.cpp)

#message(${OpenCV_LIBS})
#message(${Tensorflow::Lite})
message(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
message(${LINK_LIBRARIES_LIST})

target_link_libraries(Detector
    PRIVATE ${LINK_LIBRARIES_LIST}
    #PRIVATE Tensorflow::Lite
    #PUBLIC ${OpenCV_LIBS}
    #PUBLIC opencv_core opencv_highgui
    log
)
