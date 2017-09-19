set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_C_COMPILER g++)

set(
	LIB_SRC_LIST
	chars_segment.cpp
	func.cpp
	char_recognition.cpp
	ssd.cpp
	)

include_directories(
	${PROJECT_SOURCE_DIR}/third_party/caffe/include
	${PROJECT_SOURCE_DIR}/include
	/usr/local/cuda-8.0/include
	)
link_directories(
	${PROJECT_SOURCE_DIR}/lib
	${PROJECT_SOURCE_DIR}/third_party/caffe/build/lib
	)

add_library(swpr SHARED ${LIB_SRC_LIST})

target_link_libraries(
	swpr
	caffe
	opencv_core
	opencv_imgcodecs
	opencv_imgproc
	)
