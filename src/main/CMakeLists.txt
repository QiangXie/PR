set (CMAKE_AUTOMOC ON)
#set executable output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


include_directories(
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/third_party/caffe/include
	${OpenCV_INCLUDE_DIRS}
	/usr/local/cuda-8.0/include
	/usr/include
	)

link_directories(
	${PROJECT_SOURCE_DIR}/lib
	${PROJECT_SOURCE_DIR}/third_party/caffe/build/lib
	/usr/lib
	/usr/lib/x86_64-linux-gnu
	)
#set source file list
set(
	SEGMENT_TEST_SRC_LIST 
	segment_test.cpp
	)
set(
	SEGMENT_EXPERIMENT_SRC_LIST
	segmentExperiment.cpp
	)
set(
	RECOGNITION_TEST_SRC_LIST
	recognition_test.cpp
	)
set(
	SEG_AND_RECOG_TEST_SRC_LIST
	seg_and_recognition_test.cpp
	)
set( 
	PICK_PLATE_SRC_LIST
	pick_plate.cpp
)
set(
	END2END_TEST_LIST
	end2end_test.cpp
)

add_executable(
	segment_test 
	${SEGMENT_TEST_SRC_LIST}
	)

target_link_libraries(
	segment_test 
	swpr
	#${OpenCV_LIBS}
	)
add_executable(
	segment_experiment 
	${SEGMENT_EXPERIMENT_SRC_LIST}
	)

target_link_libraries(
	segment_experiment 
	swpr
	#${OpenCV_LIBS}
	)

add_executable(
	recognition_test
	${RECOGNITION_TEST_SRC_LIST}
	)

target_link_libraries(
	recognition_test
	swpr
	glog
	boost_system
	#${OpenCV_LIBS}
	)

add_executable(
	recognition_and_reg_test
	${SEG_AND_RECOG_TEST_SRC_LIST}
	)

target_link_libraries(
	recognition_and_reg_test
	swpr
	glog
	boost_system
	#${OpenCV_LIBS}
	)

add_executable(
	pick_plate	
	${PICK_PLATE_SRC_LIST}
	)

target_link_libraries(
	pick_plate	
	swpr
	glog
	boost_system
	#${OpenCV_LIBS}
	)
add_executable(
	end2end_test
	${END2END_TEST_LIST}
	)
target_link_libraries(
	end2end_test	
	swpr
	glog
	gflags
	boost_system
	${OpenCV_LIBS}
	)
