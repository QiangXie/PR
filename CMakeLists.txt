#set project name eg.test
Project (SWPR)
cmake_minimum_required(VERSION 2.8)
set(OpenCV_DIR "/home/swli/.local")
find_package(OpenCV REQUIRED) 
set (CMAKE_CXX_STANDARD 11)
#to generate .json file for YouCompleteMe,and set compilation_database_folder to .json file absolute path in .ycm_extra_conf.py
#(usualy is build folder)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
add_subdirectory(src)
