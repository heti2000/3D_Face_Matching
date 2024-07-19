cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)
set(CMAKE_BUILD_TYPE Release)

project(face_reconstruction)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(glog_DIR ${LIBRARY_DIR}/glog-lib/lib/cmake/glog CACHE PATH "Path to installed glog")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/CMake CACHE PATH "Path to installed Ceres")

# Set libraries for MacOS
# You need to install OpenCV, Freeimage with brew command for this to work
if(APPLE)
    set(OpenCV_DIR /usr/local/Cellar/opencv/4.9.0_1 CACHE PATH "Path to installed OpenCV")
    set(HDF5_DIR /usr/local/Cellar/hdf5/1.14.3 CACHE PATH "Path to installed HDF5")
    set(HDF5_INCLUDE_DIR /usr/local/Cellar/hdf5/1.14.3/include/ CACHE PATH "Path to HDF5 header file")
    set(HDF5_LIBRARY_DIR /usr/local/Cellar/hdf5/1.14.3/lib/ CACHE PATH "Path to HDF5 .lib/.dll folder")
endif(APPLE)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

#find_package(nanoflann REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(glog REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
get_target_property(GLOG_DLL_PATH_DEBUG glog::glog IMPORTED_LOCATION_DEBUG)
get_target_property(GLOG_DLL_PATH_RELEASE glog::glog IMPORTED_LOCATION_RELEASE)

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

set(HEADER_FILES
        lib/Eigen.h
        src/utils/io.h
        src/utils/points.h
        src/utils/render_util.h
        src/utils/util.h
        src/bfm_manager.h
        src/constants.hpp
        src/data.h
        src/debug.h
        src/dense_opt.h
        src/optimizer.h
        src/pixeltransfer.h
        src/reg_opt.h
        src/render.h
        src/sparse_opt.h
        src/parameters.h
        src/visualizer.h
)

set(SOURCE_FILES
        src/utils/util.cpp
        src/bfm_manager.cpp
        src/optimizer.cpp
        src/render.cpp
)

include_directories(src lib ${OpenCV_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} $(GLEW_INCLUDE_DIRS))
# link_directories(${FreeImage_LIBRARY_DIR})
add_executable(main main.cpp ${HEADER_FILES} ${SOURCE_FILES})
# target_link_libraries(main ceres freeimage Eigen3::Eigen ${OpenCV_LIBS} ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES} glfw)
target_link_libraries(main ceres Eigen3::Eigen ${OpenCV_LIBS} ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES} glfw OpenGL::GL GLEW::GLEW)
target_include_directories(main PUBLIC ${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR} )


add_executable(expression_transfer expression_transfer.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(expression_transfer ceres Eigen3::Eigen ${OpenCV_LIBS} ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES} glfw OpenGL::GL GLEW::GLEW)
target_include_directories(expression_transfer PUBLIC ${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR} )

add_executable(video_analyze video_analyze.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(video_analyze ceres Eigen3::Eigen ${OpenCV_LIBS} ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES} glfw OpenGL::GL GLEW::GLEW)
target_include_directories(video_analyze PUBLIC ${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR} )