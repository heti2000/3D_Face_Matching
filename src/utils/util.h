#ifndef UTIL_H   // Start of include guard
#define UTIL_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <cmath>
#include <Eigen/Geometry> // Include for Quaternion
#include <ceres/ceres.h>
#include "parameters.h"
#include "bfm_manager.h"
#include "render.h"


std::vector<Eigen::Vector2d> read_2d_landmarks(const std::string& filename);

Eigen::Matrix4d Quad2Mat(Eigen::Quaternion<double> rotation, Eigen::Vector3d translation);

template <typename T>
std::vector<Eigen::Matrix<T, 2, 1>> apply_transform(const std::vector<Eigen::Matrix<T, 2, 1>>& landmarks,
                                                const Eigen::Matrix<T, 3, 3>& transform ){
    std::vector<Eigen::Matrix<T, 2, 1>> transformed_landmarks;
    for (const auto& landmark : landmarks) {
        Eigen::Matrix<T, 3, 1> landmark_h = transform * landmark.homogeneous();
        Eigen::Matrix<T, 2, 1> transformed_landmark = landmark_h.hnormalized();
        transformed_landmarks.push_back(transformed_landmark);
    }
    return transformed_landmarks;
}


template <typename T>
Eigen::Matrix<T, 2, 1> world_to_image(const Eigen::Matrix<T, 3, 1> & landmark, 
                                        const Eigen::Matrix<T, 3, 3> & intrinsic, 
                                        const Eigen::Matrix<T, 4, 4> & pose ) {
    // projection matrix
    Eigen::Matrix<T, 3, 4 > projection_matrix = intrinsic * pose.template block<3, 4>(0, 0);
    Eigen::Matrix<T, 3, 1> landmark_2d = projection_matrix * landmark.homogeneous();

    Eigen::Matrix<T, 2, 1>  image_landmark = landmark_2d.hnormalized();
    return image_landmark;
}

template <typename T>
Eigen::Matrix<T, 3, 3>  getIntrinsic(const int imageWidth,const int imageHeight,
                                    const T focalLength, 
                                    const T sensorWidth, 
                                    const T sensorHeight) {
    Eigen::Matrix<T, 3, 3>  intrinsic = Eigen::Matrix<T, 3, 3>::Identity();
    // Convert focal length from mm to pixels
    T fx = T(imageWidth) * focalLength / sensorWidth;
    T fy = T(imageHeight) * focalLength / sensorHeight;

    // Principal point is usually at the image center
    T cx = T(imageWidth) / 2.0;
    T cy = T(imageHeight) / 2.0;
    intrinsic(0, 0) = fx;
    intrinsic(1, 1) = fy;
    intrinsic(0, 2) = cx;
    intrinsic(1, 2) = cy;
    return intrinsic;
}


template <typename T>
Eigen::Matrix<T, 4, 4> transform_matrix(Eigen::Matrix<T, 3, 1> translation, Eigen::Quaternion<T> rotation  ){
    Eigen::Matrix<T, 4, 4> transformation;
    transformation.setIdentity();
    transformation.template block<3, 3>(0, 0) = rotation.toRotationMatrix();
    transformation.template block<3, 1>(0, 3) = translation;

    return transformation;
}

template <typename T>
Eigen::Matrix<T, 4, 4> perspective_matrix(T angle, T aspect, T zNear, T zFar) {
    T const rad = angle * T(M_PI / 180.0);
    T const tanHalfFovy = tan(rad / T(2));

    Eigen::Matrix<T, 4, 4> projection_matrix = Eigen::Matrix<T, 4, 4>::Zero();

    projection_matrix(0, 0) = T(1) / (aspect * tanHalfFovy);
    projection_matrix(1, 1) = T(1) / tanHalfFovy;
    projection_matrix(2, 2) = -(zFar + zNear) / (zFar - zNear);
    projection_matrix(2, 3) = -(T(2) * zFar * zNear) / (zFar - zNear);
    projection_matrix(3, 2) = -T(1);
    return projection_matrix;
}

template <typename T>
static Eigen::Matrix<T, 4, 1> perspective_projection(int width, int height, T fov, Eigen::Matrix<T, 4, 1> transformed_point) {
    //Projecting
    T ar = T(width) /T(height);
    T n = T(0.1);
    T f = T(10000);
    Eigen::Matrix<T, 4, 4> projection_matrix = perspective_matrix(fov, ar, n, f);
    Eigen::Matrix<T, 4, 1> projected_point = projection_matrix * transformed_point;

    // result.transposeInPlace();
    return projected_point;
}

template <typename T>
static Eigen::Matrix<T, 4, 1> perspective_projection(int width, int height, T fov, Eigen::Matrix<T, 4, 4> transformation_matrix, Eigen::Matrix<T, 3, 1> mapped_vertices) {

    Eigen::Matrix<T, 4, 1> transformed_point = transformation_matrix * mapped_vertices.homogeneous();;

    //Projecting
    T ar = T(width) /T(height);
    T n = T(0.1);
    T f = T(10000);
    Eigen::Matrix<T, 4, 4> projection_matrix = perspective_matrix(fov, ar, n, f);
    Eigen::Matrix<T, 4, 1> projected_point = projection_matrix * transformed_point;

    // result.transposeInPlace();
    return projected_point;
}

static float toFloat(double val) {
    return static_cast<float>(val);
}

template <typename T, int N>
static float toFloat(const ceres::Jet<T, N>& val) {
    return static_cast<float>(val.a);
}

static int toInt(double val) {
    return static_cast<int>(val);
}

template <typename T, int N>
static int toInt(const ceres::Jet<T, N>& val) {
    return static_cast<int>(val.a);
}

#endif // UTIL_H




