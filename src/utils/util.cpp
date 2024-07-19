#include "util.h"


std::vector<Eigen::Vector2d> read_2d_landmarks(const std::string& filename) {
    std::vector<Eigen::Vector2d> landmarks;
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return landmarks;
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        Eigen::Vector2d landmark;
        if (iss >> landmark[0] >> landmark[1]) {
            landmarks.push_back(landmark);
        }
    }

    file.close();
    return landmarks;
}

template <typename T>
Eigen::Matrix<T,4,4> Quad2Mat(Eigen::Quaternion<T> rotation, Eigen::Matrix<T,3,1> translation){
    Eigen::Matrix<T,4,4> transformation;
    transformation.setIdentity();
    transformation.template block<3, 3>(0, 0) = rotation.toRotationMatrix();
    transformation.template block<3, 1>(0, 3) = translation;
    return transformation;
}