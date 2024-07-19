#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "Eigen.h"
#include "constants.hpp"

struct OptParams {
    Eigen::Matrix<double, 3, 9> sh_weights;

    VectorXd shape_weights;
    VectorXd exp_weights;
    VectorXd col_weights;
    // light params
    // camera params
    double focal_length;
    // pose params
    Vector4d rotation;
    Vector3d translation;

    static OptParams init() {
        OptParams params;
        params.sh_weights = Eigen::Matrix<double, 3, 9>::Zero();
        params.sh_weights(0, 0) = 1;
        params.sh_weights(1, 0) = 1;
        params.sh_weights(2, 0) = 1;
        params.shape_weights = Eigen::VectorXd::Zero(BFM_N_ID_PCS);
        params.exp_weights = Eigen::VectorXd::Zero(BFM_N_EXPR_PCS);
        params.col_weights = Eigen::VectorXd::Zero(BFM_N_ID_PCS);
        params.focal_length = 35.0;
        params.rotation << 1, 0, 0, 0;
        params.translation = Eigen::Vector3d::Zero();
        params.translation[2] = -400;
        return params;
    }
};

#endif