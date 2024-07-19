#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "./bfm_manager.h"
#include "./utils/util.h"
#include "./render.h"
#include <ceres/ceres.h>
#include "utils/util.h"
#include <opencv2/opencv.hpp>
#include "parameters.h"
#include "constants.hpp"
#include "dense_opt.h"
#include "sparse_opt.h"
#include "reg_opt.h"
#include "visualizer.h"

// Review the lib reference and remove the unnecessary ones

class Optimizer{
    private:
        BFM_Manager* bfm_m;
		OptParams& params;
        ceres::Solver::Options options;

    public:
        Optimizer (BFM_Manager* bfm_m, OptParams& params_);
		void optimize(const cv::Mat img, const vector<Eigen::Vector2d> imageLandmarks, bool optimizeDense, bool visualization);
		void optimize(const cv::Mat img, const vector<Eigen::Vector2d> imageLandmarks, bool optimizeDense);
};

#endif