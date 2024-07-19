#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "parameters.h"
#include "bfm_manager.h"
#include "utils/util.h"
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include "render.h"
#include "debug.h"
#include <vector>
#include "utils/render_util.h"

class VisualizationCallback : public ceres::IterationCallback {
 public:
  explicit VisualizationCallback(OptParams* params_, BFM_Manager* bfm_m, int img_w, int img_h)
      : params(params_), bfm_m_(bfm_m), img(img_h, img_w, CV_32FC3), baryCentric(img_h, img_w, CV_32FC3), meshRenderer(img_w, img_h)
        #if FACE_DEBUG_MODE == 1
        ,debug(Debug::getInstance())
        #endif 
      {
        triangleIds = new unsigned int[img_w*img_h];
      }

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {

    int iteration = summary.iteration;

    cv::Mat renderImg(img.rows, img.cols, CV_32FC3);
    // ---- for sparse ----
    // Eigen::VectorXd vertexColors = Eigen::VectorXd::Ones(3*BFM_N_VERTICES);
    // renderParamsToMat(meshRenderer, bfm_m_, *params, renderImg, vertexColors);
    // ---- for dense ----
    renderParamsToMat(meshRenderer, bfm_m_, *params, renderImg);
    cv::imshow("Rendered Image", renderImg);

    // // create the file name in the format "rendered_<iteration>.png" where iteration has padded zeros
    // std::stringstream ss;
    // ss << std::setw(3) << std::setfill('0') << iteration;
    // std::string filename = "denserender_opt/rendered_" + ss.str() + ".jpg";
    // cv::imwrite(filename, renderImg);

    cv::waitKey(1);

    int img_width = img.cols;
    int img_height = img.rows;

    #if FACE_DEBUG_MODE == 1
    // convert debug["comparison"] to a cv::Mat
    float* comparison = debug["comparison"];
    if (comparison != NULL) {
        cv::Mat comparisonMat(img_height, img_width, CV_32FC3, comparison);
        comparisonMat = comparisonMat.clone();
        cv::cvtColor(comparisonMat, comparisonMat, cv::COLOR_RGB2BGR);
        cv::flip(comparisonMat, comparisonMat, 0);
        // cv::moveWindow("Comparison", img_width, 0);
        cv::imshow("Comparison", comparisonMat);
    }

    float* error = debug["error"];
    if (error != NULL) {
        cv::Mat errorMat(img_height, img_width, CV_32FC1, error);
        errorMat = errorMat.clone();

        // cv::cvtColor(errorMat, errorMat, cv::COLOR_RGB2BGR);
        cv::flip(errorMat, errorMat, 0);
        // cv::moveWindow("Error", 2*img_width, 0);
        cv::imshow("Error", errorMat);
    }

    // visualize normals
    float* normals = debug["normal"];
    if (normals != NULL) {
        cv::Mat normalsMat(img_height, img_width, CV_32FC3, normals);
        normalsMat = normalsMat.clone();
        cv::cvtColor(normalsMat, normalsMat, cv::COLOR_RGB2BGR);
        cv::flip(normalsMat, normalsMat, 0);
        cv::imshow("Normals", normalsMat);
    }

    // visualize denseRender
    float* denseRender = debug["denseRender"];
    if (denseRender != NULL) {
        cv::Mat denseRenderMat(img_height, img_width, CV_32FC3, denseRender);
        denseRenderMat = denseRenderMat.clone();
        cv::cvtColor(denseRenderMat, denseRenderMat, cv::COLOR_RGB2BGR);
        cv::flip(denseRenderMat, denseRenderMat, 0);
        cv::imshow("DenseRender", denseRenderMat);
        // store float image
        cv::Mat denseRenderMat8U;
        denseRenderMat.convertTo(denseRenderMat8U, CV_8UC3, 255.0);
        cv::imwrite("denseRender.png", denseRenderMat8U);
    }
    #endif

    return ceres::SOLVER_CONTINUE;
  }

 private:
  OptParams* params;
  BFM_Manager* bfm_m_;
  cv::Mat img;
  cv::Mat baryCentric;
  unsigned int* triangleIds;
  MeshRenderer meshRenderer;
  #if FACE_DEBUG_MODE == 1
  Debug& debug;
  #endif
};

#endif // VISUALIZER_H