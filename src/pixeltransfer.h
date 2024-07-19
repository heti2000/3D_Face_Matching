#include "parameters.h"
#include "bfm_manager.h"
#include "Eigen.h"
#include <opencv2/opencv.hpp>
#include "constants.hpp"
#include "utils/render_util.h"

// get the color for each vertex of a specific face model from a given image using bipolar interpolation
void grabPixels(BFM_Manager* bfm_m, OptParams& params, cv::Mat& img, Eigen::Matrix<double, -1, 1>& vertexColor) {
    Eigen::Matrix<double, -1, 1> vertexPosition = bfm_m->Get_ShapeMu() + bfm_m->Get_ExprMu();
    Eigen::Matrix<double, -1, 1> renderPosition = Eigen::Matrix<double, -1, 1>::Zero(vertexPosition.size());

    const Eigen::VectorXd& vecShapeEv = bfm_m->Get_ShapeEv();
    const Eigen::MatrixXd& matShapePc = bfm_m->Get_ShapePc();

    const Eigen::VectorXd& vecExprEv = bfm_m->Get_ExprEv();
    const Eigen::MatrixXd& matExprPc = bfm_m->Get_ExprPc();

    const Eigen::VectorXd& vecTexEv= bfm_m->Get_TexEv();
    const Eigen::MatrixXd& matTexPc = bfm_m->Get_TexPc();

    // get the translation matrix
    Eigen::Quaternion<double> rotation ={params.rotation[0], params.rotation[1], params.rotation[2], params.rotation[3]};
    Eigen::Matrix<double,3,1>& translation = params.translation;
    Eigen::Matrix<double, 4, 4> transformation_matrix = transform_matrix(translation, rotation);

    Eigen::VectorXi& vecTriangleList = bfm_m->Get_CurrentFaces();

    for (int v=0; v < BFM_N_VERTICES; v++) {
        for (int i = 0; i < BFM_N_ID_PCS; i++) {
            double shapeFactor = sqrt(vecShapeEv[i]) * params.shape_weights[i];
            vertexPosition[3*v]   += shapeFactor * matShapePc(3*v  , i);
            vertexPosition[3*v+1] += shapeFactor * matShapePc(3*v+1, i);
            vertexPosition[3*v+2] += shapeFactor * matShapePc(3*v+2, i);

            double texFactor = sqrt(vecTexEv[i]) * params.col_weights[i];
            vertexColor[3*v]   += texFactor * matTexPc(3*v  , i);
            vertexColor[3*v+1] += texFactor * matTexPc(3*v+1, i);
            vertexColor[3*v+2] += texFactor * matTexPc(3*v+2, i);
        }
        
        for (int i = 0; i < BFM_N_EXPR_PCS; i++) {
            double exprFactor = sqrt(vecExprEv[i]) * params.exp_weights[i];
            vertexPosition[3*v]   += exprFactor * matExprPc(3*v  , i);
            vertexPosition[3*v+1] += exprFactor * matExprPc(3*v+1, i);
            vertexPosition[3*v+2] += exprFactor * matExprPc(3*v+2, i);
        }

        // project the point
        Eigen::Matrix<double,3,1> currentPos = vertexPosition.block<3,1>(3*v,0);
        Eigen::Matrix<double,4,1> transformedPos = transformation_matrix * currentPos.homogeneous();
        renderPosition.block<3,1>(3*v,0) = perspective_projection(img.cols, img.rows, params.focal_length, transformedPos).hnormalized();
    }

    int img_width = img.cols;
    int img_height = img.rows;

    for (int k=0; k < BFM_N_VERTICES; k++) {
        double i_screen = (renderPosition[3*k] + 1) * (img_width / 2);
        double j_screen = (renderPosition[3*k+1] + 1) * (img_height / 2);

        int i = int(i_screen);
        int j = int(j_screen);

        double w_1 = (i+1-i_screen)*(j+1-j_screen);
        double w_2 = (i_screen-i)*(j+1-j_screen);
        double w_3 = (i+1-i_screen)*(j_screen-j);
        double w_4 = (i_screen-i)*(j_screen-j);

        cv::Vec3b pixel1_ = img.at<cv::Vec3b>(img_height - toInt(j) - 1, toInt(i));
        cv::Vec3b pixel2_ = img.at<cv::Vec3b>(img_height - toInt(j) - 1, toInt(i)+1);
        cv::Vec3b pixel3_ = img.at<cv::Vec3b>(img_height - toInt(j) - 2, toInt(i));
        cv::Vec3b pixel4_ = img.at<cv::Vec3b>(img_height - toInt(j) - 2, toInt(i)+1);   
        
        Eigen::Matrix<double,3,1> pixel1 = cv2eigen<double>(pixel1_) / 255.;
        Eigen::Matrix<double,3,1> pixel2 = cv2eigen<double>(pixel2_) / 255.;
        Eigen::Matrix<double,3,1> pixel3 = cv2eigen<double>(pixel3_) / 255.;
        Eigen::Matrix<double,3,1> pixel4 = cv2eigen<double>(pixel4_) / 255.;

        double weight_sum = w_1 + w_2 + w_3 + w_4;
        vertexColor[3*k+0] = (w_1*pixel1[2] + w_2*pixel2[2] + w_3*pixel3[2] + w_4*pixel4[2]) / weight_sum;
        vertexColor[3*k+1] = (w_1*pixel1[1] + w_2*pixel2[1] + w_3*pixel3[1] + w_4*pixel4[1]) / weight_sum;
        vertexColor[3*k+2] = (w_1*pixel1[0] + w_2*pixel2[0] + w_3*pixel3[0] + w_4*pixel4[0]) / weight_sum;     
    }
}