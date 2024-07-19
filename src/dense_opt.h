#ifndef DENSE_OPT_H
#define DENSE_OPT_H

#include "Eigen.h"
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include "constants.hpp"
#include "parameters.h"
#include "render.h"
#include "debug.h"
#include "utils/util.h"
#include "utils/render_util.h"

struct DenseRGBCost {
    DenseRGBCost(BFM_Manager* bfm_m_, const cv::Mat& img_, const double weight_, const double lightWeight_) :
		bfm_m{ bfm_m_ }, img{ img_ }, img_height(img_.rows), img_width(img_.cols), weight(weight_), lightWeight(lightWeight_), meshRenderer(img_.cols, img_.rows)
        #if FACE_DEBUG_MODE == 1
        , debug(Debug::getInstance())
        #endif
	{
        renderedImage = new float[img_width*img_height*3];
        barycentricCoords = new float[img_width*img_height*3];
        triangleIds = new unsigned int[img_width*img_height];

        #if FACE_DEBUG_MODE == 1
        debug.initKey("comparison", 3*img_width*img_height);
        debug.initKey("normal", 3*img_width*img_height);
        debug.initKey("denseRender", 3*img_width*img_height);
        debug.initKey("error", img_width*img_height);
        #endif
    }

    ~DenseRGBCost() {
        delete[] renderedImage;
        delete[] barycentricCoords;
        delete[] triangleIds;
    }

    template <typename T>
	bool operator()(const T* const sh_params_c, const T* const shape_weights_c, const T* const expr_weights_c, const T* const color_weights_c, const T* const rotation_c, const T* const translation_c, const T* const focal_length, T* residuals) const {
        // Load the parameters
        const Eigen::Matrix<T, 3, 9> shParams = Eigen::Map<const Eigen::Matrix<T, 3, 9>> (sh_params_c);
        
        // get the model
        Eigen::Matrix<T, -1, 1> vertexPosition = (bfm_m->Get_ShapeMu() + bfm_m->Get_ExprMu()).cast<T>();
        Eigen::Matrix<T, -1, 1> renderPosition = Eigen::Matrix<T, -1, 1>::Zero(vertexPosition.size());
        Eigen::Matrix<T, -1, 1> vertexColor = bfm_m->Get_TexMu().cast<T>();

        const Eigen::VectorXd& vecShapeEv = bfm_m->Get_ShapeEv();
        const Eigen::MatrixXd& matShapePc = bfm_m->Get_ShapePc();

        const Eigen::VectorXd& vecExprEv = bfm_m->Get_ExprEv();
        const Eigen::MatrixXd& matExprPc = bfm_m->Get_ExprPc();

        const Eigen::VectorXd& vecTexEv= bfm_m->Get_TexEv();
        const Eigen::MatrixXd& matTexPc = bfm_m->Get_TexPc();

        Eigen::VectorXi& vecTriangleList = bfm_m->Get_CurrentFaces();

        // get the translation matrix
        Eigen::Quaternion<T> rotation ={rotation_c[0], rotation_c[1], rotation_c[2], rotation_c[3]};
		Eigen::Matrix<T,3,1> translation = {translation_c[0], translation_c[1], translation_c[2]};
		Eigen::Matrix<T, 4, 4> transformation_matrix = transform_matrix(translation, rotation);

        // std::cout << "Vertices before:" << std::endl << vertexPosition.template block<9,1>(0,0) << std::endl;
        for (int v=0; v < BFM_N_VERTICES; v++) {
            for (int i = 0; i < BFM_N_ID_PCS; i++) {
                T shapeFactor = sqrt(vecShapeEv[i]) * shape_weights_c[i];
                vertexPosition[3*v]   += shapeFactor * matShapePc(3*v  , i);
                vertexPosition[3*v+1] += shapeFactor * matShapePc(3*v+1, i);
                vertexPosition[3*v+2] += shapeFactor * matShapePc(3*v+2, i);

                T texFactor = sqrt(vecTexEv[i]) * color_weights_c[i];
                vertexColor[3*v]   += texFactor * matTexPc(3*v  , i);
                vertexColor[3*v+1] += texFactor * matTexPc(3*v+1, i);
                vertexColor[3*v+2] += texFactor * matTexPc(3*v+2, i);
            }
            
            for (int i = 0; i < BFM_N_EXPR_PCS; i++) {
                T exprFactor = sqrt(vecExprEv[i]) * expr_weights_c[i];
                vertexPosition[3*v]   += exprFactor * matExprPc(3*v  , i);
                vertexPosition[3*v+1] += exprFactor * matExprPc(3*v+1, i);
                vertexPosition[3*v+2] += exprFactor * matExprPc(3*v+2, i);
            }
        }
        
        for (int v=0; v < BFM_N_VERTICES; v++) {
            // project the point
            Eigen::Matrix<T,3,1> currentPos = vertexPosition.template block<3,1>(3*v,0);
            Eigen::Matrix<T,4,1> transformedPos = transformation_matrix * currentPos.homogeneous();
            vertexPosition.template block<3,1>(3*v,0) = transformedPos.hnormalized();
            renderPosition.template block<3,1>(3*v,0) = perspective_projection(img_width, img_height, *focal_length, transformedPos).hnormalized();
        }

		// calculate the vertex normals
		Eigen::Matrix<T, -1, 1> vertexNormals = Eigen::Matrix<T, -1, 1>::Zero(vertexPosition.size());
		calcVertexNormals(bfm_m, vertexPosition, vertexNormals);

        #if FACE_DEBUG_MODE == 1
        float* debugComparison = debug["comparison"];
        float* debugError = debug["error"];
        float* debugNormal = debug["normal"];
        float* debugRender = debug["denseRender"];

		for (int k=0; k<img_width*img_height; k++) {
			debugComparison[3*k] = 0;
			debugComparison[3*k+1] = 0;
			debugComparison[3*k+2] = 0;
			debugNormal[3*k] = 0;
			debugNormal[3*k+1] = 0;
			debugNormal[3*k+2] = 0;
			debugRender[3*k] = 0;
			debugRender[3*k+1] = 0;
			debugRender[3*k+2] = 0;
			debugError[k] = 0;
		}
        #endif

		T errorSum = T(0);
        
        for (int k=0; k < BFM_N_VERTICES; k++) {
            // get the pixel position
            T i_screen = (renderPosition[3*k] + T(1)) * T(img_width / 2);
            T j_screen = (renderPosition[3*k+1] + T(1)) * T(img_height / 2);

			Eigen::Matrix<T, 3, 1> vertexNorm = vertexNormals.template block<3,1>(3*k,0);
			Eigen::Matrix<T, 3, 1> color;

			calculatePixelLighting(shParams, vertexNorm, color);
			// color += vertexColor.template block<3,1>(3*k,0);

            T color_r = color[0] * vertexColor[3*k];
            T color_g = color[1] * vertexColor[3*k+1];
            T color_b = color[2] * vertexColor[3*k+2];

            T i = floor(i_screen);
            T j = floor(j_screen);

            T w_1 = (i+T(1)-i_screen)*(j+T(1)-j_screen);
            T w_2 = (i_screen-i)*(j+T(1)-j_screen);
            T w_3 = (i+T(1)-i_screen)*(j_screen-j);
            T w_4 = (i_screen-i)*(j_screen-j);

            cv::Vec3b pixel1_ = img.at<cv::Vec3b>(img_height - toInt(j) - 1, toInt(i));
            cv::Vec3b pixel2_ = img.at<cv::Vec3b>(img_height - toInt(j) - 1, toInt(i)+1);
            cv::Vec3b pixel3_ = img.at<cv::Vec3b>(img_height - toInt(j) - 2, toInt(i));
            cv::Vec3b pixel4_ = img.at<cv::Vec3b>(img_height - toInt(j) - 2, toInt(i)+1);

			// convert pixel colors to double colors
			Eigen::Matrix<T,3,1> pixel1 = cv2eigen<T>(pixel1_) / T(255);
			Eigen::Matrix<T,3,1> pixel2 = cv2eigen<T>(pixel2_) / T(255);
			Eigen::Matrix<T,3,1> pixel3 = cv2eigen<T>(pixel3_) / T(255);
			Eigen::Matrix<T,3,1> pixel4 = cv2eigen<T>(pixel4_) / T(255);

			T weight_sum = w_1 + w_2 + w_3 + w_4;
            T r_img = T(w_1*pixel1[2] + w_2*pixel2[2] + w_3*pixel3[2] + w_4*pixel4[2]) / weight_sum;
            T g_img = T(w_1*pixel1[1] + w_2*pixel2[1] + w_3*pixel3[1] + w_4*pixel4[1]) / weight_sum;
            T b_img = T(w_1*pixel1[0] + w_2*pixel2[0] + w_3*pixel3[0] + w_4*pixel4[0]) / weight_sum;

            T r_delta = r_img - color_r;
            T g_delta = g_img - color_g;
            T b_delta = b_img - color_b;

            T error = sqrt(r_delta*r_delta + g_delta*g_delta + b_delta*b_delta);

            #if FACE_DEBUG_MODE == 1
			debugNormal[toInt(T(3) * (T(img_width) * j + i))] = toFloat(vertexNorm[0]);
			debugNormal[toInt(T(3) * (T(img_width) * j + i) + T(1))] = toFloat(vertexNorm[1]);
			debugNormal[toInt(T(3) * (T(img_width) * j + i) + T(2))] = toFloat(vertexNorm[2]);

			debugRender[toInt(T(3) * (T(img_width) * j + i))] = toFloat(color_r);
			debugRender[toInt(T(3) * (T(img_width) * j + i) + T(1))] = toFloat(color_g);
			debugRender[toInt(T(3) * (T(img_width) * j + i) + T(2))] = toFloat(color_b);

			debugComparison[toInt(T(3) * (T(img_width) * j + i))] = toFloat(r_img);
			debugComparison[toInt(T(3) * (T(img_width) * j + i) + T(1))] = toFloat(g_img);
			debugComparison[toInt(T(3) * (T(img_width) * j + i) + T(2))] = toFloat(b_img);

			debugError[toInt(T(img_width) * j + i)] = toFloat(error);
            #endif

            errorSum += error;
        }

        residuals[0] = T(1 / sqrt(BFM_N_VERTICES)) * sqrt(errorSum) * sqrt(lightWeight);

        return true;
    }

	static ceres::CostFunction* create(BFM_Manager* bfm_m_, const cv::Mat& img_, const double weight_, const double lightWeight_) {
		return new ceres::AutoDiffCostFunction<DenseRGBCost, 1, 29, BFM_N_ID_PCS, BFM_N_EXPR_PCS, BFM_N_ID_PCS, 4, 3, 1>(
			new DenseRGBCost(bfm_m_, img_, weight_, lightWeight_));
	}

	static void addToProblem(ceres::Problem& problem, OptParams& params, BFM_Manager* model, const cv::Mat& img_, double weight_, double lightWeight_) {
		problem.AddResidualBlock(
            create(model, img_, weight_, lightWeight_),
            NULL,
            params.sh_weights.data(), params.shape_weights.data(), params.exp_weights.data(), params.col_weights.data(), params.rotation.data(), params.translation.data(), &params.focal_length
        );
	}

// should become private later
public:
    float* renderedImage;
    float* barycentricCoords;
    unsigned int* triangleIds;

private:
    BFM_Manager* bfm_m;
    const MeshRenderer meshRenderer;
    const cv::Mat& img;
	const int img_width;
	const int img_height;
    double weight;
    double lightWeight;
    #if FACE_DEBUG_MODE == 1
    Debug& debug;
    #endif
};

#endif