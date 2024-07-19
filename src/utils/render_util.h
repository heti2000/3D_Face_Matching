#ifndef RENDER_UTIL_H
#define RENDER_UTIL_H

#include "utils/util.h"
#include "constants.hpp"
#include "Eigen.h"
 
template <typename T>
static void calculatePixelLighting(const Eigen::Matrix<T,3,9>& shParameters, Eigen::Matrix<T,3,1>& lightNorm, Eigen::Matrix<T,3,1>& result) {
    T x = lightNorm[0];
    T y = lightNorm[1];
    T z = lightNorm[2];

    Eigen::Matrix<T,9,1> shBands;
    shBands[0] = T(0.2820947918) * T(3.14);
    shBands[1] = T(0.4886025119) * y * T(2.09);
    shBands[2] = T(0.4886025119) * z * T(2.09);
    shBands[3] = T(0.4886025119) * x * T(2.09);
    shBands[4] = T(1.092548431) * x * y * T(0.79);
    shBands[5] = T(1.092548431) * y * z * T(0.79);
    shBands[6] = T(0.3153915653) * (T(3) * z*z - T(1)) * T(0.79);
    shBands[7] = T(1.092548431) * x * z * T(0.79);
    shBands[8] = T(0.5462742153) * (x*x - y*y) * T(0.79);

    Eigen::Matrix<T,3,1> _result = shParameters * shBands;
    result[0] = _result[0];
    result[1] = _result[1];
    result[2] = _result[2];
}

template <typename T>
static void calcVertexNormals(BFM_Manager* bfm_m, Eigen::Matrix<T, -1, 1>& vertexPosition, Eigen::Matrix<T, -1, 1>& vertexNormals) {
    Eigen::VectorXi& vecTriangleList = bfm_m->Get_CurrentFaces();
    Eigen::VectorXi faceCounts = Eigen::VectorXi::Zero(vertexPosition.size() / 3);

    for (int i=0; i < BFM_N_FACES; i++) {
        unsigned int vid0 = vecTriangleList[i + 0 * BFM_N_FACES];
        unsigned int vid1 = vecTriangleList[i + 1 * BFM_N_FACES];
        unsigned int vid2 = vecTriangleList[i + 2 * BFM_N_FACES];

        Eigen::Matrix<T,3,1> coord0 = vertexPosition.template block<3,1>(3 * vid0, 0);
        Eigen::Matrix<T,3,1> coord1 = vertexPosition.template block<3,1>(3 * vid1, 0);
        Eigen::Matrix<T,3,1> coord2 = vertexPosition.template block<3,1>(3 * vid2, 0);

        Eigen::Matrix<T,3,1> normal = (coord1 - coord0).cross(coord2 - coord0).normalized();
        if (normal[2] < 0) {
            normal = -normal;
        }

        faceCounts[vid0] += 1;
        faceCounts[vid1] += 1;
        faceCounts[vid2] += 1;

        vertexNormals.template block<3,1>(3 * vid0, 0) += normal;
        vertexNormals.template block<3,1>(3 * vid1, 0) += normal;
        vertexNormals.template block<3,1>(3 * vid2, 0) += normal;
    }

    for (int i=0; i < vertexNormals.size() / 3; i++) {
		if (faceCounts[i] == 0) {
			vertexNormals.template block<3,1>(3 * i, 0) = Eigen::Matrix<T,3,1>::Zero();
			std::cout << "Zero triangle for vertex " << i << " - this should not happen" << std::endl;
		} else {
        	vertexNormals.template block<3,1>(3 * i, 0) /= T(faceCounts[i]);
		}
    }
}

static void renderParamsToMat(MeshRenderer& meshRenderer, BFM_Manager* bfm_m, OptParams& params, cv::Mat& img, Eigen::Matrix<double, -1, 1>& vertexColor, bool renderLighting) {
    Eigen::Matrix<double, -1, 1> vertexPosition = bfm_m->Get_ShapeMu() + bfm_m->Get_ExprMu();
    Eigen::Matrix<double, -1, 1> renderPosition = Eigen::Matrix<double, -1, 1>::Zero(vertexPosition.size());

    if (renderLighting) {
        Eigen::Matrix<double, -1, 1> vertexNormals = Eigen::Matrix<double, -1, 1>::Zero(vertexPosition.size());
        calcVertexNormals(bfm_m, vertexPosition, vertexNormals);

        for (int i=0; i < BFM_N_VERTICES; i++) {
            Eigen::Matrix<double,3,1> col;
            Eigen::Matrix<double,3,1> normal = vertexNormals.block<3,1>(3*i,0);
            calculatePixelLighting(params.sh_weights, normal, col);
            vertexColor[3*i] *= col[0];
            vertexColor[3*i+1] *= col[1];
            vertexColor[3*i+2] *= col[2];
        }
    }

    const Eigen::VectorXd& vecShapeEv = bfm_m->Get_ShapeEv();
    const Eigen::MatrixXd& matShapePc = bfm_m->Get_ShapePc();

    const Eigen::VectorXd& vecExprEv = bfm_m->Get_ExprEv();
    const Eigen::MatrixXd& matExprPc = bfm_m->Get_ExprPc();

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

    cv::Mat renderedResult = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);
    cv::Mat baryCentric = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);
    cv::Mat triangleIds = cv::Mat::zeros(img.rows, img.cols, CV_32SC1);
    
    // render the result
    meshRenderer.render(renderPosition, vertexColor, vecTriangleList, renderedResult.ptr<float>(), baryCentric.ptr<float>(), triangleIds.ptr<unsigned int>());

    // convert the result to uchar
    renderedResult.convertTo(renderedResult, CV_8UC3, 255.0);
    // flip the image
    cv::flip(renderedResult, renderedResult, 0);
    // change color space to BGR
    cv::cvtColor(renderedResult, renderedResult, cv::COLOR_RGB2BGR);

    // copy the result to the image but dont overwrite black pixels
    cv::Mat mask = renderedResult > 0;
    renderedResult.copyTo(img, mask);
}

static void renderParamsToMat(MeshRenderer& meshRenderer, BFM_Manager* bfm_m, OptParams& params, cv::Mat& img) {
    Eigen::Matrix<double, -1, 1> vertexPosition = bfm_m->Get_ShapeMu() + bfm_m->Get_ExprMu();
    Eigen::Matrix<double, -1, 1> renderPosition = Eigen::Matrix<double, -1, 1>::Zero(vertexPosition.size());
    Eigen::Matrix<double, -1, 1> vertexColor = bfm_m->Get_TexMu();
    Eigen::Matrix<double, -1, 1> vertexNormals = Eigen::Matrix<double, -1, 1>::Zero(vertexPosition.size());

    calcVertexNormals(bfm_m, vertexPosition, vertexNormals);

    for (int i=0; i < BFM_N_VERTICES; i++) {
        Eigen::Matrix<double,3,1> col;
        Eigen::Matrix<double,3,1> normal = vertexNormals.block<3,1>(3*i,0);
        calculatePixelLighting(params.sh_weights, normal, col);
        vertexColor[3*i] *= col[0];
        vertexColor[3*i+1] *= col[1];
        vertexColor[3*i+2] *= col[2];
    }

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

    cv::Mat renderedResult = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);
    cv::Mat baryCentric = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);
    cv::Mat triangleIds = cv::Mat::zeros(img.rows, img.cols, CV_32SC1);
    
    // render the result
    meshRenderer.render(renderPosition, vertexColor, vecTriangleList, renderedResult.ptr<float>(), baryCentric.ptr<float>(), triangleIds.ptr<unsigned int>());

    // convert the result to uchar
    renderedResult.convertTo(renderedResult, CV_8UC3, 255.0);
    // flip the image
    cv::flip(renderedResult, renderedResult, 0);
    // change color space to BGR
    cv::cvtColor(renderedResult, renderedResult, cv::COLOR_RGB2BGR);

    // copy the result to the image but dont overwrite black pixels
    cv::Mat mask = renderedResult > 0;
    renderedResult.copyTo(img, mask);
}

static void renderParamsToMat(BFM_Manager* bfm_m, OptParams& params, cv::Mat& img) {
    MeshRenderer meshRenderer(img.cols, img.rows);
    renderParamsToMat(meshRenderer, bfm_m, params, img);
}

static void renderParamsToMat(BFM_Manager* bfm_m, OptParams& params, cv::Mat& img, Eigen::Matrix<double, -1, 1>& vertexColor, bool renderLighting) {
    MeshRenderer meshRenderer(img.cols, img.rows);
    renderParamsToMat(meshRenderer, bfm_m, params, img, vertexColor, renderLighting);
}

template <typename T>
inline Eigen::Matrix<T,3,1> cv2eigen(cv::Vec3b vec) {
	return Eigen::Matrix<T,3,1>(T(vec[0]), T(vec[1]), T(vec[2]));
}

#endif // RENDER_UTIL_H