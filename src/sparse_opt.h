#ifndef SPARCE_OPT_H
#define SPARCE_OPT_H

#include "Eigen.h"
#include "constants.hpp"
#include "bfm_manager.h"
#include "utils/util.h"
#include "parameters.h"
#include <ceres/ceres.h>

// was SparseCost before
struct SparseCost {
	SparseCost(BFM_Manager* model, Vector2d observed_landmark_, int vertex_id_, int image_width_, int image_height_, double weight_, int landmarkSize_) :
		bfm_m{ model }, observed_landmark{ observed_landmark_ }, vertex_id{ vertex_id_ }, image_width(image_width_), image_height(image_height_), weight(weight_), landmarkSize(landmarkSize_) {}

	template <typename T>
	bool operator()(const T* const rotation_c, const T* const translation_c, const T* const focal_length, const T* const shape_weights, const T* const exp_weights, T*  residuals) const {

		unsigned int length = bfm_m->Get_nVertices()*3;
		unsigned int length_expr = bfm_m->Get_nExprPcs();

        Eigen::VectorXd& vecShapeEv = bfm_m->Get_ShapeEv();
        Eigen::VectorXd& vecShapeMu = bfm_m->Get_ShapeMu();
        Eigen::MatrixXd& matShapePc = bfm_m->Get_ShapePc();

        Eigen::VectorXd& vecExprEv = bfm_m->Get_ExprEv();
        Eigen::VectorXd& vecExprMu = bfm_m->Get_ExprMu();
        Eigen::MatrixXd& matExprPc = bfm_m->Get_ExprPc();

		Eigen::Quaternion<T> rotation ={rotation_c[0], rotation_c[1], rotation_c[2], rotation_c[3]};
		Eigen::Matrix<T,3,1> translation = {translation_c[0], translation_c[1], translation_c[2]};

        Eigen::Matrix<T, 3, 1> vertex_pos;
		int idx = vertex_id * 3;

		vertex_pos[0] = T(vecShapeMu[idx]) + T(vecExprMu[idx]);        // X
		vertex_pos[1] = T(vecShapeMu[idx + 1]) + T(vecExprMu[idx + 1]);// Y
		vertex_pos[2] = T(vecShapeMu[idx + 2]) + T(vecExprMu[idx + 2]);// Z
		
		for (int i = 0, size = bfm_m->Get_nIdPcs(); i < size; i++) {
			T value = T(sqrt(vecShapeEv[i])) * shape_weights[i];
			vertex_pos[0] += T(matShapePc(idx, i)) * value;
			vertex_pos[1] += T(matShapePc(idx + 1, i)) * value;
			vertex_pos[2] += T(matShapePc(idx + 2, i)) * value;
		}
		for (int i = 0, size = bfm_m->Get_nExprPcs(); i < size; i++) {
			T value = T(sqrt(vecExprEv[i])) * exp_weights[i];
			vertex_pos[0] += T(matExprPc(idx, i)) * value;
			vertex_pos[1] += T(matExprPc(idx + 1, i)) * value;
			vertex_pos[2] += T(matExprPc(idx + 2, i)) * value;
		}

		// compute matrix transformation 
		Eigen::Matrix<T, 4, 4> transformation_matrix = transform_matrix(translation, rotation);
		Eigen::Matrix<T, 4, 1> projected_point = perspective_projection(image_width, image_height, *focal_length, transformation_matrix, vertex_pos);
		Eigen::Matrix<T, 3, 1> transformed = projected_point.hnormalized();

		transformed[0] = (transformed[0] + T(1)) / T(2)* T(image_width);
		transformed[1] = (transformed[1] + T(1)) / T(2)* T(image_height);
		// flip image y axe
		transformed[1] = T(image_height) - transformed[1];

		residuals[0] = T(1 / sqrt(landmarkSize)) * T(transformed[0] - observed_landmark[0]) / T(image_width) * sqrt(weight);
		residuals[1] = T(1 / sqrt(landmarkSize)) * T(transformed[1] - observed_landmark[1]) / T(image_height) * sqrt(weight);

		return true;
	}

	static ceres::CostFunction* create(BFM_Manager* model, Vector2d observed_landmark_, int vertex_id_, int image_width_, int image_height_, double weight_, int landMarkSize) {
		return new ceres::AutoDiffCostFunction<SparseCost, 2, 4, 3, 1, BFM_N_ID_PCS, BFM_N_EXPR_PCS >(
			new SparseCost(model, observed_landmark_, vertex_id_, image_width_, image_height_, weight_, landMarkSize));
	}

	static void addToProblem(ceres::Problem& problem, OptParams& params, BFM_Manager* model, const std::vector<Eigen::Vector2d> imageLandmarks, int image_width, int image_height, double weight_) {
		for (int i=0; i<imageLandmarks.size(); i++) {
			problem.AddResidualBlock(
				create(model, imageLandmarks[i], model->Get_LandmarkIdx(i), image_width, image_height, weight_, imageLandmarks.size()),
				NULL,
				params.rotation.data(), params.translation.data(), &params.focal_length, params.shape_weights.data(), params.exp_weights.data()
			);
		}
	}

private:
	BFM_Manager* bfm_m;
	const Vector2d observed_landmark;
	const int vertex_id;
	const int image_width;
	const int image_height;
	const double weight;
	const int landmarkSize;
};

#endif