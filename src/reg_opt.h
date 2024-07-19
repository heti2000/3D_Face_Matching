#ifndef REG_OPT_H
#define REG_OPT_H

#include "constants.hpp"
#include "parameters.h"
#include <ceres/ceres.h>

struct ColorRegularizerCost
{
	ColorRegularizerCost(const double weight_)
		: weight{ weight_ }
	{}

	template<typename T>
	bool operator()(T const* color_weights, T* residuals) const
	{

		for (int i = 0; i < BFM_N_ID_PCS; i++) {
			residuals[i] = color_weights[i] * T(sqrt(weight));
		}
		return true;
	}

	static ceres::CostFunction* create(const double weight_) {
		return new ceres::AutoDiffCostFunction<ColorRegularizerCost, BFM_N_ID_PCS, BFM_N_ID_PCS>(
			new ColorRegularizerCost(weight_)
		);
	}
	
	static void addToProblem(ceres::Problem& problem, OptParams& params, double weight_) {
		problem.AddResidualBlock(
			create(weight_),
			NULL,
			params.col_weights.data()
		);
	}

private:
	const double weight;
};


struct ShapeRegularizerCost
{
	ShapeRegularizerCost(double weight_)
		: weight{ weight_ }
	{}

	template<typename T>
	bool operator()(T const* shape_weights, T* residuals) const
	{

		for (int i = 0; i < BFM_N_ID_PCS; i++) {
			residuals[i] = shape_weights[i] * T(sqrt(weight));
		}
		return true;
	}

	static ceres::CostFunction* create(const double weight_) {
		return new ceres::AutoDiffCostFunction<ShapeRegularizerCost, BFM_N_ID_PCS, BFM_N_ID_PCS>(
			new ShapeRegularizerCost(weight_)
		);
	}
	
	static void addToProblem(ceres::Problem& problem, OptParams& params, double weight_) {
		problem.AddResidualBlock(
			create(weight_),
			NULL,
			params.shape_weights.data()
		);
	}

private:
	const double weight;
};

struct ExprRegularizerCost
{
    ExprRegularizerCost(double weight_)
		: weight{weight_}
	{}

	template<typename T>
	bool operator()(T const* exp_weights, T* residuals) const
	{
		for (int j = 0; j < BFM_N_EXPR_PCS; j++) {
			residuals[j] = exp_weights[j] * T(sqrt(weight));
		}
		return true;
	}

	static ceres::CostFunction* create(const double weight_) {
		return new ceres::AutoDiffCostFunction<ExprRegularizerCost, BFM_N_EXPR_PCS, BFM_N_EXPR_PCS>(
			new ExprRegularizerCost(weight_)
		);
	}
	
	static void addToProblem(ceres::Problem& problem, OptParams& params, double weight_) {
		problem.AddResidualBlock(
			create(weight_),
			NULL,
			params.exp_weights.data()
		);
	}

private:
	const double weight;
};

#endif