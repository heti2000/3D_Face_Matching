#include "optimizer.h"

Optimizer::Optimizer(BFM_Manager* bfm, OptParams &params_):
			bfm_m(bfm), params(params_){
	std::cout << "----- Optimizer Init Start -----" << std::endl;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.linear_solver_type = ceres::DENSE_QR;
	// options.dense_linear_algebra_library_type = ceres::CUDA;
	options.num_threads = 8;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 100;
	options.update_state_every_iteration = true;
}

void Optimizer::optimize(const cv::Mat img, const vector<Eigen::Vector2d> imageLandmarks, bool optimizeDense) {
	optimize(img, imageLandmarks, optimizeDense, false);
}

void Optimizer::optimize(const cv::Mat img, const vector<Eigen::Vector2d> imageLandmarks, bool optimizeDense, bool visualization) {
	int img_w = img.cols;
	int img_h = img.rows;

	// initialize problem
	ceres::Problem problem;

	if (optimizeDense) {
		DenseRGBCost::addToProblem(problem, params, bfm_m, img, DENSE_WEIGHT, DENSE_WEIGHT_LIGHT);
	} 
	
	SparseCost::addToProblem(problem, params, bfm_m, imageLandmarks, img.cols, img.rows, SPARSE_WEIGHT);
	ceres::Manifold* quaternion_manifold = new ceres::QuaternionManifold;
	problem.SetManifold(params.rotation.data(), quaternion_manifold);
	problem.SetParameterLowerBound(&params.focal_length, 0, 35.);

	ColorRegularizerCost::addToProblem(problem, params, COLOR_REG_WEIGHT);
	ShapeRegularizerCost::addToProblem(problem, params, SHAPE_REG_WEIGHT);
	ExprRegularizerCost::addToProblem(problem, params, EXPR_REG_WEIGHT);

	if (visualization) {
		// add the visualizer callback
		VisualizationCallback* callback = new VisualizationCallback(&params, bfm_m, img_w, img_h);
		options.callbacks.push_back(callback);
	}

	ceres::Solver::Summary summary;

	std::cout << "Starting solve" << std::endl;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

}