#define FACE_DEBUG_MODE 0

#include <sstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "parameters.h"
#include "optimizer.h"
#include "utils/render_util.h"
#include "bfm_manager.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: main [image] [path_to_output]" << std::endl;
        std::cout << "Example: main small ./result_small.jpg" << std::endl;
        return 1;
    }
    
    std::string imageName = argv[1];
    std::string imagePath = "../datasets/" + imageName + ".jpg";
    std::string landmarkPath = "../datasets/input_" + imageName + ".anl";
    std::string outputPath = argv[2];

    std::cout << "Loading the face model" << std::endl;
    // load the basel face model
    std::shared_ptr<BFM_Manager> bfm_m = std::make_shared<BFM_Manager>();

    std::cout << "Initializing parameters" << std::endl;
    // initialize the parameters
    OptParams params = OptParams::init();
    // initialize the optimizer
    Optimizer optimizer(bfm_m.get(), params);

    std::cout << "Loading the image and landmarks" << std::endl;
    std::cout << "Image path: " << imagePath << std::endl;
    cv::Mat image = cv::imread(imagePath);
    std::vector<Eigen::Vector2d> imageLandmarks = read_2d_landmarks(landmarkPath);

    // optimize the image
    std::cout << "Optimizing the image" << std::endl;
    optimizer.optimize(image, imageLandmarks, false, false);
    optimizer.optimize(image, imageLandmarks, true, true);

    // save the result
    // For sparse optimization:
    // Eigen::VectorXd vertexColors = Eigen::VectorXd::Ones(3*BFM_N_VERTICES);
    // renderParamsToMat(bfm_m.get(), params, image, vertexColors, true);
    renderParamsToMat(bfm_m.get(), params, image);
    cv::imwrite(outputPath, image);
    cv::imshow("Result", image);
    cv::waitKey(0);

    return 0;
}