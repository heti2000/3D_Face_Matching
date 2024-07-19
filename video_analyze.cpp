#include <sstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "parameters.h"
#include "optimizer.h"
#include "utils/render_util.h"
#include "bfm_manager.h"


int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: video_analyze [path_to_images] [num_images] [output_folder]" << std::endl;
        std::cout << "Example: video_analyze ../datasets/clip 95 ./result_clip" << std::endl;
        return 1;
    }
    
    std::string dataPath = argv[1];
    int numImages = std::stoi(argv[2]);
    std::string outputPath = argv[3];

    // load the basel face model
    std::shared_ptr<BFM_Manager> bfm_m = std::make_shared<BFM_Manager>();

    // initialize the parameters
    OptParams params = OptParams::init();
    // initialize the optimizer
    Optimizer optimizer(bfm_m.get(), params);

    // load the first image
    cv::Mat image = cv::imread(dataPath + "/image_0001.jpg");
    std::vector<Eigen::Vector2d> imageLandmarks = read_2d_landmarks(dataPath + "/input_image_0001.anl");
    optimizer.optimize(image, imageLandmarks, false);
    optimizer.optimize(image, imageLandmarks, true, true);

    for (int i = 0; i < numImages; i++) {
        std::stringstream ss;
        ss << "image_" << std::setw(4) << std::setfill('0') << i+1;
        std::string imagePath = dataPath + "/" + ss.str() + ".jpg";
        std::string landmarkPath = dataPath + "/input_" + ss.str() + ".anl";

        image = cv::imread(imagePath);
        imageLandmarks = read_2d_landmarks(landmarkPath);

        // optimize the image
        optimizer.optimize(image, imageLandmarks, false);

        // save the result
        std::string outputPathOriginal = outputPath + "/" + ss.str() + ".jpg";
        renderParamsToMat(bfm_m.get(), params, image);
        cv::imwrite(outputPathOriginal, image);
        cv::imshow("Result", image);
        cv::waitKey(1);
    }

    return 0;
}