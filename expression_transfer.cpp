#include <iostream>
#include <opencv2/opencv.hpp>
#include "parameters.h"
#include "optimizer.h"
#include "utils/render_util.h"
#include "bfm_manager.h"
#include "pixeltransfer.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: expression_transfer [expression_image] [transfer_image] [otuput_name]" << std::endl;
        std::cout << "Example: expression_transfer face2 small face2_to_small" << std::endl;
        return 1;
    }

    // get image paths
    std::string image1Name = argv[1];
    std::string image2Name = argv[2];
    std::string outputPath = argv[3];

    std::string image1Path = "../datasets/" + image1Name + ".jpg";
    std::string image2Path = "../datasets/" + image2Name + ".jpg";

    std::string landmarks1Path = "../datasets/input_" + image1Name + ".anl";
    std::string landmarks2Path = "../datasets/input_" + image2Name + ".anl";

    std::string outputPathOriginal1 = outputPath + "_expression.jpg";
    std::string outputPathOriginal2 = outputPath + "_original.jpg";
    std::string outputPathTransferred = outputPath + "_transferred.jpg";
    std::string outputPathPixelRef = outputPath + "_pixel_ref.jpg";
    std::string outputPathPixelTrans = outputPath + "_pixel_transferred.jpg";

    // load the basel face model
    std::shared_ptr<BFM_Manager> bfm_m = std::make_shared<BFM_Manager>();

    // read the images
    cv::Mat image1 = cv::imread(image1Path);
    cv::Mat image2 = cv::imread(image2Path);

    // read the landmarks
    std::vector<Eigen::Vector2d> image1Landmarks = read_2d_landmarks(landmarks1Path);
    std::vector<Eigen::Vector2d> image2Landmarks = read_2d_landmarks(landmarks2Path);

    // initialize the parameters
    OptParams params1 = OptParams::init();
    OptParams params2 = OptParams::init();

    // initialize the optimizers
    Optimizer optimizer1(bfm_m.get(), params1);
    Optimizer optimizer2(bfm_m.get(), params2);

    // optimize the images
    std::cout << "Optimizing the first image" << std::endl;
    optimizer1.optimize(image1, image1Landmarks, false, true);
    // optimizer1.optimize(image1, image1Landmarks, true);
    
    // save the result
    cv::Mat imageOriginal1 = image1.clone();
    renderParamsToMat(bfm_m.get(), params1, imageOriginal1);
    cv::imshow("Original Image 1", imageOriginal1);
    cv::imwrite(outputPathOriginal1, imageOriginal1);
    cv::waitKey(1);

    std::cout << "Optimizing the second image" << std::endl;
    optimizer2.optimize(image2, image2Landmarks, false);
    // optimizer2.optimize(image2, image2Landmarks, true, true);
    
    // save the result
    cv::Mat imageOriginal2 = image2.clone();
    renderParamsToMat(bfm_m.get(), params2, imageOriginal2);
    cv::imshow("Original Image 2", imageOriginal2);
    cv::imwrite(outputPathOriginal2, imageOriginal2);
    cv::waitKey(1);

    cv::Mat emptyMat = cv::Mat::zeros(image2.rows, image2.cols, CV_8UC3);

    // get the original texture for pixel transfer
    cv::Mat imagePixelRef = image2.clone();
    Eigen::Matrix<double, -1, 1> pixelColors = Eigen::Matrix<double, -1, 1>::Zero(BFM_N_VERTICES * 3);
    grabPixels(bfm_m.get(), params2, imagePixelRef, pixelColors);
    renderParamsToMat(bfm_m.get(), params2, emptyMat, pixelColors);
    cv::imshow("Pixel reference", emptyMat);
    cv::imwrite(outputPathPixelRef, emptyMat);
    cv::waitKey(1);

    // transfer the expression from image1 to image2
    params2.exp_weights = params1.exp_weights;

    cv::Mat imageTransferred = image2.clone();
    renderParamsToMat(bfm_m.get(), params2, imageTransferred);
    cv::imshow("Transferred Image", imageTransferred);
    cv::imwrite(outputPathTransferred, imageTransferred);
    cv::waitKey(1);


    cv::Mat emptyMat2 = cv::Mat::zeros(image2.rows, image2.cols, CV_8UC3);
    cv::Mat imagePixelTrans = image2.clone();
    renderParamsToMat(bfm_m.get(), params2, imagePixelTrans, pixelColors);
    cv::imshow("Pixel transferred", imagePixelTrans);
    cv::imwrite(outputPathPixelTrans, imagePixelTrans);
    cv::waitKey(0);

    return 0;
}