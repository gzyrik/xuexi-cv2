#include <opencv2/opencv.hpp>
#include <fstream>
void main(int argc, char* argv[])
{
    std::string cfgFile("res/pano.ini");
    if (argc > 1) cfgFile = argv[1];
    std::ifstream fp(cfgFile);
    if (!fp.is_open()) {
        std::cerr << "File with  image list not found: " << cfgFile << std::endl;
        exit(-1);
    }
    std::string dir(cfgFile);
    auto pos = dir.find_last_of("/\\");
    if (pos != dir.npos)
        dir = dir.substr(0, pos+1);
    else
        dir.clear();
    std::vector<cv::Mat> images;
    while (!fp.eof()) {
        std::string name;
        std::getline(fp, name);
        if (!name.empty()) {
            auto img = cv::imread(dir + name);
            if (img.empty()){
                std::cerr << "Problem loading: " << name << std::endl;
                exit(-1);
            }
            images.emplace_back(img);
        }
    }
    fp.close();

    cv::Mat pano;
    auto stitcher = cv::Stitcher::create();
    auto status = stitcher->stitch(images, pano);
    if (status != decltype(status)::OK){
        std::cerr << "stitch failed" << std::endl;
        exit(-1);
    }
    cv::imshow("Stitching Example", pano);
    cv::waitKey(0);
}

