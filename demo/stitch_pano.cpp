#include <opencv2/opencv.hpp>
void main(int argc, char* argv[])
{
    if (argc < 2){
        std::cerr << "Problem loading image!!!" << std::endl;
        exit(-1);
    }

    std::ifstream fp(argv[1]);
    if (!fp.is_open()) {
        std::cerr << "File with  image list not found: " << argv[1] << std::endl;
        exit(-1);
    }
    std::string dir(argv[1]);
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
    auto stitcher = cv::Stitcher::createDefault(true);
    auto status = stitcher.stitch(images, pano);
    if (status != decltype(status)::OK){
        std::cerr << "stitch failed" << std::endl;
        exit(-1);
    }
    cv::imshow("Stitching Example", pano);
    cv::waitKey(0);
}

