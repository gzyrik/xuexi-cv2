#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
//对应名称列表
static std::vector<std::string> readClassNames(const std::string& labelFilename)
{
    std::ifstream fp(labelFilename);
    if (!fp.is_open()) {
        std::cerr << "File with classes labels not found: " << labelFilename << std::endl;
        exit(-1);
    }
    std::vector<std::string> classNames;
    while (!fp.eof()) {
        std::string name;
        std::getline(fp, name);
        if (name.length())
            classNames.push_back( name.substr(name.find(' ')+1) );
    }
    fp.close();
    return classNames;
}
void main(int argc, char **argv)
{
    cv::Mat image;
    if (argc < 2 || (image = cv::imread(argv[1])).empty()){
        std::cerr << argv[0] << " <image> [dir]" << std::endl;
        std::cerr << "* MUST EXIST:" << std::endl;
        std::cerr << "\t [dir]bvlc_googlenet.prototxt"   << std::endl;
        std::cerr << "\t [dir]bvlc_googlenet.caffemodel" << std::endl;
        std::cerr << "\t [dir]synset_words.txt" << std::endl;
        std::cerr << "Example:" <<std::endl;
        std::cerr << argv[0] << " space_shuttle.jpg " << " res/dnn"<<std::endl;
        exit(-1);
    }
    std::string dir = "./";
    if (argc > 2) {
        dir = argv[2];
        if (dir.back() != '/') dir.push_back('/');
    }

    //打印信息
    const std::string prototxt(dir + "bvlc_googlenet.prototxt");
    const std::string caffemodel(dir + "bvlc_googlenet.caffemodel");
    const std::string synsetxt(dir + "synset_words.txt");
    std::cout << "Load network by using the following files: " << std::endl;
    std::cout << "prototxt:   " << prototxt << std::endl;
    std::cout << "caffemodel: " << caffemodel << std::endl;
    std::cout << "synsetxt: " << synsetxt << std::endl;
    std::cout << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
    std::cout << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
    std::cout << std::endl;

    //读取1000种类别
    auto classNames = readClassNames(synsetxt);

    //创建网络
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(prototxt, caffemodel);
    if (net.empty()) {
        std::cerr << "Problem loading model!!!" << std::endl;
        exit(-1);
    }

    //GoogLeNet只接收 224x224 RGB-images
    //返回4-dimensional blob (so-called batch) with 1x3x224x224 shape 
    image = cv::dnn::blobFromImage(image, 1, cv::Size(224, 224), cv::Scalar(104, 117, 123));

    //迭代分析
    cv::Mat prob;
    net.setInput(image, "data");
    prob = net.forward("prob");

    //选择最可能的类别和正确率,即最大值的序号
    int classId;
    double classProb;
    cv::Point maxLoc;
    cv::minMaxLoc(prob.reshape(1, 1), nullptr, &classProb, nullptr, &maxLoc);
    classId = maxLoc.x;

    //打印结果
    std::cout << "Best class: #" << classId << " '" << classNames[classId] << "'" << std::endl;
    std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
}
