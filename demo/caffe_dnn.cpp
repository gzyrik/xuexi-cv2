#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
//��Ӧ�����б�
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
    std::string dir = "res/caffe";
    std::string file= "res/space_shuttle.jpg";
    if (argc > 1) {
        file = argv[1];
        if (argc > 2) dir = argv[2];
    }
    if (!dir.empty() && dir.back() != '/')
        dir.push_back('/');

    //��ӡ��Ϣ
    std::cout << "Read Image: " << file << std::endl;
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

    cv::Mat image = cv::imread(file);
    if (image.empty()) {
        std::cerr << "Problem loading imagel!!!" << std::endl;
        std::cerr << "Example:\n" << argv[0] << " space_shuttle.jpg " << " res/caffe"<<std::endl;
        exit(-1);
    }


    //��ȡ1000�����
    auto classNames = readClassNames(synsetxt);

    //��������
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(prototxt, caffemodel);
    if (net.empty()) {
        std::cerr << "Problem loading model!!!" << std::endl;
        std::cerr << "Example:\n" << argv[0] << " space_shuttle.jpg " << " res/caffe"<<std::endl;
        exit(-1);
    }

    //GoogLeNetֻ���� 224x224 RGB-images
    //����4-dimensional blob (so-called batch) with 1x3x224x224 shape 
    image = cv::dnn::blobFromImage(image, 1, cv::Size(224, 224), cv::Scalar(104, 117, 123));

    //��������
    cv::Mat prob;
    net.setInput(image, "data");
    prob = net.forward("prob");

    //ѡ������ܵ�������ȷ��,�����ֵ�����
    int classId;
    double classProb;
    cv::Point maxLoc;
    cv::minMaxLoc(prob.reshape(1, 1), nullptr, &classProb, nullptr, &maxLoc);
    classId = maxLoc.x;

    //��ӡ���
    std::cout << "Best class: #" << classId << " '" << classNames[classId] << "'" << std::endl;
    std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
}
