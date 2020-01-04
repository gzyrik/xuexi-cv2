#include <opencv2/opencv.hpp>
#include <fstream>
struct FormatSpec {
    const char* name;
    const char* suffix;//suffix list
    float   ratio;//YUV frameSize = ratio * width * height
    int     cvType;//CV_8UC1, CV_8UC3
    int     cvtCode;//cv::ColorConversionCodes
};

static const FormatSpec _format[] = {
    {"Gray",    "/Y/GRAY/8UC1/",    1,  CV_8UC1, cv::COLOR_GRAY2BGR},
    {"BGR",     "/BGR/8UC3/",       3,  CV_8UC3, -1},

    {"YUV420",  "/NV21/420SP/",     1.5,CV_8UC1, cv::COLOR_YUV2BGR_NV21},
    {"YUV420",  "/NV12/",           1.5,CV_8UC1, cv::COLOR_YUV2BGR_NV12},
    {"YUV420",  "/I420/IYUV/",      1.5,CV_8UC1, cv::COLOR_YUV2BGR_I420},
    {"YUV420",  "/YV12/",           1.5,CV_8UC1, cv::COLOR_YUV2BGR_YV12},

    {"YUV422",  "/UYVY/Y422/UYNV/", 2,  CV_8UC1, cv::COLOR_YUV2BGR_UYVY}, 
    {"YUV422",  "/YUY2/YUYV/YUNV/", 2,  CV_8UC1, cv::COLOR_YUV2BGR_YUY2}, 
    {"YUV422",  "/YVYU/",           2,  CV_8UC1, cv::COLOR_YUV2BGR_YVYU}, 
    {nullptr,   nullptr, 0, 0, 0}
};

struct Movie {
    std::ifstream stream;
    int frameCount;
    int frameSize;
    int height, width;
    char format[128];
    const FormatSpec* spec;
    Movie(const std::string& file, std::string& err){
        size_t pos = file.size();
        do {
            if (pos == 0 || pos == file.npos) {
                err = "** File name `" + file + "' not match %dx%d?%s";
                return;
            }
            pos = file.find_last_of("_/\\", pos);
            const char* p = file.c_str();
            if (pos != file.npos) p += (pos--) + 1;

            if (3 == sscanf(p, "%d%*[xX]%d%*[^a-zA-Z]%[a-zA-Z0-9]", &width, &height, format))
                break;
        } while(true);
        stream.open(file.c_str(), std::ios::in | std::ios::binary | std::ios::ate);  
        if (!stream.is_open()) {
            err = "** File not read: " + file;
            return;
        }
        std::string str = format;
        std::transform(str.begin(), str.end(), str.begin(), toupper);
        str = "/" + str + "/";//more strict match FormatSpec.suffix
        for (int i=0; _format[i].name; ++i) {
            if (strstr(_format[i].suffix, str.c_str())){
                spec = &_format[i];
                frameSize = int(width * height * _format[i].ratio);
                frameCount = int(stream.tellg()/frameSize);
                std::cout << file <<":" << width << "x" << height << " " << spec->name << std::endl;
                return;
            }
        }
        err = "** Unsupported yuv format: " + std::string(format);
    }
    void draw(cv::Mat& canvas, int streamPos, int& x) {
        cv::Mat mat(frameSize/width, width, spec->cvType);
        stream.seekg(streamPos * frameSize);
        stream.read(mat.ptr<char>(), frameSize);
        if (spec->cvtCode >= 0) {
            cv::Mat rgb;
            cv::cvtColor(mat, rgb, spec->cvtCode);
            mat = rgb;
        }
        mat.copyTo(canvas.rowRange(x, x + height));
        x += height;
    }
};
static std::vector<Movie> _movies;
static cv::Mat _canvas;
static std::string _title;
static void on_trackbar(int pos, void* )
{
    int x=0;
    for(auto& moive : _movies)
        moive.draw(_canvas, pos, x);
    if (_canvas.cols < 800) {
        cv::Mat image;
        cv::resize(_canvas, image, _canvas.size()*800/_canvas.cols); 
        cv::imshow(_title, image);
    }
    else {
        cv::imshow(_title, _canvas);
    }
}
void main(int argc, char* argv[])
{
    int height=0, width=0, frameCount=0;
    std::string err;
    if (argc < 2){
        std::cerr << "** Problem loading image!!!" << std::endl;
        goto HELP;
    }
    for (int i=1; i<argc && err.empty(); ++i)
        _movies.emplace_back(Movie(argv[i], err));
    if (!err.empty()){
        std::cerr << std::endl << err;
        goto HELP;
    }
    for(const auto& moive : _movies){
        height += moive.height;
        if (moive.width > width)
            width = moive.width;
        if (moive.frameCount > frameCount){
            frameCount = moive.frameCount;
            std::ostringstream oss;
            oss << moive.height << "x" << moive.width << " frames:"<<frameCount;
            _title = oss.str();
        }
    }
    _canvas.create(height, width, CV_8UC3);
    cv::namedWindow(_title);
    cv::createTrackbar("frame", _title, nullptr, frameCount-1, on_trackbar);
    on_trackbar(0, 0);
    cv::waitKey(0);
HELP:
    std::cerr << "Simple YUV file Viewer, support YUV Format:" << std::endl;
    for (int i=0; _format[i].name; ++i) {
        std::cerr <<_format[i].name << ":";
        std::string str = _format[i].suffix;
        std::replace(str.begin(), str.end(), '/', ' ');
        std::cerr << "\t" << str << std::endl;
    }
}
