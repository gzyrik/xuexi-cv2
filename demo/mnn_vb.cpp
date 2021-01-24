#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>
static MNN::Interpreter* _interpreter;
static MNN::Session  *_session;
static cv::Size _input_size;
static MNN::Tensor   *_input_tensor;
static MNN::Tensor   *_output_tensor;
static cv::Size GetInputblobSize(int src_w, int src_h)
{
    //W/H      
    //2.0
    //1.77		2.0		384,192
    //1.33		1.66	320,192
    //1.0		1.33	320,240
    //0.75		1		256,256
    //0.5625    0.75	240,320
    //0.5		0.6		192,320	
    if ((src_w * 9) > (16 * src_h))
        return cv::Size(384, 192);//320*192,384*240
    else if ((src_w * 3) > (4 * src_h))
        return cv::Size(320, 192);//320*192,384*240
    else if ((src_w) >= (src_h))
        return cv::Size(320, 240);
    else if ((src_w * 4) >= (3 * src_h))
        return cv::Size(256, 256);
    else if ((src_w * 16) >= (9 * src_h))
        return cv::Size(240, 320);
    else
        return cv::Size(192, 320);
}
static bool Init(const std::string& model, const cv::Size& size) 
{
    _input_size = GetInputblobSize(size.width, size.height);
    //打印信息
    std::cout << " model: " << model << std::endl;
    std::cout << "origin: " << size << std::endl;
    std::cout << " shape: " << _input_size << std::endl;
    std::cout << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 1. load model 
    _interpreter = MNN::Interpreter::createFromFile(model.c_str());
    if (!_interpreter) return false;
    // 2. create session
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_AUTO;
    _session = _interpreter->createSession(config);
    if (!_session) return false;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 3. reset input blob shape:    batch_size=1
    _input_tensor = _interpreter->getSessionInput(_session, NULL);
    if (!_input_tensor) return false;
    std::vector<int> shape = _input_tensor->shape();
    shape[0] = 1;
    shape[1] = _input_size.height;
    shape[2] = _input_size.width;
    shape[3] = 3;
    _interpreter->resizeTensor(_input_tensor, shape);
    _interpreter->resizeSession(_session);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 4. get output blob
    _output_tensor = _interpreter->getSessionOutput(_session, NULL);
    if (!_output_tensor) return false;
    return true;
}
static cv::Mat Matting(const cv::Mat& i420)
{
    std::cout << "i420:" << i420.type() << i420.size() << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 5. wrap & set: yuv -> imgBlob
    cv::Mat resized, rgb;
    // i420 -> resized
    cv::resize(i420, resized, cv::Size(_input_size.width, _input_size.height*3/2));
    std::cout << "resized:" << resized.type() << resized.size() << std::endl;
    // i420 -> rgb
    cv::cvtColor(resized, rgb, cv::COLOR_YUV2RGB_I420);
    std::cout << "rgb:" << rgb.type() << rgb.size() << std::endl;
    // uchar -> float32 
    rgb.convertTo(rgb, CV_32FC3);
    std::cout << "rgb:" << rgb.type() << rgb.size() << std::endl;
    // copy to input blob
    memcpy(_input_tensor->host<void>(), rgb.data, sizeof(float) * _input_size.width * _input_size.height * 3);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 6. Do inference        		   
    _interpreter->runSession(_session);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 5. result postprocess
    return cv::Mat(_input_size, CV_32FC2, _output_tensor->host<float>());
}
void main()
{
    const std::string model("res/mnn/vb_20210107.mnn"), bg_path="res/background.jpg";

    cv::Mat bgImage = cv::imread(bg_path);
    cv::resize(bgImage, bgImage, cv::Size(bgImage.cols&~7, bgImage.rows&~7)); //对齐

    if (!Init(model, bgImage.size())) return;

    cv::VideoCapture camera(0);
    cv::Mat frame, image, mask;

    size_t count = 0;
    double lastFps[0x100] = {0};
    while (cv::waitKey(1) == -1 && camera.read(frame) ) {

        cv::resize(frame, image, bgImage.size());
        cv::Mat i420; 
        cv::cvtColor(image, i420, cv::COLOR_BGR2YUV_I420);

        auto lastTick = (double)cv::getTickCount();
        mask = Matting(i420);
        auto nowTick = (double)cv::getTickCount();
        std::cout << "mask:" << mask.type() << mask.size() << std::endl;

        image.forEach<cv::Vec3b>([&](cv::Vec3b& element, const int position[]) {
            const auto& bg = bgImage.at<cv::Vec3b>(position);
            const auto& f = mask.at<float>(position);
            for (int i=0; i<3;++i)
                element[i] = static_cast<uint8_t>((element[i] - bg[i]) * f + bg[i]);
        });

        lastFps[count++ & 0xFF] = cv::getTickFrequency()/(nowTick - lastTick);

        //stats
        double sum = 0, minFps=1000, maxFps = -1;
        const auto num = std::min<size_t>(0x100, count);
        for (size_t i = 0; i < num; ++i) {
            const auto& f = lastFps[i];
            sum += f;
            if (f > maxFps) maxFps = f;
            if (f < minFps) minFps = f;
        }
        printf("\rAvg=%f Min=%f Max=%f", sum/num, minFps, maxFps);

        cv::imshow("vb",  image);
    }
}
