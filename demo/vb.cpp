#include <opencv2/opencv.hpp>
// 关闭编译时产生的警告
#ifdef _MSC_VER
#pragma  warning( push ) 
#pragma  warning( disable: 4251 4275 4819 )
#define getcwd _getcwd
#endif
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winherited-variadic-ctor"
#endif
#include <inference_engine.hpp>
#include <ie_compound_blob.h>
#include <extension/ext_list.hpp>
#ifdef _MSC_VER
#pragma  warning(  pop  ) 
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
static std::string _input0Name, _output0Name;
static InferenceEngine::InferRequest _inferRequest;
static uint8_t* _uvBuf = nullptr;//uv of NV12
static bool Init(const std::string& model, const cv::Size& size) 
{
    std::cout << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 1.	memory malloc
    delete _uvBuf;
    _uvBuf = new uint8_t[size.height * size.width / 2];
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 2.	Read network:	files -> network
    InferenceEngine::CNNNetReader networkReader;
    networkReader.ReadNetwork(model + ".xml");
    networkReader.ReadWeights(model + ".bin");
    if (!networkReader.isParseSuccess()) return false;
    InferenceEngine::CNNNetwork network = networkReader.getNetwork();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 3.	reshape
    InferenceEngine::ICNNNetwork::InputShapes inputShapes = network.getInputShapes();//std::map<std::string, SizeVector>
    InferenceEngine::SizeVector& tensorSize = inputShapes.begin()->second; //std::vector<size_t>
    tensorSize[0] = 1;
    tensorSize[2] = size.height;
    tensorSize[3] = size.width;
    network.reshape(inputShapes);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 4.	Prepare input blobs
    InferenceEngine::InputsDataMap inputsInfo = network.getInputsInfo();//std::map<std::string, InputInfo::Ptr>
    assert (inputsInfo.size() == 1 && "Demo supports topologies only with 1 input");
    _input0Name = inputsInfo.begin()->first;
    InferenceEngine::InputInfo& input0Info = *(inputsInfo.begin()->second);
    input0Info.setLayout(InferenceEngine::Layout::NCHW);
    input0Info.setPrecision(InferenceEngine::Precision::U8);
    InferenceEngine::PreProcessInfo& preProcess = input0Info.getPreProcess();
    preProcess.setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
    preProcess.setColorFormat(InferenceEngine::ColorFormat::NV12);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 5.	Prepare output blobs 
    InferenceEngine::OutputsDataMap outputsInfo=network.getOutputsInfo();//std::map<std::string, DataPtr>
    for (auto& item : outputsInfo) {
        if (_output0Name.empty()) _output0Name = item.first;
        InferenceEngine::DataPtr& data= item.second;
        assert (data && "output data pointer is not valid");
        data->setPrecision(InferenceEngine::Precision::FP32);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 6. Loading model to the device 
    const std::string IE_DEVICE_NAME = "CPU";
    InferenceEngine::Core ie;
    ie.AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>(), IE_DEVICE_NAME);
    InferenceEngine::ExecutableNetwork executableNetwork = ie.LoadNetwork(network, IE_DEVICE_NAME);
    _inferRequest = executableNetwork.CreateInferRequest();
    return true;
}
static cv::Mat Matting(const cv::Mat& frame)
{
    const size_t H = frame.rows, W = frame.cols;
    const size_t H_2 = (H + 1) >> 1, W_2 = (W + 1) >> 1;
    cv::Mat i420;
    cv::cvtColor(frame, i420, cv::COLOR_BGR2YUV_I420);
    uint8_t* u = i420.data+(W*H), *v = u + W_2*H_2, *d=_uvBuf;
    for (int i = 0; i < W_2*H_2; ++i) {
        *d++ = u[i], *d++ = v[i];
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 2. wrap & set: yuv -> imgBlob
    InferenceEngine::TensorDesc yDesc(InferenceEngine::Precision::U8,
        { 1, 1, H, W}, InferenceEngine::Layout::NHWC);
    InferenceEngine::TensorDesc uvDesc(InferenceEngine::Precision::U8,
        { 1, 2, H_2, W_2}, InferenceEngine::Layout::NHWC);
    InferenceEngine::Blob::Ptr yBlob = InferenceEngine::make_shared_blob<uint8_t>(yDesc, i420.data);
    InferenceEngine::Blob::Ptr uvBlob = InferenceEngine::make_shared_blob<uint8_t>(uvDesc, _uvBuf);
    InferenceEngine::Blob::Ptr imgBlob= InferenceEngine::make_shared_blob<InferenceEngine::NV12Blob>(yBlob, uvBlob);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 4. Do inference 
    _inferRequest.SetBlob(_input0Name, imgBlob);
    _inferRequest.Infer();
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 6. Process output , default channel order = NCHW
    const InferenceEngine::Blob::Ptr& outputBlob = _inferRequest.GetBlob(_output0Name);
    auto outputDims = outputBlob->getTensorDesc().getDims();
    const auto outputBuffer = outputBlob->buffer().as<float*>();
    return cv::Mat(frame.size(), CV_32FC1, outputBuffer + outputDims[2] * outputDims[3]);
}
void main()
{
    char pwd[1024];
    printf("pwd=%s", getcwd(pwd, sizeof(pwd)));

    std::string model("res/vb_mod1.0_2019r2"), bg_path="res/backgroud.jpg";

    cv::Mat bgImage = cv::imread(bg_path);
    cv::resize(bgImage, bgImage, cv::Size(bgImage.cols&~7, bgImage.rows&~7)); //对齐

    Init(model, bgImage.size());

    cv::VideoCapture camera(0);
    cv::Mat frame, image, mask;

    size_t count = 0;
    double lastFps[0x100] = {0};
    auto lastTick = (double)cv::getTickCount();
    while (cv::waitKey(1) == -1 && camera.read(frame) ) {
        cv::resize(frame, image, bgImage.size());
        mask = Matting(image);

        image.forEach<cv::Vec3b>([&](cv::Vec3b& element, const int position[]) {
            const auto& bg = bgImage.at<cv::Vec3b>(position);
            const auto& f = mask.at<float>(position);
            for (int i=0; i<3;++i)
                element[i] = static_cast<uint8_t>((element[i] - bg[i]) * f + bg[i]);
        });

        auto nowTick = (double)cv::getTickCount();
        lastFps[count++ & 0xFF] = cv::getTickFrequency()/(nowTick - lastTick);
        lastTick = nowTick;

        //stats
        double sum = 0, minFps=120, maxFps = -1;
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
