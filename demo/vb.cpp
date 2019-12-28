#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <ie_compound_blob.h>
#include <extension/ext_list.hpp>
InferenceEngine::Core ie;
std::string _input0Name, _output0Name;
InferenceEngine::InferRequest _inferRequest;
bool Init(const std::string& model, const cv::Size& size) 
{
    std::cout << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 1.	memory malloc

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
    ie.AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>(), IE_DEVICE_NAME);
    InferenceEngine::ExecutableNetwork executableNetwork = ie.LoadNetwork(network, IE_DEVICE_NAME);
    _inferRequest = executableNetwork.CreateInferRequest();
}
static int I420ToNV12(const uint8_t* src_u, int src_stride_u,
		      			const uint8_t* src_v, int src_stride_v,
						uint8_t* dst_uv, int dst_stride_uv,
						int width, int height);
static uint8_t _uvBuf[1024*1024];
cv::Mat Matting(const cv::Mat& frame)
{
    const size_t H = frame.rows, W = frame.cols;
    const size_t H_2 = (H + 1) >> 1, W_2 = (W + 1) >> 1;
    // 0. system statistic update
    cv::Mat i420;
    cv::cvtColor(frame, i420, cv::COLOR_BGR2YUV_I420);
    I420ToNV12(i420.data+(W*H), W_2, i420.data+(W*H*5/4), W_2, _uvBuf, W, W, H);
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
    auto begTime = (double)cv::getTickCount();
    int count=0;
    while (camera.read(frame) ) {
        cv::resize(frame, image, bgImage.size());
        ++count;
        mask = Matting(image);

        image.forEach<cv::Vec3b>([&](cv::Vec3b& element, const int position[]) {
            const auto& bg = bgImage.at<cv::Vec3b>(position);
            const auto& f = mask.at<float>(position);
            for (int i=0; i<3;++i)
                element[i] = (element[i] - bg[i]) * f + bg[i];
        });
        auto endTime = (double)cv::getTickCount();
        auto fps = count * cv::getTickFrequency()/(endTime - begTime);
        cv::imshow("vb",  image);
        printf("\r%f", fps);
        if (cv::waitKey(1) != -1)  break;
    }
}
typedef unsigned char uint8_t;

static void MergeUVRow_C(const uint8_t* src_u, const uint8_t* src_v, uint8_t* dst_uv, int width) 
{
  int x;
  for (x = 0; x < width - 1; x += 2) {
    dst_uv[0] = src_u[x];
    dst_uv[1] = src_v[x];
    dst_uv[2] = src_u[x + 1];
    dst_uv[3] = src_v[x + 1];
    dst_uv += 4;
  }
  if (width & 1) {
    dst_uv[0] = src_u[width - 1];
    dst_uv[1] = src_v[width - 1];
  }
}
static int I420ToNV12(	const uint8_t* src_u, int src_stride_u,
		      			const uint8_t* src_v, int src_stride_v,
						uint8_t* dst_uv, int dst_stride_uv,
						int width, int height) 
{
	// Coalesce rows.
	int halfwidth = (width + 1) >> 1;
	int halfheight = (height + 1) >> 1;
	if (!src_u || !src_v || !dst_uv ||
		width <= 0 || height <= 0) {
		return -1;
	}

	// Coalesce rows.
	if (src_stride_u == halfwidth &&
		src_stride_v == halfwidth &&
		dst_stride_uv == halfwidth * 2) {
		halfwidth *= halfheight;
		halfheight = 1;
		src_stride_u = src_stride_v = dst_stride_uv = 0;
	}

	for (int y = 0; y < halfheight; ++y) {
		// Merge a row of U and V into a row of UV.
		MergeUVRow_C(src_u, src_v, dst_uv, halfwidth);
		src_u += src_stride_u;
		src_v += src_stride_v;
		dst_uv += dst_stride_uv;
	}
	return 0;
}

    

    
