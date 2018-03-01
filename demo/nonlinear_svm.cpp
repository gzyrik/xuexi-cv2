//Support Vector Machine(SVM 支持向量机)
//主要用于解决模式识别领域中的数据分类问题，属于有监督学习算法的一种.
#include <opencv2/opencv.hpp>
using cv::ml::SVM;
void main()
{

    const int W=512, H=512;
    //样本,标签和对应训练数据(行向量)
    //训练数据:
    //  纵坐标随机范围y=[1,H), 横坐标随机范围分三块
    //  - [0,L),      x=[1, 0.4W)
    //  - [2N-L, 2N)  x=[0.6W, W)
    //  - [L,2N-L)    x=[0.4W,0.6W)
    //标签:
    //  [0,N) 为 T, [N, 2N) 为 F
    //显然中间块,不能被线性分隔的,理论最优决策面是中央线x=N

    const int N=100;
    const int L=90;
    const int T=1, F=-1;
    cv::RNG rng(100); //随机数产生器
    cv::Mat data(2*N, 2, CV_32FC1);
    cv::Mat label(2*N, 1, CV_32SC1);
    rng.fill(data.colRange(1, 2), rng.UNIFORM, cv::Scalar(1), cv::Scalar(H));//所有纵坐标
    rng.fill(data.rowRange(0, L).colRange(0, 1), rng.UNIFORM, cv::Scalar(1), cv::Scalar(0.4*W));//[0,L)的横坐标
    rng.fill(data.rowRange(2*N-L, 2*N).colRange(0, 1), rng.UNIFORM, cv::Scalar(0.6*W), cv::Scalar(W));//[2N-L, 2N)的横坐标
    rng.fill(data.rowRange(L, 2*N-L).colRange(0, 1), rng.UNIFORM, cv::Scalar(0.4*W), cv::Scalar(0.6*W));//[L,2N-L)的横坐标
    label.rowRange(0,N).setTo(T);
    label.rowRange(N, 2*N).setTo(F);

    //训练
    auto svm = SVM::create();
    svm->setC(0.1);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, (int)1e7, 1e-6)); //迭代终止条件
    svm->train(data, cv::ml::ROW_SAMPLE, label);


    //分类并绘制
    cv::Mat image = cv::Mat::zeros(W, H, CV_8UC3);
    image.forEach<cv::Vec3b>([&](cv::Vec3b& element, const int position[]) {
        cv::Mat sample = (cv::Mat_<float>(1,2) << position[1],position[0]);//只支持CV_32FC1,坐标为(列,行)
        auto response = svm->predict(sample);
        if (response == T)
            element = cv::Vec3b(0, 255, 0);
        else if (response == F)
            element = cv::Vec3b(255,0,0);
    });

    //绘制样本
    label.forEach<int>([&](int& element, const int position[]) {
        const float* v = data.ptr<float>(position[0]);
        cv::circle( image, cv::Point2f(v[0], v[1]), 5,
            element > 0 ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255), -1);
    });

    //绘制支持点
    cv::Mat sv = svm->getUncompressedSupportVectors();
    for (int i = 0; i < sv.rows; ++i) {
        const float* v = sv.ptr<float>(i);
        cv::circle( image,  cv::Point2f(v[0], v[1]), 6, cv::Scalar(0, 0, 128), 2);
    }

    cv::imshow("SVM Nonlinear Example", image);
    cv::waitKey(0);
}
