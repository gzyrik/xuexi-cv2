//Support Vector Machine(SVM 支持向量机)
//主要用于解决模式识别领域中的数据分类问题，属于有监督学习算法的一种:
//从有标签的数据学习，得到模型参数，对测试数据正确分类.
#include <opencv2/opencv.hpp>
using cv::ml::SVM;
void main()
{
    const int W=512, H=512;//图像宽高
    const int T=1, F=-1; //可能的标签值
    const cv::Vec3b green(0,255,0), blue(255,0,0);//对应的颜色
    //N个样本:标签和对应训练数据(行向量)
    //预测图像其他点的标签值,并上色
    const int N=4;
    int label[N]={T,F,F,F};
    float data[N][2]={
        {501, 10},
        {255, 10},
        {501, 255},
        {10, 501}
    };

    //训练
    auto svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR); //线性内核,没有高维空间映射,速度快
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6)); //迭代终止条件
    svm->train(
        cv::Mat(N, 2, CV_32FC1, data), //只支持CV_32FC1
        cv::ml::ROW_SAMPLE,
        cv::Mat(N, 1, CV_32SC1, label));

    //根据预测,绘制图像
    cv::Mat image = cv::Mat::zeros(W, H, CV_8UC3);//原点在左上角, Y轴向下
    image.forEach<cv::Vec3b>([&](cv::Vec3b& element, const int position[]) {
        //element是图像上点值, position是对应维数上的序号, 此处即图像坐标(列,行)
        cv::Mat xy = (cv::Mat_<float>(1,2) << position[1], position[0]);
        element = (svm->predict(xy) == T ? green : blue);
    });

    //绘制N个样本,黑圈/白圈
    const int thickness = -1;
    for(int i=0; i<N; ++i) {
        cv::circle (image, cv::Point2f(data[i][0], data[i][1]), 5,
            label[i] == T ? CV_RGB(0, 0, 0) : CV_RGB(255, 255, 255),
            thickness);
    }

    //绘制支持点(有效点),红圈
    auto& sv = svm->getUncompressedSupportVectors();
    for (int i = 0; i < sv.rows; ++i) {
        auto* v = sv.ptr<float>(i);
        cv::circle (image,  cv::Point2f(v[0], v[1]), 6, CV_RGB(128, 0, 0), 2);
    }

    cv::imshow("SVM Simple Example", image);
    cv::waitKey(0);
}
