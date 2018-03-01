//Support Vector Machine(SVM ֧��������)
//��Ҫ���ڽ��ģʽʶ�������е����ݷ������⣬�����мලѧϰ�㷨��һ��.
#include <opencv2/opencv.hpp>
using cv::ml::SVM;
void main()
{
    const int W=512, H=512;
    //����,��ǩ�Ͷ�Ӧѵ������(������)
    const int N=4;
    const int T=1, F=-1;
    int label[N]={T,F,F,F};
    float data[N][2]={
        {501, 10},
        {255, 10},
        {501, 255},
        {10, 501}
    };

    //ѵ��
    auto svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6)); //������ֹ����
    svm->train(
        cv::Mat(N, 2, CV_32FC1, data), //ֻ֧��CV_32FC1
        cv::ml::ROW_SAMPLE,
        cv::Mat(N, 1, CV_32SC1, label));

    //���ಢ����
    cv::Mat image = cv::Mat::zeros(W, H, CV_8UC3);
    image.forEach<cv::Vec3b>([&](cv::Vec3b& element, const int position[]) {
        cv::Mat sample_mat = (cv::Mat_<float>(1,2) << position[1],position[0]);//ֻ֧��CV_32FC1,����Ϊ(��,��)
        auto response = svm->predict(sample_mat);
        if (response == T)
            element = cv::Vec3b(0, 255, 0);
        else if (response == F)
            element = cv::Vec3b(255,0,0);
    });

    //��������
    for(int i=0; i<N; ++i) {
        cv::circle( image, cv::Point2f(data[i][0], data[i][1]), 5,
            label[i] > 0 ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255), -1);
    }

    //����֧�ֵ�
    cv::Mat sv = svm->getUncompressedSupportVectors();
    for (int i = 0; i < sv.rows; ++i) {
        const float* v = sv.ptr<float>(i);
        cv::circle( image,  cv::Point2f(v[0], v[1]), 6, cv::Scalar(0, 0, 128), 2);
    }

    cv::imshow("SVM Simple Example", image);
    cv::waitKey(0);
}
