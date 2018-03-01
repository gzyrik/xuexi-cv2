//Support Vector Machine(SVM ֧��������)
//��Ҫ���ڽ��ģʽʶ�������е����ݷ������⣬�����мලѧϰ�㷨��һ��.
#include <opencv2/opencv.hpp>
using cv::ml::SVM;
void main()
{

    const int W=512, H=512;
    //����,��ǩ�Ͷ�Ӧѵ������(������)
    //ѵ������:
    //  �����������Χy=[1,H), �����������Χ������
    //  - [0,L),      x=[1, 0.4W)
    //  - [2N-L, 2N)  x=[0.6W, W)
    //  - [L,2N-L)    x=[0.4W,0.6W)
    //��ǩ:
    //  [0,N) Ϊ T, [N, 2N) Ϊ F
    //��Ȼ�м��,���ܱ����Էָ���,�������ž�������������x=N

    const int N=100;
    const int L=90;
    const int T=1, F=-1;
    cv::RNG rng(100); //�����������
    cv::Mat data(2*N, 2, CV_32FC1);
    cv::Mat label(2*N, 1, CV_32SC1);
    rng.fill(data.colRange(1, 2), rng.UNIFORM, cv::Scalar(1), cv::Scalar(H));//����������
    rng.fill(data.rowRange(0, L).colRange(0, 1), rng.UNIFORM, cv::Scalar(1), cv::Scalar(0.4*W));//[0,L)�ĺ�����
    rng.fill(data.rowRange(2*N-L, 2*N).colRange(0, 1), rng.UNIFORM, cv::Scalar(0.6*W), cv::Scalar(W));//[2N-L, 2N)�ĺ�����
    rng.fill(data.rowRange(L, 2*N-L).colRange(0, 1), rng.UNIFORM, cv::Scalar(0.4*W), cv::Scalar(0.6*W));//[L,2N-L)�ĺ�����
    label.rowRange(0,N).setTo(T);
    label.rowRange(N, 2*N).setTo(F);

    //ѵ��
    auto svm = SVM::create();
    svm->setC(0.1);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, (int)1e7, 1e-6)); //������ֹ����
    svm->train(data, cv::ml::ROW_SAMPLE, label);


    //���ಢ����
    cv::Mat image = cv::Mat::zeros(W, H, CV_8UC3);
    image.forEach<cv::Vec3b>([&](cv::Vec3b& element, const int position[]) {
        cv::Mat sample = (cv::Mat_<float>(1,2) << position[1],position[0]);//ֻ֧��CV_32FC1,����Ϊ(��,��)
        auto response = svm->predict(sample);
        if (response == T)
            element = cv::Vec3b(0, 255, 0);
        else if (response == F)
            element = cv::Vec3b(255,0,0);
    });

    //��������
    label.forEach<int>([&](int& element, const int position[]) {
        const float* v = data.ptr<float>(position[0]);
        cv::circle( image, cv::Point2f(v[0], v[1]), 5,
            element > 0 ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255), -1);
    });

    //����֧�ֵ�
    cv::Mat sv = svm->getUncompressedSupportVectors();
    for (int i = 0; i < sv.rows; ++i) {
        const float* v = sv.ptr<float>(i);
        cv::circle( image,  cv::Point2f(v[0], v[1]), 6, cv::Scalar(0, 0, 128), 2);
    }

    cv::imshow("SVM Nonlinear Example", image);
    cv::waitKey(0);
}
