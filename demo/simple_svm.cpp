//Support Vector Machine(SVM ֧��������)
//��Ҫ���ڽ��ģʽʶ�������е����ݷ������⣬�����мලѧϰ�㷨��һ��:
//���б�ǩ������ѧϰ���õ�ģ�Ͳ������Բ���������ȷ����.
#include <opencv2/opencv.hpp>
using cv::ml::SVM;
void main()
{
    const int W=512, H=512;//ͼ����
    const int T=1, F=-1; //���ܵı�ǩֵ
    const cv::Vec3b green(0,255,0), blue(255,0,0);//��Ӧ����ɫ
    //N������:��ǩ�Ͷ�Ӧѵ������(������)
    //Ԥ��ͼ��������ı�ǩֵ,����ɫ
    const int N=4;
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
    svm->setKernel(SVM::LINEAR); //�����ں�,û�и�ά�ռ�ӳ��,�ٶȿ�
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6)); //������ֹ����
    svm->train(
        cv::Mat(N, 2, CV_32FC1, data), //ֻ֧��CV_32FC1
        cv::ml::ROW_SAMPLE,
        cv::Mat(N, 1, CV_32SC1, label));

    //����Ԥ��,����ͼ��
    cv::Mat image = cv::Mat::zeros(W, H, CV_8UC3);//ԭ�������Ͻ�, Y������
    image.forEach<cv::Vec3b>([&](cv::Vec3b& element, const int position[]) {
        //element��ͼ���ϵ�ֵ, position�Ƕ�Ӧά���ϵ����, �˴���ͼ������(��,��)
        cv::Mat xy = (cv::Mat_<float>(1,2) << position[1], position[0]);
        element = (svm->predict(xy) == T ? green : blue);
    });

    //����N������,��Ȧ/��Ȧ
    const int thickness = -1;
    for(int i=0; i<N; ++i) {
        cv::circle (image, cv::Point2f(data[i][0], data[i][1]), 5,
            label[i] == T ? CV_RGB(0, 0, 0) : CV_RGB(255, 255, 255),
            thickness);
    }

    //����֧�ֵ�(��Ч��),��Ȧ
    auto& sv = svm->getUncompressedSupportVectors();
    for (int i = 0; i < sv.rows; ++i) {
        auto* v = sv.ptr<float>(i);
        cv::circle (image,  cv::Point2f(v[0], v[1]), 6, CV_RGB(128, 0, 0), 2);
    }

    cv::imshow("SVM Simple Example", image);
    cv::waitKey(0);
}
