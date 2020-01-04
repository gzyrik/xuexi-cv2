//Principal Component Analysis(PCA)
//通过线性变换将原始数据变换为一组各维度(特征向量))线性无关的表示，
//可用于提取数据的主要特征分量，常用于高维数据的降维
#include <opencv2/opencv.hpp>
static void drawAxis(cv::Mat& image, const cv::Point& start, const cv::Point& offet, const cv::Scalar& color, const char* text)
{
    cv::Point p, end(start + offet);
    cv::line(image, start, end, color, 1, cv::LINE_AA);
    if (text) cv::putText(image, text, end, cv::FONT_HERSHEY_COMPLEX, 0.5, color, 1, cv::LINE_AA, true);
    //绘制箭头
    double angle = atan2(-offet.y, -offet.x);//改变方向,指向start点
    p.x = (int) (end.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (end.y + 9 * sin(angle + CV_PI / 4));
    cv::line(image, p, end, color, 1, cv::LINE_AA);

    p.x = (int) (end.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (end.y + 9 * sin(angle - CV_PI / 4));
    cv::line(image, p, end, color, 1, cv::LINE_AA);
}
static void detectAndDisplay(std::vector<cv::Point> &contour, cv::Mat &image)
{
    cv::Mat data;{
        cv::Mat contour_mat((int)contour.size(), 2, CV_32SC1, &contour.front());
        contour_mat.assignTo(data, CV_64FC1);
    }

    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    auto v = pca.mean.ptr<double>(0);
    const cv::Point2d center(v[0],v[1]);

    //特征值和特征向量
    const int rank = pca.eigenvalues.rows;
    std::vector<double> eigen_val(rank);
    std::vector<cv::Point2d> eigen_vecs(rank);
    pca.eigenvalues.forEach<double>([&](double& element, const int position[]){
        //forEach 是并行执行,不要依赖次序
        const int i = position[0];
        auto v = pca.eigenvectors.ptr<double>(i);
        eigen_vecs[i].x = v[0];
        eigen_vecs[i].y = v[1];
        eigen_val[i] = element;
    });

    //绘制,均值点和主成分
    cv::circle(image, center, 3, cv::Scalar(255, 0, 255), 2);
    drawAxis(image, center, 0.02*eigen_vecs[0] * eigen_val[0], cv::Scalar(0, 255, 0), "X");
    drawAxis(image, center, -0.1*eigen_vecs[1] * eigen_val[1], cv::Scalar(255, 255, 0), "Y");
}
void main(int argc, char** argv)
{
    cv::Mat image;
    if (argc < 2 || (image = cv::imread(argv[1])).empty()){
        std::cerr << "Problem loading image!!!" << std::endl;
        exit(-1);
    }
    cv::Mat gray_mat, binary_mat;
    cv::cvtColor( image, gray_mat, cv::COLOR_BGR2GRAY );
    cv::threshold(gray_mat, binary_mat, 50, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    //轮廓
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(binary_mat, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    for (int i=0; i >= 0; i = hierarchy[i][0] ){
        double area = cv::contourArea(contours[i]);//面积
        if (area < 1e2 || 1e5 < area) continue;

        cv::drawContours(image, contours, i, cv::Scalar(0, 0, 255), 2, 8, hierarchy, 0);
        detectAndDisplay(contours[i], image);
    }
    cv::imshow("", image);
    cv::waitKey(0);
}


