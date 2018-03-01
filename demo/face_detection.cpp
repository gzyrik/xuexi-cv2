#include <opencv2/opencv.hpp>
static cv::CascadeClassifier face_cascade, eyes_cascade;
static void detectAndDisplay( cv::Mat frame )
{
    //预处理,加快检测速度
    cv::Mat frame_scaled, frame_gray;
    cv::resize( frame, frame_scaled, cv::Size(256,256) );
    cv::cvtColor( frame_scaled, frame_gray, cv::COLOR_BGR2GRAY );
    cv::equalizeHist( frame_gray, frame_gray );

    //检测人脸
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE);
    for (auto& face : faces) {
        //检测眼睛
        std::vector<cv::Rect> eyes;
        eyes_cascade.detectMultiScale( frame_gray(face), eyes, 1.1, 2, CV_HAAR_SCALE_IMAGE);

        //绘制位置
        cv::rectangle( frame_scaled, face.tl(), face.br(), cv::Scalar( 255, 0, 255 ) );
        for (auto& eye : eyes){
            eye += face.tl();
            cv::rectangle( frame_scaled, eye.tl(), eye.br(), cv::Scalar( 0, 255, 0 ) );
        }
    }
    cv::imshow("face detection", frame_scaled);
}
void main()
{
    const std::string OpenCV_DIR = getenv("OpenCV_DIR");
    const std::string HaarCascades_DIR(OpenCV_DIR + "/etc/haarcascades/");
    std::string face_xml = HaarCascades_DIR + "haarcascade_frontalface_alt.xml";
    std::string eyes_xml = HaarCascades_DIR + "haarcascade_eye_tree_eyeglasses.xml";

    face_cascade.load(face_xml);
    eyes_cascade.load(eyes_xml);

    cv::VideoCapture camera(0);
    cv::Mat frame;
    while ( cv::waitKey(10) == -1 &&  camera.read(frame) )
        detectAndDisplay( frame );
}
