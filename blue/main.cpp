#include <stdio.h>
#include <string>
#include "opencv2/opencv.hpp"
#include "yuvblur.h"
void StartTimer();
double GetTimer();
int main() {
	/*src_I420_y u v: 		 input  I420_y u v data
	*dst_I420_y u v: 		 output I420_y u v data
	*w h: 					 width and height of the input and output I420
	*center_row center_col: the coordinate of center rotation dot
	*v:  					 times of shrink
	*blur_intensity:		 intensity of blur
	*transit_w:		 	 width of the transition area
	*origin_w:		 	 	 width of the changeless area
	*rotation:		 	 	 the rotation of blur(from 0 to PI)
	* */

    std::string imdir = "res/";
    cv::Mat im = cv::imread(imdir + "marry1280.jpg");

    int w = im.cols;
    int h = im.rows;

    cv::Mat im_yuv;
    cv::cvtColor(im, im_yuv, cv::COLOR_BGR2YUV_I420);
	StartTimer();
    uchar *src_I420_y = im_yuv.data;
    uchar *src_I420_u = src_I420_y + w * h;
    uchar *src_I420_v = src_I420_u + w * h / 4;

    cv::Mat im_yuv_blur(im_yuv.size(), im_yuv.type());
	std::cout << im_yuv.size() << ' ' << im_yuv.type();
    unsigned char *dst_I420_y = im_yuv_blur.data;
    unsigned char *dst_I420_u = dst_I420_y + w * h;
    unsigned char *dst_I420_v = dst_I420_u + w * h / 4;

    int center_row = h / 2, center_col = w / 2;
    double v = sqrt((w*h)/(640*360));
    int blur_intensity = 80, transit_w = 300, origin_w = 50;
    double rotation = 3.1415926 / 6;

    yuvblur(src_I420_y, src_I420_u, src_I420_v,
        dst_I420_y, dst_I420_u, dst_I420_v,
        w, h, center_row, center_col,
        v, blur_intensity, transit_w, origin_w, rotation);
	double costtime = GetTimer();
	printf("costime:%lf\n", costtime);
    cv::Mat im_blur;
    cv::cvtColor(im_yuv_blur, im_blur, cv::COLOR_YUV2BGR_I420);
    cv::imwrite(imdir + "marry1280_blur.png", im_blur);

    return 0;
}
