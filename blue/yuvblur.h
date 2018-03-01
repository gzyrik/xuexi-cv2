#ifndef YUV_BLUR_H
#define YUV_BLUR_H


void yuvblur(unsigned char *src_I420_y, unsigned char *src_I420_u, unsigned char *src_I420_v,
    unsigned char *dst_I420_y, unsigned char *dst_I420_u, unsigned char *dst_I420_v,
    int w, int h, int center_row, int center_col,
    double v, int blur_intensity, int transit_w, int origin_w, double rotation);

#endif
