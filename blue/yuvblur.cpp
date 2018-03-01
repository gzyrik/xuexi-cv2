#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "opencv2/opencv.hpp"

void Summed_Area_SatRotBlur(int64_t * table2,  unsigned char *newpix, int w, int h, double v,
	int center_row, int center_col, int pad, int blur_intensity, int transit_w, int origin_w, double rotation, bool isuv);
static void originpixcopy(unsigned char *src_pix,unsigned char *dst_pix,int w,int h,int center_row,int center_col,int origin_w,double rotation);
static void Summed_Area_gettable( unsigned char *pix, int64_t * table2, int w, int h, int pad);

void yuvblur(  unsigned char *src_I420_y,unsigned char *src_I420_u,unsigned char *src_I420_v,
			   unsigned char *dst_I420_y, unsigned char *dst_I420_u, unsigned char *dst_I420_v,
			   int w, int h,int center_row,int center_col,
			   double v,int blur_intensity,int transit_w,int origin_w,double rotation)
{
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

	unsigned char *dstv_y,*dstv_u,*dstv_v;
	int64_t *table_y,*table_u,*table_v;
	int pad = 21;
	dstv_y = (unsigned char *)malloc(sizeof(char) * ((int)((w*  h)/(v*v))));
	dstv_u = (unsigned char *)malloc(sizeof(char) * ( (int)((w*  h)/(4*v*v)) ));
	dstv_v = (unsigned char *)malloc(sizeof(char) * ((int)((w*  h)/(4*v*v))));
    /*------ Image is Shrinked by v times------*/
#if 0
    I420Scale(src_I420_y, w,
    	      src_I420_u, w/2,
			  src_I420_v, w/2,
			  w, h,
			  dstv_y, (int)(w/v),
    	      dstv_u, (int)(w/(v*2)),
    	      dstv_v, (int)(w/(2*v)),
			  (int)(w/v), (int)(h/v),
    	      2);
#else
    cv::Mat src_y(h, w, CV_8UC1, src_I420_y);
    cv::Mat src_u(h/2, w/2, CV_8UC1, src_I420_u);
    cv::Mat src_v(h / 2, w / 2, CV_8UC1, src_I420_v);
    int w_s = (int)(w / v);
    int h_s = (int)(h / v);
    cv::Mat small_y(h_s, w_s, CV_8UC1, dstv_y);
    cv::Mat small_u(h_s / 2, w_s / 2, CV_8UC1, dstv_u);
    cv::Mat small_v(h_s / 2, w_s / 2, CV_8UC1, dstv_v);
    cv::resize(src_y, small_y, small_y.size(), 0, 0, CV_INTER_AREA);
    cv::resize(src_u, small_u, small_u.size(), 0, 0, CV_INTER_AREA);
    cv::resize(src_v, small_v, small_v.size(), 0, 0, CV_INTER_AREA);
#endif
    printf("I420 to I420 1\n");

    /*----------Y U V Summed Area table Get-----*/
    int pad_y=pad, pad_u=(int)(pad/2), pad_v= (int)(pad/2);
    table_y = (int64_t*)malloc(sizeof(int64_t) * (int)((w/v+2*pad_y) * (h/v+2*pad_y)));
    table_u = (int64_t*)malloc(sizeof(int64_t) * (int)((w/(2*v)+2*pad_u) * (h/(2*v)+2*pad_u)));
    table_v = (int64_t*)malloc(sizeof(int64_t) * (int)((w/(2*v)+2*pad_v) * (h/(2*v)+2*pad_v)));
    Summed_Area_gettable( dstv_y,table_y,(int)(w/v),(int)(h/v), pad_y);
    printf("Summed_Area_gettable y \n");
    Summed_Area_gettable( dstv_u,table_u,(int)(w/(2*v)),(int)(h/(2*v)), pad_u);
    printf("Summed_Area_gettable u \n");
    Summed_Area_gettable( dstv_v,table_v,(int)(w/(2*v)),(int)(h/(2*v)), pad_v);
    printf("Summed_Area_gettable v\n");

    /*----------rotate the coordinate and Get y u v img blurred------------*/
    int center_row_y=(int)(center_row/v), center_row_u=(int)(center_row/(2*v)),center_row_v=(int)(center_row/(2*v));
    int center_col_y=(int)(center_col/v), center_col_u=(int)(center_col/(2*v)),center_col_v=(int)(center_col/(2*v));
    Summed_Area_SatRotBlur(table_y,dstv_y, (int)(w/v),(int)(h/v), v,center_row_y,center_col_y,pad_y,
							blur_intensity, transit_w/v,origin_w/v, rotation,false);
    Summed_Area_SatRotBlur(table_u,dstv_u, (int)(w/(2*v)),(int)(h/(2*v)), v,center_row_u,center_col_u,pad_u,
    							blur_intensity, transit_w/(2*v),origin_w/(2*v), rotation,true);
    Summed_Area_SatRotBlur(table_v,dstv_v, (int)(w/(2*v)),(int)(h/(2*v)), v,center_row_v,center_col_v,pad_v,
    							blur_intensity, transit_w/(2*v),origin_w/(2*v), rotation,true);
    printf("Summed_Area_SatRotBlur complete\n");
    /*-----------I420 zoom v times using method box filter----------------*/
#if 0
    I420Scale(dstv_y, (int)(w/v),
    		  dstv_u, (int)(w/(v*2)),
			  dstv_v, (int)(w/(2*v)),
              (int)(w/v), (int)(h/v),
			  dst_I420_y, w,
			  dst_I420_u, w/2,
			  dst_I420_v, w/2,
               w, h,3);
#else
    cv::Mat dst_y(h, w, CV_8UC1, dst_I420_y);
    cv::Mat dst_u(h / 2, w / 2, CV_8UC1, dst_I420_u);
    cv::Mat dst_v(h / 2, w / 2, CV_8UC1, dst_I420_v);
    cv::resize(small_y, dst_y, dst_y.size(), 0, 0, CV_INTER_LINEAR);
    cv::resize(small_u, dst_u, dst_u.size(), 0, 0, CV_INTER_LINEAR);
    cv::resize(small_v, dst_v, dst_v.size(), 0, 0, CV_INTER_LINEAR);
#endif
    printf("I420Scale room complete\n");
    /*-----------copy the changeless area to blurred I420----------------*/
    originpixcopy(src_I420_y,dst_I420_y, w, h, center_row, center_col,origin_w,rotation);
    originpixcopy(src_I420_u,dst_I420_u, w/2, h/2, center_row/2, center_col/2,(int)(origin_w/2),rotation);
    originpixcopy(src_I420_v,dst_I420_v, w/2, h/2, center_row/2, center_col/2,(int)(origin_w/2),rotation);
    printf("originpix copy complete\n");

    free(dstv_y);
    free(dstv_u);
    free(dstv_v);
    free(table_y);
    free(table_u);
    free(table_v);

}
static void originpixcopy(unsigned char *src_pix,unsigned char *dst_pix,int w,int h,int center_row,int center_col,int origin_w,double rotation)
 {
	 int i,j,j1,j2;
	 int rotationCenter=(int)(sin(rotation)*(center_col)+cos(rotation)*(center_row));
	 for (i=0;i<h;i++)
	 {
		 //printf("originpix copy %d",i);
        j1= (int)round(((origin_w+rotationCenter)-cos(rotation)*i)/sin(rotation));
        j2= (int)round(((-origin_w+rotationCenter)-cos(rotation)*i)/sin(rotation));
        if (j1>=w)
            j1=w-1;
         if (j2<0)
             j2=0;
         for(j=j2; j<=j1;j++)
        	 dst_pix[i*w+j]=src_pix[i*w+j];
	 }


}

static void Summed_Area_gettable( unsigned char *pix,  int64_t * table2, int w, int h, int pad){
	//int pad=ceil(40/v);
	int h_p=h+2*pad;
	int w_p=w+2*pad;
	int * table = ( int*)malloc(sizeof( int)*w_p*h_p);
	//int64_t * table2 = (int64_t*)malloc(sizeof(int64_t)*w_p*h_p);//ulong64_t
	int i,j;
	printf("Summed_Area_TableBlur %d %d %d\n",w,h,pad);
	for ( i = 0; i < h_p; i++){

		for (j = 0; j < w_p; j++){
			//if (i==190)
			//	printf("Summed_Area_TableBlur %d",j);
			int currentindex = i*w_p + j;
			int currentpixindex ;
			if  (i<pad)
			{
				if (j<pad)
					currentpixindex=(pad-1-i)*w+pad-1-j;
				else if(j>=pad&&j<pad+w)
					currentpixindex=(pad-1-i)*w+j-pad;
				else if (j>=pad+w)
					currentpixindex=(pad-1-i)*w+2*w+pad-1-j;
			}
			else if (i>=pad&&i<pad+h)
			{
				if(j<pad)
					currentpixindex=(i-pad)*w+pad-1-j;
				else if(j>=pad+w)
					currentpixindex=(i-pad)*w+2*w+pad-1-j;
				else
					currentpixindex=(i-pad)*w+j-pad;
			}
			else if (i>=pad+h)
			{
				if (j<pad)
					currentpixindex=(2*h+pad-1-i)*w+pad-1-j;
				else if (j>=pad&&j<pad+w)
					currentpixindex=(2*h+pad-1-i)*w+j-pad;
				else if (j>=pad+w)
					currentpixindex=(2*h+pad-1-i)*w+2*w+pad-1-j;
			}

			if (i == 0 && j == 0){
				table[currentindex] = pix[currentpixindex];
			}
			else if (i == 0&&j!=0){
				table[currentindex ] = pix[currentpixindex ] + table[currentindex - 1];
			}
			else if (j == 0&&i!=0){
				table[currentindex ] = pix[currentpixindex ] + table[currentindex -w_p];
				//printf(" Sat table[currentindex] %d %d", i,table[currentindex]);
			}
			else{
				table[currentindex ] = table[currentindex - 1] + pix[currentpixindex] + table[currentindex - w_p] - table[currentindex-w_p - 1];
			}
		}
	}
	printf("get sat table 1\n");
	for ( i = 0; i < h+2*pad; i++){

		//printf(" Sat table[currentindex] %d ",i);
		for (j = 0; j < w+2*pad; j++){
			int currentindex = i*w_p + j;
			if (i == 0 && j == 0){
				table2[currentindex] = table[currentindex];
				//printf(" Sat table[currentindex] %d %d %lld", sizeof( long),i,table2[currentindex]);
			}
			else if (i == 0&&j!=0){
				table2[currentindex ] = table[currentindex ] + table2[currentindex - 1];
			}
			else if (j == 0&&i!=0){
				table2[currentindex ] = table[currentindex ] + table2[currentindex -w_p];
			}

			else{
				table2[currentindex ] = table2[currentindex - 1] + table[currentindex] + table2[currentindex - w_p] - table2[currentindex-w_p - 1];
				//if (i==100&&j>300)
				//printf(" Sat table[currentindex] %d %d %lld", sizeof( long),j,table2[currentindex]);
			}
		}
	}
	printf("get sat table 2\n");
}


