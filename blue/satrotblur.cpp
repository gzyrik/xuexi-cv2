#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

void Summed_Area_SatRotBlur( int64_t * table2,  unsigned char *newpix, int w, int h, double v,
							  int center_row,int center_col,int pad,int blur_intensity,
							  int transit_w,int origin_w,double rotation,bool isuv){
	/*--------中心点变换坐标系------------*/
	double rotationCenter=sin(rotation)*(center_col+pad)+cos(rotation)*(center_row+pad);
	int w_p=w+2*pad;
	//if (v>1)
	//{
		int i,j;
		int currentindex;
		double rotationCurrent,distance;
		int current_intensity,r;
		for ( i = pad; i < h+pad; i++){
				for ( j = pad; j < w+pad; j++){
					rotationCurrent=cos(rotation)*i+sin(rotation)*j;
					distance=fabs(rotationCurrent-rotationCenter)-origin_w;
					if (distance>0)
					{
						current_intensity=blur_intensity*fabs(distance)/(transit_w);
						if (current_intensity>blur_intensity)
							current_intensity=blur_intensity;
						if (isuv==false)
							r=ceil(0.2*current_intensity/v);
						else
							r = ceil(0.1*current_intensity / v);
						if (r<=0) r=1;
						currentindex = (i-pad)*w + j-pad;
						int pre_i=i-1-r;
						int pre_j=j-1-r;
						int next_i=i-1+r;
						int next_j=j-1+r;
						newpix[currentindex ] = ((4*table2[(i-1)*w_p+j-1]-2*(table2[pre_i*w_p+j-1]+table2[(i-1)*w_p+pre_j]+table2[(i-1)*w_p+next_j]+table2[next_i*w_p+j-1]))
											+table2[next_i*w_p+next_j]+table2[pre_i*w_p+next_j]+table2[next_i*w_p+pre_j]+table2[pre_i*w_p+pre_j]
											)*1.0/(r*r*r*r);
						/*if (r==1)
							newpix[currentindex ]/=1.5;
						else if(r==2)
							newpix[currentindex ]/=2;
						else if (r==3)
							newpix[currentindex ]/=2.5;*/
					}
					//if (i==30)
					//	printf(" Sat newpix[currentindex ] %d %d", j,newpix[currentindex ] );
				}
			}
	//}



}
