#ifndef GRADIENT_CONVOLUTION_H
#define GRADIENT_CONVOLUTION_H

#include <hls_stream.h>
#include <ap_int.h>
#include <iostream>
#include <cmath>

#define MAX_IMG_H 1000
#define MAX_IMG_W 1000
#define MAX_PIXELS (MAX_IMG_H * MAX_IMG_W)

#define CELL_H 30
#define CELL_W 30
#define NB_BINS 9

typedef ap_uint<8> pixel_t;
typedef ap_int<16> grad_t;
typedef int hist_t;

void gradient_convolution(
    pixel_t* image,
    grad_t* gx,
    grad_t* gy,
    int rows,
    int cols
);

#endif