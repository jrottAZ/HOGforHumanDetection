#ifndef HOG_ACCEL_H
#define HOG_ACCEL_H

#include <ap_int.h>
#include <iostream>
#include <cmath>
#include <hls_stream.h>

#define MAX_IMG_H 500
#define MAX_IMG_W 500
#define MAX_PIXELS (MAX_IMG_H * MAX_IMG_W)

#define CELL_H 30
#define CELL_W 30
#define NB_BINS 9
#define CELL_SIZE (CELL_H * CELL_W)

#define NUM_PRIVATE 4

#define MAX_CELLS_Y (MAX_IMG_H / CELL_H)
#define MAX_CELLS_X (MAX_IMG_W / CELL_W)
#define MAX_CELLS (MAX_CELLS_Y * MAX_CELLS_X)

typedef ap_uint<8> pixel_t;
typedef ap_int<10> grad_t;
typedef ap_uint<10> mag_t;
typedef ap_uint<4> bin_t;
typedef int hist_t;

typedef struct {
    grad_t gx;
    grad_t gy;
} grad_pair_t;

void gradient_convolution(
    pixel_t* image,
    grad_t* gx,
    grad_t* gy,
    int rows,
    int cols
);

void histogram(
    grad_t* gx,
    grad_t* gy,
    hist_t* all_hists,
    int rows,
    int cols
);

void gradient_convolution_stream(
    pixel_t* image,
    hls::stream<grad_pair_t>& grad_stream,
    int rows, int cols
);

void hog_histogram_stream(
    hls::stream<grad_pair_t>& grad_stream,
    hist_t* all_hists,
    int rows, int cols
);

void hog_accel(
    pixel_t* image,
    hist_t* all_hists,
    int rows, int cols
);

#endif