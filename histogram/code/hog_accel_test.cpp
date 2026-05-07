#include "hog_accel.h"
#include <iostream>

static const int TEST_ROWS = 60;
static const int TEST_COLS = 60;

static pixel_t image_flat[MAX_PIXELS];
static grad_t gx_sw[MAX_PIXELS];
static grad_t gy_sw[MAX_PIXELS];
static grad_t gx_hw[MAX_PIXELS];
static grad_t gy_hw[MAX_PIXELS];
static hist_t all_hists_hw[MAX_CELLS * NB_BINS];
static hist_t all_hists_sw[MAX_CELLS * NB_BINS];

// gradient software reference
void gradient_convolution_sw(
    pixel_t* image,
    grad_t* gx,
    grad_t* gy,
    int rows, int cols
) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                gx[idx] = 0;
                gy[idx] = 0;
            } else {
                gx[idx] = (grad_t)image[i*cols + (j+1)] - (grad_t)image[i*cols + (j-1)];
                gy[idx] = (grad_t)image[(i+1)*cols + j] - (grad_t)image[(i-1)*cols + j];
            }
        }
    }
}

static int compute_bin_sw(int x, int y) {
    if (y < 0) { x = -x; y = -y; }
    if (x == 0 && y == 0) return 0;
    if (x > 0) {
        if (y * 1000 < x * 364) return 0;
        else if (y * 1000 < x * 839) return 1;
        else if (y * 1000 < x * 1732) return 2;
        else if (y * 1000 < x * 5671) return 3;
        else return 4;
    }
    else if (x == 0) {
        return 4;
    }
    else {
        int ax = -x;
        if (y * 1000 < ax * 364) return 8;
        else if (y * 1000 < ax * 839) return 7;
        else if (y * 1000 < ax * 1732) return 6;
        else if (y * 1000 < ax * 5671) return 5;
        else return 4;
    }
}

// histogram software reference
void histogram_sw(
    grad_t* gx,
    grad_t* gy,
    hist_t* all_hists,
    int rows, int cols
) {
    int num_cells_y = rows / CELL_H;
    int num_cells_x = cols / CELL_W;

    for (int cy = 0; cy < num_cells_y; cy++) {
        for (int cx = 0; cx < num_cells_x; cx++) {
            int hist_base = (cy * num_cells_x + cx) * NB_BINS;

            for (int b = 0; b < NB_BINS; b++)
                all_hists[hist_base + b] = 0;

            for (int py = 0; py < CELL_H; py++) {
                for (int px = 0; px < CELL_W; px++) {
                    int idx = (cy * CELL_H + py) * cols + (cx * CELL_W + px);
                    int gx_v = (int)gx[idx];
                    int gy_v = (int)gy[idx];
                    int mag = (gx_v < 0 ? -gx_v : gx_v) + (gy_v < 0 ? -gy_v : gy_v);
                    int bin = compute_bin_sw(gx_v, gy_v);
                    all_hists[hist_base + bin] += mag;
                }
            }
        }
    }
}

//main test for hls component
int main() {
    //generic test image generation for comparison
    for (int i = 0; i < TEST_ROWS; i++)
        for (int j = 0; j < TEST_COLS; j++)
            image_flat[i * TEST_COLS + j] = (pixel_t)((i * 3 + j * 2) % 256);

    //run both implementations of the convolution
    gradient_convolution_sw(image_flat, gx_sw, gy_sw, TEST_ROWS, TEST_COLS);
    gradient_convolution(image_flat, gx_hw, gy_hw, TEST_ROWS, TEST_COLS);

    //check for differences in the output from the test image
    int grad_errors = 0;
    for (int i = 0; i < TEST_ROWS * TEST_COLS; i++){
        if (gx_hw[i] != gx_sw[i] || gy_hw[i] != gy_sw[i]) {
            grad_errors++;
        }
    }

    //see if there are any errors in the code implementation
    if (grad_errors == 0) {
        std::cout << "TEST 1 PASSED: GRADIENT RESULT MATCHES GOLDEN OUTPUT" << std::endl;
    } else {
        std::cout << "TEST 1 FAILED, GRADIENT ERRORS = " << grad_errors << std::endl;
    }


    int num_cells = (TEST_ROWS / CELL_H) * (TEST_COLS / CELL_W);

    histogram_sw(gx_sw, gy_sw, all_hists_sw, TEST_ROWS, TEST_COLS);
    histogram(gx_hw, gy_hw, all_hists_hw, TEST_ROWS, TEST_COLS);
    
    int hist_errors = 0;
    for (int i = 0; i < num_cells * NB_BINS; i++) {
        if (all_hists_hw[i] != all_hists_sw[i]) {
            if (hist_errors < 5)  // print first 5 mismatches only
                std::cout << "  Mismatch at index " << i
                          << " (cell " << i/NB_BINS << ", bin " << i%NB_BINS << ")"
                          << ": hw=" << all_hists_hw[i]
                          << " sw=" << all_hists_sw[i] << std::endl;
            hist_errors++;
        }
    }

    if (hist_errors == 0)
        std::cout << "TEST 2 PASSED: HISTOGRAMS MATCH GOLDEN OUTPUT" << std::endl;
    else
        std::cout << "TEST 2 FAILED: HISTOGRAM ERRORS = " << hist_errors << std::endl;

    return (grad_errors + hist_errors == 0) ? 0 : 1;
}