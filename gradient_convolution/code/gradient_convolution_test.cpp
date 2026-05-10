#include "gradient_convolution.h"

#include <hls_stream.h>
#include <iostream>

//software test for simulation
void gradient_convolution_sw(
    pixel_t image[MAX_IMG_H][MAX_IMG_H],
    grad_t gx[MAX_IMG_H][MAX_IMG_H],
    grad_t gy[MAX_IMG_H][MAX_IMG_H]
){
    for (int i = 0; i < MAX_IMG_H; i++) {
        for (int j = 0; j < MAX_IMG_H; j++) {
            if (i == 0 || i == MAX_IMG_H - 1 || j == 0 || j == MAX_IMG_H - 1) {
                gx[i][j] = 0;
                gy[i][j] = 0;
            } else {
                gx[i][j] = (grad_t)image[i][j + 1] - (grad_t)image[i][j - 1];
                gy[i][j] = (grad_t)image[i + 1][j] - (grad_t)image[i - 1][j];
            }
        }
    }
}

//main test for hls component
int main(){
    //sample image
    pixel_t image[MAX_IMG_H][MAX_IMG_H];

    //hardware simulation results
    grad_t gx_hw[MAX_IMG_H][MAX_IMG_H];
    grad_t gy_hw[MAX_IMG_H][MAX_IMG_H];

    //software implementation results
    grad_t gx_sw[MAX_IMG_H][MAX_IMG_H];
    grad_t gy_sw[MAX_IMG_H][MAX_IMG_H];

    //generic test image generation for comparison
    for (int i = 0; i < MAX_IMG_H; i++) {
        for (int j = 0; j < MAX_IMG_H; j++) {
            image[i][j] = (pixel_t)((i * 3 + j * 2) % 256);
        }
    }

    //run both implementations of the convolution
    gradient_convolution_sw(image, gx_sw, gy_sw);
    gradient_convolution(&image[0][0], &gx_hw[0][0], &gy_hw[0][0], MAX_IMG_H, MAX_IMG_H);

    //check for differences in the output from the test image
    int errors = 0;
    for(int i = 0; i < MAX_IMG_H; i++){
        for(int j = 0; j < MAX_IMG_H; j++){
            if((gx_hw[i][j] != gx_sw[i][j]) | (gy_hw[i][j] != gy_sw[i][j])){
                errors++;
            }
        }
    }

    //see if there are any errors in the code implementation
    if (errors == 0) {
        std::cout << "TEST PASSED: RESULT MATCHES GOLDEN OUTPUT" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED, ERRORS = " << errors << std::endl;
        return 1;
    }
    
}