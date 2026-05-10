#include "gradient_convolution.h"
#include "hls_stream.h"

#define WIN_SIZE 3
#define HALF_SIZE (((WIN_SIZE) - 1) / 2)

void gradient_convolution(
    pixel_t* image,
    grad_t* gx,
    grad_t* gy,
    int rows,
    int cols
) {

#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0 depth=1000000 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=gx    offset=slave bundle=gmem1 depth=1000000 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=gy    offset=slave bundle=gmem2 depth=1000000 max_write_burst_length=64 num_write_outstanding=16

#pragma HLS INTERFACE s_axilite port=image bundle=CTRL
#pragma HLS INTERFACE s_axilite port=gx    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=gy    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=rows  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=cols  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    pixel_t line_buf[WIN_SIZE - 1][MAX_IMG_W];
    pixel_t window[WIN_SIZE][WIN_SIZE];
    pixel_t right[WIN_SIZE];

#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=window complete dim=0
#pragma HLS ARRAY_PARTITION variable=right complete

    // initialize line buffer
    init_lb_y: for (int y = 0; y < WIN_SIZE - 1; y++) {
#pragma HLS UNROLL
        init_lb_x: for (int x = 0; x < cols; x++) {
#pragma HLS PIPELINE II=1
            line_buf[y][x] = 0;
        }
    }

    // initialize window
    init_win_y: for (int y = 0; y < WIN_SIZE; y++) {
#pragma HLS UNROLL
        init_win_x: for (int x = 0; x < WIN_SIZE; x++) {
#pragma HLS UNROLL
            window[y][x] = 0;
        }
    }

    int read_count = 0;

    // main rolling-buffer scan
    for_y: for (int img_y = 0; img_y < rows; img_y++) {
        int out_row_base = (img_y - HALF_SIZE) * cols;

        for_x: for (int img_x = 0; img_x < cols; img_x++) {
#pragma HLS PIPELINE II=1

            pixel_t val_in = image[read_count];
            read_count++;

            right[0] = line_buf[0][img_x];

            shift_lb:for (int y = 1; y < WIN_SIZE - 1; y++) {
                #pragma HLS UNROLL
                right[y] = line_buf[y][img_x];
                line_buf[y - 1][img_x] = line_buf[y][img_x];
            }

            right[WIN_SIZE - 1] = val_in;
            line_buf[WIN_SIZE - 2][img_x] = val_in;

            // shift window left
            shift_win_y: for (int y = 0; y < WIN_SIZE; y++) {
                #pragma HLS UNROLL
                shift_win_x: for (int x = 0; x < WIN_SIZE - 1; x++) {
                    #pragma HLS UNROLL
                    window[y][x] = window[y][x + 1];
                }
            }

            // insert right column
            update_win: for (int y = 0; y < WIN_SIZE; y++) {
                #pragma HLS UNROLL
                window[y][WIN_SIZE - 1] = right[y];
            }

            int out_y = img_y - HALF_SIZE;
            int out_x = img_x - HALF_SIZE;

            if (out_y >= 0 && out_y < rows && out_x >= 0 && out_x < cols) {
                bool border = (out_y == 0 || out_y == rows - 1 ||
                               out_x == 0 || out_x == cols - 1);

                grad_t gx_val = 0;
                grad_t gy_val = 0;

                if (!border) {
                    gx_val = (grad_t)window[1][2] - (grad_t)window[1][0];
                    gy_val = (grad_t)window[2][1] - (grad_t)window[0][1];
                }

                int out_idx = out_row_base + out_x;

                gx[out_idx] = gx_val;
                gy[out_idx] = gy_val;
            }
        }
    }
}