#include "hog_accel.h"

#define WIN_SIZE  3
#define HALF_SIZE (((WIN_SIZE) - 1) / 2)

// orientation binning
static bin_t compute_bin(grad_t gx_in, grad_t gy_in) {
#pragma HLS INLINE
    long x = (long)gx_in;
    long y = (long)gy_in;

    if (y < 0) { x = -x; y = -y; }
    if (x == 0 && y == 0) return 0;

    bin_t bin = 0;
    if (x > 0) {
        if      (y * 1000L < x * 364L)  bin = 0;
        else if (y * 1000L < x * 839L)  bin = 1;
        else if (y * 1000L < x * 1732L) bin = 2;
        else if (y * 1000L < x * 5671L) bin = 3;
        else                             bin = 4;
    } else if (x == 0) {
        bin = 4;
    } else {
        long ax = -x;
        if      (y * 1000L < ax * 364L)  bin = 8;
        else if (y * 1000L < ax * 839L)  bin = 7;
        else if (y * 1000L < ax * 1732L) bin = 6;
        else if (y * 1000L < ax * 5671L) bin = 5;
        else                              bin = 4;
    }
    return bin;
}

static void histogram_cell(
    mag_t  cell_mag[CELL_SIZE],
    bin_t  cell_bin[CELL_SIZE],
    hist_t hist[NB_BINS]
) {
#pragma HLS INLINE off
    hist_t private_hist[NUM_PRIVATE][NB_BINS];
#pragma HLS ARRAY_PARTITION variable=private_hist complete dim=1

    init_priv: for (int p = 0; p < NUM_PRIVATE; p++) {
#pragma HLS UNROLL
        init_bins: for (int b = 0; b < NB_BINS; b++) {
#pragma HLS UNROLL
            private_hist[p][b] = 0;
        }
    }

    accum: for (int i = 0; i < CELL_SIZE; i++) {
#pragma HLS PIPELINE II=1
        int p = i % NUM_PRIVATE;
        int b = (int)cell_bin[i];
        private_hist[p][b] += (hist_t)cell_mag[i];
    }

    reduce: for (int b = 0; b < NB_BINS; b++) {
#pragma HLS PIPELINE II=1
        hist_t sum = 0;
        reduce_priv: for (int p = 0; p < NUM_PRIVATE; p++) {
#pragma HLS UNROLL
            sum += private_hist[p][b];
        }
        hist[b] = sum;
    }
}

void gradient_convolution(
    pixel_t* image,
    grad_t*  gx,
    grad_t*  gy,
    int rows, int cols
) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0 depth=16384 max_read_burst_length=64  num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=gx    offset=slave bundle=gmem1 depth=16384 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=gy    offset=slave bundle=gmem2 depth=16384 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=image  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=gx     bundle=CTRL
#pragma HLS INTERFACE s_axilite port=gy     bundle=CTRL
#pragma HLS INTERFACE s_axilite port=rows   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=cols   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    pixel_t line_buf[WIN_SIZE - 1][MAX_IMG_W];
    pixel_t window[WIN_SIZE][WIN_SIZE];
    pixel_t right[WIN_SIZE];

#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=window   complete dim=0
#pragma HLS ARRAY_PARTITION variable=right    complete

    init_lb_y: for (int y = 0; y < WIN_SIZE - 1; y++) {
        init_lb_x: for (int x = 0; x < cols; x++) {
#pragma HLS PIPELINE II=1
            line_buf[y][x] = 0;
        }
    }
    init_win_y: for (int y = 0; y < WIN_SIZE; y++) {
        init_win_x: for (int x = 0; x < WIN_SIZE; x++) {
#pragma HLS PIPELINE II=1
            window[y][x] = 0;
        }
    }

    int read_count = 0;

    for_y: for (int img_y = 0; img_y < rows; img_y++) {
        int out_row_base = (img_y - HALF_SIZE) * cols;
        for_x: for (int img_x = 0; img_x < cols; img_x++) {
#pragma HLS PIPELINE II=1
            pixel_t val_in = image[read_count++];

            right[0] = line_buf[0][img_x];
            shift_lb: for (int y = 1; y < WIN_SIZE - 1; y++) {
#pragma HLS UNROLL
                right[y]             = line_buf[y][img_x];
                line_buf[y-1][img_x] = line_buf[y][img_x];
            }
            right[WIN_SIZE - 1]           = val_in;
            line_buf[WIN_SIZE - 2][img_x] = val_in;

            shift_win_y: for (int y = 0; y < WIN_SIZE; y++) {
#pragma HLS UNROLL
                shift_win_x: for (int x = 0; x < WIN_SIZE - 1; x++) {
#pragma HLS UNROLL
                    window[y][x] = window[y][x+1];
                }
            }
            update_win: for (int y = 0; y < WIN_SIZE; y++) {
#pragma HLS UNROLL
                window[y][WIN_SIZE - 1] = right[y];
            }

            int out_y = img_y - HALF_SIZE;
            int out_x = img_x - HALF_SIZE;

            if (out_y >= 0 && out_y < rows && out_x >= 0 && out_x < cols) {
                grad_t gx_val = 0, gy_val = 0;
                bool border = (out_y == 0 || out_y == rows-1 ||
                               out_x == 0 || out_x == cols-1);
                if (!border) {
                    gx_val = (grad_t)window[1][2] - (grad_t)window[1][0];
                    gy_val = (grad_t)window[2][1] - (grad_t)window[0][1];
                }
                gx[out_row_base + out_x] = gx_val;
                gy[out_row_base + out_x] = gy_val;
            }
        }
    }
}

void histogram(
    grad_t* gx,
    grad_t* gy,
    hist_t* all_hists,
    int rows, int cols
) {
#pragma HLS INTERFACE m_axi port=gx        offset=slave bundle=gmem0 depth=16384 max_read_burst_length=64  num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=gy        offset=slave bundle=gmem1 depth=16384 max_read_burst_length=64  num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=all_hists offset=slave bundle=gmem2 depth=9801  max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=gx        bundle=CTRL
#pragma HLS INTERFACE s_axilite port=gy        bundle=CTRL
#pragma HLS INTERFACE s_axilite port=all_hists bundle=CTRL
#pragma HLS INTERFACE s_axilite port=rows      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=cols      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return    bundle=CTRL

    mag_t  cell_mag[CELL_SIZE];
    bin_t  cell_bin[CELL_SIZE];
    hist_t cell_hist[NB_BINS];

#pragma HLS ARRAY_PARTITION variable=cell_mag  cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=cell_bin  cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=cell_hist complete

    int num_cells_y = rows / CELL_H;
    int num_cells_x = cols / CELL_W;

    cell_y: for (int cy = 0; cy < num_cells_y; cy++) {
        cell_x: for (int cx = 0; cx < num_cells_x; cx++) {

            load_cell: for (int i = 0; i < CELL_SIZE; i++) {
#pragma HLS PIPELINE II=1
                int py = i / CELL_W;
                int px = i % CELL_W;
                int img_idx = (cy * CELL_H + py) * cols
                            + (cx * CELL_W + px);
                grad_t gx_val = gx[img_idx];
                grad_t gy_val = gy[img_idx];
                cell_mag[i] = (mag_t)(
                    (gx_val < 0 ? (grad_t)(-gx_val) : gx_val) +
                    (gy_val < 0 ? (grad_t)(-gy_val) : gy_val)
                );
                cell_bin[i] = compute_bin(gx_val, gy_val);
            }

            histogram_cell(cell_mag, cell_bin, cell_hist);

            write_hist: for (int b = 0; b < NB_BINS; b++) {
#pragma HLS PIPELINE II=1
                all_hists[(cy * num_cells_x + cx) * NB_BINS + b] = cell_hist[b];
            }
        }
    }
}

void gradient_convolution_stream(
    pixel_t*                  image,
    hls::stream<grad_pair_t>& grad_stream,
    int rows, int cols
) {
#pragma HLS INLINE off

    pixel_t line_buf[WIN_SIZE - 1][MAX_IMG_W];
    pixel_t window[WIN_SIZE][WIN_SIZE];
    pixel_t right[WIN_SIZE];

#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=window   complete dim=0
#pragma HLS ARRAY_PARTITION variable=right    complete

    init_lb_y: for (int y = 0; y < WIN_SIZE - 1; y++) {
        init_lb_x: for (int x = 0; x < cols; x++) {
#pragma HLS PIPELINE II=1
            line_buf[y][x] = 0;
        }
    }
init_win_y: for (int y = 0; y < WIN_SIZE; y++) {
        init_win_x: for (int x = 0; x < WIN_SIZE; x++) {
#pragma HLS PIPELINE II=1
            window[y][x] = 0;
        }
    }

    int read_count = 0;

    warmup_y: for (int img_y = 0; img_y < HALF_SIZE; img_y++) {
        warmup_x: for (int img_x = 0; img_x < cols; img_x++) {
#pragma HLS PIPELINE II=1
            pixel_t val_in = image[read_count++];

            right[0] = line_buf[0][img_x];
            wu_shift_lb: for (int y = 1; y < WIN_SIZE - 1; y++) {
#pragma HLS UNROLL
                right[y]             = line_buf[y][img_x];
                line_buf[y-1][img_x] = line_buf[y][img_x];
            }
            right[WIN_SIZE - 1]           = val_in;
            line_buf[WIN_SIZE - 2][img_x] = val_in;

            wu_shift_win: for (int y = 0; y < WIN_SIZE; y++) {
#pragma HLS UNROLL
                for (int x = 0; x < WIN_SIZE - 1; x++) {
#pragma HLS UNROLL
                    window[y][x] = window[y][x+1];
                }
            }
            wu_update_win: for (int y = 0; y < WIN_SIZE; y++) {
#pragma HLS UNROLL
                window[y][WIN_SIZE - 1] = right[y];
            }
        }
    }

    warmup_col: for (int img_x = 0; img_x < HALF_SIZE; img_x++) {
#pragma HLS PIPELINE II=1
        pixel_t val_in = image[read_count++];

        right[0] = line_buf[0][img_x];
        wc_shift_lb: for (int y = 1; y < WIN_SIZE - 1; y++) {
#pragma HLS UNROLL
            right[y]             = line_buf[y][img_x];
            line_buf[y-1][img_x] = line_buf[y][img_x];
        }
        right[WIN_SIZE - 1]           = val_in;
        line_buf[WIN_SIZE - 2][img_x] = val_in;

        wc_shift_win: for (int y = 0; y < WIN_SIZE; y++) {
#pragma HLS UNROLL
            for (int x = 0; x < WIN_SIZE - 1; x++) {
#pragma HLS UNROLL
                window[y][x] = window[y][x+1];
            }
        }
        wc_update_win: for (int y = 0; y < WIN_SIZE; y++) {
#pragma HLS UNROLL
            window[y][WIN_SIZE - 1] = right[y];
        }
    }
    
    main_y: for (int out_y = 0; out_y < rows; out_y++) {
        main_x: for (int out_x = 0; out_x < cols; out_x++) {
#pragma HLS PIPELINE II=1
            int img_x = out_x + HALF_SIZE;

            pixel_t val_in = (img_x < cols) ? image[read_count++] : (pixel_t)0;
            int bx = (img_x < cols) ? img_x : cols - 1;

            right[0] = line_buf[0][bx];
            main_shift_lb: for (int y = 1; y < WIN_SIZE - 1; y++) {
#pragma HLS UNROLL
                right[y]      = line_buf[y][bx];
                line_buf[y-1][bx] = line_buf[y][bx];
            }
            right[WIN_SIZE - 1]    = val_in;
            line_buf[WIN_SIZE - 2][bx] = val_in;

            main_shift_win: for (int y = 0; y < WIN_SIZE; y++) {
#pragma HLS UNROLL
                for (int x = 0; x < WIN_SIZE - 1; x++) {
#pragma HLS UNROLL
                    window[y][x] = window[y][x+1];
                }
            }
            main_update_win: for (int y = 0; y < WIN_SIZE; y++) {
#pragma HLS UNROLL
                window[y][WIN_SIZE - 1] = right[y];
            }

            grad_pair_t p;
            p.gx = 0;
            p.gy = 0;
            bool border = (out_y == 0 || out_y == rows-1 ||
                           out_x == 0 || out_x == cols-1);
            if (!border) {
                p.gx = (grad_t)window[1][2] - (grad_t)window[1][0];
                p.gy = (grad_t)window[2][1] - (grad_t)window[0][1];
            }
            grad_stream.write(p);
        }
    }
}

void hog_histogram_stream(
    hls::stream<grad_pair_t>& grad_stream,
    hist_t*                   all_hists,
    int rows, int cols
) {
#pragma HLS INLINE off

    mag_t  cell_mag[CELL_SIZE];
    bin_t  cell_bin[CELL_SIZE];
    hist_t cell_hist[NB_BINS];

#pragma HLS ARRAY_PARTITION variable=cell_mag  cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=cell_bin  cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=cell_hist complete

    int num_cells_y = rows / CELL_H;
    int num_cells_x = cols / CELL_W;

    grad_pair_t tile[CELL_H][MAX_IMG_W];
#pragma HLS ARRAY_PARTITION variable=tile cyclic factor=4 dim=2

    cell_y: for (int cy = 0; cy < num_cells_y; cy++) {

        read_rows: for (int py = 0; py < CELL_H; py++) {
            read_cols: for (int x = 0; x < cols; x++) {
#pragma HLS PIPELINE II=1
                tile[py][x] = grad_stream.read();
            }
        }

        cell_x: for (int cx = 0; cx < num_cells_x; cx++) {

            fill_cell: for (int i = 0; i < CELL_SIZE; i++) {
#pragma HLS PIPELINE II=1
                int py        = i / CELL_W;
                int px        = i % CELL_W;
                grad_t gx_val = tile[py][cx * CELL_W + px].gx;
                grad_t gy_val = tile[py][cx * CELL_W + px].gy;
                cell_mag[i]   = (mag_t)(
                    (gx_val < 0 ? (grad_t)(-gx_val) : gx_val) +
                    (gy_val < 0 ? (grad_t)(-gy_val) : gy_val)
                );
                cell_bin[i]   = compute_bin(gx_val, gy_val);
            }

            histogram_cell(cell_mag, cell_bin, cell_hist);

            write_hist: for (int b = 0; b < NB_BINS; b++) {
#pragma HLS PIPELINE II=1
                all_hists[(cy * num_cells_x + cx) * NB_BINS + b] = cell_hist[b];
            }
        }
    }
}

void hog_accel(
    pixel_t* image,
    hist_t*  all_hists,
    int rows, int cols
) {
#pragma HLS INTERFACE m_axi port=image     offset=slave bundle=gmem0 depth=16384 max_read_burst_length=64  num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=all_hists offset=slave bundle=gmem1 depth=9801  max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=image     bundle=CTRL
#pragma HLS INTERFACE s_axilite port=all_hists bundle=CTRL
#pragma HLS INTERFACE s_axilite port=rows      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=cols      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return    bundle=CTRL

    hls::stream<grad_pair_t> grad_stream;
#pragma HLS STREAM variable=grad_stream depth=MAX_IMG_W

#pragma HLS DATAFLOW
    gradient_convolution_stream(image, grad_stream, rows, cols);
    hog_histogram_stream(grad_stream, all_hists, rows, cols);
}