#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <vector>

typedef long long ll;
typedef long double ld;

void parallel_algo(std::vector<std::vector<ld>> &u, const std::vector<std::vector<ld>> &f, int threads, int block_size) {
    omp_set_num_threads(threads);
    int N = f.size();
    int NB = (N + block_size - 1) / block_size; 
    std::vector<ld> dm(NB);
    ld dmax = 0;
    ld h = ld(1) / N;
    ld eps = 5e-3;
    omp_lock_t dmax_lock;
    omp_init_lock(&dmax_lock); 
    do {
        dmax = 0;
        ld prev, d;
        int i, j, k, l;
        for (int nx = 0; nx < NB; ++nx) {
            dm[nx] = 0;
#pragma omp parallel for shared(nx, dm) private(i, j, k, l, prev, d)
            for (i = 0; i < nx + 1; ++i) {
                j = nx - i;
                for (k = std::max(i*block_size, 1); k < std::min((i+1)*block_size, N - 1); ++k) {
                    for (l = std::max(j*block_size, 1); l < std::min((j+1)*block_size, N - 1); ++l) {
                        prev = u[k][l];
                        u[k][l] = 0.25 * (u[k-1][l] + u[k+1][l]+ u[k][l-1] + u[k][l+1] - h*h*f[k][l]);
                        d = fabs(prev - u[k][l]);
                        if (dm[i] < d) {
                            dm[i] = d;
                        }
                    }
                }
            }
        }
        for (int nx = NB - 2; nx > -1; --nx) {
#pragma omp parallel for shared(nx, dm) private(i, j, k, l, prev, d)
            for (i = NB - 1 - nx; i < NB; ++i) {
                j = 2 * (NB - 1) - nx - i;
                for (k = i*block_size; k < std::min((i+1)*block_size, N - 1); ++k) {
                    for (l = j*block_size; l < std::min((j+1)*block_size, N - 1); ++l) {
                        prev = u[k][l];
                        u[k][l] = 0.25 * (u[k-1][l] + u[k+1][l]+ u[k][l-1] + u[k][l+1] - h*h*f[k][l]);
                        d = fabs(prev - u[k][l]);
                        if (dm[i] < d) {
                            dm[i] = d;
                        }
                    }
                }
            }
        }
#pragma omp parallel for shared(dm, dmax) private(i) 
        for (i = 0; i < NB; ++i) {
            omp_set_lock(&dmax_lock);
                if (dmax < dm[i]) {
                    dmax = dm[i];
                }
            omp_unset_lock(&dmax_lock);
        }
    } while (dmax > eps);
}


void consecutive_algo(std::vector<std::vector<ld>> &u, const std::vector<std::vector<ld>> &f, int block_size) {
    int N = f.size();
    int NB = (N + block_size - 1) / block_size; 
    std::vector<ld> dm(NB);
    ld dmax = 0;
    ld h = ld(1) / N;
    ld eps = 5e-3;
    do {
        dmax = 0;
        ld prev, d;
        int i, j, k, l;
        for (int nx = 0; nx < NB; ++nx) {
            dm[nx] = 0;
            for (i = 0; i < nx + 1; ++i) {
                j = nx - i;
                for (k = std::max(i*block_size, 1); k < std::min((i+1)*block_size, N - 1); ++k) {
                    for (l = std::max(j*block_size, 1); l < std::min((j+1)*block_size, N - 1); ++l) {
                        prev = u[k][l];
                        u[k][l] = 0.25 * (u[k-1][l] + u[k+1][l]+ u[k][l-1] + u[k][l+1] - h*h*f[k][l]);
                        d = fabs(prev - u[k][l]);
                        if (dm[i] < d) {
                            dm[i] = d;
                        }
                    }
                }
            }
        }
        for (int nx = NB - 2; nx > -1; --nx) {
            for (i = NB - 1 - nx; i < NB; ++i) {
                j = 2 * (NB - 1) - nx - i;
                for (k = i*block_size; k < std::min((i+1)*block_size, N - 1); ++k) {
                    for (l = j*block_size; l < std::min((j+1)*block_size, N - 1); ++l) {
                        prev = u[k][l];
                        u[k][l] = 0.25 * (u[k-1][l] + u[k+1][l]+ u[k][l-1] + u[k][l+1] - h*h*f[k][l]);
                        d = fabs(prev - u[k][l]);
                        if (dm[i] < d) {
                            dm[i] = d;
                        }
                    }
                }
            }
        }
        for (i = 0; i < NB; ++i) {
                if (dmax < dm[i]) {
                    dmax = dm[i];
                }
        }
    } while (dmax > eps);
}


void init(std::vector<std::vector<ld>> &u, std::vector<std::vector<ld>> &f, int N) {
    u.resize(N + 1);
    f.resize(N + 1);
    for (int i = 0; i < N + 1; ++i) {
        u[i].resize(N + 1);
        f[i].resize(N + 1);
        ld x = ld(i) / (N + 1);
        for (int j = 0; j < N + 1; ++j) {
            ld y = ld(j) / (N + 1);  
            u[i][j] = 5*x + 10*y;
            f[i][j] = 100*x - 1/(y + 0.01) + 1000*(y*y + 3*x);
        }
    }
}


int main() {
    std::vector<std::vector<ld>> u; // values to approximate
    std::vector<std::vector<ld>> f; // given function

    std::vector<int> threads = {1, 4, 8};
    std::vector<int> block_sizes = {1, 32, 64};
    std::vector<int> N_sizes = {500};
    int iters = 5;

    std::ofstream res_file("results.txt");

    for (int N: N_sizes) {
        for (int block_size: block_sizes) {
            for (int num_of_threds: threads) {
                res_file << "N=" << N << ", block_size=" << block_size << ", threads=" << num_of_threds << std::endl;
                for (int i = 0; i < iters; ++i) {
                    init(u, f, N);
                    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
                    if (num_of_threds == 1) {
                        consecutive_algo(u, f, block_size);
                    } else {
                        parallel_algo(u, f, num_of_threds, block_size);
                    }
                    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
                    ld time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000000.0;
                    res_file << "time=" << time_taken << std::endl;
                }
            }
        }
    }
}
