#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime_api.h>
#include <cudnn.h>

#include "timer.h"

#define checkCUDA(err) \
    if (err != cudaSuccess) { \
        printf("[%s:%d] CUDA error %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

#define checkCUDNN(err) \
    if (err != CUDNN_STATUS_SUCCESS) { \
        printf("[%s:%d] CUDNN error %s\n", __FILE__, __LINE__, cudnnGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

void fillData(float *d, int n) {
    for (int i = 0; i < n; ++i) {
        d[i] = rand() % 5 / 4.0;
    }
}

int equalData(float *d0, float *d1, int N, int C, int H, int W) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int x = ((n * C + c) * H + h) * W + w;
                    if ((d0[x] + d1[x] != 0 && fabs((d0[x] - d1[x]) / (d0[x] + d1[x])) > 1e-4) 
                            || (d0[x] + d1[x] == 0 && d0[x] != 0)) {
                        printf("d0 = %f, d1 = %f\n", d0[x], d1[x]);
                        return 0;
                    }
                }
            }
        }
    }
    return 1;
}

void printData(float *d, int N, int C, int H, int W, const char *name) {
    printf("%s.shape = (%d, %d, %d, %d)\n", name, N, C, H, W);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            printf("(%d, %d, :, :) =\n", n, c);
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    printf("%f ", d[((n * C + c) * H + h) * W + w]);
                }
                printf("\n");
            }
        }
    }
}

void convolution_cpu(float *inputs, float *outputs, float *filters, float *dx, float *dw, int N, int C, int H, int W, int K, int P, int Q, int R, int S, int pad) {
    timer_start(0);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int p = 0; p < P; ++p) {
                for (int q = 0; q < Q; ++q) {
                    float x = 0;
                    for (int c = 0; c < C; ++c) {
                        for (int r = 0; r < R; ++r) {
                            for (int s = 0; s < S; ++s) {
                                int h = p + r - pad, w = q + s - pad;
                                if (0 <= h && h < H && 0 <= w && w < W) {
                                    x += inputs[((n * C + c) * H + h) * W + w] * filters[((k * C + c) * R + r) * S + s];
                                }
                            }
                        }
                    }
                    outputs[((n * K + k) * P + p) * Q + q] = x;
                }
            }
        }
    }
    timer_end(0, "cpu fwd");

    timer_start(0);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    float x = 0;
                    for (int k = 0; k < K; ++k) {
                        for (int r = 0; r < R; ++r) {
                            for (int s = 0; s < S; ++s) {
                                int p = h - r + pad, q = w - s + pad;
                                if (0 <= p && p < P && 0 <= q && q < Q) {
                                    x += outputs[((n * K + k) * P + p) * Q + q] * filters[((k * C + c) * R + r) * S + s];
                                }
                            }
                        }
                    }
                    dx[((n * C + c) * H + h) * W + w] = x;
                }
            }
        }
    }
    timer_end(0, "cpu bwd data");

    timer_start(0);
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                for (int s = 0; s < S; ++s) {
                    float x = 0;
                    for (int n = 0; n < N; ++n) {
                        for (int p = 0; p < P; ++p) {
                            for (int q = 0; q < Q; ++q) {
                                int h = p + r - pad, w = q + s - pad;
                                if (0 <= h && h < H && 0 <= w && w < W) {
                                    x += outputs[((n * K + k) * P + p) * Q + q] * inputs[((n * C + c) * H + h) * W + w];
                                }
                            }
                        }
                    }
                    dw[((k * C + c) * R + r) * S + s] = x;
                }
            }
        }
    }
    timer_end(0, "cpu bwd filter");
}

void convolution_cudnn(float *inputs, float *outputs, float *filters, float *dx, float *dw, int N, int C, int H, int W, int K, int P, int Q, int R, int S, int pad, cudnnHandle_t handle) {
    void *inputsDev, *outputsDev, *filtersDev, *dxDev, *dwDev;
    checkCUDA(cudaMalloc(&inputsDev, sizeof(float) * (N * C * H * W)));
    checkCUDA(cudaMalloc(&outputsDev, sizeof(float) * (N * K * P * Q)));
    checkCUDA(cudaMalloc(&filtersDev, sizeof(float) * (K * C * R * S)));
    checkCUDA(cudaMalloc(&dxDev, sizeof(float) * (N * C * H * W)));
    checkCUDA(cudaMalloc(&dwDev, sizeof(float) * (K * C * R * S)));
    checkCUDA(cudaMemcpy(inputsDev, inputs, sizeof(float) * (N * C * H * W), cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(filtersDev, filters, sizeof(float) * (K * C * R * S), cudaMemcpyHostToDevice));

    cudnnTensorDescriptor_t inputsDesc, outputsDesc, dxDesc;
    cudnnFilterDescriptor_t filtersDesc, dwDesc;
    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&inputsDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&outputsDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dxDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&filtersDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&dwDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(inputsDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    checkCUDNN(cudnnSetTensor4dDescriptor(outputsDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, P, Q));
    checkCUDNN(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    checkCUDNN(cudnnSetFilter4dDescriptor(filtersDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S));
    checkCUDNN(cudnnSetFilter4dDescriptor(dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, pad, pad, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));

    size_t wsz;
    void *workspaceDev;
    float alpha = 1, beta = 0;

    // fwd
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, inputsDesc, filtersDesc, convDesc, outputsDesc, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED, &wsz));
    checkCUDA(cudaMalloc(&workspaceDev, wsz));
    checkCUDA(cudaStreamSynchronize(NULL));
    timer_start(0);
    checkCUDNN(cudnnConvolutionForward(handle, &alpha, inputsDesc, inputsDev, filtersDesc, filtersDev, convDesc, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED, workspaceDev, wsz, &beta, outputsDesc, outputsDev));
    checkCUDA(cudaStreamSynchronize(NULL));
    timer_end(0, "cudnn fwd");
    checkCUDA(cudaFree(workspaceDev));

    // bwd data
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filtersDesc, outputsDesc, convDesc, dxDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED, &wsz));
    checkCUDA(cudaMalloc(&workspaceDev, wsz));
    checkCUDA(cudaStreamSynchronize(NULL));
    timer_start(0);
    checkCUDNN(cudnnConvolutionBackwardData(handle, &alpha, filtersDesc, filtersDev, outputsDesc, outputsDev, convDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED, workspaceDev, wsz, &beta, dxDesc, dxDev));
    checkCUDA(cudaStreamSynchronize(NULL));
    timer_end(0, "cudnn bwd data");
    checkCUDA(cudaFree(workspaceDev));

    // bwd filter
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, inputsDesc, outputsDesc, convDesc, dwDesc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED, &wsz));
    checkCUDA(cudaMalloc(&workspaceDev, wsz));
    checkCUDA(cudaStreamSynchronize(NULL));
    timer_start(0);
    checkCUDNN(cudnnConvolutionBackwardFilter(handle, &alpha, inputsDesc, inputsDev, outputsDesc, outputsDev, convDesc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED, workspaceDev, wsz, &beta, dwDesc, dwDev));
    checkCUDA(cudaStreamSynchronize(NULL));
    timer_end(0, "cudnn bwd filter");
    checkCUDA(cudaFree(workspaceDev));

    checkCUDA(cudaMemcpy(outputs, outputsDev, sizeof(float) * (N * K * P * Q), cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(dx, dxDev, sizeof(float) * (N * C * H * W), cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(dw, dwDev, sizeof(float) * (K * C * R * S), cudaMemcpyDeviceToHost));

    checkCUDA(cudaFree(inputsDev));
    checkCUDA(cudaFree(outputsDev));
    checkCUDA(cudaFree(filtersDev));
    checkCUDA(cudaFree(dxDev));
    checkCUDA(cudaFree(dwDev));
    checkCUDNN(cudnnDestroyTensorDescriptor(inputsDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(outputsDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dxDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filtersDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(dwDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
}

void validate(int N, int C, int H, int W, int K, int P, int Q, int R, int S, int pad, cudnnHandle_t handle) {
    float *inputs = (float*)malloc(sizeof(float) * (N * C * H * W));
    float *filters = (float*)malloc(sizeof(float) * (K * C * R * S));
    fillData(inputs, N * C * H * W);
    fillData(filters, K * C * R * S);

    float *outputs_cpu = (float*)malloc(sizeof(float) * (N * K * P * Q));
    float *outputs_cudnn = (float*)malloc(sizeof(float) * (N * K * P * Q));

    float *dx_cpu = (float*)malloc(sizeof(float) * (N * C * H * W));
    float *dx_cudnn = (float*)malloc(sizeof(float) * (N * C * H * W));

    float *dw_cpu = (float*)malloc(sizeof(float) * (K * C * R * S));
    float *dw_cudnn = (float*)malloc(sizeof(float) * (K * C * R * S));

    convolution_cpu(inputs, outputs_cpu, filters, dx_cpu, dw_cpu, N, C, H, W, K, P, Q, R, S, pad);
    for (int i = 0; i < 4; ++i) {
        convolution_cudnn(inputs, outputs_cudnn, filters, dx_cudnn, dw_cudnn, N, C, H, W, K, P, Q, R, S, pad, handle);
    }
    printf("!!!!! cpu == cudnn VALIDATION FWD %s !!!!!\n", equalData(outputs_cpu, outputs_cudnn, N, K, P, Q) ? "SUCCESS" : "FAIL");
    printf("!!!!! cpu == cudnn VALIDATION BWD DATA %s !!!!!\n", equalData(dx_cpu, dx_cudnn, N, C, H, W) ? "SUCCESS" : "FAIL");
    printf("!!!!! cpu == cudnn VALIDATION FWD FILTER %s !!!!!\n", equalData(dw_cpu, dw_cudnn, K, C, R, S) ? "SUCCESS" : "FAIL");

    free(inputs);
    free(filters);
    free(outputs_cpu);
    free(outputs_cudnn);
    free(dx_cpu);
    free(dx_cudnn);
    free(dw_cpu);
    free(dw_cudnn);
}

int main() {
    cudnnHandle_t cudnnHandle;
    checkCUDNN(cudnnCreate(&cudnnHandle));
    //validate(22, 22, 22, 22, 22, 22, 22, 3, 3, 0, cudnnHandle);
    //validate(2, 128, 56, 56, 128, 56, 56, 3, 3, 1, cudnnHandle);
    checkCUDNN(cudnnDestroy(cudnnHandle));
    return 0;
}
