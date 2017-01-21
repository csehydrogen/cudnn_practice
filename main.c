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

void convolution_cpu(float *inputs, float *outputs, float *filters, int N, int C, int H, int W, int K, int P, int Q, int R, int S, int pad) {
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
    timer_end(0, "cpu");
}

void convolution_cudnn(float *inputs, float *outputs, float *filters, int N, int C, int H, int W, int K, int P, int Q, int R, int S, int pad, cudnnHandle_t handle) {
    void *inputsDev, *outputsDev, *filtersDev;
    checkCUDA(cudaMalloc(&inputsDev, sizeof(float) * (N * C * H * W)));
    checkCUDA(cudaMalloc(&outputsDev, sizeof(float) * (N * K * P * Q)));
    checkCUDA(cudaMalloc(&filtersDev, sizeof(float) * (K * C * R * S)));
    checkCUDA(cudaMemcpy(inputsDev, inputs, sizeof(float) * (N * C * H * W), cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(filtersDev, filters, sizeof(float) * (K * C * R * S), cudaMemcpyHostToDevice));

    cudnnTensorDescriptor_t inputsDesc, outputsDesc;
    cudnnFilterDescriptor_t filtersDesc;
    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&inputsDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&outputsDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&filtersDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(inputsDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    checkCUDNN(cudnnSetTensor4dDescriptor(outputsDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, P, Q));
    checkCUDNN(cudnnSetFilter4dDescriptor(filtersDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, pad, pad, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));

    size_t wsz;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, inputsDesc, filtersDesc, convDesc, outputsDesc, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED, &wsz));

    void *workspaceDev;
    checkCUDA(cudaMalloc(&workspaceDev, wsz));

    checkCUDA(cudaStreamSynchronize(NULL));
    timer_start(0);
    float alpha = 1, beta = 0;
    checkCUDNN(cudnnConvolutionForward(handle, &alpha, inputsDesc, inputsDev, filtersDesc, filtersDev, convDesc, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED, workspaceDev, wsz, &beta, outputsDesc, outputsDev));
    checkCUDA(cudaStreamSynchronize(NULL));
    timer_end(0, "cudnn");

    checkCUDA(cudaMemcpy(outputs, outputsDev, sizeof(float) * (N * K * P * Q), cudaMemcpyDeviceToHost));

    checkCUDA(cudaFree(inputsDev));
    checkCUDA(cudaFree(outputsDev));
    checkCUDA(cudaFree(filtersDev));
    checkCUDA(cudaFree(workspaceDev));
    checkCUDNN(cudnnDestroyTensorDescriptor(inputsDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(outputsDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filtersDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
}

void validate(int N, int C, int H, int W, int K, int P, int Q, int R, int S, int pad, cudnnHandle_t handle) {
    float *inputs = (float*)malloc(sizeof(float) * (N * C * H * W));
    float *filters = (float*)malloc(sizeof(float) * (K * C * R * S));
    fillData(inputs, N * C * H * W);
    fillData(filters, K * C * R * S);

    float *outputs_cpu = (float*)malloc(sizeof(float) * (N * K * P * Q));
    float *outputs_cudnn = (float*)malloc(sizeof(float) * (N * K * P * Q));
    //convolution_cpu(inputs, outputs_cpu, filters, N, C, H, W, K, P, Q, R, S, pad);
    for (int i = 0; i < 4; ++i) {
        convolution_cudnn(inputs, outputs_cudnn, filters, N, C, H, W, K, P, Q, R, S, pad, handle);
    }
    printf("!!!!! cpu == cudnn VALIDATION %s !!!!!\n", equalData(outputs_cpu, outputs_cudnn, N, K, P, Q) ? "SUCCESS" : "FAIL");
    //printData(outputs_cpu, N, K, P, Q, "outputs_cpu");
    //printData(outputs_cudnn, N, K, P, Q, "outputs_cudnn");

    free(inputs);
    free(filters);
    free(outputs_cpu);
    free(outputs_cudnn);
}

int main() {
    cudnnHandle_t cudnnHandle;
    checkCUDNN(cudnnCreate(&cudnnHandle));
    //validate(22, 22, 22, 22, 22, 22, 22, 3, 3, 1, cudnnHandle);
    validate(32, 512, 28, 28, 512, 28, 28, 3, 3, 1, cudnnHandle);
    checkCUDNN(cudnnDestroy(cudnnHandle));
    return 0;
}
