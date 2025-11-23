#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

__constant__ float e0 = 8.8541878128e-12;
__constant__ float m0 = 1.25663706212e-06;

#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))
#define INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS) (i)*(NY_FIELDS)*(NZ_FIELDS)+(j)*(NZ_FIELDS)+(k)
#define INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ) ((s)*(NX)*(NY)*(NZ) + (i)*(NY)*(NZ) + (j)*(NZ) + (k))
#define INDEX4D_RXS(s, c, t, rx, NY_RXS, N_ITER, NRX) ((s)*(NY_RXS)*(N_ITER)*(NRX) + (c)*(N_ITER)*(NRX) + (t)*(NRX) + (rx))
#define INDEX3D_RXCOORDS(s, rx, d, NRX, DIM) ((s)*(NRX)*(DIM) + (rx)*(DIM) + (d))

#define INDEX2D_R(m, n,NY_R) (m)*(NY_R)+(n)
#define INDEX3D_R(c, m, n, M, NY_R) ((c) * (M) * (NY_R) + (m) * (NY_R) + (n))

#define INDEX4D_PHI1(p, i, j, k,NX_PHI1,NY_PHI1,NZ_PHI1) (p)*(NX_PHI1)*(NY_PHI1)*(NZ_PHI1)+(i)*(NY_PHI1)*(NZ_PHI1)+(j)*(NZ_PHI1)+(k)
#define INDEX4D_PHI2(p, i, j, k,NX_PHI2,NY_PHI2,NZ_PHI2) (p)*(NX_PHI2)*(NY_PHI2)*(NZ_PHI2)+(i)*(NY_PHI2)*(NZ_PHI2)+(j)*(NZ_PHI2)+(k)

#define CUDA_CHECK() {\
    cudaError_t err = cudaGetLastError();\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
        exit(EXIT_FAILURE);\
    }\
}


__global__ void ucgetforward(const float* __restrict__ er,const float* __restrict__ se,
    const float* __restrict__ mr,
    float* __restrict__ uE0, float* __restrict__ uE1, float* __restrict__ uE4,
    float* __restrict__ uH0, float* __restrict__ uH1, float* __restrict__ uH4,
    int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,float dt,float dx) 
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long i = idx / (NY_FIELDS * NZ_FIELDS);
    long long j = (idx % (NY_FIELDS * NZ_FIELDS)) / NZ_FIELDS;
    long long k = (idx % (NY_FIELDS * NZ_FIELDS)) % NZ_FIELDS;

    if (i < (NX_FIELDS-1) && j < (NY_FIELDS-1) && k < (NZ_FIELDS-1) ) {

        float HA = m0 * mr[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] / dt;
        uH0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 1;
        uH1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = (1 / dx) * 1 / HA;
        uH4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 1 / HA;

        if (se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] > 100) {
            uE0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 0;
            uE1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 0;
            uE4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 0;

        } else {
            float EA = (e0 * er[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] / dt) + 0.5 * se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)];
            float EB = (e0 * er[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] / dt) - 0.5 * se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)];
            uE0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = EB / EA;
            uE1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = (1 / dx) * 1 / EA;
            uE4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 1 / EA;

        }
    }
}


__global__ void ucgetback(const float* __restrict__ er,const float* __restrict__ se,
    const float* __restrict__ mr,
    float* __restrict__ uE0, float* __restrict__ uE1, float* __restrict__ uE4,
    float* __restrict__ uH0, float* __restrict__ uH1, float* __restrict__ uH4,
    int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,float dt,float dx) 
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long i = idx / (NY_FIELDS * NZ_FIELDS);
    long long j = (idx % (NY_FIELDS * NZ_FIELDS)) / NZ_FIELDS;
    long long k = (idx % (NY_FIELDS * NZ_FIELDS)) % NZ_FIELDS;

    if (i < (NX_FIELDS-1) && j < (NY_FIELDS-1) && k < (NZ_FIELDS-1) ) {
        float HA = m0 * mr[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] / dt;
        uH0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 1;
        uH1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = (1 / dx) * 1 / HA;
        uH4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 1 / HA;

        if (se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] > 100) {
            uE0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 0;
            uE1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 0;
            uE4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 0;

        } else {
            float EA = (e0 * er[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] / dt) + 0.5 * se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)];
            uE0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] =(2*e0 * er[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)])/(2*e0 * er[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)]+se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)]*dt);
            uE1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = (1 / dx) * 1 / EA;
            uE4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 1 / EA;
        }
    }
}




__global__ void ucgetbackward(const float* __restrict__ er,const float* __restrict__ se,
    const float* __restrict__ mr,
    float* __restrict__ uE0, float* __restrict__ uE1, float* __restrict__ uE4,
    float* __restrict__ uH0, float* __restrict__ uH1, float* __restrict__ uH4,
    int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,float dt,float dx) 
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    long long i = idx / (NY_FIELDS * NZ_FIELDS);
    long long j = (idx % (NY_FIELDS * NZ_FIELDS)) / NZ_FIELDS;
    long long k = (idx % (NY_FIELDS * NZ_FIELDS)) % NZ_FIELDS;

    if (i < (NX_FIELDS-1) && j < (NY_FIELDS-1) && k < (NZ_FIELDS-1) ) {
        float HA = m0 * mr[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] / dt;
        uH0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 1;
        uH1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = (1 / dx) * 1 / HA;
        uH4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 1 / HA;
        if (se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] > 100) {
            uE0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 0;
            uE1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 0;
            uE4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 0;
        } else {
            float EA = (e0 * er[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] / dt) + 0.5 * se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)];
            uE0[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] =(2*e0 * er[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)])/(2*e0 * er[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)]+se[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)]*dt);
            uE1[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = (1 / dx) * 1 / EA;
            uE4[INDEX3D_FIELDS(i, j, k,NY_FIELDS,NZ_FIELDS)] = 1 / EA;

        }
    }
}



__global__ void store_outputs(
    int step, int NRX, int iteration,
    const int* __restrict__ receiverlocation,
    float* __restrict__ rxs,
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez, const float* __restrict__ Hx,
    const float* __restrict__ Hy, const float* __restrict__ Hz,
    int NX, int NY, int NZ, int N_ITER
) {
    // 全局一维索引
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = step * NRX;
    if (tid >= total) return;

    // 反算 s 和 rx
    long long s  = tid / NRX;    // 时间步
    long long rx = tid % NRX;    // 接收机编号

    // 提取接收机坐标
    long long i = receiverlocation[s * NRX * 3 + rx * 3 + 0];
    long long j = receiverlocation[s * NRX * 3 + rx * 3 + 1];
    long long k = receiverlocation[s * NRX * 3 + rx * 3 + 2];

    // 存储电场
    rxs[((s * 6 + 0) * N_ITER + iteration) * NRX + rx] = Ex[INDEX4D_FIELDS(s,i,j,k,NX,NY,NZ)];
    rxs[((s * 6 + 1) * N_ITER + iteration) * NRX + rx] = Ey[INDEX4D_FIELDS(s,i,j,k,NX,NY,NZ)];
    rxs[((s * 6 + 2) * N_ITER + iteration) * NRX + rx] = Ez[INDEX4D_FIELDS(s,i,j,k,NX,NY,NZ)];

    // 存储磁场
    rxs[((s * 6 + 3) * N_ITER + iteration) * NRX + rx] = Hx[INDEX4D_FIELDS(s,i,j,k,NX,NY,NZ)];
    rxs[((s * 6 + 4) * N_ITER + iteration) * NRX + rx] = Hy[INDEX4D_FIELDS(s,i,j,k,NX,NY,NZ)];
    rxs[((s * 6 + 5) * N_ITER + iteration) * NRX + rx] = Hz[INDEX4D_FIELDS(s,i,j,k,NX,NY,NZ)];
}





__global__ void Update_hertzian_dipole(
    int step, int iteration, float dx, 
    const int* __restrict__ sourcelocation, const float* __restrict__ srcwaveforms,
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez, const float* __restrict__ uE4,
    int NX, int NY, int NZ, int nsrc, int polarisation,int nt
) {
    long long src = blockIdx.x * blockDim.x + threadIdx.x; // 对应源维度
    long long s = blockIdx.y * blockDim.y + threadIdx.y;   // 对应 step 维度

    if (src < nsrc && s < step) {
        // 从 sourcelocation 中获取源位置信息 (i, j, k)
        long long i = sourcelocation[s * nsrc * 3 + src * 3 + 0];
        long long j = sourcelocation[s * nsrc * 3 + src * 3 + 1];
        long long k = sourcelocation[s * nsrc * 3 + src * 3 + 2];

        float dl = dx; // 每个源可能有不同的 dl 值
        float waveform_value = srcwaveforms[src * nt + iteration];// 获取第 src 个源在当前迭代的波形值
        float scale = waveform_value * dl / (dx * dx * dx);   //printf("%d %d %f \n",src,s,scale);
        // printf("%d %d %d %d %f %f %f\n",s,i,j,k,dl,waveform_value,scale);
// printf("%f %f %f\n",dl,waveform_value,scale);

            if (polarisation == 0) {
                Ex[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] -= uE4[INDEX3D_FIELDS(i, j, k, NY, NZ)] * scale;
            }
            else if (polarisation == 1) {
                Ey[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] -= uE4[INDEX3D_FIELDS(i, j, k, NY, NZ)] * scale;
            }
            else if (polarisation == 2){
                Ez[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] -= uE4[INDEX3D_FIELDS(i, j, k, NY, NZ)] * scale;
            }
        }
    }




__global__ void e_fields_updates_gpu(
    const float* __restrict__ uE0, const float* __restrict__ uE1,  
     float* __restrict__ Ex,  float* __restrict__ Ey,
     float* __restrict__ Ez,  const float* __restrict__ Hx,
     const float* __restrict__ Hy,  const float* __restrict__ Hz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS
) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_cells_per_step = NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    // int s = idx / total_cells_per_step;          // step 索引
    // int remainder = idx % total_cells_per_step;  // step 内的线性索引
    // int i = remainder / (NY_FIELDS * NZ_FIELDS);
    // int j = (remainder / NZ_FIELDS) % NY_FIELDS;
    // int k = remainder % NZ_FIELDS;

    long long s = idx / total_cells_per_step;
    long long i = idx % total_cells_per_step / (NY_FIELDS * NZ_FIELDS);
    long long j = (idx % total_cells_per_step % (NY_FIELDS * NZ_FIELDS)) / NZ_FIELDS;
    long long k = idx % total_cells_per_step % NZ_FIELDS;

    size_t total = step * NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    if (idx >= total) return;
    // // Ex 
    if (((NY_FIELDS-1) != 1 || (NZ_FIELDS-1) != 1)  && s < step && i >= 0 && i < (NX_FIELDS-1) && j > 0 && j < (NY_FIELDS-1) && k > 0 && k < (NZ_FIELDS-1)) {
        Ex[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] = 
            uE0[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] * 
            Ex[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] + 
            uE1[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            (Hz[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] - 
            Hz[INDEX4D_FIELDS(s, i, j-1, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)]) - 
            uE1[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            (Hy[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] - 
            Hy[INDEX4D_FIELDS(s, i, j, k-1, NX_FIELDS, NY_FIELDS, NZ_FIELDS)]);
    }

    // Ey 
    if (((NX_FIELDS-1) != 1 || (NZ_FIELDS-1) != 1) && s < step && i > 0 && i < (NX_FIELDS-1) && j >= 0 && j < (NY_FIELDS-1) && k > 0 && k < (NZ_FIELDS-1)) {

        Ey[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] = 
            uE0[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            Ey[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] + 
            uE1[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            (Hx[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] - 
            Hx[INDEX4D_FIELDS(s, i, j, k-1, NX_FIELDS, NY_FIELDS, NZ_FIELDS)]) - 
            uE1[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            (Hz[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] - 
            Hz[INDEX4D_FIELDS(s, i-1, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)]);
    }

    //Ez
    if (((NX_FIELDS-1) != 1 || (NY_FIELDS-1) != 1) && s < step && i > 0 && i < (NX_FIELDS-1) && j > 0 && j < (NY_FIELDS-1) && k >= 0 && k < (NZ_FIELDS-1)) {

        Ez[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] =
            uE0[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            Ez[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] +
            uE1[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            (Hy[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] -
             Hy[INDEX4D_FIELDS(s, i - 1, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)]) -
            uE1[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            (Hx[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] -
             Hx[INDEX4D_FIELDS(s, i, j - 1, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)]);
    }
}



__global__ void h_fields_updates_gpu(
    const float* __restrict__ uH0, const float* __restrict__ uH1,      
    const float* __restrict__ Ex,  const float* __restrict__ Ey,
     const float* __restrict__ Ez,  float* __restrict__ Hx,
     float* __restrict__ Hy,  float* __restrict__ Hz,
    int step, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total_cells_per_step = NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    long long s = idx / total_cells_per_step;
    long long i = idx % total_cells_per_step / (NY_FIELDS * NZ_FIELDS);
    long long j = (idx % total_cells_per_step % (NY_FIELDS * NZ_FIELDS)) / NZ_FIELDS;
    long long k = idx % total_cells_per_step % NZ_FIELDS;

    long long total = step * NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    if (idx >= total) return;
    // Hx 
    if ((NX_FIELDS-1) != 1 && s < step && i > 0 && i < (NX_FIELDS-1) && j >= 0 && j < (NY_FIELDS-1) && k >= 0 && k < (NZ_FIELDS-1)) {
        Hx[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] =
            uH0[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            Hx[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] -
            uH1[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            (Ez[INDEX4D_FIELDS(s, i, j + 1, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] -
             Ez[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)])+
             uH1[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            (Ey[INDEX4D_FIELDS(s, i, j , k+1, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] -
             Ey[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)]);

    }

    // Hy 
    if ((NY_FIELDS-1) != 1 && s < step && i >= 0 && i < (NX_FIELDS-1) && j > 0 && j < (NY_FIELDS-1) && k >= 0 && k < (NZ_FIELDS-1)) {

        Hy[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] =
            uH0[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            Hy[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] -
            uH1[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            (Ex[INDEX4D_FIELDS(s, i , j, k+1, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] -
             Ex[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)])+
            uH1[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            (Ez[INDEX4D_FIELDS(s, i + 1, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] -
             Ez[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)]);
    }

    // Hz 
    if ((NZ_FIELDS-1) != 1 && s < step  && i >= 0 && i < (NX_FIELDS-1) && j >= 0 && j < (NY_FIELDS-1) && k > 0 && k < (NZ_FIELDS-1)) {
        Hz[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] = 
            uH0[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            Hz[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] - 
            uH1[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            (Ey[INDEX4D_FIELDS(s, i+1, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] - 
            Ey[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)]) + 
            uH1[INDEX3D_FIELDS(i,j,k,NY_FIELDS,NZ_FIELDS)] *
            (Ex[INDEX4D_FIELDS(s, i, j+1, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)] - 
            Ex[INDEX4D_FIELDS(s, i, j, k, NX_FIELDS, NY_FIELDS, NZ_FIELDS)]);
    }

}






__global__ void x0H(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R,   float* __restrict__ Ex,   float* __restrict__ Ey,   float* __restrict__ Ez,   float* __restrict__ Hx,  float* __restrict__ Hy,  float* __restrict__ Hz,  float *PHI1,  float *PHI2,   const float* __restrict__ R,  float dx ,const float* __restrict__ updatecoeffsH, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step) {

    // Obtain the linear index corresponding to the current tREad
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    long long p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    long long i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    long long j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    long long k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    long long p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    long long i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    long long j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    long long k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    float RA01, RB0, RE0, RF0, dEy, dEz;
    long long ii, jj, kk;
    long long nx = xf - xs;
    long long ny = yf - ys;
    long long nz = zf - zs;
// int total1 = step * NX_PHI1 * NY_PHI1 * NZ_PHI1;
// int total2 = step * NX_PHI2 * NY_PHI2 * NZ_PHI2;
// int total  = max(total1, total2);
// if (idx >= total) return;
    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = xf - (i1 + 1);
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        // RA01 = RA[INDEX2D_R(0,i1,NY_R)] - 1;
        // RB0 = RB[INDEX2D_R(0,i1,NY_R)];
        // RE0 = RE[INDEX2D_R(0,i1,NY_R)];
        // RF0 = RF[INDEX2D_R(0,i1,NY_R)];

        long long m = 0; // 目前你还是只有一层，就写0，将来可以改
        RA01 = R[INDEX3D_R(0,m,i1,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,i1,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,i1,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,i1,1,NY_R)];      // RF
        // Hy
        dEz = (Ez[INDEX4D_FIELDS(p1,ii+1,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ez[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dx;

        Hy[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hy[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsH[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEz + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);

        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dEz;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = xf - (i2 + 1);
        jj = j2 + ys;
        kk = k2 + zs;
        long long m=0;
        // PML coefficients
        // RA01 = RA[INDEX2D_R(0,i2,NY_R)] - 1;
        // RB0 = RB[INDEX2D_R(0,i2,NY_R)];
        // RE0 = RE[INDEX2D_R(0,i2,NY_R)];
        // RF0 = RF[INDEX2D_R(0,i2,NY_R)];
        RA01 = R[INDEX3D_R(0,m,i2,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,i2,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,i2,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,i2,1,NY_R)];      // RF
        // Hz

        dEy = (Ey[INDEX4D_FIELDS(p2,ii+1,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ey[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dx;

        Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsH[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEy + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);

        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dEy;


        // printf("(%d %d %d %d) \n",p2,i2,j2,k2);

    }
}




__global__ void xmH(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R,   float* __restrict__ Ex,   float* __restrict__ Ey,   float* __restrict__ Ez,   float* __restrict__ Hx,  float *Hy,  float *Hz,  float *PHI1,  float *PHI2,   const float* __restrict__ R,  float d,float *updatecoeffsH, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step) {


    // Obtain the linear index corresponding to the current tREad
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    long long p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    long long i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    long long j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    long long k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    long long p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    long long i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    long long j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    long long k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

     float RA01, RB0, RE0, RF0, dEy, dEz;
     float dx = d;
    long long ii, jj, kk;
    long long nx = xf - xs;
    long long ny = yf - ys;
    long long nz = zf - zs;
    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,i1,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,i1,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,i1,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,i1,1,NY_R)];      // RF
        // Hy
         
        dEz = (Ez[INDEX4D_FIELDS(p1,ii+1,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ez[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dx;
        Hy[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hy[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsH[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEz + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);

        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dEz;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,i2,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,i2,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,i2,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,i2,1,NY_R)];      // RF
        // Hz
         
        dEy = (Ey[INDEX4D_FIELDS(p2,ii+1,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ey[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dx;
        Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsH[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEy + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);

        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dEy;
    }
}





__global__ void y0H(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R,    float* __restrict__ Ex,   float* __restrict__ Ey,   float* __restrict__ Ez,  float* __restrict__ Hx,   float* __restrict__ Hy,  float* __restrict__ Hz,  float *PHI1,  float *PHI2,  const float* __restrict__ R,  float d,float* __restrict__ updatecoeffsH, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step) {

    
    // Obtain the linear index corresponding to the current tREad
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    long long p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    long long i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    long long j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    long long k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    long long p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    long long i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    long long j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    long long k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

     float RA01, RB0, RE0, RF0, dEx, dEz;
     float dy = d;
    long long ii, jj, kk;
    long long nx = xf - xs;
    long long ny = yf - ys;
    long long nz = zf - zs;

    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = yf - (j1 + 1);
        kk = k1 + zs;
        // printf("(%d %d %d %d) \n",p1,ii,jj,kk);
        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,j1,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,j1,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,j1,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,j1,1,NY_R)];      // RF
        // Hx
         
        dEz = (Ez[INDEX4D_FIELDS(p1,ii,jj+1,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ez[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dy;
        Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsH[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEz + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);
        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dEz;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {
        ii = i2 + xs;
        jj = yf - (j2 + 1);
        kk = k2 + zs;

        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,j2,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,j2,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,j2,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,j2,1,NY_R)];      // RF
        // Hz
         
        dEx = (Ex[INDEX4D_FIELDS(p2,ii,jj+1,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ex[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dy;
        Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsH[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEx + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);
        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dEx;
    }
}




__global__ void ymH(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R,  float* __restrict__ Ex,   float* __restrict__ Ey,   float* __restrict__ Ez,  float* __restrict__ Hx,   float* __restrict__ Hy,  float* __restrict__ Hz,  float *PHI1,  float *PHI2, const float* __restrict__ R,  float d, float* __restrict__ updatecoeffsH, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step) {

    // Obtain the linear index corresponding to the current tREad
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    long long p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    long long i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    long long j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    long long k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    long long p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    long long i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    long long j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    long long k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

     float RA01, RB0, RE0, RF0, dEx, dEz;
     float dy = d;
    long long ii, jj, kk;
    long long nx = xf - xs;
    long long ny = yf - ys;
    long long nz = zf - zs;
    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,j1,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,j1,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,j1,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,j1,1,NY_R)];      // RF
        // Hx
         
        dEz = (Ez[INDEX4D_FIELDS(p1,ii,jj+1,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ez[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dy;
        Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsH[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEz + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);
        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dEz;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {

        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;


        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,j2,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,j2,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,j2,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,j2,1,NY_R)];      // RF
        // Hz
         
        dEx = (Ex[INDEX4D_FIELDS(p2,ii,jj+1,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ex[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dy;
        Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hz[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsH[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEx + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);
        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dEx;
    }
}





__global__ void z0H(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R,    float* __restrict__ Ex,   float* __restrict__ Ey,   float* __restrict__ Ez,  float *Hx,  float *Hy,   float* __restrict__ Hz,  float *PHI1,  float *PHI2,  const float* __restrict__ R,  float d,float *updatecoeffsH, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step) {

    // Obtain the linear index corresponding to the current tREad
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    long long p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    long long i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    long long j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    long long k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    long long p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    long long i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    long long j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    long long k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

     float RA01, RB0, RE0, RF0, dEx, dEy;
     float dz = d;
    long long ii, jj, kk;
    long long nx = xf - xs;
    long long ny = yf - ys;
    long long nz = zf - zs;
// long long total1 = step * NX_PHI1 * NY_PHI1 * NZ_PHI1;
// long long total2 = step * NX_PHI2 * NY_PHI2 * NZ_PHI2;
// long long total  = max(total1, total2);
// if (idx >= total) return;
    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = zf - (k1 + 1);

        // PML coefficients
        // RA01 = RA[INDEX2D_R(0,k1,NY_R)] - 1;
        // RB0 = RB[INDEX2D_R(0,k1,NY_R)];
        // RE0 = RE[INDEX2D_R(0,k1,NY_R)];
        // RF0 = RF[INDEX2D_R(0,k1,NY_R)];
        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,k1,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,k1,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,k1,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,k1,1,NY_R)];      // RF
        // Hx
         
        dEy = (Ey[INDEX4D_FIELDS(p1,ii,jj,kk+1,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ey[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dz;
        Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsH[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEy + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);
        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dEy;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {

        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = zf - (k2 + 1);

        // PML coefficients
        // RA01 = RA[INDEX2D_R(0,k2,NY_R)] - 1;
        // RB0 = RB[INDEX2D_R(0,k2,NY_R)];
        // RE0 = RE[INDEX2D_R(0,k2,NY_R)];
        // RF0 = RF[INDEX2D_R(0,k2,NY_R)];
        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,k2,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,k2,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,k2,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,k2,1,NY_R)];      // RF
        // Hy
         
        dEx = (Ex[INDEX4D_FIELDS(p2,ii,jj,kk+1,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ex[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dz;
        Hy[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hy[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsH[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEx + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);
        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dEx;
    }
}


__global__ void zmH(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R,    float* __restrict__ Ex,   float* __restrict__ Ey,   float* __restrict__ Ez,  float *Hx,  float *Hy,   float* __restrict__ Hz,  float *PHI1,  float *PHI2,  const float* __restrict__ R,  float d,float *updatecoeffsH, int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,int step) {


    // Obtain the linear index corresponding to the current tREad
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    long long p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    long long i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    long long j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    long long k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    long long p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    long long i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    long long j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    long long k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

     float RA01, RB0, RE0, RF0, dEx, dEy;
     float dz = d;
    long long ii, jj, kk;
    long long nx = xf - xs;
    long long ny = yf - ys;
    long long nz = zf - zs;
// long long total1 = step * NX_PHI1 * NY_PHI1 * NZ_PHI1;
// long long total2 = step * NX_PHI2 * NY_PHI2 * NZ_PHI2;
// long long total  = max(total1, total2);
// if (idx >= total) return;
    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        // RA01 = RA[INDEX2D_R(0,k1,NY_R)] - 1;
        // RB0 = RB[INDEX2D_R(0,k1,NY_R)];
        // RE0 = RE[INDEX2D_R(0,k1,NY_R)];
        // RF0 = RF[INDEX2D_R(0,k1,NY_R)];
        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,k1,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,k1,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,k1,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,k1,1,NY_R)];      // RF
        // Hx
         
        dEy = (Ey[INDEX4D_FIELDS(p1,ii,jj,kk+1,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ey[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dz;
        Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hx[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsH[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEy + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);
        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dEy;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {
      // printf("1");
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        // RA01 = RA[INDEX2D_R(0,k2,NY_R)] - 1;
        // RB0 = RB[INDEX2D_R(0,k2,NY_R)];
        // RE0 = RE[INDEX2D_R(0,k2,NY_R)];
        // RF0 = RF[INDEX2D_R(0,k2,NY_R)];
        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,k2,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,k2,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,k2,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,k2,1,NY_R)];      // RF
        // Hy
         
        dEx = (Ex[INDEX4D_FIELDS(p2,ii,jj,kk+1,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Ex[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dz;
        Hy[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Hy[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsH[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dEx + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);
        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dEx;
    }
}




__global__ void x0E(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1,
int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R,  float* __restrict__ Ex, float *Ey, float *Ez,  float* __restrict__ Hx,
float* __restrict__ Hy,  float* __restrict__ Hz, float *PHI1, float *PHI2, const float* __restrict__ R, float d,float *updatecoeffsE, int NX_FIELDS, int NY_FIELDS,  int NZ_FIELDS,int step) {
    
    // Obtain the linear index corresponding to the current thread
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    long long p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    long long i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    long long j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    long long k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    long long p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    long long i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    long long j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    long long k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    float RA01, RB0, RE0, RF0, dHy, dHz;
    float dx = d;
    long long ii, jj, kk ;
    long long nx = xf - xs;
    long long ny = yf - ys;
    long long nz = zf - zs;
// int total1 = step * NX_PHI1 * NY_PHI1 * NZ_PHI1;
// int total2 = step * NX_PHI2 * NY_PHI2 * NZ_PHI2;
// int total  = max(total1, total2);
// if (idx >= total) return;
    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = xf - i1;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        // RA01 = RA[INDEX2D_R(0,i1,NY_R)] - 1;
        // RB0 = RB[INDEX2D_R(0,i1,NY_R)];
        // RE0 = RE[INDEX2D_R(0,i1,NY_R)];
        // RF0 = RF[INDEX2D_R(0,i1,NY_R)];
        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,i1,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,i1,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,i1,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,i1,1,NY_R)];      // RF
        // Ey

        dHz = (Hz[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Hz[INDEX4D_FIELDS(p1,ii-1,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dx;
        Ey[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Ey[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsE[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dHz + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);
        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dHz;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {

        // Subscripts for field arrays
        ii = xf - i2;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        // RA01 = RA[INDEX2D_R(0,i2,NY_R)] - 1;
        // RB0 = RB[INDEX2D_R(0,i2,NY_R)];
        // RE0 = RE[INDEX2D_R(0,i2,NY_R)];
        // RF0 = RF[INDEX2D_R(0,i2,NY_R)];
        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,i2,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,i2,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,i2,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,i2,1,NY_R)];      // RF
        // Ez
          
        dHy = (Hy[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Hy[INDEX4D_FIELDS(p2,ii-1,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dx;
        Ez[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Ez[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsE[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dHy + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);
        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dHy;
    }
}






__global__ void xmE(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R,  float* __restrict__ Ex, float *Ey, float *Ez,  float* __restrict__ Hx,  float* __restrict__ Hy,  float* __restrict__ Hz, float *PHI1, float *PHI2, const float* __restrict__ R, float d,float *updatecoeffsE, int NX_FIELDS, int NY_FIELDS,  int NZ_FIELDS,int step) {


    // Obtain the linear index corresponding to the current thread
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    long long p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    long long i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    long long j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    long long k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    long long p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    long long i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    long long j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    long long k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    float RA01, RB0, RE0, RF0, dHy, dHz;
    float dx = d;
    long long ii, jj, kk ;
    long long nx = xf - xs;
    long long ny = yf - ys;
    long long nz = zf - zs;

    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,i1,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,i1,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,i1,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,i1,1,NY_R)];      // RF
        // Ey

        dHz = (Hz[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Hz[INDEX4D_FIELDS(p1,ii-1,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dx;

        Ey[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Ey[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsE[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dHz + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);
        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dHz;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,i2,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,i2,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,i2,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,i2,1,NY_R)];      // RF
        // Ez
          
        dHy = (Hy[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Hy[INDEX4D_FIELDS(p2,ii-1,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dx;

        Ez[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Ez[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsE[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dHy + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);
        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dHy;
    }
}



__global__ void y0E(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, float *Ex,  float* __restrict__ Ey, float *Ez,  float* __restrict__ Hx,  float* __restrict__ Hy,  float* __restrict__ Hz, float *PHI1, float *PHI2, const float* __restrict__ R, float d,float* __restrict__ updatecoeffsE, int NX_FIELDS, int NY_FIELDS,  int NZ_FIELDS,int step) {

    // Obtain the linear index corresponding to the current thread
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    long long p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    long long i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    long long j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    long long k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    long long p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    long long i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    long long j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    long long k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    float RA01, RB0, RE0, RF0, dHx, dHz;
    float dy = d;
    long long ii, jj, kk ;
    long long nx = xf - xs;
    long long ny = yf - ys;
    long long nz = zf - zs;

    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = yf - j1;
        kk = k1 + zs;

        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,j1,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,j1,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,j1,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,j1,1,NY_R)];      // RF
        // Ex

        dHz = (Hz[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Hz[INDEX4D_FIELDS(p1,ii,jj-1,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dy;

        Ex[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Ex[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsE[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dHz + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);
        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dHz;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {

        ii = i2 + xs;
        jj = yf - j2;
        kk = k2 + zs;
        
        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,j2,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,j2,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,j2,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,j2,1,NY_R)];      // RF
        // Ez
          
        dHx = (Hx[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Hx[INDEX4D_FIELDS(p2,ii,jj-1,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dy;
        Ez[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Ez[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsE[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dHx + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);
        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dHx;
    }
}



__global__ void ymE(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, float *Ex,  float* __restrict__ Ey, float *Ez,  float* __restrict__ Hx,  float* __restrict__ Hy,  float* __restrict__ Hz, float *PHI1, float *PHI2, const float* __restrict__ R, float d,float *updatecoeffsE, int NX_FIELDS, int NY_FIELDS,  int NZ_FIELDS,int step) {

    // Obtain the linear index corresponding to the current thread
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    long long p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    long long i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    long long j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    long long k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    long long p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    long long i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    long long j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    long long k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    float RA01, RB0, RE0, RF0, dHx, dHz;
    float dy = d;
    long long ii, jj, kk ;
    long long nx = xf - xs;
    long long ny = yf - ys;
    long long nz = zf - zs;
// int total1 = step * NX_PHI1 * NY_PHI1 * NZ_PHI1;
// int total2 = step * NX_PHI2 * NY_PHI2 * NZ_PHI2;
// int total  = max(total1, total2);
// if (idx >= total) return;
    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        // PML coefficients
        // RA01 = RA[INDEX2D_R(0,j1,NY_R)] - 1;
        // RB0 = RB[INDEX2D_R(0,j1,NY_R)];
        // RE0 = RE[INDEX2D_R(0,j1,NY_R)];
        // RF0 = RF[INDEX2D_R(0,j1,NY_R)];
        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,j1,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,j1,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,j1,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,j1,1,NY_R)];      // RF
        // Ex

        dHz = (Hz[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Hz[INDEX4D_FIELDS(p1,ii,jj-1,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dy;

        Ex[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Ex[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsE[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dHz + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);
        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dHz;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        // PML coefficients
        // RA01 = RA[INDEX2D_R(0,j2,NY_R)] - 1;
        // RB0 = RB[INDEX2D_R(0,j2,NY_R)];
        // RE0 = RE[INDEX2D_R(0,j2,NY_R)];
        // RF0 = RF[INDEX2D_R(0,j2,NY_R)];
        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,j2,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,j2,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,j2,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,j2,1,NY_R)];      // RF
        // Ez
          
        dHx = (Hx[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Hx[INDEX4D_FIELDS(p2,ii,jj-1,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dy;
        Ez[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Ez[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsE[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dHx + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);
        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dHx;
    }
}




__global__ void z0E(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, float *Ex, float *Ey,  float* __restrict__ Ez,  float* __restrict__ Hx,  float* __restrict__ Hy,  float* __restrict__ Hz, float *PHI1, float *PHI2, const float* __restrict__ R, float d,float *updatecoeffsE, int NX_FIELDS, int NY_FIELDS,  int NZ_FIELDS,int step) {

    // Obtain the linear index corresponding to the current thread
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    long long p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    long long i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    long long j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    long long k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    long long p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    long long i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    long long j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    long long k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    float RA01, RB0, RE0, RF0, dHx, dHy;
    float dz = d;
    long long ii, jj, kk ;
    long long nx = xf - xs;
    long long ny = yf - ys;
    long long nz = zf - zs;
// long long total1 = step * NX_PHI1 * NY_PHI1 * NZ_PHI1;
// int total2 = step * NX_PHI2 * NY_PHI2 * NZ_PHI2;
// int total  = max(total1, total2);
// if (idx >= total) return;
    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = zf - k1;

        // PML coefficients
        // RA01 = RA[INDEX2D_R(0,k1,NY_R)] - 1;
        // RB0 = RB[INDEX2D_R(0,k1,NY_R)];
        // RE0 = RE[INDEX2D_R(0,k1,NY_R)];
        // RF0 = RF[INDEX2D_R(0,k1,NY_R)];
        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,k1,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,k1,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,k1,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,k1,1,NY_R)];      // RF
        // Ex

        dHy = (Hy[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Hy[INDEX4D_FIELDS(p1,ii,jj,kk-1,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dz;

        Ex[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Ex[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsE[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dHy + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);
        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dHy;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {
        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = zf - k2;

        // PML coefficients
        // RA01 = RA[INDEX2D_R(0,k2,NY_R)] - 1;
        // RB0 = RB[INDEX2D_R(0,k2,NY_R)];
        // RE0 = RE[INDEX2D_R(0,k2,NY_R)];
        // RF0 = RF[INDEX2D_R(0,k2,NY_R)];
        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,k2,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,k2,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,k2,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,k2,1,NY_R)];      // RF
        // Ey
        dHx = (Hx[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Hx[INDEX4D_FIELDS(p2,ii,jj,kk-1,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dz;
        Ey[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Ey[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsE[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dHx + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);
        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dHx;
    }
}


__global__ void zmE(int xs, int xf, int ys, int yf, int zs, int zf, int NX_PHI1, int NY_PHI1, int NZ_PHI1, int NX_PHI2, int NY_PHI2, int NZ_PHI2, int NY_R, float *Ex, float *Ey,  float* __restrict__ Ez,  float* __restrict__ Hx,  float* __restrict__ Hy,  float* __restrict__ Hz, float *PHI1, float *PHI2, const float* __restrict__ R, float d,float *updatecoeffsE, int NX_FIELDS, int NY_FIELDS,  int NZ_FIELDS,int step) {


    // Obtain the linear index corresponding to the current thread
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for PML PHI1 (4D) arrays
    long long p1 = idx / (NX_PHI1 * NY_PHI1 * NZ_PHI1);
    long long i1 = (idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) / (NY_PHI1 * NZ_PHI1);
    long long j1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) / NZ_PHI1;
    long long k1 = ((idx % (NX_PHI1 * NY_PHI1 * NZ_PHI1)) % (NY_PHI1 * NZ_PHI1)) % NZ_PHI1;

    // Convert the linear index to subscripts for PML PHI2 (4D) arrays
    long long p2 = idx / (NX_PHI2 * NY_PHI2 * NZ_PHI2);
    long long i2 = (idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) / (NY_PHI2 * NZ_PHI2);
    long long j2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) / NZ_PHI2;
    long long k2 = ((idx % (NX_PHI2 * NY_PHI2 * NZ_PHI2)) % (NY_PHI2 * NZ_PHI2)) % NZ_PHI2;

    float RA01, RB0, RE0, RF0, dHx, dHy;
    float dz = d;
    long long ii, jj, kk ;
    long long nx = xf - xs;
    long long ny = yf - ys;
    long long nz = zf - zs;
    if (p1 <step && i1 < nx && j1 < ny && k1 < nz) {
        // Subscripts for field arrays
        ii = i1 + xs;
        jj = j1 + ys;
        kk = k1 + zs;

        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,k1,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,k1,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,k1,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,k1,1,NY_R)];      // RF
        // Ex

        dHy = (Hy[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Hy[INDEX4D_FIELDS(p1,ii,jj,kk-1,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dz;

        Ex[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Ex[INDEX4D_FIELDS(p1,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - updatecoeffsE[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dHy + RB0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)]);
        PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] = RE0 * PHI1[INDEX4D_PHI1(p1,i1,j1,k1,NX_PHI1,NY_PHI1,NZ_PHI1)] - RF0 * dHy;
    }

    if (p2 <step && i2 < nx && j2 < ny && k2 < nz) {

        // Subscripts for field arrays
        ii = i2 + xs;
        jj = j2 + ys;
        kk = k2 + zs;

        long long m = 0; 
        RA01 = R[INDEX3D_R(0,m,k2,1,NY_R)] - 1;  // RA
        RB0  = R[INDEX3D_R(1,m,k2,1,NY_R)];      // RB
        RE0  = R[INDEX3D_R(2,m,k2,1,NY_R)];      // RE
        RF0  = R[INDEX3D_R(3,m,k2,1,NY_R)];      // RF
        // Ey

        dHx = (Hx[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] - Hx[INDEX4D_FIELDS(p2,ii,jj,kk-1,NX_FIELDS,NY_FIELDS,NZ_FIELDS)]) / dz;
        Ey[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] = Ey[INDEX4D_FIELDS(p2,ii,jj,kk,NX_FIELDS,NY_FIELDS,NZ_FIELDS)] + updatecoeffsE[INDEX3D_FIELDS(ii,jj,kk,NY_FIELDS,NZ_FIELDS)] * (RA01 * dHx + RB0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)]);
        PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] = RE0 * PHI2[INDEX4D_PHI2(p2,i2,j2,k2,NX_PHI2,NY_PHI2,NZ_PHI2)] - RF0 * dHx;
    }
}



__global__ void Back_source(
    int step, int iteration, float dx,
    const int* __restrict__ sourcelocation,
    const float* __restrict__ srcwaveforms,
    float* Ex, float* Ey, float* Ez, float* uE4,
    int NX, int NY, int NZ, int nsr, int polarisation, int iterations
){
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;   // 一维全局索引
    if (tid >= step * nsr) return;

    // 由tid反推原来的(s, src)
    long long s   = tid / nsr;   // step 维度
    long long src = tid % nsr;   // 源维度

    // 计算源波形索引
    long long index = s * (iterations * nsr) + iteration * nsr + src;

    // 获取源位置 (i, j, k)
    long long i = sourcelocation[s * nsr * 3 + src * 3 + 0];
    long long j = sourcelocation[s * nsr * 3 + src * 3 + 1];
    long long k = sourcelocation[s * nsr * 3 + src * 3 + 2];

    float waveform_value = srcwaveforms[index];

    // 调试打印（需要时保留）
    // printf("%d %d %d %d %.10lf\n", s, i, j, src, waveform_value);

    // 根据极化方向注入
    if (polarisation == 0) {
        Ex[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] -= waveform_value;
    } else if (polarisation == 1) {
        Ey[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] -= waveform_value;
    } else if (polarisation == 2) {
        Ez[INDEX4D_FIELDS(s, i, j, k, NX, NY, NZ)] -= waveform_value;
    }
}




#include <cfloat>

__global__ void check_nan_inf6_print(
    const float* Ex, const float* Ey, const float* Ez,
    const float* Hx, const float* Hy, const float* Hz,
    int NX, int NY, int NZ, int NZ1)
{
    long long total = NX * NY * NZ;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float max_val = -FLT_MAX;
        float min_val =  FLT_MAX;

        for (long long idx = 0; idx < total; idx++) {
            float ex = Ex[idx], ey = Ey[idx], ez = Ez[idx];
            float hx = Hx[idx], hy = Hy[idx], hz = Hz[idx];

            // 检查 NaN/Inf
            // if (!isfinite(ex)) printf("NaN/Inf %d in Ex[%d]: %f\n", NZ1, idx, ex);
            // if (!isfinite(ey)) printf("NaN/Inf %d in Ey[%d]: %f\n", NZ1, idx, ey);
            // if (!isfinite(ez)) printf("NaN/Inf %d in Ez[%d]: %f\n", NZ1, idx, ez);
            // if (!isfinite(hx)) printf("NaN/Inf %d in Hx[%d]: %f\n", NZ1, idx, hx);
            // if (!isfinite(hy)) printf("NaN/Inf %d in Hy[%d]: %f\n", NZ1, idx, hy);
            // if (!isfinite(hz)) printf("NaN/Inf %d in Hz[%d]: %f\n", NZ1, idx, hz);

            // 逐个更新最大最小
            max_val = fmaxf(max_val, ex);
            max_val = fmaxf(max_val, ey);
            max_val = fmaxf(max_val, ez);
            max_val = fmaxf(max_val, hx);
            max_val = fmaxf(max_val, hy);
            max_val = fmaxf(max_val, hz);

            min_val = fminf(min_val, ex);
            min_val = fminf(min_val, ey);
            min_val = fminf(min_val, ez);
            min_val = fminf(min_val, hx);
            min_val = fminf(min_val, hy);
            min_val = fminf(min_val, hz);
        }

        // printf("Step %d : Global min = %e , Global max = %e\n",
        //        NZ1, min_val, max_val);
    }
}


__global__ void copy_to_Eall_single(
    float* __restrict__ Eall,   // [nt, step, NX-1, NY-1, NZ-1]
    int t,                      // 当前的时间索引 (1 ~ nt-1)
    const float* __restrict__ E, 
    int step, int NX, int NY, int NZ)
{
    // 内部网格大小
    const long long nx1 = NX - 1;
    const long long ny1 = NY - 1;
    const long long nz1 = NZ - 1;
    const unsigned long long  total = (unsigned long long)step * nx1 * ny1 * nz1;

    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {


        // 展开 idx -> [s, i, j, k]
        long long s   = idx / (nx1 * ny1 * nz1);
        long long rem = idx % (nx1 * ny1 * nz1);
        long long i   = rem / (ny1 * nz1);
        long long j   = (rem % (ny1 * nz1)) / nz1;
        long long k   = rem % nz1;

    if (s < 0 || s >= step || i >= NX-1 || j >= NY-1 || k >= NZ-1) {
        // printf("Index error: s=%d i=%d j=%d k=%d\n", s, i, j, k);
        return;}

        // 在原始 E (step,NX,NY,NZ) 中取内部点
        unsigned long long src_idx = ((unsigned long long)s * NX * NY * NZ)
                    + i * NY * NZ
                    + j * NZ
                    + k;

        // 目标 Eall(nt, step, NX-1, NY-1, NZ-1) 的线性索引
        unsigned long long dst_idx = ((unsigned long long)(t * step + s) * (nx1 * ny1 * nz1))
                       + i * (ny1 * nz1)
                       + j * nz1
                       + k;

        // 拷贝单个分量
        Eall[dst_idx] = E[src_idx];

    }
}



__global__ void accumulate_gradients_1d_safe(
    const float* __restrict__ Ez,
    const float* __restrict__ Eall,
    float* __restrict__ grader,
    float* __restrict__ gradse,
    int i, int step, int NX, int NY, int NZ, float dt,int errequiregrad,int serequiregrad
) {

    long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    long long sx = (NX - 1);
    long long sy = (NY - 1);
    long long sz = (NZ - 1);

    long long per_step = sx * sy * sz;
    unsigned long long total_threads = step * per_step;

    if (tid >= total_threads) return;

    unsigned long long tmp = tid;
    long long iz = (tmp % sz); tmp /= sz;
    long long iy = (tmp % sy); tmp /= sy;
    long long ix = (tmp % sx); tmp /= sx;
    long long s  = tmp;
    // printf("%d %d %d %d\n",s,ix,iy,iz);
    long long idx = ix * sy * sz + iy * sz + iz;

    long long idx_Ez = s * NX * NY * NZ
                + ix * NY * NZ
                + iy * NZ
                + iz;

    unsigned long long idx_curr = i * step*per_step + s * per_step + idx;
    unsigned long long idx_prev = (i-1) * step*per_step + s * per_step + idx;
    if (errequiregrad==1)
        {atomicAdd(&grader[idx], (Eall[idx_curr] - Eall[idx_prev]) * Ez[idx_Ez]/dt);}
    if (serequiregrad==1)
        {atomicAdd(&gradse[idx], Eall[idx_curr]* Ez[idx_Ez]*dt);}

}





extern "C" {


void forward(const float* __restrict__ er, const float* __restrict__ se,
             const float* __restrict__ mr,  
             float* __restrict__ Eall,
             float* __restrict__ Ex,  float* __restrict__ Ey,
             float* __restrict__ Ez,  float* __restrict__ Hx,
             float* __restrict__ Hy,  float* __restrict__ Hz,
             
             float* __restrict__ uE0, float* __restrict__ uE1, float* __restrict__ uE4,
             float* __restrict__ uH0, float* __restrict__ uH1, float* __restrict__ uH4,

            float* __restrict__ x0EPhi1,float* __restrict__ x0EPhi2,
            float* __restrict__ x0HPhi1,float* __restrict__ x0HPhi2,
            float* __restrict__ xmEPhi1,float* __restrict__ xmEPhi2,
            float* __restrict__ xmHPhi1,float* __restrict__ xmHPhi2,
            float* __restrict__ y0EPhi1,float* __restrict__ y0EPhi2,
            float* __restrict__ y0HPhi1,float* __restrict__ y0HPhi2,
            float* __restrict__ ymEPhi1,float* __restrict__ ymEPhi2,
            float* __restrict__ ymHPhi1,float* __restrict__ ymHPhi2,
            float* __restrict__ z0EPhi1,float* __restrict__ z0EPhi2,
            float* __restrict__ z0HPhi1,float* __restrict__ z0HPhi2,
            float* __restrict__ zmEPhi1,float* __restrict__ zmEPhi2,
            float* __restrict__ zmHPhi1,float* __restrict__ zmHPhi2,

            int pml0,int pml1,int pml2,int pml3,int pml4,int pml5,

            const float* __restrict__ x0ER,const float* __restrict__ xmER,
            const float* __restrict__ y0ER,const float* __restrict__ ymER,
            const float* __restrict__ z0ER,const float* __restrict__ zmER,
            const float* __restrict__ x0HR,const float* __restrict__ xmHR,
            const float* __restrict__ y0HR,const float* __restrict__ ymHR,
            const float* __restrict__ z0HR,const float* __restrict__ zmHR,

             float dt, int nt, int step, int nrx, float dx,
             const int* __restrict__ receiverlocation, float* __restrict__ rxs, 

             int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS, int nsrc, 
             const int* __restrict__ sourcelocation, const float* __restrict__ srcwaveforms,
             int polarisation
)
{
    long long blockSize = 256;

    long long total_ucget = NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    dim3 grid_ucget(CEIL_DIV(total_ucget, blockSize));
    ucgetforward<<<grid_ucget, blockSize>>>(
            er, se, mr, 
            uE0, uE1, uE4,
            uH0, uH1, uH4,
            NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, dx
            );
            cudaDeviceSynchronize();   
    

     
    long long total_x0H = step * pml0 * NY_FIELDS * (NZ_FIELDS );
    dim3 grid_x0H(CEIL_DIV(total_x0H, blockSize));
    long long total_xmH = step * pml1 * NY_FIELDS * (NZ_FIELDS );
    dim3 grid_xmH(CEIL_DIV(total_xmH, blockSize));

    long long total_y0H = step * NX_FIELDS * pml2 * (NZ_FIELDS );
    dim3 grid_y0H(CEIL_DIV(total_y0H, blockSize));
    long long total_ymH = step * NX_FIELDS * pml3 * (NZ_FIELDS );
    dim3 grid_ymH(CEIL_DIV(total_ymH, blockSize));

    long long total_z0H = step * NX_FIELDS * (NY_FIELDS ) * pml4;
    dim3 grid_z0H(CEIL_DIV(total_z0H, blockSize));
    long long total_zmH = step * NX_FIELDS * (NY_FIELDS ) * pml5;
    dim3 grid_zmH(CEIL_DIV(total_zmH, blockSize));

    long long total_x0E = step * (pml0+1) * (NY_FIELDS) * (NZ_FIELDS);
    dim3 grid_x0E(CEIL_DIV(total_x0E, blockSize));
    long long total_xmE = step * (pml1+1) * (NY_FIELDS) * (NZ_FIELDS);
    dim3 grid_xmE(CEIL_DIV(total_xmE, blockSize));

    long long total_y0E = step * NX_FIELDS * (pml2+1) * (NZ_FIELDS);
    dim3 grid_y0E(CEIL_DIV(total_y0E, blockSize));
    long long total_ymE = step * NX_FIELDS * (pml3+1) * (NZ_FIELDS);
    dim3 grid_ymE(CEIL_DIV(total_ymE, blockSize));

    long long total_z0E = step * NX_FIELDS * (NY_FIELDS) * (pml4+1);
    dim3 grid_z0E(CEIL_DIV(total_z0E, blockSize));
    long long total_zmE = step * NX_FIELDS * (NY_FIELDS) * (pml5+1);
    dim3 grid_zmE(CEIL_DIV(total_zmE, blockSize));
  


    for (int i = 0; i < nt; i++)
    {

    // {   // 检测代码块
    //     int total = NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    //     dim3 grid((total + 255) / 256);
    //     check_nan_inf6_print<<<grid, 256>>>(Ex,Ey,Ez,Hx,Hy,Hz,NX_FIELDS,NY_FIELDS,NZ_FIELDS,i);
    //     cudaDeviceSynchronize();
    // }
        // printf("Forward step %d / %d \n", i+1, nt);
        {
            long long total = step * nrx;
            long long blockSize = 256;
            long long gridSize  = (total + blockSize - 1) / blockSize;
            // printf("%d %d %d %d \n",step, nrx, i, nt);
            store_outputs<<<gridSize, blockSize>>>(
                    step, nrx, i,
                    receiverlocation,
                    rxs,
                    Ex, Ey, Ez,
                    Hx, Hy, Hz,
                    NX_FIELDS, NY_FIELDS, NZ_FIELDS, nt
                                );
            cudaDeviceSynchronize();CUDA_CHECK();
        }

        {        
            long long total_h = step * NX_FIELDS * NY_FIELDS * NZ_FIELDS;
            dim3 grid_h(CEIL_DIV(total_h, blockSize));
            h_fields_updates_gpu<<<grid_h, blockSize>>>(
                    uH0, uH1,      
                    Ex, Ey, Ez,   
                    Hx, Hy, Hz,
                    step,  NX_FIELDS,  NY_FIELDS,  NZ_FIELDS
                );
            cudaDeviceSynchronize();CUDA_CHECK();
        } 

        {        

        if (pml0>0)
            {x0H<<<grid_x0H, blockSize>>>(0,pml0,0,NY_FIELDS-1,0,NZ_FIELDS-1,
                pml0, NY_FIELDS, NZ_FIELDS-1,
                pml0, NY_FIELDS-1, NZ_FIELDS,
                pml0,
                Ex,Ey,Ez,Hx,Hy,Hz,
                x0HPhi1,x0HPhi2,
                x0HR,
                dx, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);
            cudaDeviceSynchronize();CUDA_CHECK();
            }
        } 

        {        

        if (pml1>0)
            {xmH<<<grid_xmH, blockSize>>>(NX_FIELDS-1-pml1,NX_FIELDS-1,0,NY_FIELDS-1,0,NZ_FIELDS-1,
                pml1, NY_FIELDS, NZ_FIELDS-1,
                pml1, NY_FIELDS-1, NZ_FIELDS,
                pml1,
                Ex,Ey,Ez,Hx,Hy,Hz, 
                xmHPhi1,xmHPhi2,
                xmHR,
                dx, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);
            cudaDeviceSynchronize();CUDA_CHECK();
            }
        } 

        {        
        if (pml2>0)
            {y0H<<<grid_y0H, blockSize>>>(
                 0,NX_FIELDS-1,0,pml2,0,NZ_FIELDS-1,
                NX_FIELDS, pml2, NZ_FIELDS-1,
                NX_FIELDS-1, pml2, NZ_FIELDS,
                pml2,
                Ex,Ey,Ez,Hx,Hy,Hz,
                y0HPhi1,y0HPhi2,
                y0HR,
                dx, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);
            cudaDeviceSynchronize();CUDA_CHECK();
            }
        } 
    
        {        
        if (pml3>0)
            {ymH<<<grid_ymH, blockSize>>>(
                0,NX_FIELDS-1,NY_FIELDS-1-pml3,NY_FIELDS-1,0,NZ_FIELDS-1,
                NX_FIELDS, pml3, NZ_FIELDS-1,
                NX_FIELDS-1, pml3, NZ_FIELDS,
                pml3,
                Ex,Ey,Ez,Hx,Hy,Hz,
                ymHPhi1,ymHPhi2,
                ymHR,
                dx, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);
            cudaDeviceSynchronize();CUDA_CHECK();
            }
        } 

        {        
        if (pml4>0)
            {z0H<<<grid_z0H, blockSize>>>(0,NX_FIELDS-1,0,NY_FIELDS-1,0,pml4,
                NX_FIELDS, NY_FIELDS-1, pml4,
                NX_FIELDS-1, NY_FIELDS, pml4,
                pml4,
                Ex,Ey,Ez,Hx,Hy,Hz,
                z0HPhi1,z0HPhi2,
                z0HR,
                dx, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);
            cudaDeviceSynchronize();CUDA_CHECK();  
            }
        } 
        // printf("%d %d %d %d\n",NX_FIELDS-1,NY_FIELDS-1,NZ_FIELDS-1-pml5,NZ_FIELDS-1); 
        
        {        
        if (pml5>0)
            {zmH<<<grid_zmH, blockSize>>>(0,NX_FIELDS-1,0,NY_FIELDS-1,NZ_FIELDS-1-pml5,NZ_FIELDS-1,
            NX_FIELDS, NY_FIELDS-1, pml5,
            NX_FIELDS-1, NY_FIELDS, pml5,
            pml5,
            Ex,Ey,Ez,Hx,Hy,Hz,
            zmHPhi1,zmHPhi2,
            zmHR,
            dx, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);}
        cudaDeviceSynchronize();  
        CUDA_CHECK();
        } 

        {               

            size_t total_e = step * NX_FIELDS * NY_FIELDS * NZ_FIELDS;
            dim3 grid_e(CEIL_DIV(total_e, blockSize));
            e_fields_updates_gpu<<<grid_e, blockSize>>>(uE0, uE1,  
                    Ex,  Ey, Ez,  
                    Hx, Hy, Hz,
                    step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
            cudaDeviceSynchronize();CUDA_CHECK();
        } 



        {        
         if (pml0>0)
            {x0E<<<grid_x0E, blockSize>>>(0,pml0,0,NY_FIELDS-1,0,NZ_FIELDS-1,
                pml0+1, NY_FIELDS-1, NZ_FIELDS,
                pml0+1, NY_FIELDS, NZ_FIELDS-1,
                pml0,
                Ex,Ey,Ez,Hx,Hy,Hz,
                x0EPhi1,x0EPhi2,
                x0ER,
                dx, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);
            CUDA_CHECK();
            }
        } 

        {        
        if (pml1>0)
            {xmE<<<grid_xmE, blockSize>>>(NX_FIELDS-1-pml1,NX_FIELDS-1,0,NY_FIELDS-1,0,NZ_FIELDS-1,
                pml1+1, NY_FIELDS-1, NZ_FIELDS,
                pml1+1, NY_FIELDS, NZ_FIELDS-1,
                pml1,
                Ex,Ey,Ez,Hx,Hy,Hz, 
                xmEPhi1,xmEPhi2,
                xmER,
                dx, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);
                CUDA_CHECK();
            }
        } 

        {        
        if (pml2>0)
            {y0E<<<grid_y0E, blockSize>>>(
                 0,NX_FIELDS-1,0,pml2,0,NZ_FIELDS-1,
                NX_FIELDS-1, pml2+1, NZ_FIELDS,
                NX_FIELDS, pml2+1, NZ_FIELDS-1,
                pml2,
                Ex,Ey,Ez,Hx,Hy,Hz,
                y0EPhi1,y0EPhi2,
                y0ER,
                dx, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);
            CUDA_CHECK();
            }
        } 

        {        
        if (pml3>0)
            {ymE<<<grid_ymE, blockSize>>>(
                0,NX_FIELDS-1,NY_FIELDS-1-pml3,NY_FIELDS-1,0,NZ_FIELDS-1,
                NX_FIELDS-1, pml3+1, NZ_FIELDS,
                NX_FIELDS, pml3+1, NZ_FIELDS-1,
                pml3,
                Ex,Ey,Ez,Hx,Hy,Hz,
                ymEPhi1,ymEPhi2,
                ymER,
                dx, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);
                CUDA_CHECK();
            }
        } 

        {        
        if (pml4>0)
            {z0E<<<grid_z0E, blockSize>>>(0,NX_FIELDS-1,0,NY_FIELDS-1,0,pml4,
                NX_FIELDS-1, NY_FIELDS, pml4+1,
                NX_FIELDS, NY_FIELDS-1, pml4+1,
                pml4,
                Ex,Ey,Ez,Hx,Hy,Hz,
                z0EPhi1,z0EPhi2,
                z0ER,
                dx, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);
            CUDA_CHECK();
            }
        } 

        {        
        if (pml5>0)
            {zmE<<<grid_zmE, blockSize>>>(0,NX_FIELDS-1,0,NY_FIELDS-1,NZ_FIELDS-1-pml5,NZ_FIELDS-1,
            NX_FIELDS-1, NY_FIELDS, pml5+1,
            NX_FIELDS, NY_FIELDS-1, pml5+1,
            pml5,
            Ex,Ey,Ez,Hx,Hy,Hz,
            zmEPhi1,zmEPhi2,
            zmER,
            dx, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);}
        cudaDeviceSynchronize(); CUDA_CHECK();
        } 

        {

        int bx = 64;
        int by = 16;
        dim3 block_src(bx, by);
        dim3 grid_src( (nsrc + bx - 1) / bx, (step + by - 1) / by );
        Update_hertzian_dipole<<<grid_src, block_src>>>(
            step, i, dx,
            sourcelocation, srcwaveforms,
            Ex, Ey, Ez, uE4,
            NX_FIELDS, NY_FIELDS, NZ_FIELDS, nsrc, polarisation, nt);
        cudaDeviceSynchronize();CUDA_CHECK();

        }
        
        {
   
            size_t total_copy = step * (NX_FIELDS - 1) * (NY_FIELDS - 1) * (NZ_FIELDS - 1);
            dim3 grid_copy((total_copy + blockSize - 1) / blockSize);

            copy_to_Eall_single<<<grid_copy, blockSize>>>(
                Eall, i,
                Ez,
                step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
            cudaDeviceSynchronize();CUDA_CHECK();
        }
    }
}





void backward(const float* __restrict__ er, const float* __restrict__ se,
             const float* __restrict__ mr,  
             const float* __restrict__ Eall,
             float* __restrict__ Ex,  float* __restrict__ Ey,
             float* __restrict__ Ez,  float* __restrict__ Hx,
             float* __restrict__ Hy,  float* __restrict__ Hz,
             
             float* __restrict__ uE0, float* __restrict__ uE1, float* __restrict__ uE4,
             float* __restrict__ uH0, float* __restrict__ uH1, float* __restrict__ uH4,

            float* __restrict__ x0EPhi1,float* __restrict__ x0EPhi2,
            float* __restrict__ x0HPhi1,float* __restrict__ x0HPhi2,
            float* __restrict__ xmEPhi1,float* __restrict__ xmEPhi2,
            float* __restrict__ xmHPhi1,float* __restrict__ xmHPhi2,
            float* __restrict__ y0EPhi1,float* __restrict__ y0EPhi2,
            float* __restrict__ y0HPhi1,float* __restrict__ y0HPhi2,
            float* __restrict__ ymEPhi1,float* __restrict__ ymEPhi2,
            float* __restrict__ ymHPhi1,float* __restrict__ ymHPhi2,
            float* __restrict__ z0EPhi1,float* __restrict__ z0EPhi2,
            float* __restrict__ z0HPhi1,float* __restrict__ z0HPhi2,
            float* __restrict__ zmEPhi1,float* __restrict__ zmEPhi2,
            float* __restrict__ zmHPhi1,float* __restrict__ zmHPhi2,

            int pml0,int pml1,int pml2,int pml3,int pml4,int pml5,

            float* __restrict__ x0ER,float* __restrict__ xmER,
            float* __restrict__ y0ER,float* __restrict__ ymER,
            float* __restrict__ z0ER,float* __restrict__ zmER,
            float* __restrict__ x0HR,float* __restrict__ xmHR,
            float* __restrict__ y0HR,float* __restrict__ ymHR,
            float* __restrict__ z0HR,float* __restrict__ zmHR,

             float dt, int nt, int step, int nrx, float dx,
             int NX_FIELDS, int NY_FIELDS, int NZ_FIELDS,
             int nsrc, const int* __restrict__ sourcelocation, const float* __restrict__ srcwaveforms,
             int polarisation, 
             float*__restrict__ grad_er,float*__restrict__ grad_se, int  errequiregrad, int  serequiregrad
)
{
    int blockSize = 256;

    int total_ucget = NX_FIELDS * NY_FIELDS * NZ_FIELDS;
    dim3 grid_ucget(CEIL_DIV(total_ucget, blockSize));
    ucgetbackward<<<grid_ucget, blockSize>>>(
            er, se, mr, 
            uE0, uE1, uE4,
            uH0, uH1, uH4,
            NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt, dx
            );
            cudaDeviceSynchronize();   
    CUDA_CHECK();

     
    int total_x0H = step * pml0 * NY_FIELDS * (NZ_FIELDS );
    dim3 grid_x0H(CEIL_DIV(total_x0H, blockSize));
    int total_xmH = step * pml1 * NY_FIELDS * (NZ_FIELDS );
    dim3 grid_xmH(CEIL_DIV(total_xmH, blockSize));

    int total_y0H = step * NX_FIELDS * pml2 * (NZ_FIELDS);
    dim3 grid_y0H(CEIL_DIV(total_y0H, blockSize));
    int total_ymH = step * NX_FIELDS * pml3 * (NZ_FIELDS );
    dim3 grid_ymH(CEIL_DIV(total_ymH, blockSize));

    int total_z0H = step * NX_FIELDS * (NY_FIELDS ) * pml4;
    dim3 grid_z0H(CEIL_DIV(total_z0H, blockSize));
    int total_zmH = step * NX_FIELDS * (NY_FIELDS ) * pml5;
    dim3 grid_zmH(CEIL_DIV(total_zmH, blockSize));

    int total_x0E = step * (pml0+1) * (NY_FIELDS) * (NZ_FIELDS);
    dim3 grid_x0E(CEIL_DIV(total_x0E, blockSize));
    int total_xmE = step * (pml1+1) * (NY_FIELDS) * (NZ_FIELDS);
    dim3 grid_xmE(CEIL_DIV(total_xmE, blockSize));

    int total_y0E = step * NX_FIELDS * (pml2+1) * (NZ_FIELDS);
    dim3 grid_y0E(CEIL_DIV(total_y0E, blockSize));
    int total_ymE = step * NX_FIELDS * (pml3+1) * (NZ_FIELDS);
    dim3 grid_ymE(CEIL_DIV(total_ymE, blockSize));

    int total_z0E = step * NX_FIELDS * (NY_FIELDS) * (pml4+1);
    dim3 grid_z0E(CEIL_DIV(total_z0E, blockSize));
    int total_zmE = step * NX_FIELDS * (NY_FIELDS) * (pml5+1);
    dim3 grid_zmE(CEIL_DIV(total_zmE, blockSize));
  


    for (int i = nt-1; i >0; i--)
    {

        {
            int threads = 256;                                   // 每个block的线程数，可根据硬件调节
            int total   = step * nsrc;                           // 总元素数
            int blocks  = (total + threads - 1) / threads;        // gridDim.x

            Back_source<<<blocks, threads>>>(
                step, i, dx,
                sourcelocation, srcwaveforms,
                Ex, Ey, Ez, uE4,
                NX_FIELDS, NY_FIELDS, NZ_FIELDS,
                nsrc, polarisation, nt
            );
            cudaDeviceSynchronize();
            CUDA_CHECK();
        }
        {               
            size_t total_h = step * NX_FIELDS * NY_FIELDS * NZ_FIELDS;
            dim3 grid_h(CEIL_DIV(total_h, blockSize));
            e_fields_updates_gpu<<<grid_h, blockSize>>>(uE0, uE1,  
                    Ex,  Ey, Ez,  
                    Hx, Hy, Hz,
                    step, NX_FIELDS, NY_FIELDS, NZ_FIELDS);
            cudaDeviceSynchronize();CUDA_CHECK();
        } 

        {        
         if (pml0>0)
            {x0E<<<grid_x0E, blockSize>>>(0,pml0,0,NY_FIELDS-1,0,NZ_FIELDS-1,
                pml0+1, NY_FIELDS-1, NZ_FIELDS,
                pml0+1, NY_FIELDS, NZ_FIELDS-1,
                pml0,
                Ex,Ey,Ez,Hx,Hy,Hz,
                x0EPhi1,x0EPhi2,
                x0ER,
                dx, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);}CUDA_CHECK();
        } 

        {        
        if (pml1>0)
            {xmE<<<grid_xmE, blockSize>>>(NX_FIELDS-1-pml1,NX_FIELDS-1,0,NY_FIELDS-1,0,NZ_FIELDS-1,
                pml1+1, NY_FIELDS-1, NZ_FIELDS,
                pml1+1, NY_FIELDS, NZ_FIELDS-1,
                pml1,
                Ex,Ey,Ez,Hx,Hy,Hz, 
                xmEPhi1,xmEPhi2,
                xmER,
                dx, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);}
                CUDA_CHECK();
        } 

        {        
        if (pml2>0)
            {y0E<<<grid_y0E, blockSize>>>(
                 0,NX_FIELDS-1,0,pml2,0,NZ_FIELDS-1,
                NX_FIELDS-1, pml2+1, NZ_FIELDS,
                NX_FIELDS, pml2+1, NZ_FIELDS-1,
                pml2,
                Ex,Ey,Ez,Hx,Hy,Hz,
                y0EPhi1,y0EPhi2,
                y0ER,
                dx, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);}CUDA_CHECK();
        } 

        {        
        if (pml3>0)
            {ymE<<<grid_ymE, blockSize>>>(
                0,NX_FIELDS-1,NY_FIELDS-1-pml3,NY_FIELDS-1,0,NZ_FIELDS-1,
                NX_FIELDS-1, pml3+1, NZ_FIELDS,
                NX_FIELDS, pml3+1, NZ_FIELDS-1,
                pml3,
                Ex,Ey,Ez,Hx,Hy,Hz,
                ymEPhi1,ymEPhi2,
                ymER,
                dx, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);}CUDA_CHECK();
        } 

        {        
        if (pml4>0)
            {z0E<<<grid_z0E, blockSize>>>(0,NX_FIELDS-1,0,NY_FIELDS-1,0,pml4,
                NX_FIELDS-1, NY_FIELDS, pml4+1,
                NX_FIELDS, NY_FIELDS-1, pml4+1,
                pml4,
                Ex,Ey,Ez,Hx,Hy,Hz,
                z0EPhi1,z0EPhi2,
                z0ER,
                dx, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);}CUDA_CHECK();
        } 

        {        
        if (pml5>0)
            {zmE<<<grid_zmE, blockSize>>>(0,NX_FIELDS-1,0,NY_FIELDS-1,NZ_FIELDS-1-pml5,NZ_FIELDS-1,
            NX_FIELDS-1, NY_FIELDS, pml5+1,
            NX_FIELDS, NY_FIELDS-1, pml5+1,
            pml5,
            Ex,Ey,Ez,Hx,Hy,Hz,
            zmEPhi1,zmEPhi2,
            zmER,
            dx, uE4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);}
        
        } 
        cudaDeviceSynchronize(); CUDA_CHECK();

        {        
            size_t total_h = step * NX_FIELDS * NY_FIELDS * NZ_FIELDS;
            dim3 grid_h(CEIL_DIV(total_h, blockSize));
            h_fields_updates_gpu<<<grid_h, blockSize>>>(
                    uH0, uH1,      
                    Ex, Ey, Ez,   
                    Hx, Hy, Hz,
                    step,  NX_FIELDS,  NY_FIELDS,  NZ_FIELDS
                );
            cudaDeviceSynchronize();CUDA_CHECK();
        } 

        {        

        if (pml0>0)
            {x0H<<<grid_x0H, blockSize>>>(0,pml0,0,NY_FIELDS-1,0,NZ_FIELDS-1,
                pml0, NY_FIELDS, NZ_FIELDS-1,
                pml0, NY_FIELDS-1, NZ_FIELDS,
                pml0,
                Ex,Ey,Ez,Hx,Hy,Hz,
                x0HPhi1,x0HPhi2,
                x0HR,
                dx, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);
            
            }
            CUDA_CHECK();
        } 

        {        

        if (pml1>0)
            {xmH<<<grid_xmH, blockSize>>>(NX_FIELDS-1-pml1,NX_FIELDS-1,0,NY_FIELDS-1,0,NZ_FIELDS-1,
                pml1, NY_FIELDS, NZ_FIELDS-1,
                pml1, NY_FIELDS-1, NZ_FIELDS,
                pml1,
                Ex,Ey,Ez,Hx,Hy,Hz, 
                xmHPhi1,xmHPhi2,
                xmHR,
                dx, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);}
                CUDA_CHECK();
        } 

        {        
        if (pml2>0)
            {y0H<<<grid_y0H, blockSize>>>(
                 0,NX_FIELDS-1,0,pml2,0,NZ_FIELDS-1,
                NX_FIELDS, pml2, NZ_FIELDS-1,
                NX_FIELDS-1, pml2, NZ_FIELDS,
                pml2,
                Ex,Ey,Ez,Hx,Hy,Hz,
                y0HPhi1,y0HPhi2,
                y0HR,
                dx, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);}
                CUDA_CHECK();
        } 

        {        
        if (pml3>0)
            {ymH<<<grid_ymH, blockSize>>>(
                0,NX_FIELDS-1,NY_FIELDS-1-pml3,NY_FIELDS-1,0,NZ_FIELDS-1,
                NX_FIELDS, pml3, NZ_FIELDS-1,
                NX_FIELDS-1, pml3, NZ_FIELDS,
                pml3,
                Ex,Ey,Ez,Hx,Hy,Hz,
                ymHPhi1,ymHPhi2,
                ymHR,
                dx, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);}
                CUDA_CHECK();
        } 

        {        
        if (pml4>0)
            {z0H<<<grid_z0H, blockSize>>>(0,NX_FIELDS-1,0,NY_FIELDS-1,0,pml4,
                NX_FIELDS, NY_FIELDS-1, pml4,
                NX_FIELDS-1, NY_FIELDS, pml4,
                pml4,
                Ex,Ey,Ez,Hx,Hy,Hz,
                z0HPhi1,z0HPhi2,
                z0HR,
                dx, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);}
                CUDA_CHECK();
        } 

        {        
        if (pml5>0)
            {zmH<<<grid_zmH, blockSize>>>(0,NX_FIELDS-1,0,NY_FIELDS-1,NZ_FIELDS-1-pml5,NZ_FIELDS-1,
            NX_FIELDS, NY_FIELDS-1, pml5,
            NX_FIELDS-1, NY_FIELDS, pml5,
            pml5,
            Ex,Ey,Ez,Hx,Hy,Hz,
            zmHPhi1,zmHPhi2,
            zmHR,
            dx, uH4, NX_FIELDS, NY_FIELDS, NZ_FIELDS, step);}
        
        } 
        cudaDeviceSynchronize();  
        CUDA_CHECK();

        {
            int total_threads = step * (NX_FIELDS-1) * (NY_FIELDS-1) * (NZ_FIELDS-1);
            int numBlocks = (total_threads + blockSize - 1) / blockSize;

            accumulate_gradients_1d_safe<<<numBlocks, blockSize>>>(
                Ez, Eall, grad_er, grad_se, i, step, NX_FIELDS, NY_FIELDS, NZ_FIELDS, dt,errequiregrad,serequiregrad
            );
            cudaDeviceSynchronize();  
            CUDA_CHECK();
        }
        

    }
}






















}






