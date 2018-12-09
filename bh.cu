/*
0. Read input data and transfer to GPU
for each timestep do {
1. Compute bounding box around all bodies
2. Build hierarchical decomposition by inserting each body into octree
3. Summarize body information in each internal octree node
4. Approximately sort the bodies by spatial distance
5. Compute forces acting on each body with help of octree
6. Update body positions and velocities
}
7. Transfer result to CPU and output
*/
#include <iostream>
#include <fstream>
//#include <chrono>
#include <algorithm>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>


#define TOTAL_V_NUM 10000
#define CELL_NUM 4
#define LOCK -2
#define NOTHING -1

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...)    printf(fmt, ## args)
#else
# define DEBUG_PRINT(fmt, args...) do {} while (false)
#endif

float find_min(float* nums, int N){
    thrust::device_ptr<float> ptr(nums);
    int result_offset = thrust::min_element(ptr, ptr + N) - ptr;
    float min_x = *(ptr + result_offset);
    printf("min: %f\n", min_x);
    return min_x;
}

float find_max(float* nums, int N){
    thrust::device_ptr<float> ptr(nums);
    int result_offset = thrust::max_element(ptr, ptr + N) - ptr;
    float max_x = *(ptr + result_offset);
    return max_x;
}

void batch_set(int* raw_ptr, int N, int target){
    thrust::device_ptr<int> dev_ptr(raw_ptr);
    thrust::fill(dev_ptr, dev_ptr + N, (int) target);
}

__global__ void BuildTreeKernel(float* posx, float* posy, int* child, int* _bottom, float radius, int N, int node_num,
                     float rootx, float rooty){
    int threadId = blockIdx.x*blockDim.x + threadIdx.x;
    int threadNum = blockDim.x*gridDim.x;//TODO
    int step = threadNum;
    bool newBody = true;
    float x;
    float y;
    float curRad;
    int rootIndex;
    int curIndex;
    int lastIndex;
    int i = threadId;
    int path;
    int locked;
    int old_node_path;
    rootIndex = node_num;
    if(threadId == 0){
        posx[rootIndex] = rootx;
        posy[rootIndex] = rooty;
        printf("bottom %d\n",*_bottom);
    }

    while(i < N){
        // initialize

        if(newBody){
            x = posx[i];
            y = posy[i];
            path = 0;
            if(rootx < x){
                path += 1;
            }
            if(rooty < y){
                path += 2;
            }
            lastIndex = rootIndex;
            newBody = false;
            curRad = radius;
            printf("new body %d x %f y %f\n", i, x, y);
        }
        curIndex = child[CELL_NUM*lastIndex+path];//TODO
        while(curIndex >= N){
            lastIndex = curIndex;
            path = 0;
            if(posx[lastIndex] < x){
                path += 1;
            }
            if(posx[lastIndex] < y){
                path += 2;
            }
            curIndex = child[CELL_NUM*curIndex+path];
            curRad *= 0.5;
        }

        if (curIndex != LOCK) {
            locked = CELL_NUM*lastIndex+path;
            if (curIndex == atomicCAS((int*)&child[locked], curIndex, LOCK)) {
                printf("add to tree node %d posx %f posy %f\n", lastIndex, posx[lastIndex], posy[lastIndex]);
                if (curIndex == NOTHING) {
                    child[locked] = i; // insert body and release lock
                    printf("empty, add %d to %d path %d success\n", i, lastIndex, path);
                } else {
                    int old_node = curIndex;
                    float old_node_x = posx[old_node];
                    float old_node_y = posy[old_node];
                    int cell = atomicSub((int*)_bottom, 1) - 1;
                    int new_cell = cell;
                    printf("old node %d x:%f y:%f new cell:%d\n", old_node, old_node_x, old_node_y, cell);
                    do{
                        if(cell<N){
                            printf("error break\n");
                            break;
                        }
                        child[CELL_NUM*lastIndex+path] = cell;
                        posx[cell] = posx[lastIndex] - curRad*0.5 + (path&1)*curRad;
                        posy[cell] = posy[lastIndex] - curRad*0.5 + ((path>>1)&1)*curRad;

                        curRad *= 0.5;

                        path = 0;
                        if(posx[cell] < x){
                            path += 1;
                        }
                        if(posx[cell] < y){
                            path += 2;
                        }
                        old_node_path = 0;
                        if(posx[cell] < old_node_x){
                            old_node_path += 1;
                        }
                        if(posx[cell] < old_node_y){
                            old_node_path += 2;
                        }
                        printf("new cell %d x:%f y:%f rad:%f old id %d path %d new id %d path %d\n",
                                cell, posx[cell], posy[cell], curRad, old_node, old_node_path, i, path);
                        if(path != old_node_path){
                            child[cell*CELL_NUM+path] = i;
                            child[cell*CELL_NUM+old_node_path] = old_node;
                            printf("new cell %d x:%f y:%f rad:%f old id %d path %d new id %d path %d break\n",
                                   cell, posx[cell], posy[cell], curRad, old_node, old_node_path, i, path);
                            break;
                        }else{
                            lastIndex = cell;
                            cell = atomicSub((unsigned int*)_bottom, 1) - 1;
                        }
                    }while(true);
                    __threadfence();
                    child[locked] = new_cell;
                }
                newBody = true;
                i += step;
            }
        }
        //__syncthreads();
    }


}

__global__ void SummarizeTreeKernel(float* posx, float* posy, int* child, int* count, int* _bottom, int node_num, int N){
    
    int missing = 0;
    int child_node;
    int cache[CELL_NUM] = {0};
    int cache_tail = 0;
    int step = blockDim.x*gridDim.x;//TODO

    int child_num;
    int tmp_count;
    int tmp_c;
    float sum_x;
    float sum_y;
    int threadId = blockIdx.x*blockDim.x + threadIdx.x;
    int node_id = threadId + *_bottom;
    if(threadId == 0){
        for(int i=0; i<N; i++){
            printf("count %d:%d\n", i, count[i]);
        }
    }

    while(node_id<node_num){
        
        if(missing == 0){
            child_num = 0;
            sum_x = 0.0;
            sum_y = 0.0;
            tmp_count = 0;
            cache_tail = 0;
            for(int i=CELL_NUM*node_id; i<CELL_NUM*node_id+CELL_NUM; i++){
                int child_node = child[i];
                if(child_node >= 0){
                    if(count[child_node] > 0){
                        sum_x += posx[child_node];
                        sum_y += posy[child_node];
                        tmp_count += count[child_node];
                    }else{
                        missing++;
                        //TODO cache index
                        cache[cache_tail++] = child_node;
                    }
                    child_num++;
                }
                
            }
        }
        if(missing != 0){

            do{
                child_node = cache[cache_tail-1];
                tmp_c = count[child_node];
                if(tmp_c > 0){
                    missing--;
                    sum_x += posx[child_node];
                    sum_y += posy[child_node];
                    tmp_count += tmp_c;
                    cache_tail--;
                }
            }while(missing != 0 && tmp_c > 0);
        }
        if(missing == 0){
            printf("%d before: x %f y %f\n", node_id, posx[node_id], posy[node_id]);
            posx[node_id] = sum_x/tmp_count;
            posy[node_id] = sum_y/tmp_count;
            //FENCE
            __threadfence();
            count[node_id] = tmp_count;
            printf("%d after: x %f y %f count %d\n", node_id, posx[node_id], posy[node_id], count[node_id]);
            node_id += step;
        }
    }
}

void printTree(float* posx, float* posy, int* child, int node_num, int N){
    bool flag;
    for(int i=node_num; i>=N; i--){
        flag = false;
        printf("node %d posx %f posy %f \n", i, posx[i], posy[i]);
        for(int j=i*CELL_NUM; j<i*CELL_NUM+CELL_NUM; j++){

            printf(" child:%d", child[j]);
            if(child[j] > 0) flag=true;
        }

        printf("\n");
        if(!flag) break;
    }
    for(int i=0; i<N; i++){
        printf("node %d posx %f posy %f \n", i, posx[i], posy[i]);
    }
}

void BH(float* hostx, float* hosty, int N, int timesteps){
    int gridDim = 16;
    int blockDim = 16;
    float* posx;
    float* posy;
    int* child;
    int* count;
    int* _bottom;
    float minx, miny;
    float maxx, maxy;
    float radius;
    float rootx;
    float rooty;
    int node_num = N*2;
    int iter;
    //===debug===//
    int* host_child = new int [(node_num+1)*CELL_NUM];

    cudaMalloc(&posx, sizeof(float)*(node_num+1));
    cudaMalloc(&posy, sizeof(float)*(node_num+1));
    cudaMalloc(&child, sizeof(int)*CELL_NUM*(node_num+1));


    cudaMalloc(&count, sizeof(int)*(node_num+1));
    cudaMalloc(&_bottom, sizeof(int));
    //INIT
    cudaMemcpy(posx, hostx, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(posy, hosty, sizeof(float)*N, cudaMemcpyHostToDevice);

    //FORCE DIRECTED
    for (iter = 0; iter < timesteps; iter++) {
        //Calculating repulsive force
        //Calculating bounding box
        printf("iter %d\n", iter);
        minx = find_min(posx, N);
        maxx = find_max(posx, N);
        miny = find_min(posy, N);
        maxy = find_max(posy, N);
        radius = std::max(maxx-minx,maxy-miny);
        radius *= 0.5;
        rootx = (minx+maxx)*0.5;
        rooty = (miny+maxy)*0.5;
        printf("rootx %f rooty %f radius %f\n", rootx, rooty, radius);
        cudaMemset(_bottom, node_num, 1);
        //Build Tree
        batch_set(child, CELL_NUM*(node_num+1), -1);
        BuildTreeKernel<<<gridDim, blockDim>>>(posx, posy, child, _bottom, radius, N, node_num, rootx, rooty);
        cudaDeviceSynchronize();
        printf("build tree success!\n");

        //===debug====//
        cudaMemcpy(host_child, child, sizeof(int)*CELL_NUM*(node_num+1),cudaMemcpyDeviceToHost);
        cudaMemcpy(hostx, posx, sizeof(float)*(node_num+1), cudaMemcpyDeviceToHost);
        cudaMemcpy(hosty, posy, sizeof(float)*(node_num+1), cudaMemcpyDeviceToHost);
        printTree(hostx, hosty, host_child, node_num, N);


        //Summerize Tree
        batch_set(count+N, node_num-N, -1);
        batch_set(count, N, 1);
        SummarizeTreeKernel<<<gridDim, blockDim>>>(posx, posy, child, count, _bottom, node_num, N);
        cudaDeviceSynchronize();

        //===debug====//
        cudaMemcpy(host_child, child, sizeof(int)*CELL_NUM*(node_num+1),cudaMemcpyDeviceToHost);
        cudaMemcpy(hostx, posx, sizeof(float)*(node_num+1), cudaMemcpyDeviceToHost);
        cudaMemcpy(hosty, posy, sizeof(float)*(node_num+1), cudaMemcpyDeviceToHost);
        printTree(hostx, hosty, host_child, node_num, N);
        //Compute force

        //Calculating attractive force

        //Update


    }
}

__global__ void print_kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main(){
    int N = 8;
    float *hostx = new float[N];
    float *hosty = new float[N];
    print_kernel<<<10, 10>>>();
    cudaDeviceSynchronize();
    for(int i=0; i<N; i++){
        hostx[i] = i*0.1f;//(float(rand())/RAND_MAX-0.5f);
        hosty[i] = i*0.1f;(float(rand())/RAND_MAX-0.5f);
    }
    BH(hostx, hosty, N, 1);
}

