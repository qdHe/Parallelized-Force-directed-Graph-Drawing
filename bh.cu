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
#define MAXDEPTH 10
#define THREADS5 16
#define WARPSIZE 32
#define W 10
#define H 10
int N = 8;//tentative
int M = 8;
float K = sqrt(1.0*W*H/N);

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...)    printf(fmt, ## args)
#else
# define DEBUG_PRINT(fmt, args...) do {} while (false)
#endif

struct GlobalConstants{
    int N;
    int M;
    float K;
    float Alpha;
    float Eps;
    float Thr;
};

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
__constant__ GlobalConstants deviceParams;
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

__global__ void SummarizeTreeKernel(float* posx, float* posy, int* child, int* count, int* _bottom, int node_num){

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
            printf("%d after: x %f y %f count\n", node_id, posx[node_id], posy[node_id], count[node_id]);
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

/******************************************************************************/
/*** sort bodies **************************************************************/
/******************************************************************************/

__global__ void SortKernel(int* startd, int *sort, int *child, int *count, 
			int *_bottom, int node_num) {

	int N = deviceParams.N;
	int bottom = *_bottom;
    int gridSize = blockDim.x * gridDim.x;
    int cell = node_num + 1 - gridSize + threadIdx.x + blockIdx.x * blockDim.x;

    // iterate over all cells assigned to thread
    while (cell >= bottom) {
        int start = startd[cell];
        if (start >= 0) {
            for (int i = 0; i < 4; ++i) {
                int childIdx = child[cell*4+i];
				if(childIdx == 15){
					printf("child[15], count = %d\n", count[15]);
				}
                if (childIdx >= N) {
					printf("   #case1: start = %d, i = %d, childIdx = %d\n",
								start, i, childIdx);
                    // child is a cell
                    startd[childIdx] = start;  // set start ID of child
                    start += count[childIdx];  // add #bodies in subtree
                } else if (childIdx >= 0) {
					printf("   #case2: start = %d, i = %d, childIdx = %d\n",
								start, i, childIdx);
                    // child is a body
                    sort[start] = childIdx;  // record body in 'sorted' array
                    ++start;
                }
            }
            cell -= gridSize;  // move on to next cell
        }
        __syncthreads();  // throttle
    }
}


/******************************************************************************/
/*** compute force ************************************************************/
/******************************************************************************/

__device__ __inline__ float repulsive_force(float dist){
    return deviceParams.K*deviceParams.K/dist/deviceParams.N/10000;
}

__device__ __inline__ float attractive_force(float dist){
    return dist*dist/deviceParams.K/deviceParams.N;
}

__global__
void ForceCalculationKernel(float* posx, float* posy, int* child, int* count, int *sort, int *E, int *Idx,
        float *dispX, float *dispY, int node_num, int *_maxdepth, float radius)
{
    __shared__ volatile int pos[MAXDEPTH * THREADS5/WARPSIZE], node[MAXDEPTH * THREADS5/WARPSIZE];
    __shared__ float dq[MAXDEPTH * THREADS5/WARPSIZE];
    float alpha = deviceParams.Alpha, eps = deviceParams.Eps;
	int maxdepth = *_maxdepth;

    if (0 == threadIdx.x) {
        // precompute values that depend only on tree level
        dq[0] = radius * radius * alpha;
        for (int i = 1; i < maxdepth; i++) {
            dq[i] = dq[i - 1] * 0.25f;
            dq[i - 1] += eps;
        }
        dq[maxdepth - 1] += eps;

        if (maxdepth > MAXDEPTH) {
            // error
            printf("ERROR");
        }
    }
    __syncthreads();

    if (maxdepth <= MAXDEPTH) {
        // figure out first thread in each warp (lane 0)
        int base = threadIdx.x / WARPSIZE;
        int sbase = base * WARPSIZE;
        int j = base * MAXDEPTH;

        int diff = threadIdx.x - sbase;
        // make multiple copies to avoid index calculations later
        if (diff < MAXDEPTH) {
            dq[diff+j] = dq[diff];
        }
        __syncthreads();

        // iterate over all bodies assigned to thread
        for (int k = threadIdx.x + blockIdx.x * blockDim.x; k < deviceParams.N; k += blockDim.x * gridDim.x) {
            int v = sort[k];  // get permuted/sorted index
            // cache position info
            float px = posx[v];
            float py = posy[v];
            float dispx = 0.;
            float dispy = 0.;
            // initialize iteration stack, i.e., push root node onto stack
            int depth = j;
            if (sbase == threadIdx.x) {
                node[j] = node_num;//nnodesd;
                pos[j] = 0;
            }

            while (depth >= j) {
                // stack is not empty
                int t;
                while ((t = pos[depth]) < 4) {
                    // node on top of stack has more children to process
                    int childIdx = child[node[depth]*4+t];  // load child pointer
                    if (sbase == threadIdx.x) {
                        // I'm the first thread in the warp
                        pos[depth] = t + 1;
                    }
                    if (childIdx >= 0) {
                        float dx = posx[childIdx] - px;
                        float dy = posy[childIdx] - py;
                        float dist = dx*dx + dy*dy + eps;  // compute distance squared (plus softening)
                        if ((childIdx < deviceParams.N) || __all(dist >= dq[depth])) {  // check if all threads agree that cell is far enough away (or is a body)
                            dist = rsqrtf(dist);  // compute distance
                            float rf = repulsive_force(dist)*count[childIdx];
                            dispx += dx/dist*rf;//disp_x
                            dispy += dy/dist*rf;//disp_y
                        } else {
                            // push cell onto stack
                            depth++;
                            if (sbase == threadIdx.x) {
                                node[depth] = childIdx;
                                pos[depth] = 0;
                            }
                        }
                    } else {
                        depth = max(j, depth - 1);  // early out because all remaining children are also zero
                    }
                }
                depth--;  // done with this level
            }

            int start = 0;
            if (v > 0){
                start = Idx[v-1];
            }
            for(int e=start; e<Idx[v]; e+=2){
                int u = E[e+1];
                float dx = px-posx[u];
                float dy = py-posy[u];
                float dist = sqrt(dx*dx+dy*dy+eps);
                float af = attractive_force(dist);
                //if(v==10) printf("u=%d, att_force= %f\n", u, af);
                dispx -= dx/dist*af;
                dispy -= dy/dist*af;
            }

            dispX[v] = dispx;
            dispY[v] = dispy;
        }
    }
}

__global__
void UpdatePosKernel(float* posx, float* posy, float *dispX, float *dispY){
    float  thr = deviceParams.Thr;
    float dispx, dispy, px, py;
    for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < deviceParams.N; v += blockDim.x * gridDim.x) {
        dispx = dispX[v];
        dispy = dispY[v];
		px = posx[v];
		py = posy[v];
        float dist = sqrt(dispx * dispx + dispy * dispy);
        px += (dist > thr) ? dispx / dist * thr : dispx;
        py += (dist > thr) ? dispy / dist * thr : dispy;
        posx[v] = min(W / 2., max(-W / 2., px));
        posy[v] = min(H / 2., max(-H / 2., py));
    }
}


void BH(float* hostx, float* hosty, int *E, int *Idx, int N, int timesteps){
    int gridDim = 16;
    int blockDim = 16;
    float alpha = 4;
    float eps = 0.0025;
    float thr = W+H;
    float* posx;
    float* posy;
    float* dispx;
    float* dispy;
    int* child;
    int* count;
    int* debug_start = new int[2*N+1]();
    int* start;
    int* debug_sort = new int[N]();
    int* sort;
    int *deviceEdge;
    int *deviceIdx;
    int* _bottom;
    int *_maxdepth = new int(4); //TODO
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
    cudaMalloc(&start, sizeof(int)*(node_num+1));
    cudaMalloc(&sort, sizeof(int)*N);
    cudaMalloc(&deviceEdge, sizeof(int)*4*M);
    cudaMalloc(&deviceIdx, sizeof(int)*N);
    cudaMalloc(&dispx, sizeof(float)*N);
    cudaMalloc(&dispy, sizeof(float)*N);

    cudaMalloc(&count, sizeof(int)*(node_num+1));
    cudaMalloc(&_bottom, sizeof(int));
    //INIT
    cudaMemcpy(posx, hostx, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(posy, hosty, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceEdge, E, sizeof(int)*4*M, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceIdx, Idx, sizeof(int)*N, cudaMemcpyHostToDevice);

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

        GlobalConstants params;
        params.N = N;
        params.M = M;
        params.K = K;
        params.Thr = thr;
        params.Alpha = alpha;
        params.Eps = eps;
        cudaMemcpyToSymbol(deviceParams, &params, sizeof(GlobalConstants));



        //Build Tree
        batch_set(child, CELL_NUM*(node_num+1), -1);
        BuildTreeKernel<<<gridDim, blockDim>>>(posx, posy, child, _bottom, radius, N, node_num, rootx, rooty);
        cudaDeviceSynchronize();
        printf("build tree success!\n");

        //===debug====//
        //cudaMemcpy(host_child, child, sizeof(int)*CELL_NUM*(node_num+1),cudaMemcpyDeviceToHost);
        //cudaMemcpy(hostx, posx, sizeof(float)*(node_num+1), cudaMemcpyDeviceToHost);
        //cudaMemcpy(hosty, posy, sizeof(float)*(node_num+1), cudaMemcpyDeviceToHost);
        //printTree(hostx, hosty, host_child, node_num, N);


        //Summerize Tree
        batch_set(count+N, node_num-N, -1);
        batch_set(count, N, 1);
        SummarizeTreeKernel<<<gridDim, blockDim>>>(posx, posy, child, count, _bottom, node_num);
        cudaDeviceSynchronize();

        //===debug====//
        //cudaMemcpy(host_child, child, sizeof(int)*CELL_NUM*(node_num+1),cudaMemcpyDeviceToHost);
        //cudaMemcpy(hostx, posx, sizeof(float)*(node_num+1), cudaMemcpyDeviceToHost);
        //cudaMemcpy(hosty, posy, sizeof(float)*(node_num+1), cudaMemcpyDeviceToHost);
        //printTree(hostx, hosty, host_child, node_num, N);
        //Sort
		cudaMemset(start,0,sizeof(int)*(node_num+1));
        batch_set(start+N, N, -1);
        cudaMemcpy(debug_start, start, sizeof(float)*(node_num+1), cudaMemcpyDeviceToHost);
		for(int i = 0; i<=node_num; ++i){
			printf("%d ", debug_start[i]);
		}
        printf("\nPOS1\n");
		//printf("node_num: %d, bottom: %d\n", node_num, (*_bottom));
        SortKernel<<<gridDim, blockDim>>>(start, sort, child, count, _bottom, node_num);
        cudaDeviceSynchronize();
        cudaMemcpy(debug_sort, sort, sizeof(float)*N, cudaMemcpyDeviceToHost);
		for(int i = 0; i<N; ++i){
			printf("%d ", debug_sort[i]);
		}
        printf("\nPOS2\n");
        //Compute force //TODO try separate repulsive force and attractive force calculation
        ForceCalculationKernel<<<gridDim, blockDim>>>(posx, posy, child, count, sort, deviceEdge, deviceIdx,
                dispx, dispy, node_num, _maxdepth, radius);
        cudaDeviceSynchronize();
        thr *= 0.99; //TODO
        //Update
        UpdatePosKernel<<<gridDim, blockDim>>>(posx, posy, dispx, dispy);
        cudaDeviceSynchronize();


    }
}

__global__ void print_kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main(){
    int N = 8, M = 8;
    float *hostx = new float[N];
    float *hosty = new float[N];
    int *E = new int[4*M];
    int *Idx = new int[N]();
	for(int i=0; i<N; ++i){
		Idx[i] = 4*(i+1);
		E[4*i] = i;
		E[4*i] = (i+1)%N;
		E[4*i] = i;
		E[4*i] = (i-1+N)%N;
	}

    print_kernel<<<10, 10>>>();
    cudaDeviceSynchronize();
    for(int i=0; i<N; i++){
        hostx[i] = i*0.1f;//(float(rand())/RAND_MAX-0.5f);
        hosty[i] = i*0.1f;//(float(rand())/RAND_MAX-0.5f);
    }

    BH(hostx, hosty, E, Idx, N, 1);
}
