#include <iostream>
#include <fstream>
#include <chrono>
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
using namespace std;

#define TOTAL_V_NUM 10000
#define CELL_NUM 4
#define LOCK -2
#define NOTHING -1
#define MAXDEPTH 30
#define THREADS5 16
#define WARPSIZE 32
#define BLOCK_SIZE 256
#define W 10
#define H 10



struct GlobalConstants{
    int N;
    int M;
    float K;
    float Alpha;
    float Eps;
    float Thr;
};

class Edge{
public:
    int idx1;
    int idx2;

    Edge():idx1(-1), idx2(-1){}
    Edge(int a, int b):idx1(a), idx2(b){}
};

bool cmp(Edge a, Edge b) {
    if (a.idx1<b.idx1) return true;
    else if (a.idx1 == b.idx1 && a.idx2 < b.idx2) return true;
    return false;
}

float find_min(float* nums, int N){
    thrust::device_ptr<float> ptr(nums);
    int result_offset = thrust::min_element(ptr, ptr + N) - ptr;
    float min_x = *(ptr + result_offset);
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

__global__ void init(int* _bottom, int* maxDepth, int node_num){
    *_bottom = node_num;
    *maxDepth = 0;
}


/******************************************************************************/
/*** build tree ***************************************************************/
/******************************************************************************/
__global__ void BuildTreeKernel(float* posx, float* posy, int* child, int* _bottom, int* _maxDepth, float radius, int N, int node_num,
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
    int local_depth = 0;
    int local_max_depth = 0;
    rootIndex = node_num;
    if(threadId == 0){
        posx[rootIndex] = rootx;
        posy[rootIndex] = rooty;
    }

    while(i < N){
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

            local_depth = 1;
        }
        curIndex = child[CELL_NUM*lastIndex+path];
        // follow a path to leave node
        while(curIndex >= N){
            lastIndex = curIndex;
            path = 0;
            if(posx[lastIndex] < x){
                path += 1;
            }
            if(posy[lastIndex] < y){
                path += 2;
            }
            curIndex = child[CELL_NUM*curIndex+path];
            local_depth++;
            curRad *= 0.5;
        }

        if (curIndex != LOCK) {
            locked = CELL_NUM*lastIndex+path;
            // try locking the node
            if (curIndex == atomicCAS((int*)&child[locked], curIndex, LOCK)) {
                if (curIndex == NOTHING) {
                    child[locked] = i; // insert body and release lock
                    local_max_depth = max(local_depth, local_max_depth);
                } else {
                    // malloc new cell nodes
                    int old_node = curIndex;
                    float old_node_x = posx[old_node];
                    float old_node_y = posy[old_node];
                    int cell = atomicSub((int*)_bottom, 1) - 1;
                    int new_cell = cell;
                    do{
                        if(cell<N){
                            printf("error break\n");
                            break;
                        }
                        if(cell != new_cell)
                            child[CELL_NUM*lastIndex+path] = cell;
                        posx[cell] = posx[lastIndex] - curRad*0.5 + (path&1)*curRad;
                        posy[cell] = posy[lastIndex] - curRad*0.5 + ((path>>1)&1)*curRad;

                        curRad *= 0.5;
                        local_depth++;

                        path = 0;
                        if(posx[cell] < x){
                            path += 1;
                        }
                        if(posy[cell] < y){
                            path += 2;
                        }
                        old_node_path = 0;
                        if(posx[cell] < old_node_x){
                            old_node_path += 1;
                        }
                        if(posy[cell] < old_node_y){
                            old_node_path += 2;
                        }
                        if(path != old_node_path){
                            child[cell*CELL_NUM+path] = i;
                            child[cell*CELL_NUM+old_node_path] = old_node;
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
                local_max_depth = max(local_depth, local_max_depth);
            }
        }
    }
    // gather maximum depth
    atomicMax((int *)_maxDepth, local_max_depth);


}

/******************************************************************************/
/*** summarize nodes **********************************************************/
/******************************************************************************/
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

    while(node_id<=node_num){

        if(missing == 0){
            child_num = 0;
            sum_x = 0.0;
            sum_y = 0.0;
            tmp_count = 0;
            cache_tail = 0;
            for(int i=CELL_NUM*node_id; i<CELL_NUM*node_id+CELL_NUM; i++){
                int child_node = child[i];
                if(child_node >= 0){
                    tmp_c = count[child_node];
                    if(tmp_c > 0){
                        sum_x += posx[child_node]*tmp_c;
                        sum_y += posy[child_node]*tmp_c;
                        tmp_count += tmp_c;
                    }else{
                        // add node to stack
                        missing++;
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
                    sum_x += posx[child_node]*tmp_c;
                    sum_y += posy[child_node]*tmp_c;
                    tmp_count += tmp_c;
                    cache_tail--;
                }
            }while(missing != 0 && tmp_c > 0);
        }
        // all children are ready, update current node
        if(missing == 0){
            posx[node_id] = sum_x/tmp_count;
            posy[node_id] = sum_y/tmp_count;
            //FENCE
            __threadfence();
            count[node_id] = tmp_count;
            node_id += step;
        }
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
                if (childIdx >= N) {
                    // child is a cell
                    startd[childIdx] = start;  // set start ID of child
                    start += count[childIdx];  // add #bodies in subtree
                } else if (childIdx >= 0) {
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
    return deviceParams.K*deviceParams.K/dist/deviceParams.N/100.;
}

__device__ __inline__ float attractive_force(float dist){
    return dist*dist/deviceParams.K/deviceParams.N;
}

__global__
void ForceCalculationKernel(float* posx, float* posy, int* child, int* count, int *sort, int *E, int *Idx,
                            float *dispX, float *dispY, int node_num, int *_maxdepth, float radius)
{
    __shared__ volatile int pos[MAXDEPTH * BLOCK_SIZE/WARPSIZE], node[MAXDEPTH * BLOCK_SIZE/WARPSIZE];
    __shared__ float dq[MAXDEPTH * BLOCK_SIZE/WARPSIZE];
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
                        float dx = px - posx[childIdx];
                        float dy = py - posy[childIdx];
                        float dist = dx*dx + dy*dy;  // compute distance squared (plus softening)
                        if ((childIdx < deviceParams.N) || __all(dist >= dq[depth])) {  // check if all threads agree that cell is far enough away (or is a body)
							if (childIdx != v){ 	
								dist = sqrt(dist);  // compute distance
								dist = max(dist, 0.001);
								float rf = repulsive_force(dist)*count[childIdx];
								dispx += dx/dist*rf;//disp_x
								dispy += dy/dist*rf;//disp_y
							}
						} else {
                            // push cell onto stack
                            depth++;
                            if (sbase == threadIdx.x) {
								node[depth] = childIdx;
                                pos[depth] = 0;
                            }
                        }
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
                float dist = sqrt(dx*dx+dy*dy);
				dist = max(dist, 0.001);
                float af = attractive_force(dist);
                dispx -= dx/dist*af;
            dispy  -= dy/dist*af;
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

void BH(float* hostx, float* hosty, Edge *E, int *Idx, int N, int M, float K, int timesteps){
    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(min((N*2+1+BLOCK_SIZE-1)/BLOCK_SIZE, 32));
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
    int* _maxDepth;
    float minx, miny;
    float maxx, maxy;
    float radius;
    float rootx;
    float rooty;
    int node_num = N*2;
    int iter;

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
    cudaMalloc(&_maxDepth, sizeof(int));
    //INIT
    cudaMemcpy(posx, hostx, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(posy, hosty, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceEdge, E, sizeof(int)*4*M, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceIdx, Idx, sizeof(int)*N, cudaMemcpyHostToDevice);

    GlobalConstants params;
    params.N = N;
    params.M = M;
    params.K = K;
    params.Thr = thr;
    params.Alpha = alpha;
    params.Eps = eps;
    cudaMemcpyToSymbol(deviceParams, &params, sizeof(GlobalConstants));

    //FORCE DIRECTED
    auto calc_start = Clock::now();
    for (iter = 0; iter < timesteps; iter++) {
        //Calculating bounding box
        auto bounding_start = Clock::now();
        cudaDeviceSynchronize();
        minx = find_min(posx, N);
        maxx = find_max(posx, N);
        miny = find_min(posy, N);
        maxy = find_max(posy, N);
        radius = std::max(maxx-minx,maxy-miny);
        radius *= 0.5;
        rootx = (minx+maxx)*0.5;
        rooty = (miny+maxy)*0.5;
        cudaMemset(_bottom, node_num, 1);
        cudaMemset(_maxDepth, 0, 1);
        cout << "bounding box:" << duration_cast<dsec>(Clock::now() - bounding_start).count() << endl;

        //Build Tree
        auto build_start = Clock::now();
        init<<<1, 1>>>(_bottom, _maxDepth, node_num);
        cudaDeviceSynchronize();
        batch_set(child, CELL_NUM*(node_num+1), -1);
        BuildTreeKernel<<<gridDim, blockDim>>>(posx, posy, child, _bottom, _maxDepth, radius, N, node_num, rootx, rooty);
        cudaDeviceSynchronize();
        cout << "build tree:" << duration_cast<dsec>(Clock::now() - build_start).count() << endl;

        //Summerize Tree
        auto summerize_start = Clock::now();
        batch_set(count+N, node_num-N, -1);
        batch_set(count, N, 1);
        SummarizeTreeKernel<<<gridDim, blockDim>>>(posx, posy, child, count, _bottom, node_num, N);
        cudaDeviceSynchronize();
        cout << "summerize tree:" << duration_cast<dsec>(Clock::now() - summerize_start).count() << endl;

        //Sort
        auto sort_start = Clock::now();
        batch_set(start,(node_num+1),0);
        batch_set(start+N, N, -1);
        SortKernel<<<gridDim, blockDim>>>(start, sort, child, count, _bottom, node_num);
        cudaDeviceSynchronize();
        cout << "sort nodes:" << duration_cast<dsec>(Clock::now() - sort_start).count() << endl;

        //Compute force
        auto force_start = Clock::now();
        ForceCalculationKernel<<<gridDim, blockDim>>>(posx, posy, child, count, sort, deviceEdge, deviceIdx,
                dispx, dispy, node_num, _maxDepth, radius);
        cudaDeviceSynchronize();
        cout << "compute force:" << duration_cast<dsec>(Clock::now() - force_start).count() << endl;

        //Update
        auto update_start = Clock::now();
        UpdatePosKernel<<<gridDim, blockDim>>>(posx, posy, dispx, dispy);
        cudaDeviceSynchronize();
        cout << "update positions:" << duration_cast<dsec>(Clock::now() - update_start).count() << endl;

    }
    double calc_time = duration_cast<dsec>(Clock::now() - calc_start).count();

    cout << "Time: " << calc_time << endl;
    cudaMemcpy(hostx, posx, sizeof(float)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(hosty, posy, sizeof(float)*N, cudaMemcpyDeviceToHost);
}


int main(int argc, char* argv[]){
    int N, M;


    ifstream infile;
    infile.open(argv[1]);
    infile >> N >> M;
    float K = sqrt(1.0*W*H/N);
    Edge *E = new Edge[2*M];
    int *Idx = new int[N]();
    float *hostx = new float[N];
    float *hosty = new float[N];

    Edge e;
    for(int i=0; i<2*M; i+=2){
        infile >> e.idx1 >> e.idx2;
        E[i] = e;
        swap(e.idx1, e.idx2);
        E[i+1] = e;
    }
    sort(E, E+2*M, cmp);
    cout << "Total Edges = " << M << endl;

    for(int i=0; i<2*M; ++i) {
        Idx[E[i].idx1] += 2;
    }

    for(int i=1; i<N; ++i) {
        Idx[i] += Idx[i-1]; //End Index
    }
    cout << "Complete Initialization" << endl;
    int iteration = atoi(argv[2]);
    for(int i=0; i<N; i++){
        hostx[i] = (float(rand())/RAND_MAX-0.5f);
        hosty[i] = (float(rand())/RAND_MAX-0.5f);
    }
    BH(hostx, hosty, E, Idx, N, M, K, iteration);
    ofstream outfile("Vertex_Pos_pl.txt");
    for (int v=0; v<N; ++v){
        outfile << hostx[v] <<' '<<hosty[v]<<endl;
    }
    outfile.close();
};
