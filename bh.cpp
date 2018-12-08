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
#define TOTAL_V_NUM 10000
#define CELL_NUM 4
#define LOCK -2
#define NOTHING -1
float find_min(float* nums, int N){
    thrust::device_ptr<float> ptr =  thrust::device_pointer_cast(nums);
    int result_offset = thrust::min_element(ptr, ptr + N) - ptr;
    float min_x = *(ptr + result_offset);
    return min_x;
}

float find_max(float* nums, int N){
    thrust::device_ptr<float> ptr =  thrust::device_pointer_cast(nums);
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
    int threadNum = blockDim.x*threadDim.x;//TODO
    int step = threadNum;
    bool newBody = true;
    float x;
    float y;
    float curRad;
    int rootIndex;
    int curIndex;
    int lastIndex;
    int i = threadId;
    rootIndex = node_num - 1;
    if(threadId == 0){
        posx[rootIndex] = rootx;
        posy[rootIndex] = rooty;
    }

    while(i < N){
        // initialize
        x = posx[i];
        y = posy[i];
        path = 0;
        if(newBody){
            if(rootx < x){
                path += 1;
            }
            if(rooty < y){
                path += 2;
            }
            lastIndex = rootIndex;
            newBody = false;
            curRad = radius;
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
                if (curIndex == NOTHING) {
                    child[locked] = body; // insert body and release lock
                } else {
                    int old_node = curIndex;
                    int old_node_x = posx[old_node];
                    int old_node_y = posy[old_node];
                    int cell = atomicDec(_bottom, 1) - 1;
                    int new_cell = cell;
                    do{
                        child[CELL_NUM*lastIndex+path] = cell;
                        posx[cell] = posx[lastIndex] - curRad*0.5 + (path&1)*curRad;
                        posy[cell] = posy[lastIndex] - curRad*0.5 + (path&2)*curRad;
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
                        if(path != old_path){
                            child[cell*CELL_NUM+path] = i;
                            child[cell*CELL_NUM+old_path] = old_node;
                            break;
                        }else{
                            lastIndex = cell;
                            cell = atom_dec(_bottom) - 1;
                        }
                    }while(true);
                    __threadfence();
                    child[locked] = new_cell;
                }
                new_cell = true;
                i += step;
            }
        }
        __syncthreads();
    }


}

__global__ void SummarizeTreeKernel(float* posx, float* posy, int* child, int* count, int* _bottom){
    
    int missing = 0;
    int child_node;
    int cache[CELL_NUM] = {0};
    int cache_tail = 0;
    int step = blockDim.x*threadDim.x;//TODO

    int child_num;
    int tmp_count;
    int tmp_c;
    float sum_x;
    float sum_y;
    int threadId = blockIdx.x*blockDim.x + threadIdx.x;
    int node_id = threadId + *_bottom;

    while(node_id<=TOTAL_NODE_NUM){
        
        if(missing == 0){
            child_num = 0;
            sum_x = 0.0;
            sum_y = 0.0;
            tmp_count = 0;
            cache_tail = 0;
            for(int i=CELL_NUM*node_id; i<CELL_NUM*node_id+CELL_NUM; i++){
                int child_node = count[i];
                if(child_node > 0){
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
            posx[node_id] = sum_x/tmp_count;
            posy[node_id] = sum_y/tmp_count;
            //FENCE
            __threadfence();
            count[node_id] = tmp_count;
            node_id += step;
        }
    }
}



int BH(float* hostx, float* hosty, int N){

    float* posx;
    float* posy;
    float* centerx;
    float* centery;
    int* child;
    int* count;
    int* _bottom;
    int minx, miny;
    int maxx, maxy;
    int radius;
    int rootx;
    int rooty;
    int node_num = N*2;
    cudaMalloc(&posx, sizeof(float)*node_num);
    cudaMalloc(&posy, sizeof(float)*node_num);
    cudaMalloc(&child, sizeof(int)*CELL_NUM*node_num);

    cudaMalloc(&centerx, sizeof(float)*node_num);
    cudaMalloc(&centery, sizeof(float)*node_num);
    cudaMalloc(&count, sizeof(int)*node_num);
    cudaMalloc(&_bottom, sizeof(int));
    //INIT
    cudaMemcpy(posx, hostx, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(posy, hosty, sizeof(int)*N, cudaMemcpyHostToDevice);

    //FORCE DIRECTED
    for (step = 0; step < timesteps; step++) {
        //Calculating repulsive force
        //Calculating bounding box
        minx = find_min(posx, N);
        maxx = find_max(posx, N);
        miny = find_min(posy, N);
        maxy = find_max(posy, N);
        radius = max(maxx-minx,maxy-miny);
        rootx = (minx+maxx)*0.5;
        rooty = (miny+maxy)*0.5;
        cudaMemset(_bottom, node_num-1, 1);
        //Build Tree
        batch_set(child, CELL_NUM*node_num, -1);
        BuildTreeKernel(posx, posy, child, _bottom, radius, N, rootx, rooty);
        //Summerize Tree
        batch_set(count+N, node_num-N, -1);
        batch_set(count, N, 1);
        SummarizeTreeKernel(posx, posy, child, count, centerx, centery, _bottom);
        //Compute force

        //Calculating attractive force

        //Update


    }
}