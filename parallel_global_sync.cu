#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

#define W 10
#define H 10
#define BLOCK_SIZE 256

struct GlobalConstants{
	int N;
	int M;
	float K;

	int Iteration;
	int Thr;
};

class Vertex{
public:
    float x;
    float y;

    Vertex():x((float(rand())/RAND_MAX-0.5)), y((float(rand())/RAND_MAX-0.5)){}
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


__constant__ GlobalConstants deviceParams;
//__device__ __inline__ float max(float a, float b){
//    return (a>b)? a:b;
//}
//__device__ __inline__ float min(float a, float b){
//    return (a<b)? a:b;
//}
__device__ __inline__ float repulsive_force(float dist){
    return deviceParams.K*deviceParams.K/dist/deviceParams.N/100.0;
}

__device__ __inline__ float attractive_force(float dist){
    return dist*dist/deviceParams.K/deviceParams.N;
}

__global__ void kernelForceDirected(float  *V, int *E, int *Idx, float *disp, float thr){


    int v = blockIdx.x*blockDim.x + threadIdx.x;
	if(v >= deviceParams.N) return;
	float disp_x, disp_y;
	float2 end1 = *(float2*)(&V[2*v]), end2;
	//for(int itr=0; itr<deviceParams.Iteration; ++itr){
        //if (itr%(Iteration/10) == 0) cout << "Iteration = " << itr+1 << endl;
    disp_x = 0;
    disp_y = 0;
    end1 = *(float2*)(&V[2*v]);
    for(int u=0; u<deviceParams.N; ++u){
        end2 = *(float2*)(&V[2*u]);
        float d_x = end1.x-end2.x;
        float d_y = end1.y-end2.y;
        float dist = sqrt(d_x*d_x+d_y*d_y);
        dist = max(dist, 0.001);
        float rf = repulsive_force(dist);
        //if(v==10) printf("u=%d, repulsive_force= %f\n", u, rf);
        //if(v%1000 && u%1000) cout<<rf<<' '<<af<<' '<<dist<<endl;
        disp_x += d_x/dist*rf;//disp_x
        disp_y += d_y/dist*rf;//disp_y
    }
    int start = 0;
    if (v > 0){
        start = Idx[v-1];
    }
    for(int e=start; e<Idx[v]; ++e){
        int u = E[2*e+1];
        end2 = *(float2*)(&V[2*u]);
        float d_x = end1.x-end2.x;
        float d_y = end1.y-end2.y;
        float dist = sqrt(d_x*d_x+d_y*d_y);
        dist = max(dist, 0.001);
        float af = attractive_force(dist);
        //if(v==24) printf("e=%d, u=%d, att_force= %f\n", e, u, af);
        disp_x -= d_x/dist*af;
        disp_y -= d_y/dist*af;
    }

    disp[2*v] = disp_x;
    disp[2*v+1] = disp_y;
}

__global__ void kernelUpdate(float* V, float* disp){
    float  thr = deviceParams.Thr;
    int v = threadIdx.x + blockIdx.x * blockDim.x;
    float2 end1 = *(float2*)(&V[2*v]);
    float disp_x = disp[2*v], disp_y = disp[2*v+1];
    float dist = sqrt(disp_x*disp_x + disp_y*disp_y);
    end1.x += (dist > thr)? disp_x/dist*thr : disp_x;
    end1.y += (dist > thr)? disp_y/dist*thr : disp_y;
    end1.x = min(W/2., max(-W/2.,end1.x));
    end1.y = min(H/2., max(-H/2.,end1.y));
    *(float2*)&V[2*v] = end1;

}



int main(int argc, char* argv[]) {

	using namespace std::chrono;
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::duration<double> dsec;

	int N, M;
    
	ifstream infile;
    infile.open(argv[1]);
	infile >> N >> M;
	Vertex *V = new Vertex[N];
    Edge *E = new Edge[2*M];
    int *Idx = new int[N]();

    Edge e;
    for(int i=0; i<2*M; i+=2){
        infile >> e.idx1 >> e.idx2;
		//cout<<e.idx1<<' '<<e.idx2<<endl;
        E[i] = e;
        swap(e.idx1, e.idx2);
        E[i+1] = e;
    }
    sort(E, E+2*M, cmp);
	float K = sqrt(1.0*W*H/N);
    cout << "Total Edges = " << M << endl;

    for(int i=0; i<2*M; ++i) {
        Idx[E[i].idx1] += 1;
    }

    for(int i=1; i<N; ++i) {
        Idx[i] += Idx[i-1]; //End Index
    }
	//cout<<"]"<<endl;
    cout << "Complete Initialization" << endl;
    int iteration = atoi(argv[2]);
    int thr = W+H;
    //force_directed(V, E, Idx, N, iteration, thr);

    float *deviceVertex;
    int *deviceEdge;
    int *deviceIdx;
    float *deviceDisp;
    cudaMalloc(&deviceVertex, sizeof(float)*2*N);
    cudaMalloc(&deviceEdge, sizeof(int)*4*M);
    cudaMalloc(&deviceIdx, sizeof(int)*N);
    cudaMalloc(&deviceDisp, sizeof(float)*2*N);
    cudaMemcpy(deviceVertex, V, sizeof(int)*2*N, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceEdge, E, sizeof(int)*4*M, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceIdx, Idx, sizeof(int)*N, cudaMemcpyHostToDevice);
	
	GlobalConstants params;
	params.N = N;
	params.M = M;
	params.K = K;
	params.Iteration = iteration;
	params.Thr = thr;
	cudaMemcpyToSymbol(deviceParams, &params, sizeof(GlobalConstants));
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N+BLOCK_SIZE-1)/BLOCK_SIZE);
    auto calc_start = Clock::now();
    for(int itr=0; itr<iteration; ++itr) {
        kernelForceDirected << < gridDim, blockDim >> > (deviceVertex, deviceEdge,
                deviceIdx, deviceDisp, thr);
        //thr *= 0.99;
        cudaDeviceSynchronize();
        kernelUpdate << < gridDim, blockDim >> > (deviceVertex, deviceDisp);
        cudaDeviceSynchronize();
    }
    double calc_time = duration_cast<dsec>(Clock::now() - calc_start).count();
    cout << "Time: " << calc_time << endl;

    cudaMemcpy(V, deviceVertex, sizeof(float)*2*N,  cudaMemcpyDeviceToHost);
    cudaFree(deviceVertex);
    cudaFree(deviceEdge);
    cudaFree(deviceIdx);


    ofstream outfile("Vertex_Pos_pl.txt");
    for (int v=0; v<N; ++v){
        outfile << V[v].x <<' '<<V[v].y<<endl;
    }
    outfile.close(); 
    return 0;
}
