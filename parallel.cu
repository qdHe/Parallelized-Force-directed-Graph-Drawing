#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

#define W 10
#define H 10
#define BLOCK_SIZE 16
int N = 25;//tentative
int M = 35;
int K = sqrt(W*H/N);

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
    return deviceParams.K*deviceParams.K/dist/deviceParams.N/10000;
}

__device__ __inline__ float attractive_force(float dist){
    return dist*dist/deviceParams.K/deviceParams.N;
}

__global__ void kernelForceDirected(float  *V, int *E, int *Idx){

	float thr = deviceParams.Thr;

    int v = blockIdx.x*blockDim.x + threadIdx.x;
	if(v >= deviceParams.N) return;
	float disp_x, disp_y;
	float2 end1 = *(float2*)(&V[2*v]), end2;
   
	for(int itr=0; itr<deviceParams.Iteration; ++itr){
        //if (itr%(Iteration/10) == 0) cout << "Iteration = " << itr+1 << endl;
        if(v==0) printf("End1:(%f,%f)\n", end1.x, end1.y);
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
				//if(v==0) printf("u=%d, repulsive_force= %f\n", u, rf);
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
					//if(v==0) printf("u=%d, att_force= %f\n", u, af);
                    disp_x -= d_x/dist*af;
                    disp_y -= d_y/dist*af;
                }
            

            //if(v%1000==0) cout<<V[v].disp_x<<' '<<V[v].disp_y<<endl;
            float dist = sqrt(disp_x*disp_x + disp_y*disp_y);
            end1.x += (dist > thr)? disp_x/dist*thr : disp_x;
            end1.y += (dist > thr)? disp_y/dist*thr : disp_y;
            end1.x = min(W/2., max(-W/2.,end1.x));
            end1.y = min(H/2., max(-H/2.,end1.y));
			*(float2*)&V[2*v] = end1;
        __syncthreads();
        thr *= 0.99;
    }
}



int main(int argc, char* argv[]) {
    Vertex *V = new Vertex[N];
    Edge *E = new Edge[2*M];
    int *Idx = new int[N]();

    ifstream infile;
	cout<<argv[0]<<endl;
    infile.open(argv[1]);
    //int idx1, idx2, w;
    int ct = 0;
    Edge e;
    while(!infile.eof()){
        infile >> e.idx1 >> e.idx2;
        //cout<<idx1<<' '<<idx2<<endl;
        E[ct++] = e;
        swap(e.idx1, e.idx2);
        E[ct++] = e;
    }
    sort(E, E+ct, cmp);
    cout << "Total Edges = " << ct/2 << endl;
	//for(int i=0; i<20; ++i){
	//	cout<<E[i].idx1<<' '<<E[i].idx2<<endl;
	//}
    for(int i=0; i<ct; ++i) {
        Idx[E[i].idx1] += 1;
    }
	//cout<<"IDX:[";
    for(int i=1; i<N; ++i) {
        Idx[i] += Idx[i-1]; //End Index
		//cout<<Idx[i]<<' ';
    }
	//cout<<"]"<<endl;
    //cout << "Complete Initialization" << endl;
    int iteration = atoi(argv[2]);
    int thr = W+H;
    //force_directed(V, E, Idx, N, iteration, thr);

    float *deviceVertex;
    int *deviceEdge;
    int *deviceIdx;
    cudaMalloc(&deviceVertex, sizeof(float)*2*N);
    cudaMalloc(&deviceEdge, sizeof(int)*2*M);
    cudaMalloc(&deviceIdx, sizeof(int)*N);
    cudaMemcpy(deviceVertex, V, sizeof(int)*2*N, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceEdge, E, sizeof(int)*2*M, cudaMemcpyHostToDevice);
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
    kernelForceDirected<<<gridDim, blockDim>>>(deviceVertex, deviceEdge,
				deviceIdx);
    cudaDeviceSynchronize();

    cudaMemcpy(V, deviceVertex, sizeof(int)*2*N,  cudaMemcpyDeviceToHost);
    cudaFree(deviceVertex);
    cudaFree(deviceEdge);
    cudaFree(deviceIdx);


    ofstream outfile("Vertex_Pos.txt");
    for (int v=0; v<N; ++v){
        outfile << V[v].x <<' '<<V[v].y<<endl;
    }
    outfile.close(); 
    return 0;
}
