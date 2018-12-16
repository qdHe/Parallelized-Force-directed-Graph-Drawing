#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "mic.h"
using namespace std;

#define W 10
#define H 10

class Vertex{
public:
    float x;
    float y;
    float disp_x;
    float disp_y;

    Vertex():x((float(rand())/RAND_MAX-0.5)), y((float(rand())/RAND_MAX-0.5)), disp_x(0), disp_y(0){}
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

float max(float a, float b){
    return (a>b)? a:b;
}
float min(float a, float b){
    return (a<b)? a:b;
}
inline float repulsive_force(float dist, float K, int N){
    return K*K/dist/N/10000;
}

inline float attractive_force(float dist, float K, int N){
    return dist*dist/K/N;
}

void force_directed(float *posx, float *posy, int *edges, int *Idx, int N, int M, float K, int Iteration, float thr){
 	using namespace std::chrono;
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::duration<double> dsec;

    float *disp_x = new float[N]();
    float *disp_y = new float[N]();
    printf("N = %d, M = %d\n", N, M);
    auto calc_start = Clock::now();
    for(int itr=0; itr<Iteration; ++itr){
        #pragma omp parallel for default(shared) schedule(dynamic)
        for(int v=0; v<N; ++v){
            for(int u=0; u<N; ++u){
                float d_x = posx[v]-posx[u];
                float d_y = posy[v]-posy[u];
                float dist = sqrt(d_x*d_x+d_y*d_y);
                dist = max(dist, 0.001);
                float rf = repulsive_force(dist, K, N);
                disp_x[v] += d_x/dist*rf;
                disp_y[v] += d_y/dist*rf;
            }
            int e = 0;
            if (v > 0){
                e = Idx[v-1];
            }
            for(; e<Idx[v]; ++e){
                int u = edges[2*e+1];
                float d_x = posx[v]-posx[u];
                float d_y = posy[v]-posy[u];
                float dist = sqrt(d_x*d_x+d_y*d_y);
                dist = max(dist, 0.001);
                float af = attractive_force(dist, K, N);
                disp_x[v] -= d_x/dist*af;
                disp_y[v] -= d_y/dist*af;
            }
        }

        #pragma omp parallel for default(shared) schedule(dynamic)
        for(int v=0; v<N; ++v){
            float dist = sqrt(disp_x[v]*disp_x[v] + disp_y[v]*disp_y[v]);
            posx[v] += (dist > thr)? disp_x[v]/dist*thr : disp_x[v];
            posy[v] += (dist > thr)? disp_y[v]/dist*thr : disp_y[v];
            posx[v] = min(W/2, max(-W/2,posx[v]));
            posy[v] = min(H/2, max(-H/2,posy[v]));
            disp_x[v] = 0;
            disp_y[v] = 0;
        }
        thr *= 0.99;
    }
    double calc_time = duration_cast<dsec>(Clock::now() - calc_start).count();
    cout << "Time: " << calc_time << endl;
    free(disp_x);
    free(disp_y);
}

int main(int argc, char* argv[]) {
   
    int N;
    int M;

    ifstream infile;
    infile.open(argv[1]);
    infile >> N >> M;
    float K = sqrt(1.0*W*H/N);
    float *posx = new float[N];
    float *posy = new float[N];
    Edge *E = new Edge[2*M];
    int *edges = new int[4*M];
    int *Idx = new int[N]();
   
    Edge e;
    for(int i=0; i<2*M; i+=2){
        infile >> e.idx1 >> e.idx2;
        E[i] = e;
        swap(e.idx1, e.idx2);
        E[i+1] = e;
    }
    sort(E, E+2*M, cmp);
    for(int i=0; i<2*M; ++i) {
        Idx[E[i].idx1] += 1;
    }
    for(int i=1; i<N; ++i) {
        Idx[i] += Idx[i-1]; //End Index
    }
    for(int i=1; i<2*M; ++i) {
        edges[2*i] = E[i].idx1;
        edges[2*i+1] = E[i].idx2;
    }
    int iteration = atoi(argv[2]), numProcs = atoi(argv[3]);
    int thr = W+H;


#ifdef RUN_MIC /* Use RUN_MIC to distinguish between the target of compilation */

  /* This pragma means we want the code in the following block be executed in 
   * Xeon Phi.
   */
#pragma offload target(mic) \
    inout(posx: length(N) INOUT)    \
    inout(posy: length(N) INOUT)    \
    inout(edges: length(4*M) INOUT)     \
    inout(Idx: length(N) INOUT)
#endif
    {
        cout << "max procs = " << omp_get_num_procs() << endl;
        omp_set_num_threads(numProcs);
        force_directed(posx, posy, edges, Idx, N, M, K, iteration, thr);
    }
    
    ofstream outfile("Vertex_Pos.txt");
    for (int v=0; v<N; ++v){
        outfile << posx[v] << ' ' << posy[v] <<endl;
    }
    outfile.close();

    free(posx);
    free(posy);
    free(E);
    free(edges);
    free(Idx); 
    return 0;
}
