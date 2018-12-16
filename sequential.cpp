#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <stdlib.h>
#include <math.h>
using namespace std;

#define W 10
#define H 10
int N;
int M;
float K;

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
inline float repulsive_force(float dist){
    return K*K/dist/N/100.0;
}

inline float attractive_force(float dist){
    return dist*dist/K/N;
}

void force_directed(Vertex *V, Edge *E, int *Idx, int N, int Iteration, float thr){
    for(int itr=0; itr<Iteration; ++itr){
        for(int v=0; v<N; ++v){
			// Repulsive Force
            for(int u=0; u<N; ++u){
                float d_x = V[v].x-V[u].x;
                float d_y = V[v].y-V[u].y;
                float dist = sqrt(d_x*d_x+d_y*d_y);
                dist = max(dist, 0.001);
                float rf = repulsive_force(dist);
                V[v].disp_x += d_x/dist*rf;
                V[v].disp_y += d_y/dist*rf;
            }

			// Attractive Force
            if (v == 0){
                for(int e=0; e<Idx[v]; ++e){
                    int u = E[e].idx2;
                    float d_x = V[v].x-V[u].x;
                    float d_y = V[v].y-V[u].y;
                    float dist = sqrt(d_x*d_x+d_y*d_y);
                    dist = max(dist, 0.001);
                    float af = attractive_force(dist);
                    V[v].disp_x -= d_x/dist*af;
                    V[v].disp_y -= d_y/dist*af;
                }
            }
            else {
                for(int e=Idx[v-1]; e<Idx[v]; ++e){
                    int u = E[e].idx2;
                    float d_x = V[v].x-V[u].x;
                    float d_y = V[v].y-V[u].y;
                    float dist = sqrt(d_x*d_x+d_y*d_y);
                    dist = max(dist, 0.001);
                    float af = attractive_force(dist);
                    V[v].disp_x -= d_x/dist*af;
                    V[v].disp_y -= d_y/dist*af;
                }
            }
        }

        // Update
		for(int v=0; v<N; ++v){
            float dist = sqrt(V[v].disp_x*V[v].disp_x + V[v].disp_y*V[v].disp_y);
            V[v].x += (dist > thr)? V[v].disp_x/dist*thr : V[v].disp_x;
            V[v].y += (dist > thr)? V[v].disp_y/dist*thr : V[v].disp_y;
            V[v].x = min(W/2, max(-W/2,V[v].x));
            V[v].y = min(H/2, max(-H/2,V[v].y));
            V[v].disp_x = 0;
            V[v].disp_y = 0;
        }
        thr *= 0.99;
    }
}

int main(int argc, char* argv[]) {

    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;


    ifstream infile;
	cout<<argv[0]<<endl;
    infile.open(argv[1]);
    infile >> N >> M;
    K = sqrt(1.0*W*H/N);

    Vertex *V = new Vertex[N]();
    Edge *E = new Edge[2*M];
    int *Idx = new int[N]();

	// Get E array
    Edge e;
    for(int i=0; i<2*M; i+=2){
        infile >> e.idx1 >> e.idx2;
        E[i] = e;
        swap(e.idx1, e.idx2);
        E[i+1] = e;
    }
    sort(E, E+2*M, cmp);
    
	// Get Idx array
	for(int i=0; i<2*M; ++i) {
        Idx[E[i].idx1] += 1;
    }
    for(int i=1; i<N; ++i) {
        Idx[i] += Idx[i-1]; //End Index
    }
    
	
	int iteration = atoi(argv[2]);
    int thr = W+H;
    
	// Key Computation
	auto calc_start = Clock::now();
    force_directed(V, E, Idx, N, iteration, thr);
    double calc_time = duration_cast<dsec>(Clock::now() - calc_start).count();
    cout << "Time: " << calc_time << endl;
    
	// Output results
	ofstream outfile("Vertex_Pos.txt");
    for (int v=0; v<N; ++v){
        outfile << V[v].x <<' '<<V[v].y<<endl;
    }
    outfile.close(); 
    
	return 0;
}
