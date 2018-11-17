#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdlib.h>
#include <cmath>
using namespace std;

#define K 0.1
#define W 10
#define H 10

class Vertex{
public:
    float x;
    float y;
    float disp_x;
    float disp_y;

    Vertex():x(float(rand())/RAND_MAX), y(float(rand())/RAND_MAX), disp_x(0), disp_y(0){}
};

float max(float a, float b){
    return (a>b)? a:b;
}
float min(float a, float b){
    return (a<b)? a:b;
}
inline float repulsive_force(float dist){
    return K*K/dist/10000;
}

inline float attractive_force(float dist){
    return dist*dist/K/10000;
}

void force_directed(Vertex *V, int **E, int N, int Iteration, float thr){
    for(int itr=0; itr<Iteration; ++itr){
        cout << "Iteration = " << itr+1 << endl;
        for(int v=0; v<N; ++v){
            for(int u=v+1; u<N; ++u){
                float d_x = V[v].x-V[u].x;
                float d_y = V[v].y-V[u].y;
                float dist = sqrt(d_x*d_x+d_y*d_y);
                dist = max(dist, 0.0001);
                float rf = repulsive_force(dist);
                float af = E[v][u]*attractive_force(dist);
                //if(v%1000 && u%1000) cout<<rf<<' '<<af<<' '<<dist<<endl;
                V[v].disp_x += d_x/dist*(rf-af);
                V[v].disp_y += d_y/dist*(rf-af);
                V[u].disp_x += d_x/dist*(af-rf);
                V[u].disp_y += d_y/dist*(af-rf);
            }
        }
        for(int v=0; v<N; ++v){
            //if(v%1000==0) cout<<V[v].disp_x<<' '<<V[v].disp_y<<endl;
            float dist = sqrt(V[v].disp_x*V[v].disp_x + V[v].disp_y*V[v].disp_y);
            V[v].x += (dist > thr)? V[v].disp_x/dist*thr : V[v].disp_x;
            V[v].y += (dist > thr)? V[v].disp_y/dist*thr : V[v].disp_y;
            V[v].x = min(1, max(0,V[v].x));
            V[v].y = min(1, max(0,V[v].y));
            V[v].disp_x = 0;
            V[v].disp_y = 0;
        }
        thr -= 0.01;
    }
}

int main() {
    int N = 9000;//tentative
    Vertex *V = new Vertex[N];
    int **E = new int*[N];
    for (int i=0; i<N; ++i){
        E[i] = new int[N]();
    }
    ifstream infile;
    infile.open("data/Wiki-Vote.txt");
    int idx1, idx2;
    while(!infile.eof()){
        infile >> idx1 >> idx2;
        //cout<<idx1<<' '<<idx2<<endl;
        E[idx1][idx2] = E[idx2][idx1] = 1;
    }
    cout << "Complete Initialization" << endl;
    int iteration = 10;
    int thr = 1;
    force_directed(V, E, N, iteration, thr);
    ofstream outfile("Vertex_Pos.txt");
    for (int v=0; v<N; ++v){
        outfile << V[v].x <<' '<<V[v].y<<endl;
    }
    outfile.close(); 
    return 0;
}
