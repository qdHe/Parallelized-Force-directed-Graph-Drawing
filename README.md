# CUDA forced-directed graph drawing - Project Proposal
Parallel network data visualization using GPU.

## Summary

We are going to parallelize a forced-directed graph drawing algorithm that consider a force between two nodes to draw a aesthetically-pleasig graph. We will compare the speedup and quality of our parallel algorithm with the sequential version.

## Background

_Graph drawing_ shows a graph based on the topological relationship between vertices and edges. One category of typical algorithms to draw graphs in an aesthetically-pleasing way is called forced-directed graph drawing. The idea of a force directed layout algorithm is to consider a force between any two nodes. In this project, we want to implement and optimize a specific version called Fruchterman-Reingold. The nodes are represented by steel rings and the edges are springs between them. The attractive force is analogous to the spring force and the repulsive force is analogous to the electrical force. The basic idea is to minimize the energy of the system by moving the nodes and changing the forces between them.

<img src="" alt="img0" width="800" align="middle" />

Suppose k is , attractive force is:

~~~ golang
area := W * L; //W and L are the width and length of the frame
G := (V, E); //the vertices are assigned random initial positions 
k :=  sqrt(area/V)    
func fa(x) := return x^2/k 
func fr(x) := return k^2/x
for i := 1 -> iterations
    //calculate repulsive forces
    for v in V //each vertex has two vectors: .pos and .disp
        v.disp := 0; 
        for u in V
            d := v.pos - u.pos
            v.disp += (d/|d|) * fr(|d|)
    //calculate attractive forces
    for e in E //each edge is an ordered pair of vertices .v and .u 
        d := e.v.pos – e.u.pos 
        e.v.disp –= (d/|d|) * fa(|d|) 
        e.u.disp += (d/|d|) * fa(|d|)
    //limit the maximum displacement to the temperature t
    //prevent from being displaced outside frame
    for v in V
        v.pos += (v.disp/|v.disp|) * min(v.disp, t)
        v.pos.x = min(W/2, max(-W/2, v.pos.x))
        v.pos.y = min(L/2, max(–L/2, v.pos.y)) 
    //reduce the temperature as the layout approaches a better configuration
    t := cool(t)
~~~




## Challenges

The first challenge is that how to assign jobs evenly, since the amount of work per body is not uniform in each iteration. The amount of work for a body in a group with multiple of bodies is different from a body that's far away with most of the body groups. Also, bodies will move during two iterations. So the cost and communication patterns will also change over time. 
The second challenge is that how to handle collision between different bodies. Currently, our team hasn't decided how to handle this case. If we need to consider collision in this problem, it will make it more challenging, because the position update of one body may need to consider other bodies' influence and re-calculate where the body will go after that. 

### Dependency


### Memory Access


### Communication



## Resources
We will build the program from scratch in C++. We follow the guidance from the following references:

[1] FRUCHTERMAN T M J, REINGOLD E M. Graph drawing by force-directed placement[J]. Software, practice & experience, 1991, 21(11): 1129-1164.

[2] Jacomy M, Venturini T, Heymann S, et al. ForceAtlas2, a continuous graph layout algorithm for handy network visualization designed for the Gephi software[J]. PloS one, 2014, 9(6): e98679.

[3] https://www.boost.org/doc/libs/1_55_0/boost/graph/distributed/fruchterman_reingold.hpp

We also need NVIDIA GPU resourse as we want to parallel the algorithm through CUDA.

## Goals and Deiverables

### Plan to achieve
- 1: Write the CUDA version of force-directed algorithm and run on GPU.
- 2: Parallelize the algorithm and greatly imrpove the performance. 
- 3: The complexity of the original algorithm is O(n^3), we will reduce the total complexity using algorithm such as QuadTree.

### Hope to achieve
- 1: Propose massive data in real time.
- 2: Write the parallel CPU version of force-directed algorithm with OpenMP and run on Xeon Phi. Compare the performance of parallel GPU version and CPU version.

### Demo
- 1: We will show our speedup graphs which compare the performance of different versions of algorithms/different size of input data.
- 2: We will show the output visualization images of our program and guarantee the quality is similar to sequential version.
- 3: We may also show how the visualization images evolve with iterations.

## Platform

We plan to use the Latedays cluster to run our code, which will use Tesla K40 GPU. One of the main disadvantage of force-directed algorithms is high running time with large amounts of data. Therefore, we want to parallelize the algorithm with GPU to imrpove the performance as GPU has good ability of computing and parallel. Also, Latedays has Xeon Phi which enables parallel through OpenMP as well. For consistency, we will also run the sequential version on Latedays. 

## Schedule

* **Week 1 11.5--11.11**

Understand the algorithm and the code in C++ boost library. Start the sequential implementation in C++.

* **Week 2 11.12--11.18**

Finish the sequential version of the program and analyze the performance by fine-grained timing. Identify the bottleneck and come up with an approach to parallelize the program.

* **Week 3 11.19--11.25**

Start programming the parallel version of the program.

* **Week 4 11.26--12.2**

Iterate on the parallel version and optimize performance.

* **Week 5 12.3--12.9**

Finish parallelizing the program and generate results. If there is time left, write the parallel CPU version using OpenMP and compare performance.

* **Week 6 12.10--12.16**

Wrap up the project. Write final report. Prepare video and poster for demo.


## Authors

* **Qidu He** -  [qiduh]
* **Di Jin** -  [djin2]
