## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/qdHe/15618-final-project/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/qdHe/15618-final-project/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

# CUDA Fruchterman-Reingold - Project Proposal

Parallel network data visualization using GPU.

## Summary

We are going to parallelize an image stitching program that aligns a set of images and stitch them together to produce a panorama. We will compare the speedup and quality of our parallel algorithm with the sequential version.

## Background

_Graph drawing_ shows a graph based on the topological relationship between vertices and edges. One category of typical algorithms to draw graphs in an aesthetically-pleasing way is called forced-directed graph drawing. The idea of a force directed layout algorithm is to consider a force between any two nodes. In this project, we want to implement and optimize a specific version called Fruchterman-Reingold. The nodes are represented by steel rings and the edges are springs between them. The attractive force is analogous to the spring force and the repulsive force is analogous to the electrical force. The basic idea is to minimize the energy of the system by moving the nodes and changing the forces between them.

<img src="https://user-images.githubusercontent.com/16803685/32248395-f8698b64-be5b-11e7-933c-25ecd84771af.png" alt="img0" width="800" align="middle" />

Suppose k is , attractive force is:
re
算法初始时给每个顶点分配一个随机位置(可以组成圆形，可以是网格，也可以是其他布局算法的输出结果，但不能排在一条直线上(想一想为什么))，核心是个迭代过程，计算出所有点对间的斥力，再对于每个顶点，考虑和它关连的弹簧对它产生的引力。每一轮迭代枚举每个顶点，根据它受到的合力向量让它的位置发生改变。当所有顶点位置不发生改变或者迭代次数超过预设的某个阈值后算法结束。
伪代码如下：





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

