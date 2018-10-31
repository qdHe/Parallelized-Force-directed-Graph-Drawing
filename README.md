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

Automated panoramic image stitching is an interesting topic in computer vision. It consists of an interest point detector, a feature descriptor such as SIFT and BRIEF, and an image stitching tool based on feature matching. It extends the limitation of physical camera to capture scenes that cannot be captured in one frame, and easily generates the effects that would otherwise be expensive to produce. The program can be broken down into several highly parallelizable stages:

<img src="https://user-images.githubusercontent.com/16803685/32248395-f8698b64-be5b-11e7-933c-25ecd84771af.png" alt="img0" width="800" align="middle" />

**1) Interest Point Detection**: interest points provide an efficient representation of the image. Interest points are found using Difference of Gaussian (DoG), which can be obtained by subtracting adjacent levels of a Gaussian Pyramid.

<img src="https://user-images.githubusercontent.com/16803685/32247308-4e742ca2-be58-11e7-87ef-81cdaab4260b.png" alt="img2" width="800" align="middle" />

<img src="https://user-images.githubusercontent.com/16803685/32247312-523eccd4-be58-11e7-9b6c-e5fa2cc07e3a.png" alt="img3" width="500" align="middle" />


**2) Feature Descriptor**: feature descriptor characterizes the local information about an interest point. We will use either SIFT (Scale invariant feature descriptor) or BRIEF(Binary Robust Independent Elementary Features) as our choice of descriptor. If time allowed, we can experiment with different descriptors and compare the results.

**3) Matching Interest points**: match interest points using the distance of their descriptors.

<img src="https://user-images.githubusercontent.com/16803685/32247324-5722f748-be58-11e7-885f-cfc13b3831cb.png" alt="img4" width="800" align="middle" />

**4) Align images**: compute the alignment of image pairs by estimating their homography (projection from one image to another).

**5) Stitching**: crop and blend the aligned images to produce the final result. If time allowed, we will address the problem of vertical “drifting” between image pairs by applying some “straightening” algorithm.

<img src="https://user-images.githubusercontent.com/16803685/32247454-ce36ce90-be58-11e7-9cee-5a417001f309.png" alt="img5" width="800" align="middle" />

## Challenges

Building and parallelizing a panorama stitching program from scratch is a challenging work. The program is complex and consists of several stages that are dependent on each other. The stages in the process should be carried out sequentially, so after parallelizing each stage, we have to synchronize all the processors before proceeding to the next stage. This could incur large synchronization overhead. One challenge is to experiment with the trade-off between computation cost and synchronization cost.

To compute the interest points in each image, we need to generate a sequence of blurred versions of the original image, so when the size of the image or the number of images is large, the working set will inevitably not fit in cache. One challenge is to hide the memory access latency and reducing cache misses to achieve lower computation time.

From running the sequential version of the program in MATLAB, we believe that the performance bottleneck of the program lies in detecting the interest points, which involves convolving multiple filters and applying local neighborhood operations on the image. It is still a brute-force algorithm that searches every pixel as a potential interest point and thus incurs lots of computation. The highly expensive operation of processing local neighborhood of each pixel fits well with the data-parallel model. Therefore, we believe that the program will benefit greatly from parallel implementation.

### Dependency
The five stages in the pipeline are dependent on each other and should be carried out sequentially, yet there is a lot of parallelism in each stage, both within a single image and across multiple images.

### Memory Access
Processing pixels within an image exploits spatial locality.

### Communication
Communication happens when each image needs to match and stitch with its neighboring image. 


## Resources
We will build the program from scratch in C++. We follow the guidance from the following references:

[1] FRUCHTERMAN T M J, REINGOLD E M. Graph drawing by force-directed placement[J]. Software, practice & experience, 1991, 21(11): 1129-1164.

[2] Jacomy M, Venturini T, Heymann S, et al. ForceAtlas2, a continuous graph layout algorithm for handy network visualization designed for the Gephi software[J]. PloS one, 2014, 9(6): e98679.

[3] https://www.boost.org/doc/libs/1_55_0/boost/graph/distributed/fruchterman_reingold.hpp

We also need NVIDIA GPU resourses as we want to parallel the algorithm through CUDA.

## Goals and Deiverables

We plan to complete the implementation of the parallel image stitching algorithm using BRIEF descriptor on Xeon Phi and compare the speedup against sequential version of the program. Apart from the performance, we also care about the quality of the resulting image after stitching. We plan to have reasonable quality panorama-like image after performing the algorithm.

If the project goes well, we hope to implement the algorithm with other descriptors like SIFT and ORB[3]. We hope to achieve better image stitching quality (rotation-variant and noise-resistant) with these descriptors or better speedup. If the work goes slowly, 

For demo, we plan to present our result by a video as well as speedup graphs. We plan to demonstrate the process of processing a sequence of input images and results in the final panorama image via a video. In addition, we will show the speedup graphs of our parallel algorithm against the sequential version.


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

