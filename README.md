# multi-granularity_seletive_search

## This is the code base and explanation of the paper [Candidate region acquisition optimization algorithm based on multi-granularity data enhancement](https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=51yJbQ0AAAAJ&citation_for_view=51yJbQ0AAAAJ:IjCSPb-OGe4C).

### Abstract
Given the deepening network hierarchy of deep learning, improving the accuracy of the candidate region acquisition algorithm can help save time and improve operational efciency in subsequent work. Since the traditional methods overly rely on single-grain size, color and texture features of images, which can easily lead to candidate frames cutting of the foreground object when acquiring candidate regions, this paper proposes a multi-granularity selective search algorithm (MGSS) for candidate region acquisition by extracting the main features such as outline, texture and color of images with multiple grain sizes and improving the subgraph similarity calculation method.This paper mainly compares the performance of previous common algorithms on Pascal VOC 2012 and 2007 datasets, and the experiments show that the method used in this paper maintains the Mean Average Best Overlap (MABO) values of 0.909 and 0.890, which is 9.55% and 2.05% better than the Selective Search (SS)“Fast” and SS “Quality” results, respectively. The experiments show that both R-CNN and Fast R-CNN algorithms improve mAP (mean Average Precision) values by 1.5, 0.8 and 0.6 % with MGSS respectively, over with the traditional SS algorithm and RPN algorithm.

### Keywords:
Multi-granularity · Object detection · Watershed algorithm · Perceptual hashing

* Multi‑Granularity Selective Search Algorithm

* Coarse‑grained images for outline feature

![watershed algorithm](https://github.com/Alan-D-Chen/multi-granularity_seletive_search/blob/main/pics/%E6%88%AA%E5%B1%8F2023-03-16%2010.58.16.png)

![multi-granularity images](https://github.com/Alan-D-Chen/multi-granularity_seletive_search/blob/main/pics/%E6%88%AA%E5%B1%8F2023-03-16%2010.55.56.png)

* Color similarity of images

* Texture similarity of images

* Experiments and analysis

![conclusion show](https://github.com/Alan-D-Chen/multi-granularity_seletive_search/blob/main/pics/%E6%88%AA%E5%B1%8F2023-03-16%2010.56.59.png)


### Conclusion
This paper focuses on the acquisition method of candidate regions at the basic level, and creatively enriches the acquisition method of candidate region for object detection, MGSS, which improves 9.55% and 2.05% compared with the results of SS “Fast” and SS “Quality”, respectively. We also verify that mAP values of R-CNN and Fast R-CNN algorithms improve by 1.5 and 0.8 %, respectively, compared with the traditional SS algorithm. The MGSS is more adept at identifying classes with complex backgrounds or hierarchical complexity compared to the RPN method.
