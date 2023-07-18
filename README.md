# PrecipitationFusion
 

 This repository contains codes for the paper entitled <a href="https://www.sciencedirect.com/science/article/pii/S0034425723002742?via%3Dihub#f0025)" target="_blank">"A deep learning-based framework for multi-source precipitation fusion"</a> published in Remote Sensing of Environment Journal. The paper was authored by Keyhan Gavahi, Ehsan Foroumandi, and Hamid Moradkhani. In this paper, we proposed a DL framework for multi-source precipitation data fusion and developed a fused product, PDFN, that improved the accuracy by 35%.

<p align="center">
<img src="figures/boxplots.png" width="600" height="400">
</p>


<p align="center">
<img src="figures/20150103_comparison22_rev.png" width="600" height="675">
</p>


```
@article{GAVAHI2023113723,
title = {A deep learning-based framework for multi-source precipitation fusion},
journal = {Remote Sensing of Environment},
volume = {295},
pages = {113723},
year = {2023},
issn = {0034-4257},
doi = {https://doi.org/10.1016/j.rse.2023.113723},
url = {https://www.sciencedirect.com/science/article/pii/S0034425723002742},
author = {Keyhan Gavahi and Ehsan Foroumandi and Hamid Moradkhani},
keywords = {Precipitation fusion, Remote sensing, Deep learning, Convolutional neural networks (CNN), Convolutional long short-term memory (ConvLSTM)},
abstract = {Accurate quantitative precipitation estimation (QPE) is essential for various applications, including land surface modeling, flood forecasting, drought monitoring and prediction. In situ precipitation datasets, remote sensing-based estimations, and reanalysis products have heterogeneous uncertainty. Numerous models have been developed to merge precipitation estimations from different sources to improve the accuracy of QPE. However, many of these attempts are mainly focused on spatial or temporal correlations between various remote sensing sources and/or gauge data separately, and thus, the developed model cannot fully capture the inherent spatiotemporal dependencies that could potentially improve the precipitation estimations. In this study, we developed a general framework that can simultaneously merge and downscale multiple user-defined precipitation products by using rain gauge observations as target values. A novel deep learning-based convolutional neural network architecture, namely, the precipitation data fusion network (PDFN), that combines multiple layers of 3D-CNN and ConvLSTM was developed to fully exploit the spatial and temporal patterns of precipitation. This architecture benefits from techniques such as batch normalization, data augmentation schemes, and dropout layers to avoid overfitting and address skewed class proportions due to the highly imbalanced nature of the precipitation datasets. The results showed that the fused daily product remarkably improved the mean square error (MSE) and Pearson correlation coefficient (PCC) by 35% and 16%, respectively, compared to the best-performing product.}
}
```
