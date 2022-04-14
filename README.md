# driver2vec_Reproduced
A reproduction of the paper [Driver2vec: Driver Identification from Automotive Data](https://arxiv.org/abs/2102.05234). This paper aims to identify drivers
through their driving habits, making use of a triplet loss function when comparing the drivers within a set. This model also makes use of a Temporal Convolutional
Network (TCN) to output a driver's driving "fingerprint" that can be used to identify a driver within a set of drivers with varying degrees of accuracy depending on the
number of drivers in a given set. After the TCN, the model then makes use of LightGBM, a gradient boosting method that is used to ultimately predict the correct driver
from a small sample of the total driving time within a set.
