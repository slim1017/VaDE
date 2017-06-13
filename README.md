# VaDE
This repository contains the Python implementation for our generative clustering method VaDE. 

Further details about VaDE can be found in our paper:

**Variational Deep Embedding : A Generative Approach to Clustering**
Requirements
=================
* Python-3.4.4
* keras-1.1.0
* scikit-learn-1.17.1

**Replace** `keras/engine/training.py` by `training.py` in this repository!!

(The modification version of `keras/engine/training.py` enables the simultaneous updating of the gmm parameters and the network parameters in our model.)

Usage
=================

* To train the VaDE model on the MNIST, Reuters, HHAR datasets:
```shell
python ./VaDE.py db
```
db can be one of mnist,reuters10k,har.

* To achieve the 94.46% clustering accuracy on the MNIST dataset and generate the class-specified digits (Note that: unlike Conditional GAN, we do not use any supervised information during training):
```shell
python ./VaDE_test_mnist.py
```

* To achieve the 79.38% clustering accuracy on the Reuters(685K) dataset:
```shell
cd $VaDE_ROOT/dataset/reuters
./get_data.sh
cd $VaDE_ROOT
python ./VaDE_test_reuters_all.py
```

**Note**: the data preprocessing code for the Reuters dataset is taken from (https://github.com/piiswrong/dec).


Face generation by VaDE on CelebA
=================
(DCGAN-like network architecture)
* Attribute-conditioned generation (sampled from each cluster) without using any supervised information

1-6 rows: 1.black/short hair, man; 2.black/long hair, woman; 3.gold/long hair, woman; 4.bald, sunglasses, man; 5.left side face, woman; 6.right side face, woman;
![image](https://github.com/slim1017/VaDE/blob/master/cluster_generation.jpg)

* Interpolation between cluster centers in latent space
![image](https://github.com/slim1017/VaDE/blob/master/interpolation.jpg)

* Vector arithmetic in latent space：right + left = front
![image](https://github.com/slim1017/VaDE/blob/master/arithmetic.jpg)
