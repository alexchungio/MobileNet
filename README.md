# MobileNet
MobileNet 

## dataset
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
## image_preprocessing
https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
## mobile  pre train model
### mobilenet.ckpt
https://github.com/alexchungio/models/tree/master/research/slim/nets/mobilenet

## train and validation result

### hyper parameter config
* batch size: 32
* learning rate: 0.0001
* decay rate: 0.9
* num epoch percent decay: 20
* weight decay: 0.00004
* dropout keep prob: 0.8
* epoch: 30

### depth multiplier == 1.4
| |loss| accuracy
---|---|---
train| 1.0293749570846558|0.96875
val| 1.0631169080734253|0.9320651888847351

### depth multiplier == 1.0
| |loss| accuracy
---|---|---
train| 1.0218572616577148|0.96875
val| 1.0486029386520386|0.9429348111152649

### depth multiplier == 0.75
| |loss| accuracy
---|---|---
train|  1.0015695095062256|0.96875
val| 1.030778408050537| 0.936141312122345

### depth multiplier == 0.50
| |loss| accuracy
---|---|---
train|  0.9928182363510132|0.9375
val| 1.0167617797851562| 0.926630437374115