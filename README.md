<h1>Rock Scissors Paper Classification</h1>

Simple CNN architecture demo.

This repository includes small size of training and test data. 
All data was pictured by me and my friends.

<h2>Project details</h2>

|||
|---|---|
|Period|2021.01 ~ 2021.01|
|Team|None|
|Accuracy|0.82|

|Tech|Detail|
|---|---|
|<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>|Main algorism|
|<img src="https://img.shields.io/badge/Tensorflow-FF6F00?style=flat-square&logo=Tensorflow&logoColor=white"/>|Prediction model|

<h2>Model Summary</h2>

```model summary
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
                                                                 
 max_pooling2d_2 (MaxPooling  multiple                 0         
 2D)                                                             
                                                                 
 conv2d_8 (Conv2D)           multiple                  1792      
                                                                 
 conv2d_9 (Conv2D)           multiple                  36928     
                                                                 
 conv2d_10 (Conv2D)          multiple                  73856     
                                                                 
 conv2d_11 (Conv2D)          multiple                  147584    
                                                                 
 flatten_2 (Flatten)         multiple                  0         
 
 dropout_2 (Dropout)         multiple                  0         
                                                                 
 dense_4 (Dense)             multiple                  131328    
                                                                 
 dense_5 (Dense)             multiple                  771       
                                                                 
=================================================================
Total params: 392,259
Trainable params: 392,259
Non-trainable params: 0
_________________________________________________________________
```


<h2>Limitation</h2>

1. image augmentation 기법으로 데이터를 늘리기는 하지만, 여전히 작은 크기의 dataset으로 overfit이 크게 발생합니다.
이는 미리 학습된 resnet 등의 모델을 backbone으로 활용해 transfer learning을 시도하여 개선할 수 있습니다.
2. 본 코드는 CPU에서 구동되므로 GPU를 활용하는 것에 비해 학습 속도가 매우 느립니다.