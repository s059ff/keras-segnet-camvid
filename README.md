# keras-segnet-camvid
CamVid データセットを使用したSegNetによる自動車のセグメンテーションのテストです。

## 実行環境
* Tensorflow 1.12.0
* Keras 2.2.4
* OpenCV 3.4.2

## 実行方法
* ネットワークのトレーニング

    ```
    > python train.py
    ```

    ```
    > python train.py -h
    usage: train.py [-h] [-e EPOCHS] [--checkpoint_interval CHECKPOINT_INTERVAL] [--batch_size BATCH_SIZE] [--onmemory]

    optional arguments:
    -h, --help            show this help message and exit
    -e EPOCHS, --epochs EPOCHS
                          The number of times of learning. default: 100
    --checkpoint_interval CHECKPOINT_INTERVAL
                          The frequency of saving model. default: 10
    --batch_size BATCH_SIZE
                          The number of samples contained per mini batch. default: 1
    --onmemory            Whether store all data to GPU. If not specified this option, use both CPU memory and GPU memory.
    ```

* ネットワークのテスト

    ```
    > python evaluate.py
    ```
    
    ```
    > python evaluate.py -h
    usage: evaluate.py [-h] [-m MODEL] [-n NUM]

    optional arguments:
    -h, --help            show this help message and exit
    -m MODEL, --model MODEL
                          The model file path pattern. You can use symbol of * and **.
    -n NUM, --num NUM     The number of samples to evaluate. default: 10
    ```

## 実行結果
入力データ, 出力データ, 入力データと出力データの合成, 教師データ

<!-- ![](./examples/test-2-input.png) -->
<!-- ![](./examples/test-2-prediction.png) -->
<!-- ![](./examples/test-2-prediction+.png) -->
<!-- ![](./examples/test-2-teaching.png) -->

![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-2-input.png)
![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-2-prediction.png)
![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-2-prediction+.png)
![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-2-teaching.png)

<!-- ![](./examples/test-4-input.png) -->
<!-- ![](./examples/test-4-prediction.png) -->
<!-- ![](./examples/test-4-prediction+.png) -->
<!-- ![](./examples/test-4-teaching.png) -->

![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-4-input.png)
![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-4-prediction.png)
![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-4-prediction+.png)
![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-4-teaching.png)

<!-- ![](./examples/test-6-input.png) -->
<!-- ![](./examples/test-6-prediction.png) -->
<!-- ![](./examples/test-6-prediction+.png) -->
<!-- ![](./examples/test-6-teaching.png) -->

![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-6-input.png)
![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-6-prediction.png)
![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-6-prediction+.png)
![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-6-teaching.png)
