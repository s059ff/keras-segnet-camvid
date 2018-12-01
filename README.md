# keras-segnet-camvid
CamVid データセットを使用したSegNetによる自動車のセグメンテーションのテストです。

## 実行環境
* Tensorflow 1.12.0
* Keras 2.2.4
* OpenCV 3.4.2

## 実行方法
* ネットワークのトレーニング
    ```bash
    python train.py
    ```
    * 各エポックにおいて、ネットワークのパラメータが./temp/model-####.h5に保存されます。

* ネットワークのテスト
    ```bash
    python evaluate.py --model "./temp/model-####.h5"
    ```
    または
    ```bash
    python evaluate.py -m "./temp/model-####.h5"
    ```

## 実行結果
入力データ, 予測データ, 入力データと予測データの合成, 教師データ

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
