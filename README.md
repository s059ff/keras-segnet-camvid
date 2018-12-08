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

* ネットワークのテスト

    ```bash
    python validate.py
    ```

## 実行結果
入力データ, 予測データ, 正解データ

![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-0-input.png)
![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-0-prediction.png)
![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-0-teacher.png)

![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-1-input.png)
![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-1-prediction.png)
![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-1-teacher.png)

![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-2-input.png)
![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-2-prediction.png)
![](https://github.com/s059ff/keras-segnet-camvid/blob/master/examples/test-2-teacher.png)
