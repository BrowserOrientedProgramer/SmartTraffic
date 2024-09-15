# SmartTraffic

## 介绍
SmartTraffic是一个交通场景感知系统，能够实时识别车辆，并进行车流量、车速、车距等实时监测。

## 使用工具
- [YOLO](https://docs.ultralytics.com/)
- [supervision](https://supervision.roboflow.com/latest/)

## 创建环境
```shell
conda create -n smarttraffic python=3.10
conda activate smarttraffic
pip install -r requirements.txt
```

## Downloading Video Assets
- vehicles.mp4
```python
from supervision.assets import VideoAssets, download_assets
download_assets(VideoAssets.VEHICLES)
```
- driving.mp4

链接: https://pan.baidu.com/s/1GCggxFWG3scO094XvAglwA?pwd=m4hi 提取码: m4hi

## 使用
车辆计数：
```python
python count.py
```
行车测距：
```python
python distance.py
```
车辆测速：
```python
python speed_estimation.py
```
车辆计数与测速：
```python
python speed_count.py
```

## 结果展示
车辆计数与测速：

https://github.com/user-attachments/assets/5afaa5c6-f979-427d-bf28-0bc710cdea23

行车测距：

https://github.com/user-attachments/assets/d5c8bf95-9ef8-421e-89ba-c312b3c25c6b


