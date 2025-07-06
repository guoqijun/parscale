# 复现parscale


pip install transformers==4.48.1

## 从parscale模型结构初始化模型
### 初始化模型
python init_model.py
### 进行PT （注释掉设置freeze的代码）
python train/demo/demo_freeze.py

loss不正常
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 4.464285714285714e-07, 'epoch': 0.0}                                                          
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 8.928571428571428e-07, 'epoch': 0.0}                                                          


## 从Qwen2模型结构初始化模型
python init_model_2.py
### 进行PT （注释掉设置freeze的代码）
python train/demo/demo_freeze.py
正常训练


## 从parscale模型结构初始化模型，Qwen2复制权重
### 初始化模型
python init_model_3.py
### 进行PT （注释掉设置freeze的代码）
python train/demo/demo_freeze.py

loss不正常
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 4.464285714285714e-07, 'epoch': 0.0}                                                          
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 8.928571428571428e-07, 'epoch': 0.0}                                                          
