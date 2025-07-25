# 复现parscale


pip install transformers==4.48.1

## 从parscale模型结构初始化模型，Qwen2复制权重
### 初始化模型
python init_model_3.py
### 进行PT （注释掉设置freeze的代码）
python train/demo/demo_freeze.py
