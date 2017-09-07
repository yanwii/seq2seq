## 基于Pytorch的中文聊天机器人 集成BeamSearch算法  

Pytorch 厉害了！    

---
Requirements:   
[**Python3**](https://www.python.org/)  
[**Pytorch-0.1.11**](https://github.com/pytorch/pytorch)   
[**Jieba分词**](https://github.com/fxsjy/jieba)

---

### Pytorch 安装
        python2.7
        pip install http://download.pytorch.org/whl/cu80/torch-0.1.11.post5-cp27-none-linux_x86_64.whl
        pip install torchvision
        
        python3.5
        pip install http://download.pytorch.org/whl/cu80/torch-0.1.11.post5-cp35-cp35m-linux_x86_64.whl
        pip install torchvision
        
        python3.6
        pip install http://download.pytorch.org/whl/cu80/torch-0.1.11.post5-cp36-cp36m-linux_x86_64.whl
        pip install torchvision

### 关于BeamSearch算法
很经典的贪心算法，在很多领域都有应用。

![](./img/beamsearch.png)


在这个引用中 我们引入了惩罚因子
![](./img/beamsearch2.jpeg)


![](./img/1.png)


---

### 用法  

        # 准备数据
        python3 preprocessing.py
        # 训练
        python3 seq2seq.py train
        # 预测
        python3 seq2seq.py predict
        # 重新训练
        python3 seq2seq.py retrain

### 以下是k=5时的结果, 越接近1，结果越好

        me > 我是谁
        drop [3, 1], 1
        drop [1, 6, 1], 2
        drop [7, 6, 1], 3
        drop [4, 5, 6, 1], 4
        drop [7, 6, 8, 1], 5
        ai >  __UNK__ -1.92623626371
        ai >   -1.41548742168
        ai >  关你 -1.83084125204
        ai >  我是你 0.0647218796512
        ai >  关你屁事 -0.311924366579

