# HPCC-Hazardous-Driving
## 数据说明
数据存放于43.41服务器中`/data/VOLVO/HPCC_modelling_data`目录下，三个目录说明：
 - `data`：运动学参数，以npy格式存储，读取时使用`numpy`库的np.load方法
 - `label`：标签0/1（正常/危险驾驶场景），csv文件
 - `img`：motion profile的img图像，每一个视频对应一张图片
## 模型说明
`model.py`文件中各个模型类说明：
 - `BuildAlexNet`：原AlexNet模型
 - `BuildAlexNetSimple`：轻量级CNN模型
 - `BuildCBAMAlexNet`：轻量级CNN + CBAM注意力机制
 - `BuildCBAMAlexNetAll`：轻量级CNN + CBAM注意力机制 + 运动学参数（LSTM）
## 论文
 
    @inproceedings{guo2020towards,
        title =        {A Lightweight VK-Net Based on Motion Profiles for Hazardous Driving Scenario Identification},
        author =       {Gao, Zhen and Xu, Jingning and Zheng, Jiang Yu and Fan, Hongfei and Yu, Rongjie and Zong, Jiaqi and Li, Xinyi},
        booktitle =    {2021 IEEE 23rd Int Conf on High Performance Computing \& Communications; 7th Int Conf on Data Science \& Systems; 19th Int Conf on Smart City; 7th Int Conf on Dependability in Sensor, Cloud \& Big Data Systems \& Application (HPCC/DSS/SmartCity/DependSys)},
        year =         {2021},
        pages =        {908--913},
        organization = {IEEE}
    }
