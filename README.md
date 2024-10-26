# 处理 fire_xiaojie.mp4 数据

处理 fire_xiaojie.mp4 数据的程序在 Fire_Video_Dataprocessor.py中

Fire_Original_Dataset : 将 fire_xiaojie.mp4 转换成 图片序列得到的数据集，有140张图片
Fire_Interpolation_Dataset : 对原140张图片进行插帧（4倍帧率），得到插帧后的数据集，有560张图片

# Result : 
1. 包含10张图片，每张图片由 5000个事件累加得到
2. events_images.txt : 保存了上述构成10张图片的所有事件数据。          shape：（10*5000，4）
3. voxel_images.txt : 保存最终生成的 voxel grids 数据。              shape：（5，420，420）
4. normaled_voxel_images.txt : 保存了正则化后的 voxel grids 数据。   shape：（5，420，420）


## Final Results
最终得到 hdf5 文件，其数据结构如下：
<img width="486" alt="image" src="https://github.com/user-attachments/assets/7ee6f5ad-eb01-46eb-87a1-13858e2bd70c">


 # Run Results
 获得上述 hdf5文件后 模型能够成功运行，然而却报错 “电脑的CUDA内存不足，错误信息显示，程序试图分配4.78GiB的内存，然而GPU只有487.44MiB的闲置空间”，具体如下图：
![截图 2024-10-26 17-38-36](https://github.com/user-attachments/assets/0e8256a1-b50c-4c00-bf72-5aa46acd5ef8)# EvTexture_DataProcessor
![截图 2024-10-26 17-39-05](https://github.com/user-attachments/assets/02e41482-3956-434b-84d5-0e6498697682)


 






