# EvTexture_DataProcessor

The original_dataset : REDS dataset/Lower Resolution/test_sharp_bcubic/000   (100 pictures)

The frame_interpolation_dataset : use pretrained RIFE to interpolate frames  (397 pictures)

# 处理 fire_xiaojie.mp4 数据

处理 fire_xiaojie.mp4 数据的程序在 Fire_Video_Dataprocessor.py中

Fire_Original_Dataset : 将 fire_xiaojie.mp4 转换成 图片序列得到的数据集，有140张图片
Fire_Interpolation_Dataset : 对原140张图片进行插帧（4倍帧率），得到插帧后的数据集，有560张图片

# Result : 
1. 包含10张图片，每张图片由 5000个事件累加得到
2. events_images.txt : 保存了上述构成10张图片的所有事件数据。          shape：（10*5000，4）
3. voxel_images.txt : 保存最终生成的 voxel grids 数据。              shape：（5，420，420）
4. normaled_voxel_images.txt : 保存了正则化后的 voxel grids 数据。   shape：（5，420，420）


