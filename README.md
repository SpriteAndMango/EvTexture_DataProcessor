# EvTexture_DataProcessor

The original_dataset : REDS dataset/Lower Resolution/test_sharp_bcubic/000   (100 pictures)

The frame_interpolation_dataset : use pretrained RIFE to interpolate frames  (397 pictures)


Error：使用esim.generateFromVideo()为视频生成events时报错: 
        ‘MemoryError: std::bad_alloc’
        。 github 上早些时候也有人遇到类似的问题，后面解决之后似乎最近又有这种情况出现。
