import random
import esim_py
import numpy as np
import cv2
import math
import os
import torch
from datetime import datetime, timedelta
from abc import ABCMeta, abstractmethod
import h5py


config_images = {
	'refractory_period': 1e-4,
	'CT_range': [0.05, 0.5],
	'max_CT': 0.5,
	'min_CT': 0.02,
	'mu': 1,
	'sigma': 0.1,
	'H': 180,
	'W': 240,
	'log_eps': 1e-3,
	'use_log': True,
}

config_video = {
	'refractory_period': 1e-4,
	'CT_range': [0.05, 0.5],
	'max_CT': 0.5,
	'min_CT': 0.02,
	'mu': 1,
	'sigma': 0.1,
	'H': 420,
	'W': 420,
	'log_eps': 1e-3,
	'use_log': True,
}



def render(x, y, t, p, shape):
    img = np.full(shape=shape + [3], fill_value=255, dtype="uint8")
    img[y, x, :] = 0
    img[y, x, p] = 255
    return img

def render_events(events, count=5000, shape=[180, 320]):
    x = events[:, 0].astype(np.int32)
    y = events[:, 1].astype(np.int32)
    # t = events[:, 2]
    p = events[:, 3].astype(np.int32)
    idx_begin = 0
    cnt = events.shape[0] // count
    for i in range(cnt):
        img = render(x[idx_begin:idx_begin+count], y[idx_begin:idx_begin+count], None, p[idx_begin:idx_begin+count], shape)
        idx_begin += count
        yield img


def render_events_save(events, save_path ,count=5000, shape=[180, 320]):
    x = events[:, 0].astype(np.int32)
    y = events[:, 1].astype(np.int32)
    # t = events[:, 2]
    p = events[:, 3].astype(np.int32)
    idx_begin = 0
    cnt = events.shape[0] // count
    for i in range(cnt):
        img = render(x[idx_begin:idx_begin+count], y[idx_begin:idx_begin+count], None, p[idx_begin:idx_begin+count], shape)
        idx_begin += count
        print(img.shape)
        yield img
        image_path = os.path.join(save_path,f'{i:04d}.png')
        print(image_path)
        # cv2.imwrite(image_path,img)
        success = cv2.imwrite(image_path, img)
        if not success:
            print(f"Failed to save image at {image_path}")


def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    @param pxs Numpy array of integer typecast x coords of events
    @param pys Numpy array of integer typecast y coords of events
    @param dxs Numpy array of residual difference between x coord and int(x coord)
    @param dys Numpy array of residual difference between y coord and int(y coord)
    @returns Image
    """
    img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)
    return img



def binary_search_torch_tensor(t, l, r, x, side='left'):
    """
    Binary search implemented for pytorch tensors (no native implementation exists)
    @param t The tensor
    @param x The value being searched for
    @param l Starting lower bound (0 if None is chosen)
    @param r Starting upper bound (-1 if None is chosen)
    @param side Which side to take final result for if exact match is not found
    @returns Index of nearest event to 'x'
    """
    if r is None:
        r = len(t)-1
    while l <= r:
        mid = l + (r - l)//2;
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r



def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation=None, padding=True, default=0):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
    @param xs Tensor of x coords of events
    @param ys Tensor of y coords of events
    @param ps Tensor of event polarities/weights
    @param device The device on which the image is. If none, set to events device
    @param sensor_size The size of the image sensor/output image
    @param clip_out_of_range If the events go beyond the desired image size,
       clip the events to fit into the image
    @param interpolation Which interpolation to use. Options=None,'bilinear'
    @param padding If bilinear interpolation, allow padding the image by 1 to allow events to fit:
    @returns Event image from the events
    """
    if device is None:
        device = xs.device
    if interpolation == 'bilinear' and padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = (torch.ones(img_size)*default).to(device)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs-pxs).float()
        dys = (ys-pys).float()
        pxs = (pxs*mask).long()
        pys = (pys*mask).long()
        masked_ps = ps.squeeze()*mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        try:
            mask = mask.long().to(device)
            xs, ys = xs*mask, ys*mask
            img.index_put_((ys, xs), ps, accumulate=True)
        except Exception as e:
            print("Unable to put tensor {} positions ({}, {}) into {}. Range = {},{}".format(
                ps.shape, ys.shape, xs.shape, img.shape,  torch.max(ys), torch.max(xs)))
            raise e
    return img

def events_to_voxel_torch(xs, ys, ts, ps, B, device=None, sensor_size=(180, 240), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    @param xs List of event x coordinates (torch tensor)
    @param ys List of event y coordinates (torch tensor)
    @param ts List of event timestamps (torch tensor)
    @param ps List of event polarities (torch tensor)
    @param B Number of bins in output voxel grids (int)
    @param device Device to put voxel grid. If left empty, same device as events
    @param sensor_size The size of the event sensor/output voxels
    @param temporal_bilinear Whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    @returns Voxel of the events between t0 and t1
    """
    if device is None:
        device = xs.device
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    bins = []
    dt = ts[-1]-ts[0]
    t_norm = (ts-ts[0])/dt*(B-1)
    zeros = torch.zeros(t_norm.size())

    for bi in range(B):
        if temporal_bilinear:
            bilinear_weights = torch.max(zeros, 1.0-torch.abs(t_norm-bi))
            weights = ps*bilinear_weights
            vb = events_to_image_torch(xs, ys,
                    weights, device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        else:
            tstart = ts[0] + dt*bi
            tend = tstart + dt
            beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart)
            end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend)
            vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                    ps[beg:end], device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        bins.append(vb)
    bins = torch.stack(bins)
    return bins



def voxel_normalization(voxel):
    """
        normalize the voxel same as https://arxiv.org/abs/1912.01584 Section 3.1
        Params:
            voxel: torch.Tensor, shape is [num_bins, H, W]

        return:
            normalized voxel
    """
    # check if voxel all element is 0
    a,b,c = voxel.shape
    tmp = torch.zeros(a, b, c)
    if torch.equal(voxel, tmp):
        return voxel
    abs_voxel, _ = torch.sort(torch.abs(voxel).view(-1, 1).squeeze(1))
    first_non_zero_idx = torch.nonzero(abs_voxel)[0].item()
    non_zero_voxel = abs_voxel[first_non_zero_idx:]
    norm_idx = math.floor(non_zero_voxel.shape[0] * 0.98)
    ones = torch.ones_like(voxel)
    normed_voxel = torch.where(torch.abs(voxel) < non_zero_voxel[norm_idx], voxel / non_zero_voxel[norm_idx], voxel)
    normed_voxel = torch.where(normed_voxel >= non_zero_voxel[norm_idx], ones, normed_voxel)
    normed_voxel = torch.where(normed_voxel <= -non_zero_voxel[norm_idx], -ones, normed_voxel)
    return normed_voxel


def video_to_images(video_path,output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video =  cv2.VideoCapture(video_path)
    frame_index = 0
    while True:
        ret,frame = video.read()
        if not ret:
            break
        image_filename = os.path.join(output_folder,f'{frame_index:04d}.png')
        cv2.imwrite(image_filename,frame)
        frame_index += 1
    video.release()


def save_events_images(events_images,output_folder,num=10,count=5000,shape=[420,420]):
    # num represents the number of events to be saved
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    indice = num*count
    print(events_images[:indice].shape)
    # render_events_save(events_images[:indice],output_folder,count,shape)
    i = 0
    for img in render_events(events_images[:indice],count,shape):
        cv2.imwrite(os.path.join(output_folder,f'{i:04d}.png'),img)
        i += 1




class packager():

    __metaclass__ = ABCMeta

    def __init__(self, name, output_path, max_buffer_size=1000000):
        self.name = name
        self.output_path = output_path
        self.max_buffer_size = max_buffer_size

    @abstractmethod
    def package_events(self, xs, ys, ts, ps):
        pass

    @abstractmethod
    def package_image(self, frame, timestamp):
        pass

    @abstractmethod
    def package_flow(self, flow, timestamp):
        pass

    @abstractmethod
    def add_metadata(self, num_events, num_pos, num_neg,
            duration, t0, tk, num_imgs, num_flow):
        pass

    @abstractmethod
    def set_data_available(self, num_images, num_flow):
        pass

class hdf5_packager(packager):
    """
    This class packages data to hdf5 files
    """
    def __init__(self, output_path, max_buffer_size=1000000):
        packager.__init__(self, 'hdf5', output_path, max_buffer_size)
        print("CREATING FILE IN {}".format(output_path))
        self.events_file = h5py.File(output_path, 'w')
        self.event_xs = self.events_file.create_dataset("events/xs", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.event_ys = self.events_file.create_dataset("events/ys", (0, ), dtype=np.dtype(np.int16), maxshape=(None, ), chunks=True)
        self.event_ts = self.events_file.create_dataset("events/ts", (0, ), dtype=np.dtype(np.float64), maxshape=(None, ), chunks=True)
        self.event_ps = self.events_file.create_dataset("events/ps", (0, ), dtype=np.dtype(np.bool_), maxshape=(None, ), chunks=True)

    def append_to_dataset(self, dataset, data):
        dataset.resize(dataset.shape[0] + len(data), axis=0)
        if len(data) == 0:
            return
        dataset[-len(data):] = data[:]

    def package_events(self, xs, ys, ts, ps):
        self.append_to_dataset(self.event_xs, xs)
        self.append_to_dataset(self.event_ys, ys)
        self.append_to_dataset(self.event_ts, ts)
        self.append_to_dataset(self.event_ps, ps)

    def package_image(self, image, timestamp, img_idx):
        image_dset = self.events_file.create_dataset("images/{:06d}".format(img_idx),					### 原代码是 {image/:{09d}}与EvTexture model要求的不符故做了修改
                data=image, dtype=np.dtype(np.uint8))
        image_dset.attrs['size'] = image.shape
        image_dset.attrs['timestamp'] = timestamp
        image_dset.attrs['type'] = "greyscale" if image.shape[-1] == 1 or len(image.shape) == 2 else "color_bgr" 


    



    def package_flow(self, flow_image, timestamp, flow_idx):
        flow_dset = self.events_file.create_dataset("flow/flow{:09d}".format(flow_idx),
                data=flow_image, dtype=np.dtype(np.float32))
        flow_dset.attrs['size'] = flow_image.shape
        flow_dset.attrs['timestamp'] = timestamp

    def add_event_indices(self):
        datatypes = ['images', 'flow']
        for datatype in datatypes:
            if datatype in self.events_file.keys():
                s = 0
                added = 0
                ts = self.events_file["events/ts"][s:s+self.max_buffer_size]
                for image in self.events_file[datatype]:
                    img_ts = self.events_file[datatype][image].attrs['timestamp']
                    event_idx = np.searchsorted(ts, img_ts)
                    if event_idx == len(ts):
                        added += len(ts)
                        s += self.max_buffer_size
                        ts = self.events_file["events/ts"][s:s+self.max_buffer_size]
                        event_idx = np.searchsorted(ts, img_ts)
                    event_idx = max(0, event_idx-1)
                    self.events_file[datatype][image].attrs['event_idx'] = event_idx + added

    def add_metadata(self, num_pos, num_neg,
            duration, t0, tk, num_imgs, num_flow, sensor_size):
        self.events_file.attrs['num_events'] = num_pos+num_neg
        self.events_file.attrs['num_pos'] = num_pos
        self.events_file.attrs['num_neg'] = num_neg
        self.events_file.attrs['duration'] = tk-t0
        self.events_file.attrs['t0'] = t0
        self.events_file.attrs['tk'] = tk
        self.events_file.attrs['num_imgs'] = num_imgs
        self.events_file.attrs['num_flow'] = num_flow
        self.events_file.attrs['sensor_resolution'] = sensor_size
        self.add_event_indices()

    def set_data_available(self, num_images, num_flow):
        if num_images > 0:
            self.image_dset = self.events_file.create_group("images")
            self.image_dset.attrs['num_images'] = num_images
        if num_flow > 0:
            self.flow_dset = self.events_file.create_group("flow")
            self.flow_dset.attrs['num_images'] = num_flow

def add_voxel_to_hdf5(hdf5_path,segment_voxel,segment_voxel_b):
    with h5py.File(hdf5_path,'a') as f:
        # xs = f['events/xs']
        # ys = f['events/ys']
        # ts = f['events/ts']
        # ps = f['events/ps']

        # voxel_data = events_to_voxel_torch(xs,ys,ts,ps,bins,device=device,sensor_size=sensor_size)
        # normed_voxel_data = voxel_normalization(voxel_data)
        # down_sampling = torch.nn.functional.interpolate(normed_voxel_data,scale_factor=0.5,mode='bilinear')

        # if 'images' not in f:
            

        if 'voxels_f' not in f:
            print("Creating new voxels_f dataset")
            for index,voxel in enumerate(segment_voxel):
                print(f"Creating voxel dataset: {index:06d}")
                f.create_dataset(f'voxels_f/{index:06d}',data=voxel.numpy())
        else:
            print("voxels_f dataset already exists")
            pass
        

        if 'voxels_b' not in f:
            for index,voxel in enumerate(segment_voxel_b):
                print(f"Creating voxel_b dataset: {index:06d}")
                f.create_dataset(f'voxels_b/{index:06d}',data=voxel.numpy())
        else:
            print("voxels_b dataset already exists")
            pass

        # t_start = ts[0]
        # t_end = ts[-1]

        # if backward:
        #     xs = torch.flip(xs, dims=[0])
        #     ys = torch.flip(ys, dims=[0])
        #     ts = torch.flip(t_end - ts + t_start, dims=[0]) # t_end and t_start represent the timestamp range of the events to be flipped, typically the timestamps of two consecutive frames.
        #     ps = torch.flip(-ps, dims=[0])


        # voxel_data_backward = events_to_voxel_torch(xs,ys,ts,ps,bins,device=device,sensor_size=sensor_size)
        # normed_voxel_data_backward = voxel_normalization(voxel_data_backward)
        # down_sampling_backward = torch.nn.functional.interpolate(normed_voxel_data_backward,scale_factor=0.5,mode='bilinear')



if __name__ == "__main__":
	
    Cp = random.uniform(config_video['CT_range'][0], config_video['CT_range'][1])
    Cn = random.gauss(config_video['mu'], config_video['sigma']) * Cp
    Cp = min(max(Cp, config_video['min_CT']), config_video['max_CT'])
    Cn = min(max(Cn, config_video['min_CT']), config_video['max_CT'])
    esim = esim_py.EventSimulator(Cp,
                                Cn,
                                config_video['refractory_period'],
                                config_video['log_eps'],
                                config_video['use_log'])



     fire_xiaojie_path = "./fire_xiaojie.mp4"
     output_folder = './Fire_Original_Dataset/'	
     # video_to_images(fire_xiaojie_path,output_folder)




###  生成时间戳

     video_frames = 140
     video_time = 6
     fps = video_frames / video_time
     start_time = round(0.0,4)
     interval = round(1/(4*fps),4)
     timestamps = []

     for i in range(4*video_frames+1):
         timestamp = start_time + i * interval
         timestamps.append(round(timestamp, 4))
    

     image_folder = "./Fire_Interpolation_Dataset/"
     timestamps_file = "./video_timestamps.txt"
    
     # with open(timestamps_file, 'w') as f:
     #     for ts in timestamps:
     #         f.write(f"{ts}\n")
     # f.close()


###   为  “图片序列” 生成 events

     events_images = esim.generateFromFolder(image_folder, timestamps_file) # Generate events with shape [N, 4]


###   获得前向与反向的 （xs,ys,ts,ps）

     xs = torch.from_numpy(events_images[:, 0].astype(np.int32))
     ys = torch.from_numpy(events_images[:, 1].astype(np.int32))
     ts = torch.from_numpy(events_images[:, 2].astype(np.float32))
     ps = torch.from_numpy(events_images[:, 3].astype(np.int32))

     backward = True
     t_start = ts[0]
     t_end = ts[-1]

     if backward:
         xs_b = torch.flip(xs, dims=[0])
         ys_b = torch.flip(ys, dims=[0])
         ts_b = torch.flip(t_end - ts + t_start, dims=[0]) # t_end and t_start represent the timestamp range of the events to be flipped, typically the timestamps of two consecutive frames.
         ps_b = torch.flip(-ps, dims=[0])

  

###   根据时间戳，为相邻帧 生成 event ； 最终关于forward与backward 分别获得两个列表，长度为 帧数-1 

     time_min = ts.min()
     time_max = ts.max()
     seg_length = (time_max-time_min)/560

     time_interval = [(time_min+i*seg_length,time_min+(i+1)*seg_length) for i in range(560)]
    
     segmented_events = []
     segmented_events_b = []
     for time_start,time_end in time_interval:
         mask = (ts>time_start) & (ts<time_end)
         events_in_segment = (xs[mask],ys[mask],ts[mask],ps[mask])
         mask_b = (ts_b>time_start) & (ts_b<time_end)
         events_in_segment_b = (xs_b[mask_b],ys_b[mask_b],ts_b[mask_b],ps_b[mask_b])
         segmented_events.append(events_in_segment) 
         segmented_events_b.append(events_in_segment_b)       


    
     print(len(segmented_events))
     print(len(segmented_events_b))


###   生成为 上述相邻帧的每个 event 生成 voxel 数据，同样地返回两个列表，每个列表的长度为 帧数-1， 每个元素shape为 （bins,H,W）即 （5,420,420）

     bins = 5
     sensor_size = [420,420]
     segmented_voxel = []
     for event in segmented_events:
         voxel_data = events_to_voxel_torch(event[0],event[1],event[2],event[3],bins,device=None,sensor_size=sensor_size)
         normed_voxel_data = voxel_normalization(voxel_data)
         segmented_voxel.append(normed_voxel_data)
    
     print(len(segmented_voxel))
     print('segmented_voxel finished')

     segmented_voxel_b = []
     for event in segmented_events_b:
         voxel_data = events_to_voxel_torch(event[0],event[1],event[2],event[3],bins,device=None,sensor_size=sensor_size)
         normed_voxel_data = voxel_normalization(voxel_data)
         segmented_voxel_b.append(normed_voxel_data)

     print('segmented_voxel_b finished')


###   生成一个 hdf5 格式的文件

    packager_path = '.Results/fire_h5_file.h5'

     if os.path.exists(packager_path):
         os.remove(packager_path)


     with h5py.File(packager_path,'w') as f:
         pass

    
     packager = hdf5_packager(packager_path)

###   将 images（560张） 添加到 hdf5 中

     for img_idx,img_name in enumerate(sorted(os.listdir(image_folder))):
         img_path = os.path.join(image_folder,img_name)
         img = cv2.imread(img_path)
         packager.package_image(img,timestamps[img_idx],img_idx)
    

###   将 forwar与 backward 的 voxel 数据添加到 hdf5中
     add_voxel_to_hdf5(packager_path,segmented_voxel,segmented_voxel_b)


###   查看 hdf5 文件的数据结构

    with h5py.File(packager_path,'r') as f:
        def print_structure(name,obj,paths):
            paths.append(name)
        paths = []        
        f.visititems(lambda name,obj: print_structure(name,obj,paths))
        print(' '.join(paths))




   

