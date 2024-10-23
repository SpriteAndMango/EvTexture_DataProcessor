import random
import esim_py
import numpy as np
import cv2
import math
import torch
from datetime import datetime, timedelta



config = {
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



if __name__ == "__main__":
    Cp = random.uniform(config['CT_range'][0], config['CT_range'][1])
    Cn = random.gauss(config['mu'], config['sigma']) * Cp
    Cp = min(max(Cp, config['min_CT']), config['max_CT'])
    Cn = min(max(Cn, config['min_CT']), config['max_CT'])
    esim = esim_py.EventSimulator(Cp,
                                Cn,
                                config['refractory_period'],
                                config['log_eps'],
                                config['use_log'])

    start_time = round(0.0,4)
    interval = round(1/(4*30),4)
    timestamps = []

    for i in range(397):
        timestamp = start_time + i * interval
        timestamps.append(round(timestamp, 4))
    for ts in timestamps:
        print(ts)


    image_folder = "./frame_interpolation_dataset"
    timestamps_file = "./timestamps.txt"
    
    with open(timestamps_file, 'w') as f:
        for ts in timestamps:
            f.write(f"{ts}\n")
    f.close()

    events = esim.generateFromFolder(image_folder, timestamps_file) # Generate events with shape [N, 4]
    backward = False

    xs = torch.from_numpy(events[:, 0].astype(np.int32)).reshape(-1, 1)
    ys = torch.from_numpy(events[:, 1].astype(np.int32)).reshape(-1, 1)
    ts = torch.from_numpy(events[:, 2].astype(np.float32)).reshape(-1, 1)
    ps = torch.from_numpy(events[:, 3].astype(np.int32)).reshape(-1, 1)
    t_start = ts[0]
    t_end = ts[-1]
    bins = 10
    sensor_size = (180, 320)


    if backward:
        xs = torch.flip(xs, dims=[0])
        ys = torch.flip(ys, dims=[0])
        ts = torch.flip(t_end - ts + t_start, dims=[0]) # t_end and t_start represent the timestamp range of the events to be flipped, typically the timestamps of two consecutive frames.
        ps = torch.flip(-ps, dims=[0])
    voxel = events_to_voxel_torch(xs, ys, ts, ps, bins, device=None, sensor_size=sensor_size)
    normed_voxel = voxel_normalization(voxel)
    print(voxel.shape)

    # print(events)
    for img in render_events(events):
        cv2.imshow("img", img)
        if cv2.waitKey(0) & 0xFF == 27:  # 如果按下的是Esc键
            break
