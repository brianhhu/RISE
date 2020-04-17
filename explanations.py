import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm


class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]

    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)

        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        return sal


class RISEBatch(RISE):
    def forward(self, x):
        # Apply array of filters to the image
        N = self.N
        B, C, H, W = x.size()
        stack = torch.mul(self.masks.view(N, 1, H, W),
                          x.data.view(B * C, H, W))
        stack = stack.view(B * N, C, H, W)
        stack = stack

        #p = nn.Softmax(dim=1)(model(stack)) in batches
        p = []
        for i in range(0, N*B, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N*B)]))
        p = torch.cat(p)
        CL = p.size(1)
        p = p.view(N, B, CL)
        sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, H * W))
        sal = sal.view(B, CL, H, W)
        return sal


class SBSM(nn.Module):
    def __init__(self, model, thresh, input_size, gpu_batch=100):
        super(SBSM, self).__init__()
        self.model = model
        self.thresh = thresh  # TODO: make this a property of visualization?
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, window_size, stride, savepath='masks.npy'):
        """
        Generates sliding window type binary masks used in augment() to 
        mask an image. The Images are resized to 224x224 to 
        enable re-use of masks Generating the sliding window style masks.
        :param int window_size: the block window size 
        (with value 0, other areas with value 1)
        :param int stride: the sliding step
        :param tuple image_size: the mask size which should be the 
        same to the image size
        :return: the sliding window style masks
        :rtype: numpy.ndarray
        """

        rows = np.arange(0 + stride - window_size, self.input_size[0], stride)
        cols = np.arange(0 + stride - window_size, self.input_size[1], stride)

        mask_num = len(rows) * len(cols)
        self.masks = np.ones(
            (mask_num, self.input_size[0], self.input_size[1]), dtype=np.float64)
        i = 0
        for r in rows:
            for c in cols:
                if r < 0:
                    r1 = 0
                else:
                    r1 = r
                if r + window_size > self.input_size[0]:
                    r2 = self.input_size[0]
                else:
                    r2 = r + window_size
                if c < 0:
                    c1 = 0
                else:
                    c1 = c
                if c + window_size > self.input_size[1]:
                    c2 = self.input_size[1]
                else:
                    c2 = c + window_size
                self.masks[i, r1:r2, c1:c2] = 0
                i += 1
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.window_size = window_size
        self.stride = stride

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]

    def weighted_avg(self, scalar_vec):
        count = self.N - self.masks.sum(dim=(0, 1))
        sal = (1 - self.masks).permute(2, 3, 1, 0) * \
            scalar_vec.clamp(min=0)
        sal = sal.sum(dim=-1).permute(2, 0, 1) / count
        # TODO: make min local vs. global
        sal = sal.clamp(min=(sal.max() * self.thresh))

        return sal

    def forward(self, x_q, x):
        _, _, H, W = x.size()

        # Get embedding of query and retrieval image
        x_q = self.model(x_q.data)
        x_r = self.model(x.data)
        o_dist = torch.cdist(x_q, x_r)

        # Apply array of masks to the image
        stack = torch.mul(self.masks, x.data)

        p = []
        for i in range(0, self.N, self.gpu_batch):
            x = self.model(stack[i:min(i + self.gpu_batch, self.N)])
            p.append(torch.cdist(x_q, x))
        p = torch.cat(p, dim=1)

        # Compute saliency
        m_dist = p - o_dist
        sal = self.weighted_avg(m_dist)

        return sal


class SBSMBatch(SBSM):
    def forward(self, x_q, x):
        B, C, H, W = x.size()

        # Get embedding of query and retrieval images
        x_q = self.model(x_q.data)
        x_r = self.model(x.data)
        o_dist = torch.cdist(x_q, x_r)

        # Apply array of masks to the image
        stack = torch.mul(self.masks.view(self.N, 1, H, W),
                          x.data.view(B * C, H, W))
        stack = stack.view(B * self.N, C, H, W)

        p = []
        for i in range(0, self.N*B, self.gpu_batch):
            x = self.model(stack[i:min(i + self.gpu_batch, self.N)])
            p.append(torch.cdist(x_q, x))
        p = torch.cat(p, dim=1)
        p = p.view(B, self.N)

        # Compute saliency
        m_dist = p - o_dist
        sal = self.weighted_avg(m_dist)

        return sal

# To process in batches
# def explain_all_batch(data_loader, explainer):
#     n_batch = len(data_loader)
#     b_size = data_loader.batch_size
#     total = n_batch * b_size
#     # Get all predicted labels first
#     target = np.empty(total, 'int64')
#     for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Predicting labels')):
#         p, c = torch.max(nn.Softmax(1)(explainer.model(imgs.cuda())), dim=1)
#         target[i * b_size:(i + 1) * b_size] = c
#     image_size = imgs.shape[-2:]
#
#     # Get saliency maps for all images in val loader
#     explanations = np.empty((total, *image_size))
#     for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Explaining images')):
#         saliency_maps = explainer(imgs.cuda())
#         explanations[i * b_size:(i + 1) * b_size] = saliency_maps[
#             range(b_size), target[i * b_size:(i + 1) * b_size]].data.cpu().numpy()
#     return explanations
