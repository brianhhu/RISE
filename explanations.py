import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm

from gradcam import ModelOutputs


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
        self.N = self.masks.shape[0]
        self.window_size = window_size
        self.stride = stride

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]

    def weighted_avg(self, K):
        count = self.N - self.masks.sum(dim=(0, 1))
        sal = K.sum(dim=-1).permute(2, 0, 1) / count
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

        m_dist = []
        for i in range(0, self.N, self.gpu_batch):
            x = self.model(stack[i:min(i + self.gpu_batch, self.N)])
            m_dist.append(torch.cdist(x_q, x))
        m_dist = torch.cat(m_dist, dim=1)

        # Compute saliency
        K = (1 - self.masks).permute(2, 3, 1, 0) * \
            (m_dist - o_dist).clamp(min=0)
        sal = self.weighted_avg(K)

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

        m_dist = []
        for i in range(0, self.N*B, self.gpu_batch):
            x = self.model(stack[i:min(i + self.gpu_batch, self.N)])
            m_dist.append(torch.cdist(x_q, x))
        m_dist = torch.cat(m_dist, dim=1)
        m_dist = m_dist.view(B, self.N)

        # Compute saliency
        K = (1 - self.masks).permute(2, 3, 1, 0) * \
            (m_dist - o_dist).clamp(min=0)
        sal = self.weighted_avg(K)

        return sal


class SimAtt(nn.Module):
    def __init__(self, model, feature_module, target_layer_names):
        super(SimAtt, self).__init__()
        self.model = model
        self.feature_module = feature_module

        self.extractor = ModelOutputs(
            self.model, self.feature_module, target_layer_names)

    def forward(self, x_a, x_p=None, x_n=None):
        # Consider all possible conditions:
        # 1. anchor + positive (Siamese)
        # 2. anchor + negative (Siamese)
        # 3. anchor + positive + negative (triplet)
        # 4. anchor + positive + negative1 + negative2 (quadruplet)

        # concatenate all inputs
        x = x_a
        if x_p is not None:
            x = torch.cat((x, x_p))
        if x_n is not None:
            x = torch.cat((x, x_n))

        # extract intermediate activations and outputs
        A, x = self.extractor(x)

        # compute positive and negative weights
        w = torch.abs(x[0] - x[1:])
        if x_p is not None:
            w[0] = 1 - w[0]

        # take elementwise product
        w = torch.prod(w, dim=0)

        # compute sample scores
        s = torch.matmul(x, w)

        # loop through sample scores
        feats = A[-1].data  # choose last set of features
        M = torch.zeros(feats.shape[0], feats.shape[2],
                        feats.shape[3], device=x_a.device)
        for i, s_i in enumerate(s):
            self.feature_module.zero_grad()
            self.model.zero_grad()
            s_i.backward(retain_graph=True)
            s_i_grad = self.extractor.get_gradients()[-1]
            weights = s_i_grad.mean(dim=(0, 2, 3))

            # loop through channels
            for j, w in enumerate(weights):
                M[i] += w * feats[i, j, :, :]

        # apply ReLU
        M.clamp(min=0)

        # upsample
        M = nn.functional.interpolate(M.unsqueeze(1), size=(
            x_a.shape[2], x_a.shape[3]), mode='bilinear').squeeze()

        return M


class SimCAM(nn.Module):
    """
    Adapted from: https://github.com/Jeff-Zilence/Explain_Metric_Learning/blob/master/Face_Verification/demo.py
    """

    def __init__(self, model, feature_module, target_layer_names, fc=None, bn=None):
        super(SimCAM, self).__init__()
        self.model = model
        self.feature_module = feature_module
        self.fc = fc
        self.bn = bn

        self.extractor = ModelOutputs(
            self.model, self.feature_module, target_layer_names, return_gradients=False)

    def Point_Specific(self, decom, point=[0, 0], size=(224, 224)):
        """
            Generate the point-specific activation map
            We assume the query point is always on the query image (image 1)
        """
        decom_padding = nn.functional.pad(decom.permute(
            2, 3, 0, 1), (1, 1, 1, 1), mode='replicate').permute(2, 3, 0, 1)

        # compute the transformed coordinates
        x = (point[0] + 0.5) / size[0] * (decom_padding.shape[0]-2)
        y = (point[1] + 0.5) / size[1] * (decom_padding.shape[1]-2)
        x = x + 0.5
        y = y + 0.5
        x_min = int(np.floor(x))
        y_min = int(np.floor(y))
        x_max = x_min + 1
        y_max = y_min + 1
        dx = x - x_min
        dy = y - y_min
        interpolation = decom_padding[x_min, y_min]*(1-dx)*(1-dy) + \
            decom_padding[x_max, y_min]*dx*(1-dy) + \
            decom_padding[x_min, y_max]*(1-dx)*dy + \
            decom_padding[x_max, y_max]*dx*dy

        return interpolation.clamp(min=0)

    def forward(self, x_q, x, point=None):
        # concatenate all inputs
        x = torch.cat((x_q, x))

        # extract intermediate activations and outputs
        A, _ = self.extractor(x.data)

        x = A[-1].data
        x = x.permute(0, 2, 3, 1)

        x_reshape = torch.reshape(x, [-1, x.shape[-1]])
        # x_embed = torch.zeros(
        #     [x_reshape.shape[0], fc.weight.data.shape[0]])

        # consider all operations as one linear transformation, compute the equivalent feature for each position
        weight = 1
        bias = 0

        if self.fc is not None:
            weight *= torch.reshape(self.fc.weight.data, [
                                    self.fc.weight.data.shape[0], x_reshape.shape[-1], x_reshape.shape[0]])
            bias += self.fc.bias.data / x_reshape.shape[0] / x_reshape.shape[1]
        if self.bn is not None:
            weight /= torch.sqrt(self.bn.running_var.data).view(-1, 1, 1)
            bias = (bias - self.bn.running_mean.data) / \
                torch.sqrt(self.bn.running_var.data)

            weight *= self.bn.weight.data.view(-1, 1, 1)
            bias = bias * self.bn.weight.data + self.bn.bias.data

        # # compute the transformed feature, break apart to avoid too large matrix operation in Memory
        # for i in range(x_reshape.shape[0]):
        #     x_embed[i] = torch.matmul(
        #         x_reshape[i], weight[:, :, i].t())  # + bias

        # compute the transformed feature
        x_embed = x_reshape * weight

        # reshape back
        x_embed = torch.reshape(
            x_embed, [x.shape[0], x.shape[1], x.shape[2], -1])

        Decomposition = torch.zeros(
            [x.shape[1], x.shape[2], x.shape[1], x.shape[2]], device=x_q.device)
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                for k in range(x.shape[1]):
                    for l in range(x.shape[2]):
                        Decomposition[i, j, k, l] = torch.sum(
                            x_embed[0, i, j]*x_embed[1, k, l])
        Decomposition = Decomposition / torch.max(Decomposition)

        # apply ReLU
        Decomposition.clamp(min=0)

        # map for query image
        decom_1 = torch.sum(Decomposition, dim=(2, 3))

        # map for retrieval image
        # do point specific calculation here if needed
        if point is not None:
            decom_2 = self.Point_Specific(
                Decomposition, point, size=(x_q.shape[2], x_q.shape[3]))
        else:
            decom_2 = torch.sum(Decomposition, dim=(0, 1))

        # upsample
        Decomposition = nn.functional.interpolate(torch.stack((decom_1, decom_2)).unsqueeze(
            1), size=(x_q.shape[2], x_q.shape[3]), mode='bilinear').squeeze()

        return Decomposition


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
