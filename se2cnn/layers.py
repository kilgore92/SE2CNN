"""
Pytorch implementation of SE(2) group equivariant convolutions

Based on:
E. J. Bekkers, M. W. Lafarge, M. Veta, K. A. Eppenhof, J. P. Pluim, and R. Duits, “Roto-Translation Covariant Convolutional Networks for Medical Image Analysis,” arXiv:1804.03393 [cs, math], Jun. 2018, Accessed: Oct. 13, 2020. [Online]. Available: http://arxiv.org/abs/1804.03393.

Tensorflow implementation: https://github.com/tueimage/SE2CNN

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from math import floor, sin, cos, pi, degrees
import numpy as np

class SE2Conv2d(nn.Module):
    """
    Parent class for the SE(2) lifiting and group convolutions

    """

    def __init__(self,
                 in_channels:int = 1,
                 out_channels:int = 32,
                 n_orientations:int = 8,
                 kernel_size:int = 3,
                 padding:int = 1,
                 periodicity=2*pi):

        super(SE2Conv2d, self).__init__()
        # Initialize weight and bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_orientations = n_orientations
        self.padding=padding
        self.kernel_size=kernel_size
        self.periodicity = periodicity

        # Angles specified in radians
        rotation_step = self.periodicity/self.n_orientations
        self.angles = [idx*rotation_step for idx in range(self.n_orientations)]

        # Create rotation matrix
        self.multi_rotation_matrix = self.create_multirotation_rotation_matrix()


    @staticmethod
    def rotate_coordinate(coord, theta, kernel_size):
        """
        Find the rotated version of a coordinate (about center).
        We rotate the coordinate by -theta, since we want to rotate
        the kernel (function) by theta
        params:
            coord: list containting [y, x] coord
            theta: angle to rotate in radians
            kernel_size: size of kernel (kernel assumed square)
        returns:
            rotated_coord

        """
        center = floor(kernel_size/2)

        # Rotate the coordinates about the center
        rotated_coord = torch.zeros(size=(2,), dtype=torch.float32)
        rotated_coord[0] = (coord[0] - center)*cos(theta) + (coord[1] - center)*sin(theta) + center
        rotated_coord[1] = -1*(coord[0] - center)*sin(theta) + (coord[1] - center)*cos(theta) + center

        return rotated_coord

    @staticmethod
    def find_flat_coord(coord, kernel_size):
        """
        Find corresponding position of a 2-D coordinate in
        a flatteneded array

        """
        pos_idx = coord[0]*kernel_size + coord[1]
        return pos_idx

    @staticmethod
    def interpolation_indices_and_weights(coord, kernel_size):
        """ Returns, given a target index (i,j), the 4 neighbouring indices and
            their corresponding weights used for linear interpolation.

            INPUT:
                - coord, a list of length 2 containing the i and j coordinate
                  as [i,j]
                - kernel_size: Height/width of a square kernel

            OUTPUT:
                - indicesAndWeights, a list index-weight pairs as [[i0,j0,w00],
                  [i0,j1,w01],...]
        """

        # The index where want to obtain the value
        i = coord[0]
        j = coord[1]

        # The neighbouring indices
        i1 = int(floor(i))  # -- to integer format
        i2 = i1 + 1
        j1 = int(floor(j))  # -- to integer format
        j2 = j1 + 1

        # The 1D weights
        ti = i - i1
        tj = j - j1

        # The 2D weights
        w11 = (1 - ti) * (1 - tj)
        w12 = (1 - ti) * tj
        w21 = ti * (1 - tj)
        w22 = ti * tj

        # Only add indices and weights if they fall in the range of the image with
        # dimensions kernel_size x kernel_size
        indicesAndWeights = []
        if (0 <= i1 < kernel_size) and (0 <= j1 < kernel_size):
            indicesAndWeights.append([i1, j1, w11])
        if (0 <= i1 < kernel_size) and (0 <= j2 < kernel_size):
            indicesAndWeights.append([i1, j2, w12])
        if (0 <= i2 < kernel_size) and (0 <= j1 < kernel_size):
            indicesAndWeights.append([i2, j1, w21])
        if (0 <= i2 < kernel_size) and (0 <= j2 < kernel_size):
            indicesAndWeights.append([i2, j2, w22])

        return indicesAndWeights

    def create_rotation_matrix(self, kernel_size, theta, diskMask=True):
        """
            Create rotation matrix containing interpolation weights. In this
            function we calculate the corresponding points in the original kernel
            for each point in the rotated kernel via a rotation by theta in the
            opposite direction. Erik's implementation is based on creating a
            sparse tensor, but Pytorch faces issues in loading of a saved model
            containing sparse tensors, see: https://discuss.pytorch.org/t/unable-to-load-sparse-tensor/84598

            INPUT:
                - kernel_size(int) : Specify height or width of kernel (square kernel assumed)
                - theta(float): a real number specifying the rotation angle in radians

            INPUT (optional):
                - diskMask = True, by default values outside a circular mask are set
                  to zero.

            OUTPUT:
                rotation_matrix (np.ndarray): Rotation matrix for a given angle of rotation

        """

        cij = floor(kernel_size / 2)  # center

        rotation_matrix = np.zeros((kernel_size*kernel_size, kernel_size*kernel_size), dtype=np.float32)
        for i in range(0, kernel_size):
            for j in range(0, kernel_size):
                # Apply a circular mask (disk matrix) if desired
                if not(diskMask) or ((i - cij) * (i - cij) + (j - cij) * (j - cij) <= (cij + 0.5) * (cij + 0.5)):
                    # The row index of the operator matrix
                    linij = self.find_flat_coord([i, j], kernel_size)
                    # The interpolation points
                    ijOld = self.rotate_coordinate([i, j], theta=theta, kernel_size=kernel_size)
                    # The indices used for interpolation and their weights
                    linIntIndicesAndWeights = self.interpolation_indices_and_weights(ijOld, kernel_size)

                    # Fill the weights in the rotationMatrix
                    for indexAndWeight in linIntIndicesAndWeights:
                        indexOld = [indexAndWeight[0], indexAndWeight[1]]
                        linIndexOld = self.find_flat_coord(indexOld, kernel_size)
                        weight = indexAndWeight[2]
                        rotation_matrix[linij, linIndexOld] = weight

        return rotation_matrix


    def create_multirotation_rotation_matrix(self):
        idx = ()
        vals = ()
        multi_rotation_matrix = [None]*(len(self.angles))
        for r, angle in enumerate(self.angles):
            multi_rotation_matrix[r] = self.create_rotation_matrix(kernel_size=self.kernel_size,
                                                                   theta=angle)

        multi_rotation_matrix = np.concatenate(multi_rotation_matrix, axis=0)
        return torch.from_numpy(multi_rotation_matrix)


    def forward(self, x):
        raise NotImplementedError



class SE2LiftingConv2d(SE2Conv2d):

    def __init__(self,
                 in_channels: int = 1,
                 out_channels:int = 32,
                 n_orientations: int = 8,
                 kernel_size: int = 3,
                 padding: int = 0):

        super(SE2LiftingConv2d, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               n_orientations=n_orientations,
                                               kernel_size=kernel_size,
                                               padding=padding)

        # Weights and biases require their gradients to be stored for backprop
        self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = Parameter(torch.Tensor(self.out_channels))

    def forward(self, x):
        # Created rotated copies of filters

        # Transpose axes for behavior consistent with Erik's code (channel-last)
        channel_first_weights = self.weight.permute(2, 3, 1, 0)

        flattened_kernel = torch.reshape(channel_first_weights,
                                         (self.kernel_size*self.kernel_size, self.in_channels*self.out_channels))

        # Rotate kernels via (sparse) matrix multiplication with the rotation matrix
        set_of_rotated_kernels = torch.mm(self.multi_rotation_matrix, flattened_kernel)
        rotated_kernels_reshaped = torch.reshape(set_of_rotated_kernels,
                                                 (self.n_orientations, self.kernel_size, self.kernel_size, self.in_channels, self.out_channels))

        # Get it back to torch friendly axes ordering
        rotated_kernels_transposed = rotated_kernels_reshaped.permute(0, 4, 3, 1, 2)

        # Merge out channels and n_orientations axes to perform 2-D convolution so each rotated kernel is convolved separately
        rotated_kernels_pre_conv = torch.reshape(rotated_kernels_transposed, (self.n_orientations*self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))

        # Perform 2-D convolution
        lifting_conv = F.conv2d(input=x,
                                weight=rotated_kernels_pre_conv,
                                bias=None,
                                padding=self.padding)


        # Reshape convolution output
        lifting_conv = torch.reshape(lifting_conv, (lifting_conv.shape[0], self.out_channels, self.n_orientations, lifting_conv.shape[-2], lifting_conv.shape[-1]))


        # Add bias (one bias term per output channel i.e. all rotated versions of a single channel share the bias term)
        # See PyTorch broadcasting rules: https://pytorch.org/docs/stable/notes/broadcasting.html
        bias_broadcast = self.bias[:, None, None, None]
        conv_output = lifting_conv + bias_broadcast

        return conv_output, rotated_kernels_transposed


class SE2GroupConv2d(SE2Conv2d):
    """
    Convolution where the function and kernel are defined on the SE(2) group (x, y, theta)

    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 32,
                 n_orientations: int = 8,
                 kernel_size: int = 3,
                 padding:int = 0):


        super(SE2GroupConv2d, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             n_orientations=n_orientations,
                                             kernel_size=kernel_size,
                                             padding=padding)

        self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels, self.n_orientations, self.kernel_size, self.kernel_size))
        self.bias = Parameter(torch.Tensor(self.out_channels))


    def forward(self, x):
        # Part-1 : Rotation of kernels defined on SE(2)
        # Created rotated copies of filters
        # Transpose axes for behavior consistent with Erik's code (channel-last)
        channel_first_weights = self.weight.permute(3, 4, 2, 1, 0) # (kH, kW, n_ori, inChan, outChan)
        flattened_kernel = torch.reshape(channel_first_weights,
                                         (self.kernel_size*self.kernel_size, self.n_orientations*self.in_channels*self.out_channels))

        # Rotate kernels via (sparse) matrix multiplication with the rotation matrix
        kernel_rotated_planar = torch.sparse.mm(self.multi_rotation_matrix, flattened_kernel)
        kernel_rotated_planar_reshaped = torch.reshape(kernel_rotated_planar,
                                                       (self.n_orientations, self.kernel_size, self.kernel_size, self.n_orientations, self.in_channels, self.out_channels))



        # Part-2 : Shift of kernels along the theta axis
        set_of_rotated_kernels = [None] * self.n_orientations
        for orientation in range(self.n_orientations):
            # [kernelSizeH,kernelSizeW,orientations_nb,channelsIN,channelsOUT]
            kernels_temp = kernel_rotated_planar_reshaped[orientation]
            # [kernelSizeH,kernelSizeW,channelsIN,channelsOUT,orientations_nb]
            kernels_temp = kernels_temp.permute(0, 1, 3, 4, 2)
            # [kernelSizeH*kernelSizeW*channelsIN*channelsOUT*orientations_nb]
            kernels_temp = torch.reshape(
                    kernels_temp, (self.kernel_size * self.kernel_size * self.in_channels * self.out_channels, self.n_orientations))
            # Roll along the orientation axis
            roll_matrix = torch.from_numpy(np.roll(np.identity(self.n_orientations, dtype=np.float32), orientation, axis=1))
            kernels_temp = torch.matmul(kernels_temp, roll_matrix)
            kernels_temp = torch.reshape(
                kernels_temp, (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels, self.n_orientations))  # [kH,kW,in_chan,out_chan,n_ori]
            kernels_temp = kernels_temp.permute(0, 1, 4, 2, 3) # [kH, kW, n_ori, in_chan, out_chan]
            set_of_rotated_kernels[orientation] = kernels_temp

        # Shape: (n_orientations, kH, kW, inChan, outChan, n_orientations)
        set_of_rotated_kernels = torch.stack(set_of_rotated_kernels)


        # Part-3 : Convolution with feature map defined on SE(2)

        # Merge out channels and n_orientations axes to perform 2-D convolution so each rotated kernel is convolved separately
        # Get it back to torch friendly axes ordering
        set_of_rotated_kernels = set_of_rotated_kernels.permute(0, 4, 3, 5, 1, 2) # Shape: (n_orientations, outChan, inChan, n_orientations, kH, kW)
        rotated_kernels_pre_conv = torch.reshape(set_of_rotated_kernels,
                                                 (self.n_orientations*self.out_channels, self.in_channels*self.n_orientations, self.kernel_size, self.kernel_size))


        # Reshape input to perform 2-D convolution : Merge in_channel and n_orientation axes
        x_reshaped = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4]))

        # Perform 2-D convolution
        # Output shape: (batch_size, n_orientations*outChan, kH, kW)
        g_conv = F.conv2d(input=x_reshaped,
                          weight=rotated_kernels_pre_conv,
                          bias=None,
                          padding=self.padding)

        g_conv = torch.reshape(g_conv, (g_conv.shape[0],self.out_channels, self.n_orientations, g_conv.shape[-2], g_conv.shape[-1]))
        # See PyTorch broadcasting rules: https://pytorch.org/docs/stable/notes/broadcasting.html
        bias_broadcast = self.bias[:, None, None, None]
        out = g_conv + bias_broadcast

        return out


class SpatialMaxPool2d(nn.Module):

    def __init__(self,
                 kernel_size:int=2,
                 stride:int=2,
                 padding:int=0):
        """
        Max pooling performed over spatial dimensions of the feature map defined on SE(2)

        """
        super(SpatialMaxPool2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):

        # Shape of x : (batch_id, in_channels, n_orientations, kH, kW)
        # Changed to : (batch_id, n_orientations, in_channels, kH, kW)
        x_orientation_first = x.permute(0, 2, 1, 3, 4)
        n_orientations = x_orientation_first.shape[0]

        pooled_responses = [None]*n_orientations
        for ori in range(n_orientations):
            pooled_responses[ori] = F.max_pool2d(input=x_orientation_first[ori],
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding)

        pooled_responses = torch.stack(pooled_responses)

        # Transpose axes to channel first
        out = pooled_responses.permute(0, 2, 1, 3, 4)
        return out







