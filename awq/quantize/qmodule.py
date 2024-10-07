import math
import torch
import torch.nn as nn
import awq_inference_engine  # with CUDA kernels
import numpy as np

class Packer:
    def __init__(self):
        self.s = torch.from_numpy(np.array([1, 2, 4, 8, 16, 32, 64, 128])).view(
            [-1, 1])
        if torch.cuda.is_available():
            self.s = self.s.cuda()
        self.w_pool = {}

    def __get_weight(self, shape, dtype):
        key = np.prod(shape)
        if key not in self.w_pool.keys():
            self.w_pool[key] = torch.zeros(shape, dtype=dtype)
            if torch.cuda.is_available():
                self.w_pool[key] = self.w_pool[key].cuda()
        return self.w_pool[key].reshape(shape)

    def pack(self, b):
        shape = b.shape
        p_b = b
        if torch.cuda.is_available():
            p_b = p_b.cuda()
        p_b = (p_b + 1) / 2  # (-1., +1.) -> (0, 1)
        p_b = torch.reshape(p_b, [8, -1]).type(torch.uint8)
        p_b = p_b * self.s
        p_b = p_b.sum(0)
        p_b = p_b.type(torch.uint8)
        return p_b, shape

    def unpack(self, pb, shape, dtype=torch.float16):
        b = self.__get_weight(shape, dtype).view([8, -1])
        for i in range(8):
            b[i] = (pb & 1)  # (pB%2)
            pb = pb >> 1  # //2
        b = b * 2 - 1
        b = b.reshape(shape)
        return b

PACKER = Packer()
def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError

    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width

def convert_bcq_format(scale, zero, quant_data, qbits, do_packing=False, in_ch_wise=False):
    global PACKER

    zero   = scale * zero #O ,#G,1
    upack  = torch.Tensor([[2**(i) for i in range(qbits)]]).to(torch.device('cuda:0'))
    scale  = scale / 2.0
    scale  = torch.matmul(scale, upack) #O G B

    offset = scale.sum(-1).unsqueeze(-1) - zero #O G 1
    offset= offset.reshape(offset.shape[0],-1)
    binary = torch.zeros(list(quant_data.shape) + [qbits])
    binary_shape = binary.shape
    
    quant_data = quant_data.to(torch.int)
    for i in range(qbits):
        binary[:, :, i] = ((quant_data >> i) & 1) * 2 - 1
        # O I B

    K = binary.shape[1] #input
    N = binary.shape[0] #output

    scale_ = scale.permute(1,2,0).contiguous() # G B O
    binary_ = binary.permute(1,2,0).contiguous().to(torch.int64).to(torch.device('cuda:0'))
    offset_ = offset.permute(1,0).contiguous() # G O

    bW = torch.zeros([K // 32, qbits, N], dtype=torch.int64,device ='cuda')

    #if do_packing == True:
    #    for n in range(N):
    #        for b in range(qbits):
    #            for k in range(0, K, 32):
    #                s = 0
    #                for t in range(32):
    #                    if binary_[n][b][k + t] == 1:
    #                        s |= (1 << t)  # 비트를 설정
    #                bW[k // 32][b][n] = (s & 0xFFFFFFFF)
    for b in range(qbits):
        for n in range(N):
            for k in range(0, K, 32):
                # torch.int32로 변환
                binary_chunk = binary_[k:k+32, b, n].to(torch.int64).to(torch.device('cuda:0'))
                bit_values = torch.tensor([1 << i for i in range(32)], dtype=torch.int64, device='cuda')
                
                # 원소 곱셈 후 합산을 통해 내적 연산을 수행
                s = torch.sum(binary_chunk * bit_values)
                bW[k // 32, b, n] = s & 0xFFFFFFFF
    bW = bW.to(torch.int32)
    return scale_, bW, binary_shape, offset_

def pack_intweight(unpacked_qweight, interleave, kstride):
    # unpacked_qweight: [N, K]
    N = unpacked_qweight.shape[0]
    K = unpacked_qweight.shape[1]

    Packed_Kernel = unpacked_qweight.cpu().numpy().reshape(N, K // 32, 32)
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)

    # reorder each 8 weights for fast dequantization
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)

    # interleaving every four rows
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, interleave, K // kstride, kstride
    )
    # N // 4, K // 64, 4, 64
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, K // kstride, kstride, interleave
    )
    # Packing -> (N // 4, K // 64, 64)
    Packed_Kernel = (
        Packed_Kernel[..., 0]
        | (Packed_Kernel[..., 1] << 4)
        | (Packed_Kernel[..., 2] << 8)
        | (Packed_Kernel[..., 3] << 12)
    )
    # reshape to (N // 4, K), FP16 format
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K)
    qweight = (
        torch.tensor(Packed_Kernel.astype("int16"))
        .to(unpacked_qweight.device)
        .contiguous()
    )
    return qweight


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


class WQLinear(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        #if w_bit not in [4]:
        #    raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8
        self.interleave = 4
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        #assert out_features % (32 // self.w_bit) == 0
        pack_num = 32 // 4
        int16_pack_num = 16 // 4

        assert out_features % (self.interleave) == 0
    
        self.register_buffer(
            "q_bias",
            torch.zeros(
                (
                    in_features//128,
                    out_features
                ),
                dtype=torch.float32,
                device=dev,
            ),
        )
        self.register_buffer(
            "alpha",
            torch.zeros(
                (
                    in_features//128,
                    self.w_bit,
                    out_features
                ),
                dtype=torch.float32,
                device=dev,
            ),
        )
        self.register_buffer(
            "binary",
            torch.zeros(
                (
                    in_features//32,
                    self.w_bit,
                    out_features
                ),
                dtype=torch.int32,
                device=dev,
            ),
        )    

        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=torch.float16, device=dev)
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size,data, init_only=False, scales=None, zeros=None
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        alpha, binary, binary_shape, offset = convert_bcq_format(
            scales, zeros, data, qbits=w_bit,
            do_packing=True, in_ch_wise=False)

        awq_linear.binary = binary
        awq_linear.alpha = alpha
        awq_linear.q_bias = offset

        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        # out_shape = x.shape[:-1] + (self.out_features,)
        # inputs = x.reshape(-1, x.shape[-1])
        inputs = x
        if inputs.numel() / inputs.shape[-1] < 8:
            out = awq_inference_engine.gemv_forward_cuda_new(
                inputs,
                self.qweight,
                self.scales,
                self.scaled_zeros,
                inputs.numel() // inputs.shape[-1],
                self.out_features,
                self.in_features,
                self.group_size,
            )
        else:
            out = awq_inference_engine.gemm_forward_cuda_new(
                inputs, self.qweight, self.scales, self.scaled_zeros
            )  # - 8.0 * self.scales)
        out = out + self.bias if self.bias is not None else out
        # print(out)
        # assert 0
        return out

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )
