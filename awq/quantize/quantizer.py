import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from .qmodule import ScaledActivation
from ..utils.module import set_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class CompressionParameter(nn.Parameter):
    def compress(self, **kwargs):
        raise NotImplemented

    def decompress(self, **kwargs):
        raise NotImplemented
class RTNParameter(CompressionParameter):
    def compress(self, in_ch_wise=False, **kwargs):
        data_shape = self.data.shape
        group_size = -1
        if 'group_size' in kwargs:
            group_size = kwargs.pop('group_size')
        out_ch = data_shape[0]
        in_ch = data_shape[1]

        quant = Quantizer()
        quant.configure(**kwargs)
        if in_ch_wise == False:
            data = self.data
            if group_size > 0:
                data = data.reshape([-1, group_size])
            quant.find_params(data, weight=True)
            quant_data  = torch.clamp(torch.round(data / quant.scale) + quant.zero, 0, quant.maxq)
            quant_data  = quant_data.reshape([out_ch, -1]).to(torch.int)
            quant.scale = quant.scale.reshape([out_ch, -1, 1])
            quant.zero  = quant.zero.reshape([out_ch, -1, 1])
        else:
            data = self.data.T
            if group_size > 0:
                data = data.reshape([-1, group_size])
            quant.find_params(data, weight=True)
            quant_data = torch.clamp(torch.round(data / quant.scale) + quant.zero, 0, quant.maxq)
            quant_data = quant_data.reshape([in_ch, -1, group_size]).to(torch.int)
            quant.scale = quant.scale.reshape([in_ch, -1, 1])
            quant.zero  = quant.zero.reshape([in_ch, -1, 1])

        return quant.scale, quant.zero, quant_data, quant_data.shape

    def decompress(self, scale, zero, quant_data, quant_data_shape, in_ch_wise=False):
        # w.shape = [out_ch, in_ch]
        # in_ch_wise == True
        #   -> quant_data.shape = [in_ch, out_ch//group_size, group_size]
        #   -> scale.shape      = [in_ch, out_ch//group_size, 1]
        #   -> zero.shape       = [in_ch, out_ch//group_size, 1]
        # in_ch_wise == False
        #   -> quant_data.shape = [out_ch, in_ch//group_size, group_size]
        #   -> scale.shape      = [out_ch, in_ch//group_size, 1]
        #   -> zero.shape       = [out_ch, in_ch//group_size, 1]

        if in_ch_wise == True:
            out_ch = quant_data_shape[1] * quant_data_shape[2]
            decomp_w = scale * (quant_data - zero)
            decomp_w = decomp_w.reshape([-1, out_ch]).T
        else:
            out_ch = quant_data_shape[0]
            decomp_w = scale * (quant_data - zero)
            decomp_w = decomp_w.reshape([out_ch, -1])
        self.data = decomp_w

    def convert_bcq_format(self, scale, zero, quant_data, qbits, do_packing=False, in_ch_wise=False):
        global PACKER

        zero   = scale * zero #O ,#G,1
        upack  = torch.Tensor([[2**(i) for i in range(qbits)]])
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

        scale = scale.permute(1,2,0).contiguous() # G B O
        binary = binary.permute(1,2,0).contiguous() # I B O
        offset = offset.permute(1,0).contiguous() # G O

        bW = torch.zeros([K // 32, qbits, N], dtype=torch.int64)
    
        if do_packing == True:
            for n in range(N):
                for b in range(qbits):
                    for k in range(0, K, 32):
                        s = 0
                        for t in range(32):
                            if binary[k + t][b][n] == 1:
                                s |= (1 << t)  # 비트를 설정
                        bW[k // 32][b][n] = (s & 0xFFFFFFFF)

        bW = bW.to(torch.int32).contiguous()
        return scale, bW, binary_shape, offset

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        qbits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
        self.maxq = torch.tensor(2 ** qbits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits:
            self.maxq = torch.tensor(-1) 

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def scale_activations(module):
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.gelu_impl, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)
    elif "mptblock" in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif "falcon" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "bigcode" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "neox" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)


# core quantization method (simulated quantization)
def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

def compress(data, in_ch_wise=False, **kwargs):
    data_shape = data.shape
    group_size = -1
    if 'group_size' in kwargs:
        group_size = kwargs.pop('group_size')
    out_ch = data_shape[0]
    in_ch = data_shape[1]

    quant = Quantizer()
    quant.configure(**kwargs)
    if in_ch_wise == False:
        if group_size > 0:
            data_ = data.reshape([-1, group_size])
        quant.find_params(data, weight=True)
        quant_data  = torch.clamp(torch.round(data / quant.scale) + quant.zero, 0, quant.maxq)
        quant_data  = quant_data.reshape([out_ch, -1]).to(torch.int)
        quant.scale = quant.scale.reshape([out_ch, -1, 1])
        quant.zero  = quant.zero.reshape([out_ch, -1, 1])

    return quant.scale, quant.zero, quant_data, quant_data.shape
@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.cuda()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, **q_config
            )
            m.cpu()


@torch.no_grad()
def real_quantize_model_weight(model, w_bit, q_config, init_only=False):
    from .qmodule import WQLinear
    from .pre_quant import get_blocks, get_named_linears

    assert q_config["zero_point"], "We only support zero_point quantization now."

    layers = get_blocks(model)
    for i in tqdm(
        range(len(layers)),
        desc="real weight quantization..." + ("(init only)" if init_only else ""),
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        scale_activations(layer)

        for name, module in named_linears.items():
            if init_only:
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
            else:
                

                w_rtn = RTNParameter(module.weight.data)
                scales, zeros, data, w_quant_shape = w_rtn.compress(in_ch_wise=False, qbits=4, group_size=128, perchannel=True, sym=False)

                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"],data, False,scales, zeros
                )
                module.cpu()
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
                torch.cuda.empty_cache()
                gc.collect()

    torch.cuda.empty_cache()
    gc.collect()
