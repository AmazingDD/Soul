from energy_sim.img2col import conv_forward_with_sparsity
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

thr = 0.2
dataset = "cifar10dvs"
T = 10
module_sop_dict = {}

def lsar_register(net,unstructured_pruning=False):
    m_dict = dict(net.named_modules())
    for key in m_dict.keys():
        if key == "":
            continue
        m = m_dict[key]
        if isinstance(m,torch.nn.Conv2d):
            m.register_forward_hook(ops_hook_fn(key+".weight",unstructured_pruning))


# this function is especially prepared for lasr and ac attr
def ops_hook_fn(module_name,unstructured_pruning):
    def hook(m,inputs,outputs):
        import torch
        inputs = inputs[0]
        max_v = torch.max(inputs)
        is_mac = max_v.dtype == torch.int64 or not torch.floor(max_v) == max_v
        ran = max_v.detach().cpu().numpy().astype(int)
        lsar = 0
        stride = m.stride[0]
        padding = m.padding[0]
        kn,kn = m.kernel_size
        hn,wn = inputs.shape[-2:]
        in_channels = m.in_channels
        out_channels = m.out_channels
        T,B,C,H,W = inputs.shape
        for i in range(1, ran + 1):
            lsar += len(torch.where(inputs == float(i))[0]) / inputs.numel() * i
        if lsar == 0:
            lsar = inputs.count_nonzero() / inputs.numel()
        if unstructured_pruning and isinstance(m,torch.nn.Conv2d):
            weight = m.weight.data
            weight = weight.detach().cpu().numpy()
            # inputs = inputs.reshape(B,C,H,W)
            inputs = inputs.reshape(-1,C,H,W)
            inputs = inputs.detach().cpu().numpy()
            _, lsar = conv_forward_with_sparsity(inputs,weight,0,stride,padding)
            pass
        if module_name not in module_sop_dict.keys():
            module_sop_dict[module_name] = lsar * kn * kn * hn * wn *  in_channels * out_channels * T
        else:
            module_sop_dict[module_name] += lsar * kn * kn * hn * wn *  in_channels * out_channels * T
    return hook



if __name__ == "__main__":
    import torch
    from model.vgg import SpikingVGG9
    size = {"cifar10":32,
            "cifar10dvs":128}
    model = SpikingVGG9(input_shape=(2,128,128),num_classes=10,T=T)
    sp_model = torch.load(f"./sparse_weight/raw/best_SpikingVGG9_{dataset}_T{T}.pth",map_location="cpu")
    model.load_state_dict(sp_model)
    sp_model = model
    # sp_model = torch.load(f"./sparse_weight/raw/best_sparse_model_SpikingVGG9_{dataset}_T{T}_thr{thr}.pth",map_location="cpu")
    inputs = torch.load(f"../../samples/{dataset}-T{T}-size{size[dataset]}.pt",map_location="cpu")
    lsar_register(sp_model,unstructured_pruning=False)
    for single_sample in inputs:
        single_sample = single_sample.unsqueeze(0)
        single_sample.transpose(0,1)
        sp_model(single_sample)
    total_sops = 0
    for k,v in module_sop_dict.items():
        total_sops += v / 10
    print(total_sops)
