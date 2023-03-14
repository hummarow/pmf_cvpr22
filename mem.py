import torch
import inspect
import gc

def get_tensor_memory_usage(tensor):
    return tensor.element_size() * tensor.nelement()

def get_variable_memory_usage(var):
    if hasattr(var, 'data'):
        return get_tensor_memory_usage(var.data)
    elif isinstance(var, list):
        return sum(get_variable_memory_usage(v) for v in var)
    elif isinstance(var, dict):
        return sum(get_variable_memory_usage(v) for v in var.values())
    else:
        return 0

def get_memory_usage():
    torch.cuda.empty_cache()
    gc.collect()
    mem_usage = {}
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            mem_usage[id(obj)] = get_tensor_memory_usage(obj)
        elif isinstance(obj, torch.nn.Module):
            mem_usage[id(obj)] = sum(get_tensor_memory_usage(p) for p in obj.parameters())
    return mem_usage

def get_variable_name(obj_id):
    frame = inspect.currentframe().f_back
    while frame:
        for name, obj in frame.f_locals.items():
            if id(obj) == obj_id:
                return name
        frame = frame.f_back
    return None
# def get_variable_name(obj_id):
#     for referrer in gc.get_referrers(obj_id):
#         if isinstance(referrer, dict):
#             for name, variable in referrer.items():
#                 if id(variable) == obj_id:
#                     return name
#     return None

if __name__=='__main__':
    device = torch.device("cuda")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(2000, 2000, device=device)
    z = torch.randn(3000, 3000, device=device)

    mem_usage = get_memory_usage()
    for obj_id, obj_mem in mem_usage.items():
        print(f"Object ID {obj_id} uses {obj_mem/1024/1024:.2f} MB of memory")
