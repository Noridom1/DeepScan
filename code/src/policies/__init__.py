from .vstar import policy_map as vstar_policy_map
# from .pope import policy_map as pope_policy_map
# from .hrbench import policy_map as hrbench_policy_map
# from .treebench import policy_map as treebench_policy_map

# Merge four policy_maps
policy_map = {}
for name, cls in vstar_policy_map.items():
    policy_map[f"vstar.{name}"] = cls

# for name, cls in pope_policy_map.items():
#     policy_map[f"pope.{name}"] = cls
    
# for name, cls in hrbench_policy_map.items():
#     policy_map[f"hrbench.{name}"] = cls    
    
# for name, cls in treebench_policy_map.items():
#     policy_map[f"treebench.{name}"] = cls