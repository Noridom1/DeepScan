from .vstar import policy_map as vstar_policy_map
from .vstar_zf import policy_map as vstar_zf_policy_map
from .pope import policy_map as pope_policy_map
from .pope_zf import policy_map as pope_zf_policy_map


# Merge four policy_maps
policy_map = {}
for name, cls in vstar_policy_map.items():
    policy_map[f"vstar.{name}"] = cls
    
for name, cls in vstar_zf_policy_map.items():
    policy_map[f"vstar_zf.{name}"] = cls    
    
for name, cls in pope_policy_map.items():
    policy_map[f"pope.{name}"] = cls
    
for name, cls in pope_zf_policy_map.items():
    policy_map[f"pope_zf.{name}"] = cls