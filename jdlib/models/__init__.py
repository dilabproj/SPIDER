from .slot_attention import SlotAttention
from .slot_tcn import SlotFormer, SlotFormerPos, SlotFormerPos2
from .global_branches import GlobalAverPooling1d, SingleTransEncoder
from .local_branches import TemporalSegmentation, SlotEncoder


__all__ = [
    # global branches
    "SlotAttention",
    "SlotFormer",
    "SlotFormerPos",
    "SlotFormerPos2", 
    
    # global branches
    "GlobalAverPooling1d",
    "SingleTransEncoder",
    
    # local branches
    "TemporalSegmentation",
    "SlotEncoder",
    
]
