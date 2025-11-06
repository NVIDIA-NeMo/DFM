import os

from .wan_t2v_1_3B import t2v_1_3B
from .wan_t2v_14B import t2v_14B

os.environ["TOKENIZERS_PARALLELISM"] = "false"


WAN_CONFIGS = {
    "t2v-14B": t2v_14B,
    "t2v-1.3B": t2v_1_3B,
}

SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
    "1024*1024": (1024, 1024),
}

MAX_AREA_CONFIGS = {
    "720*1280": 720 * 1280,
    "1280*720": 1280 * 720,
    "480*832": 480 * 832,
    "832*480": 832 * 480,
}

SUPPORTED_SIZES = {
    "t2v-14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "t2v-1.3B": ("480*832", "832*480"),
}
