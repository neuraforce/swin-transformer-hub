dependencies = ['torch', "timm"]

import torch
import swin

MODEL_URLS = {
    "swin_base_patch4_window12_384": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth",
    "swin_base_patch4_window12_384_22k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
    "swin_base_patch4_window12_384_22kto1k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth",
    "swin_base_patch4_window7_224": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth",
    "swin_base_patch4_window7_224_22k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth",
    "swin_base_patch4_window7_224_22kto1k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth",
    "swin_large_patch4_window12_384_22k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth",
    "swin_large_patch4_window12_384_22kto1k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth",
    "swin_large_patch4_window7_224_22k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth",
    "swin_large_patch4_window7_224_22kto1k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth",
    "swin_small_patch4_window7_224": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
    "swin_tiny_patch4_window7_224": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
}

def swin_base_patch4_window12_384(pretrained=False):
    model = swin.SwinTransformer()
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_base_patch4_window12_384"], map_location="cpu")['model'])
    return model

def swin_base_patch4_window12_384_22k(pretrained=False):
    model = swin.SwinTransformer()
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_base_patch4_window12_384_22k"], map_location="cpu")['model'])
    return model

def swin_base_patch4_window12_384_22kto1k(pretrained=False):
    model = swin.SwinTransformer()
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_base_patch4_window12_384_22kto1k"], map_location="cpu")['model'])
    return model

def swin_base_patch4_window7_224(pretrained=False):
    model = swin.SwinTransformer()
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_base_patch4_window7_224"], map_location="cpu")['model'])
    return model

def swin_base_patch4_window7_224_22k(pretrained=False):
    model = swin.SwinTransformer()
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_base_patch4_window7_224_22k"], map_location="cpu")['model'])
    return model

def swin_base_patch4_window7_224_22kto1k(pretrained=False):
    model = swin.SwinTransformer()
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_base_patch4_window7_224_22kto1k"], map_location="cpu")['model'])
    return model

def swin_large_patch4_window12_384_22k(pretrained=False):
    model = swin.SwinTransformer()
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_large_patch4_window12_384_22k"], map_location="cpu")['model'])
    return model

def swin_large_patch4_window12_384_22kto1k(pretrained=False):
    model = swin.SwinTransformer()
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_large_patch4_window12_384_22kto1k"], map_location="cpu")['model'])
    return model

def swin_large_patch4_window7_224_22k(pretrained=False):
    model = swin.SwinTransformer()
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_large_patch4_window7_224_22k"], map_location="cpu")['model'])
    return model

def swin_large_patch4_window7_224_22kto1k(pretrained=False):
    model = swin.SwinTransformer()
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_large_patch4_window7_224_22kto1k"], map_location="cpu")['model'])
    return model

def swin_small_patch4_window7_224(pretrained=False):
    model = swin.SwinTransformer()
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_small_patch4_window7_224"], map_location="cpu")['model'])
    return model

def swin_tiny_patch4_window7_224(pretrained=False):
    model = swin.SwinTransformer()
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_tiny_patch4_window7_224"], map_location="cpu")['model'])
    return model
