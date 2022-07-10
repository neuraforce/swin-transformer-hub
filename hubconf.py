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
    model = swin.SwinTransformer(img_size=384, patch_size=4, windows_size=12, depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32])
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_base_patch4_window12_384"], map_location="cpu")['model'])
    return model

def swin_base_patch4_window12_384_22k(pretrained=False):
    model = swin.SwinTransformer(img_size=384, patch_size=4, windows_size=12, depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32], num_classes=21841)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_base_patch4_window12_384_22k"], map_location="cpu")['model'])
    return model

def swin_base_patch4_window12_384_22kto1k(pretrained=False):
    model = swin.SwinTransformer(img_size=384, patch_size=4, windows_size=12, depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32])
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_base_patch4_window12_384_22kto1k"], map_location="cpu")['model'])
    return model

def swin_base_patch4_window7_224(pretrained=False):
    model = swin.SwinTransformer(img_size=224, patch_size=4, windows_size=7, depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32])
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_base_patch4_window7_224"], map_location="cpu")['model'])
    return model

def swin_base_patch4_window7_224_22k(pretrained=False):
    model = swin.SwinTransformer(img_size=224, patch_size=4, windows_size=7, depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32], num_classes=21841)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_base_patch4_window7_224_22k"], map_location="cpu")['model'])
    return model

def swin_base_patch4_window7_224_22kto1k(pretrained=False):
    model = swin.SwinTransformer(img_size=224, patch_size=4, windows_size=7, depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32])
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_base_patch4_window7_224_22kto1k"], map_location="cpu")['model'])
    return model

def swin_large_patch4_window12_384_22k(pretrained=False):
    model = swin.SwinTransformer(img_size=384, patch_size=4, windows_size=12, depths=[2, 2, 18, 2], embed_dim=192, num_heads=[6, 12, 24, 48], num_classes=21841)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_large_patch4_window12_384_22k"], map_location="cpu")['model'])
    return model

def swin_large_patch4_window12_384_22kto1k(pretrained=False):
    model = swin.SwinTransformer(img_size=384, patch_size=4, windows_size=12, depths=[2, 2, 18, 2], embed_dim=192, num_heads=[6, 12, 24, 48])
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_large_patch4_window12_384_22kto1k"], map_location="cpu")['model'])
    return model

def swin_large_patch4_window7_224_22k(pretrained=False):
    model = swin.SwinTransformer(img_size=224, patch_size=4, windows_size=7, depths=[2, 2, 18, 2], embed_dim=192, num_heads=[6, 12, 24, 48], num_classes=21841)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_large_patch4_window7_224_22k"], map_location="cpu")['model'])
    return model

def swin_large_patch4_window7_224_22kto1k(pretrained=False):
    model = swin.SwinTransformer(img_size=224, patch_size=4, windows_size=7, depths=[2, 2, 18, 2], embed_dim=192, num_heads=[6, 12, 24, 48])
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_large_patch4_window7_224_22kto1k"], map_location="cpu")['model'])
    return model

def swin_small_patch4_window7_224(pretrained=False):
    model = swin.SwinTransformer(img_size=224, patch_size=4, windows_size=7, depths=[2, 2, 18, 2], embed_dim=96, num_heads=[3, 6, 12, 24])
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_small_patch4_window7_224"], map_location="cpu")['model'])
    return model

def swin_tiny_patch4_window7_224(pretrained=False):
    model = swin.SwinTransformer(img_size=224, patch_size=4, windows_size=7)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS["swin_tiny_patch4_window7_224"], map_location="cpu")['model'])
    return model
