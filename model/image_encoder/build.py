from timm.models import create_model
from .swin_transformer import SwinTransformer

def build_model(config):
    model_type = config.TYPE
    print(f"Creating model: {model_type}")
    
    if "swin" in model_type:
        model = SwinTransformer(
            num_classes=0,
            img_size=config.IMG_SIZE,
            patch_size=config.SWIN.PATCH_SIZE,
            in_chans=config.SWIN.IN_CHANS,
            embed_dim=config.SWIN.EMBED_DIM,
            depths=config.SWIN.DEPTHS,
            num_heads=config.SWIN.NUM_HEADS,
            window_size=config.SWIN.WINDOW_SIZE,
            mlp_ratio=config.SWIN.MLP_RATIO,
            qkv_bias=config.SWIN.QKV_BIAS,
            qk_scale=config.SWIN.QK_SCALE,
            drop_rate=config.DROP_RATE,
            drop_path_rate=config.DROP_PATH_RATE,
            ape=config.SWIN.APE,
            patch_norm=config.SWIN.PATCH_NORM,
            use_checkpoint=False
            ) 
    elif "vit" in model_type:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            img_size=config.DATA.IMG_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
        )
    elif "resnet" in model_type:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            num_classes=config.MODEL.NUM_CLASSES
        )
    else:
        model = create_model(
            model_type,
            pretrained=is_pretrained,
            num_classes=config.MODEL.NUM_CLASSES
        )        
    return model
