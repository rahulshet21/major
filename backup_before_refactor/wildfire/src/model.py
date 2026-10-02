import segmentation_models_pytorch as smp


def build_unet(in_channels, encoder_name="resnet34"):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
        activation=None,
    )
    return model
