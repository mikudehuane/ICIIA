from efficientnet_pytorch import EfficientNet


def get_efficientnet_base_params(module: EfficientNet):
    for name, param in module.named_parameters():
        if name not in ('_fc.weight', '_fc.bias'):
            yield param


def get_efficientnet_top_params(module: EfficientNet):
    for name, param in module.named_parameters():
        if name in ('_fc.weight', '_fc.bias'):
            yield param
