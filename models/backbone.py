import torchvision.models as models

def get_backbone(pretrained=True):
     backbone = models.densenet121(pretrained=pretrained).features
    return backbone
