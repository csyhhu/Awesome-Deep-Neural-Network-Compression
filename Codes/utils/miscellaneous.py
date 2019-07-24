def get_layer(net, layer_idx):
    """
    Given layer idx: ['layer3', 2, 'conv1'], this function return specific weight and bias (if possible)
    :param net:
    :param layer_idx:
    :return:
    """
    # Parser layer information
    if len(layer_idx) == 1 or (not isinstance(layer_idx, list)):
        layer = getattr(net, layer_idx)
    elif len(layer_idx) == 3:
        if type(layer_idx[2]) == int:
            layer = getattr(net, layer_idx[0])[layer_idx[1]][layer_idx[2]]
        else:
            layer = getattr(getattr(net, layer_idx[0])[layer_idx[1]], layer_idx[2])
    elif len(layer_idx) == 4:
        layer = getattr(getattr(net, layer_idx[0])[layer_idx[1]], layer_idx[2])[layer_idx[3]]
    elif len(layer_idx) == 2:
        layer = get_layer(net, layer_idx[0])[layer_idx[1]]
    else:
        print(layer_idx)
        raise NotImplementedError

    return layer