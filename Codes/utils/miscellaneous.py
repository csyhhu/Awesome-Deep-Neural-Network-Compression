def get_layer(net, layer_info):

    layer = net

    try:
        for info in layer_info:
            if type(info) == str:
                layer = getattr(layer, info)
            elif type(info) == int:
                layer = layer[info]
    except:
        print (layer_info)
        raise RuntimeError

    return layer