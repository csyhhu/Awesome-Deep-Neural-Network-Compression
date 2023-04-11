import torch

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


def save_checkpoint(_net: torch.nn.Module, _optimizer: torch.optim, _epoch, _best_acc, _ckpt_path):

    checkpoint = {
        'net': _net.state_dict(),
        'optimizer': _optimizer.state_dict() if _optimizer is not None else None,
        'epoch': _epoch,
        'best_acc': _best_acc
    }

    torch.save(checkpoint, _ckpt_path)
    print('Save checkpoint')


def load_checkpoint(_net: torch.nn.Module, _optimizer: torch.optim.SGD, _ckpt_path):
    print('Loading checkpoint from %s..' % _ckpt_path)
    checkpoint = torch.load(_ckpt_path)
    """
    try:
        _net.load_state_dict(checkpoint, strict=False)
        print('Load state dict finish.')
    except:
    """
    _net.load_state_dict(checkpoint['net'], strict=False)
    if _optimizer is not None:
        try:
            _optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(e)

    print('Load checkpoint finish. Start training from %d with lr: %.3e' % (checkpoint['epoch'], _optimizer.param_groups[0]['lr']))
    return checkpoint['epoch'], checkpoint['best_acc']