def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break


import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        pass
    elif _backend == "pytorch":
        from .ngcf.NGCF import NGCF
        from .lightgcn.LightGCN import LightGCN
        from .mmgcn.MMGCN import MMGCN
        from .dgcf.DGCF import DGCF
        from .bprmf.BPRMF import BPRMF
        from .vbpr.VBPR import VBPR
        from .grcn.GRCN import GRCN
        from .mgat.MGAT import MGAT
        from .lattice.LATTICE import LATTICE
        from .uuii.UUII import UUII
        from .ultragcn import UltraGCN
