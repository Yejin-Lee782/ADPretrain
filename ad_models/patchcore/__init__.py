import torch
from .common import FaissNN
from .patchcore import PatchCore 
from .sampler import IdentitySampler, GreedyCoresetSampler, ApproximateGreedyCoresetSampler 


def get_patchcore(encoder,
                  projectors,
                  device,
                  dtype=torch.float,
                  residual=True,
                  input_shape=(3, 224, 224),
                  pretrain_embed_dimension=1024,
                  target_embed_dimension=1024,
                  patchsize=3,
                  anomaly_scorer_num_nn=1,
                  faiss_on_gpu=True, 
                  faiss_num_workers=8):
    sampler = get_sampler(name='approx_greedy_coreset', percentage=0.01, device=device)
    
    nn_method = FaissNN(faiss_on_gpu, faiss_num_workers)

    patchcore_instance = PatchCore(device, dtype=dtype)
    patchcore_instance.load(
        encoder=encoder,
        projectors=projectors,
        layers_to_extract_from=None,
        device=device,
        residual=residual,
        input_shape=input_shape,
        pretrain_embed_dimension=pretrain_embed_dimension,
        target_embed_dimension=target_embed_dimension,
        patchsize=patchsize,
        featuresampler=sampler,
        anomaly_scorer_num_nn=anomaly_scorer_num_nn,
        nn_method=nn_method,
    )
    
    return patchcore_instance

    
def get_sampler(name, percentage, device):
    if name == "identity":
        return IdentitySampler()
    elif name == "greedy_coreset":
        return GreedyCoresetSampler(percentage, device)
    elif name == "approx_greedy_coreset":
        return ApproximateGreedyCoresetSampler(percentage, device)
