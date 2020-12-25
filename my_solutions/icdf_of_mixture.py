import torch

from torch.distributions.normal import Normal
from torch.distributions.mixture_same_family import Distribution, MixtureSameFamily
from torch.distributions.categorical import Categorical

# mix = Categorical(probs=torch.FloatTensor([0.1, 0.6, 0.1, 0.1, 0.1]))
# comp = Normal(torch.randn(5,), torch.rand(5,))
# gmm = MixtureSameFamily(mix, comp)
#
# gmm.icdf(torch.FloatTensor([1]))


dist = Normal(0,1)
dist.cdf(0.5) #0.69146246