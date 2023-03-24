import numpy as np
import scipy.stats
import torch
import matplotlib.pyplot as plt

def highest_density_region(sample, alpha=0.05, roundto=2):
    temp = np.asarray(sample)
    temp = temp[~np.isnan(temp)]
    l = np.min(temp)
    u = np.max(temp)
    density = scipy.stats.gaussian_kde(temp,'scott')
    x = np.linspace(l, u, 500)
    y = density.evaluate(x)
    xy_zipped = zip(x, y / np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1 - alpha):
            break
    hdv.sort()
    diff = (u-l)/20  # differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i] - hdv[i - 1] >= diff:
            hpd.append(round(hdv[i - 1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    return hpd

def compute_expected_coverage(reference_samples, tested_samples,grid = 50):
    list_ = []
    for alpha in range(0,grid + 1):
        hpd = highest_density_region(reference_samples, 1-alpha/grid)
        sum = 0
        for mode in hpd:
            sum += ((tested_samples > mode[0]) * (tested_samples < mode[1])).float().mean()
        list_.append(sum.unsqueeze(0))
    return torch.cat(list_), torch.linspace(0,1,grid+1)

def plot_expected_coverage_1d_samples(reference_samples, tested_samples, label = None, figsize = (5,5)):
    assert reference_samples.shape[-1] ==1,'Dimension >= 1 not supported'
    assert tested_samples.shape[-1] == 1,'Dimension >= 1 not supported'
    to_plot, range = compute_expected_coverage(reference_samples, tested_samples)
    fig = plt.figure(figsize=figsize)
    plt.plot(range.numpy(), to_plot.numpy(), label = label)
    plt.plot(range.numpy(),range.numpy(), linestyle = '--', color = 'grey', alpha =.6)
    plt.show()