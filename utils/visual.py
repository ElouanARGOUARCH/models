from density_estimation import FlowDensityEstimation
from variational_inference import FlowSampler

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import ListedColormap
import seaborn as sns

N = 256
orange = np.ones((N, 4))
orange[:, 0] = np.geomspace(255 / 256, 1, N)  # R = 255
orange[:, 1] = np.geomspace(165 / 256, 1, N)  # G = 165
orange[:, 2] = np.geomspace(0.001 / 256, 1, N)  # B = 0
orange_cmap = ListedColormap(orange[::-1])

orange_color = "#FFA500"

red = np.ones((N, 4))
red[:, 0] = np.geomspace(255 / 256, 1, N)  # R = 255
red[:, 1] = np.geomspace(0.001 / 256, 1, N)  # G = 0
red[:, 2] = np.geomspace(0.001 / 256, 1, N)  # B = 0
red_cmap = ListedColormap(red[::-1])

red_color = "#FF0000"

blue = np.ones((N, 4))
blue[:, 0] = np.geomspace(0.001 / 256, 1, N)  # R = 0
blue[:, 1] = np.geomspace(0.001 / 256, 1, N)  # G = 0
blue[:, 2] = np.geomspace(255 / 256, 1, N)  # B = 255
blue_cmap = ListedColormap(blue[::-1])

blue_color = "#0000FF"

green = np.ones((N, 4))
green[:, 0] = np.geomspace(0.001 / 256, 1, N)  # R = 0
green[:, 1] = np.geomspace(128 / 256, 1, N)  # G = 128
green[:, 2] = np.geomspace(0.001 / 256, 1, N)  # B = 128
green_cmap = ListedColormap(green[::-1])

green_color = "#008000"

pink = np.ones((N, 4))
pink[:, 0] = np.geomspace(255 / 256, 1, N)  # R = 255
pink[:, 1] = np.geomspace(0.001 / 256, 1, N)  # G = 0
pink[:, 2] = np.geomspace(211 / 256, 1, N)  # B = 211
pink_cmap = ListedColormap(pink[::-1])

pink_color = "#FF00D3"

purple = np.ones((N, 4))
purple[:, 0] = np.geomspace(51 / 256, 1, N)  # R = 102
purple[:, 1] = np.geomspace(0.001 / 256, 1, N)  # G = 0
purple[:, 2] = np.geomspace(51 / 256, 1, N)  # B = 102
purple_cmap = ListedColormap(purple[::-1])

purple_color = "#660066"

def plot_1d_unormalized_function(f,range = [-10,10], bins=100):
    tt =torch.linspace(range[0],range[1],bins)
    with torch.no_grad():
        values = f(tt)
    plot_1d_unormalized_values(values,tt)

def plot_1d_unormalized_values(values,tt):
    x_min, x_max, bins = tt[0], tt[-1], tt.shape[0]
    plt.plot(tt, values*bins/(torch.sum(values)*(x_max - x_min)))

def plot_2d_function(f,range = [[-10,10],[-10,10]], bins = [50,50], alpha = 0.7, figsize = (10,10)):
    fig = plt.figure(figsize=figsize)
    with torch.no_grad():
        tt_x = torch.linspace(range[0][0], range[0][1], bins[0])
        tt_y = torch.linspace(range[1][0],range[1][1], bins[1])
        mesh = torch.cartesian_prod(tt_x, tt_y)
        with torch.no_grad():
            plt.pcolormesh(tt_x,tt_y,f(mesh).numpy().reshape(bins[0],bins[1]).T, cmap = matplotlib.cm.get_cmap('viridis'), alpha = alpha, lw = 0)
    plt.show()

def plot_likelihood_function(log_likelihood, range = [[-10,10],[-10,10]], bins = [50,50], levels = 2 , alpha = 0.7, figsize = (10,10)):
    fig = plt.figure(figsize=figsize)
    with torch.no_grad():
        tt_x = torch.linspace(range[0][0], range[0][1], bins[0])
        tt_y = torch.linspace(range[1][0],range[1][1], bins[1])
        tt_x_plus = tt_x.unsqueeze(0).unsqueeze(-1).repeat(tt_y.shape[0],1,1)
        tt_y_plus = tt_y.unsqueeze(1).unsqueeze(-1).repeat(1,tt_x.shape[0], 1)
        with torch.no_grad():
            plt.contourf(tt_x,tt_y,torch.exp(log_likelihood(tt_y_plus, tt_x_plus)), levels = levels, cmap = matplotlib.cm.get_cmap('viridis'), alpha = alpha, lw = 0)
    plt.show()

def plot_2d_points(samples, figsize = (10,10)):
    assert samples.shape[-1] == 2, 'Requires 2-dimensional points'
    fig = plt.figure(figsize=figsize)
    plt.scatter(samples[:,0], samples[:,1])
    plt.show()

def plot_image_2d_points(samples, bins=(200, 200), range=None, figsize=(10,10)):
    assert samples.shape[-1] == 2, 'Requires 2-dimensional points'
    fig = plt.figure(figsize=figsize)
    hist, x_edges, y_edges = numpy.histogram2d(samples[:, 0].numpy(), samples[:, 1].numpy(), bins,
                                                                range)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.imshow(torch.tensor(hist),
               extent=[0, bins[1], 0, bins[0]])
    plt.show()

def flow_visual(flow_to_visualize):
    num_samples = 5000
    if isinstance(flow_to_visualize, FlowSampler):
        if flow_to_visualize.p == 1:
            linspace = 500
            with torch.no_grad():
                samples = flow_to_visualize.reference.sample([num_samples])
                backward_samples = [samples]
                tt = torch.linspace(torch.min(backward_samples[0]), torch.max(backward_samples[0]), linspace).unsqueeze(
                    1)
                backward_density = [torch.exp(flow_to_visualize.model[-1].q_log_prob(tt))]
                backward_linspace = [tt]
                forward_density = [torch.exp(flow_to_visualize.model[-1].latent_log_prob(tt))]
                for i in range(flow_to_visualize.N - 1, -1, -1):
                    samples = flow_to_visualize.model[i].sample_backward(samples)
                    tt = torch.linspace(torch.min(samples), torch.max(samples), linspace).unsqueeze(1)
                    b_density = torch.exp(flow_to_visualize.model[i].log_prob(tt))
                    f_density = torch.exp(flow_to_visualize.model[i].p_log_prob(tt))
                    backward_samples.insert(0, samples)
                    backward_linspace.insert(0, tt)
                    backward_density.insert(0, b_density)
                    forward_density.insert(0, f_density)

            fig = plt.figure(figsize=((flow_to_visualize.N + 1) * 8, 2 * 7))
            ax = fig.add_subplot(2, flow_to_visualize.N + 1, 1)
            ax.plot(backward_linspace[0].cpu(), forward_density[0].cpu(), color='red',
                    label="Input Model density")
            ax.legend(loc=1)
            for i in range(1, flow_to_visualize.N):
                ax = fig.add_subplot(2, flow_to_visualize.N + 1, i + 1)
                ax.plot(backward_linspace[i].cpu(), forward_density[i].cpu(), color='magenta',
                        label="Intermediate density")
                ax.legend(loc=1)
            ax = fig.add_subplot(2, flow_to_visualize.N + 1, flow_to_visualize.N + 1)
            ax.plot(backward_linspace[-1].cpu(), forward_density[-1].cpu(), color='orange',
                    label="Proxy density")
            ax.legend(loc=1)
            ax = fig.add_subplot(2, flow_to_visualize.N + 1, flow_to_visualize.N + 1 + 1)
            sns.histplot(backward_samples[0][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='blue',
                         label="Model Samples")
            ax.plot(backward_linspace[0].cpu(), backward_density[0].cpu(), color='blue',
                    label="Output Model density")
            ax.legend(loc=1)
            for i in range(1, flow_to_visualize.N):
                ax = fig.add_subplot(2, flow_to_visualize.N + 1, flow_to_visualize.N + 1 + i + 1)
                sns.histplot(backward_samples[i][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='purple',
                             label="Intermediate Samples")
                ax.plot(backward_linspace[i].cpu(), backward_density[i].cpu(), color='purple',
                        label="Intermediate density")
                ax.legend(loc=1)
            ax = fig.add_subplot(2, flow_to_visualize.N + 1, flow_to_visualize.N + 1 + flow_to_visualize.N + 1)
            ax.plot(backward_linspace[-1].cpu(), backward_density[-1].cpu(), color='green', label="Reference density")
            sns.histplot(backward_samples[-1][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='green',
                         label="Reference Samples")
            ax.legend(loc=1)
            plt.show()
        elif flow_to_visualize.p > 1:
            delta = 200
            with torch.no_grad():
                backward_samples = [flow_to_visualize.reference.sample([num_samples])]
                grid = torch.cat((torch.cartesian_prod(
                    torch.linspace(torch.min(backward_samples[0][:, 0]).item(),
                                   torch.max(backward_samples[0][:, 0]).item(), delta),
                    torch.linspace(torch.min(backward_samples[0][:, 1]).item(),
                                   torch.max(backward_samples[0][:, 1]).item(), delta)),
                                  torch.mean(backward_samples[0][:, 2:], dim=0) * torch.ones(delta**2,flow_to_visualize.p - 2)),dim=-1)
                backward_density = [torch.exp(flow_to_visualize.reference.log_prob(grid)).reshape(delta, delta).T.cpu().detach()]
                forward_density = [torch.exp(flow_to_visualize.model[-1].latent_log_prob(grid)).reshape(delta, delta).T.cpu().detach()]
                x_range = [[torch.min(backward_samples[0][:, 0]).item(), torch.max(backward_samples[0][:, 0]).item()]]
                y_range = [[torch.min(backward_samples[0][:, 1]).item(), torch.max(backward_samples[0][:, 1]).item()]]
                for i in range(flow_to_visualize.N - 1, -1, -1):
                    backward_samples.insert(0, flow_to_visualize.model[i].sample_backward(backward_samples[0]))
                    grid = torch.cat((torch.cartesian_prod(
                        torch.linspace(torch.min(backward_samples[0][:, 0]).item(),
                                       torch.max(backward_samples[0][:, 0]).item(), delta),
                        torch.linspace(torch.min(backward_samples[0][:, 1]).item(),
                                       torch.max(backward_samples[0][:, 1]).item(), delta)),
                                      torch.mean(backward_samples[0][:, 2:], dim=0) * torch.ones(delta**2,flow_to_visualize.p - 2)), dim=-1)
                    backward_density.insert(0, torch.exp(flow_to_visualize.model[i].log_prob(grid)).reshape(delta,
                                                                                                    delta).T.cpu().detach())
                    forward_density.insert(0, torch.exp(flow_to_visualize.model[i].p_log_prob(grid)).reshape(delta,
                                                                                                   delta).T.cpu().detach())
                    x_range.insert(0, [torch.min(backward_samples[0][:, 0]).item(), torch.max(backward_samples[0][:, 0]).item()])
                    y_range.insert(0, [torch.min(backward_samples[0][:, 1]).item(), torch.max(backward_samples[0][:, 1]).item()])

            fig = plt.figure(figsize=((flow_to_visualize.N + 1) * 8, 3 * 7))
            ax = fig.add_subplot(3, flow_to_visualize.N + 1, 1)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.pcolormesh(torch.linspace(torch.min(backward_samples[0][:, 0]).item(),
                                         torch.max(backward_samples[0][:, 0]).item(), delta),
                          torch.linspace(torch.min(backward_samples[0][:, 1]).item(),
                                         torch.max(backward_samples[0][:, 1]).item(), delta), forward_density[0],
                          cmap=red_cmap, shading='auto')
            ax.set_xlim((x_range[0][0], x_range[0][1]))
            ax.set_ylim((y_range[0][0], y_range[0][1]))
            ax.set_title(r'$P$ density')
            for i in range(1, flow_to_visualize.N):
                ax = fig.add_subplot(3, flow_to_visualize.N + 1, i + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.set_xlim((x_range[i][0], x_range[i][1]))
                ax.set_ylim((y_range[i][0], y_range[i][1]))
                ax.pcolormesh(torch.linspace(torch.min(backward_samples[i][:, 0]).item(),
                                             torch.max(backward_samples[i][:, 0]).item(), delta),
                              torch.linspace(torch.min(backward_samples[i][:, 1]).item(),
                                             torch.max(backward_samples[i][:, 1]).item(), delta), forward_density[i],
                              cmap=pink_cmap, shading='auto')
            ax = fig.add_subplot(3, flow_to_visualize.N + 1, flow_to_visualize.N + 1)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.pcolormesh(torch.linspace(torch.min(backward_samples[-1][:, 0]).item(),
                                         torch.max(backward_samples[-1][:, 0]).item(), delta),
                          torch.linspace(torch.min(backward_samples[-1][:, 1]).item(),
                                         torch.max(backward_samples[-1][:, 1]).item(), delta), forward_density[-1],
                          cmap=orange_cmap, shading='auto')
            ax.set_xlim((x_range[-1][0], x_range[-1][1]))
            ax.set_ylim((y_range[-1][0], y_range[-1][1]))
            ax.set_title(r'$\Phi$ Density')

            ax = fig.add_subplot(3, flow_to_visualize.N + 1, flow_to_visualize.N + 1 + 1)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.scatter(backward_samples[0][:, 0].cpu(), backward_samples[0][:, 1].cpu(), alpha=0.5, color=blue_color)
            ax.set_xlim((x_range[0][0], x_range[0][1]))
            ax.set_ylim((y_range[0][0], y_range[0][1]))
            ax.set_title(r'$\Psi$ Samples')
            ax = fig.add_subplot(3, flow_to_visualize.N + 1, 2 * (flow_to_visualize.N + 1) + 1)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.pcolormesh(torch.linspace(torch.min(backward_samples[0][:, 0]).item(),
                                         torch.max(backward_samples[0][:, 0]).item(), delta),
                          torch.linspace(torch.min(backward_samples[0][:, 1]).item(),
                                         torch.max(backward_samples[0][:, 1]).item(), delta), backward_density[0],
                          cmap=blue_cmap, shading='auto')
            ax.set_xlim((x_range[0][0], x_range[0][1]))
            ax.set_ylim((y_range[0][0], y_range[0][1]))
            ax.set_title(r'$\Psi$ Density')
            for i in range(1, flow_to_visualize.N):
                ax = fig.add_subplot(3, flow_to_visualize.N + 1, flow_to_visualize.N + 1 + i + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.scatter(backward_samples[i][:, 0].cpu(), backward_samples[i][:, 1].cpu(), alpha=0.5,
                           color=purple_color)
                ax.set_xlim((x_range[i][0], x_range[i][1]))
                ax.set_ylim((y_range[i][0], y_range[i][1]))
                ax = fig.add_subplot(3, flow_to_visualize.N + 1, 2 * (flow_to_visualize.N + 1) + i + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.pcolormesh(torch.linspace(torch.min(backward_samples[i][:, 0]).item(),
                                             torch.max(backward_samples[i][:, 0]).item(), delta),
                              torch.linspace(torch.min(backward_samples[i][:, 1]).item(),
                                             torch.max(backward_samples[i][:, 1]).item(), delta), backward_density[i],
                              cmap=purple_cmap, shading='auto')
                ax.set_xlim((x_range[i][0], x_range[i][1]))
                ax.set_ylim((y_range[i][0], y_range[i][1]))
            ax = fig.add_subplot(3, flow_to_visualize.N + 1, flow_to_visualize.N + 1 + flow_to_visualize.N + 1)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.scatter(backward_samples[-1][:, 0].cpu(), backward_samples[-1][:, 1].cpu(), alpha=0.5,
                       color=green_color)
            ax.set_xlim((x_range[-1][0], x_range[-1][1]))
            ax.set_ylim((y_range[-1][0], y_range[-1][1]))
            ax.set_title(r'$Q$ Samples')
            ax = fig.add_subplot(3, flow_to_visualize.N + 1, 2 * (flow_to_visualize.N + 1) + flow_to_visualize.N + 1)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.pcolormesh(torch.linspace(torch.min(backward_samples[-1][:, 0]).item(),
                                         torch.max(backward_samples[-1][:, 0]).item(), delta),
                          torch.linspace(torch.min(backward_samples[-1][:, 1]).item(),
                                         torch.max(backward_samples[-1][:, 1]).item(), delta), backward_density[-1],
                          cmap=green_cmap, shading='auto')
            ax.set_xlim((x_range[-1][0], x_range[-1][1]))
            ax.set_ylim((y_range[-1][0], y_range[-1][1]))
            ax.set_title(r'$Q$ Density')
            plt.show()
    elif isinstance(flow_to_visualize, FlowDensityEstimation):
        if flow_to_visualize.p ==1:
            linspace = 500
            with torch.no_grad():
                backward_samples = [flow_to_visualize.reference.sample([num_samples])]
                tt = torch.linspace(torch.min(backward_samples[0]), torch.max(backward_samples[0]), linspace).unsqueeze(
                    1)
                backward_density = [torch.exp(flow_to_visualize.reference.log_prob(tt))]
                backward_linspace = [tt]
                for i in range(flow_to_visualize.N - 1, -1, -1):
                    samples = flow_to_visualize.model[i].sample_backward(backward_samples[0])
                    tt = torch.linspace(torch.min(samples), torch.max(samples), linspace).unsqueeze(1)
                    density = torch.exp(flow_to_visualize.model[i].log_prob(tt))
                    backward_samples.insert(0, samples)
                    backward_linspace.insert(0, tt)
                    backward_density.insert(0, density)

                forward_samples = [flow_to_visualize.target_samples[:num_samples]]
                for i in range(flow_to_visualize.N):
                    forward_samples.append(flow_to_visualize.model[i].sample_forward(forward_samples[-1]))

                fig = plt.figure(figsize=((flow_to_visualize.N + 1) * 8, 2 * 7))
                ax = fig.add_subplot(2, flow_to_visualize.N + 1, 1)
                sns.histplot(forward_samples[0][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color=red_color,
                             label="Input Target Samples")
                ax.legend(loc=1)
                for i in range(1, flow_to_visualize.N):
                    ax = fig.add_subplot(2, flow_to_visualize.N + 1, i + 1)
                    sns.histplot(forward_samples[i][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color=pink_color,
                                 label="Intermediate Samples")
                    ax.legend(loc=1)
                ax = fig.add_subplot(2, flow_to_visualize.N + 1, flow_to_visualize.N + 1)
                sns.histplot(forward_samples[-1][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color=orange_color,
                             label="Proxy Samples")
                ax.legend(loc=1)

                ax = fig.add_subplot(2, flow_to_visualize.N + 1, flow_to_visualize.N + 1 + 1)
                sns.histplot(backward_samples[0][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color=blue_color,
                             label="Model Samples")
                ax.plot(backward_linspace[0].cpu(), backward_density[0].cpu(), color=blue_color,
                        label="Output Model density")
                ax.legend(loc=1)
                for i in range(1, flow_to_visualize.N):
                    ax = fig.add_subplot(2, flow_to_visualize.N + 1, flow_to_visualize.N + 1 + i + 1)
                    sns.histplot(backward_samples[i][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color=purple_color,
                                 label="Intermediate Samples")
                    ax.plot(backward_linspace[i].cpu(), backward_density[i].cpu(), color=purple_color,
                            label="Intermediate density")
                    ax.legend(loc=1)
                ax = fig.add_subplot(2, flow_to_visualize.N + 1, flow_to_visualize.N + 1 + flow_to_visualize.N + 1)
                ax.plot(backward_linspace[-1].cpu(), backward_density[-1].cpu(), color='green', label="Reference density")
                sns.histplot(backward_samples[-1][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color=green_color,
                             label="Reference Samples")
                ax.legend(loc=1)
                plt.show()
        elif flow_to_visualize.p > 1:
            delta = 200
            with torch.no_grad():
                samples = flow_to_visualize.reference.sample([num_samples])
                backward_samples = [samples]
                grid = torch.cartesian_prod(
                    torch.linspace(torch.min(samples[:, 0]).item(), torch.max(samples[:, 0]).item(), delta),
                    torch.linspace(torch.min(samples[:, 1]).item(), torch.max(samples[:, 1]).item(), delta))
                grid = torch.cat(
                    (grid, torch.mean(samples[:, 2:], dim=0) * torch.ones(grid.shape[0], flow_to_visualize.p - 2)),
                    dim=-1)
                density = torch.exp(flow_to_visualize.reference.log_prob(grid)).reshape(delta, delta).T.detach()
                backward_density = [density]
                x_range = [[torch.min(samples[:, 0]).item(), torch.max(samples[:, 0]).item()]]
                y_range = [[torch.min(samples[:, 1]).item(), torch.max(samples[:, 1]).item()]]
                for i in range(flow_to_visualize.N - 1, -1, -1):
                    samples = flow_to_visualize.model[i].sample_backward(backward_samples[0])
                    backward_samples.insert(0, samples)
                    grid = torch.cartesian_prod(
                        torch.linspace(torch.min(samples[:, 0]).item(), torch.max(samples[:, 0]).item(), delta),
                        torch.linspace(torch.min(samples[:, 1]).item(), torch.max(samples[:, 1]).item(), delta))
                    grid = torch.cat((grid, torch.zeros(grid.shape[0], flow_to_visualize.p - 2)), dim=-1)
                    density = torch.exp(flow_to_visualize.model[i].log_prob(grid)).reshape(delta, delta).T.detach()
                    backward_density.insert(0, density)
                    x_range.insert(0, [torch.min(samples[:, 0]).item(), torch.max(samples[:, 0]).item()])
                    y_range.insert(0, [torch.min(samples[:, 1]).item(), torch.max(samples[:, 1]).item()])


                    forward_samples = [flow_to_visualize.target_samples[:num_samples]]
                    for i in range(flow_to_visualize.N):
                        forward_samples.append(flow_to_visualize.model[i].sample_forward(forward_samples[-1]))

                fig = plt.figure(figsize=((flow_to_visualize.N + 1) * 8, 3 * 7))
                ax = fig.add_subplot(3, flow_to_visualize.N + 1, 1)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                ax.scatter(forward_samples[0][:, 0], forward_samples[0][:, 1], alpha=0.5, color=red_color)
                ax.set_title(r'$P$ samples')
                ax.set_xlim((x_range[0][0], x_range[0][1]))
                ax.set_ylim((y_range[0][0], y_range[0][1]))
                for i in range(1, flow_to_visualize.N):
                    ax = fig.add_subplot(3, flow_to_visualize.N + 1, i + 1)
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    ax.scatter(forward_samples[i][:, 0], forward_samples[i][:, 1], alpha=0.5,
                               color=pink_color)
                    ax.set_xlim((x_range[i][0], x_range[i][1]))
                    ax.set_ylim((y_range[i][0], y_range[i][1]))
                ax = fig.add_subplot(3, flow_to_visualize.N + 1, flow_to_visualize.N + 1)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                ax.scatter(forward_samples[-1][:, 0], forward_samples[-1][:, 1], alpha=0.5,
                           color=orange_color)
                ax.set_xlim((x_range[-1][0], x_range[-1][1]))
                ax.set_ylim((y_range[-1][0], y_range[-1][1]))
                ax.set_title(r'$\Phi$ Samples')

                ax = fig.add_subplot(3, flow_to_visualize.N + 1, flow_to_visualize.N + 1 + 1)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                ax.scatter(backward_samples[0][:, 0], backward_samples[0][:, 1], alpha=0.5, color=blue_color)
                ax.set_xlim((x_range[0][0], x_range[0][1]))
                ax.set_ylim((y_range[0][0], y_range[0][1]))
                ax.set_title(r'$\Psi$ Samples')
                ax = fig.add_subplot(3, flow_to_visualize.N + 1, 2 * (flow_to_visualize.N + 1) + 1)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                ax.pcolormesh(
                    torch.linspace(torch.min(backward_samples[0][:, 0]).item(), torch.max(backward_samples[0][:, 0]).item(),
                                   200),
                    torch.linspace(torch.min(backward_samples[0][:, 1]).item(), torch.max(backward_samples[0][:, 1]).item(),
                                   200), backward_density[0], cmap=blue_cmap, shading='auto')
                ax.set_xlim((x_range[0][0], x_range[0][1]))
                ax.set_ylim((y_range[0][0], y_range[0][1]))
                ax.set_title(r'$\Psi$ Density')
                for i in range(1, flow_to_visualize.N):
                    ax = fig.add_subplot(3, flow_to_visualize.N + 1, flow_to_visualize.N + 1 + i + 1)
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    ax.scatter(backward_samples[i][:, 0], backward_samples[i][:, 1], alpha=0.5,
                               color=purple_color)
                    ax.set_xlim((x_range[i][0], x_range[i][1]))
                    ax.set_ylim((y_range[i][0], y_range[i][1]))
                    ax = fig.add_subplot(3, flow_to_visualize.N + 1, 2 * (flow_to_visualize.N + 1) + i + 1)
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    ax.set_xlim((x_range[i][0], x_range[i][1]))
                    ax.set_ylim((y_range[i][0], y_range[i][1]))
                    ax.pcolormesh(torch.linspace(torch.min(backward_samples[i][:, 0]).item(),
                                                 torch.max(backward_samples[i][:, 0]).item(), 200),
                                  torch.linspace(torch.min(backward_samples[i][:, 1]).item(),
                                                 torch.max(backward_samples[i][:, 1]).item(), 200), backward_density[i],
                                  cmap=purple_cmap, shading='auto')

                ax = fig.add_subplot(3, flow_to_visualize.N + 1, flow_to_visualize.N + 1 + flow_to_visualize.N + 1)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                ax.scatter(backward_samples[-1][:, 0], backward_samples[-1][:, 1], alpha=0.5,
                           color=green_color)
                ax.set_xlim((x_range[-1][0], x_range[-1][1]))
                ax.set_ylim((y_range[-1][0], y_range[-1][1]))
                ax.set_title(r'$Q$ samples')
                ax = fig.add_subplot(3, flow_to_visualize.N + 1, 2 * (flow_to_visualize.N + 1) + flow_to_visualize.N + 1)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                ax.pcolormesh(torch.linspace(torch.min(backward_samples[-1][:, 0]).item(),
                                             torch.max(backward_samples[-1][:, 0]).item(), 200),
                              torch.linspace(torch.min(backward_samples[-1][:, 1]).item(),
                                             torch.max(backward_samples[-1][:, 1]).item(), 200), backward_density[-1],
                              cmap=green_cmap, shading='auto')
                ax.set_xlim((x_range[-1][0], x_range[-1][1]))
                ax.set_ylim((y_range[-1][0], y_range[-1][1]))
                ax.set_title(r"$Q$ Density")
                plt.show()