import numpy as np

TASKS = ['humanoid-stand', 'walker-walk', 'dog-stand', 'finger-spin', 'cheetah-run', 'quadruped-walk', 'cartpole-swingup']
TASKS = ['cheetah-run', 'dog-stand', 'quadruped-walk']

qrt = 2

def load_datapoints(task, variable, utd, bs, lr):
    directory = f'./data/results/results/{variable}/BRO_{bs}_{utd}_{lr}/{task}.npy'
    data = np.load(directory)
    if data.shape[1] == 11:
        data = data[:,1:]
    return data

def get_task_id(tasks, task):
    for i, task_ in enumerate(tasks):
        if task == task_:
            return i

def load_array(rr, bs, lr, task, performance_to_reach=800):
    name = f'./data/results/results/returns/BRO_{bs}_{rr}_{lr}/{task}.npy'
    data = np.load(name)
    data = data.mean(-1)
    return data

def _get_time_to_performance(data, performance=500.0):
    indx = np.argmin(np.abs(data - performance), axis=0)
    steps = indx
    return steps

def get_time_to_performance(data, performances):
    feature_matrix = np.zeros((performances.shape[0]))
    for i, perf in enumerate(performances):
        feature_matrix[i] = _get_time_to_performance(data, perf)
    return feature_matrix

def get_single_task_single_hypers_features(rr, bs, lr, task):
    bs_encoder = {32: 1, 64:2, 128:3, 256:4, 512:5}
    lr_encoder = {'15e-5' : 1, '3e-4' : 2, '6e-4' : 3, '12e-3': 4}
    rr_encoder = {1 : 1, 2 : 2, 3 : 3, 4 : 4, 5: 5, 8: 8}
    data = load_array(rr, bs, lr, task)
    performances = np.arange(800, 801, 50)
    feature_matrix = get_time_to_performance(data, performances)
    features = np.concatenate((performances[:, None]/1000, feature_matrix[:, None]), -1)
    bs_ = np.zeros((features.shape[0], 1)) + bs_encoder[bs]
    lr_ = np.zeros((features.shape[0], 1)) + lr_encoder[lr]
    rr_ = np.zeros((features.shape[0], 1)) + rr_encoder[rr]
    task_ = np.zeros((features.shape[0], 1)) + get_task_id(TASKS, task)
    features = np.concatenate((features, bs_, lr_, rr_, task_), axis=-1)
    return features

def get_data_(rr, bs, lr, task):
    print(rr, bs, lr, task)
    data = get_single_task_single_hypers_features(rr, bs, lr, task).mean(0)
    pnorms = load_datapoints(task, 'pnorms', rr, bs, lr).mean(-1)
    len_ = pnorms.shape[0]
    pnorms = pnorms[len_//qrt:]
    train_loss = load_datapoints(task, 'train_loss', rr, bs, lr).mean(-1)[len_//qrt:]
    val_loss = load_datapoints(task, 'validation_loss', rr, bs, lr).mean(-1)[len_//qrt:]
    if train_loss.shape[0] != val_loss.shape[0]:
        if train_loss.shape[0] > val_loss.shape[0]:
            train_loss = train_loss[:val_loss.shape[0]]
        else:
            val_loss = val_loss[:train_loss.shape[0]]
    overfit = (val_loss - train_loss)/train_loss
    data = np.concatenate((data, pnorms.mean()[None], overfit.mean()[None]))
    # other overfitting
    train_loss2 = load_datapoints(task, 'train_loss', rr, bs, lr)[len_//qrt:]
    val_loss2 = load_datapoints(task, 'validation_loss', rr, bs, lr)[len_//qrt:]
    if train_loss2.shape[0] != val_loss2.shape[0]:
        if train_loss2.shape[0] > val_loss2.shape[0]:
            train_loss2 = train_loss2[:val_loss2.shape[0]]
        else:
            val_loss2 = val_loss2[:train_loss2.shape[0]]
    train_loss_delta = train_loss2[1:] - train_loss2[:-1]
    val_loss_delta = val_loss2[1:] - val_loss2[:-1]
    val_loss_increase = np.where(val_loss_delta > 0, 1, 0)
    train_loss_decrease = np.where(train_loss_delta < 0, 1, 0)
    overfitting2 = val_loss_increase * train_loss_decrease
    overfitting2 = overfitting2.mean(-1)
    data = np.concatenate((data, overfitting2.mean()[None]))
    return data

lr = '3e-4'
bs = 32
rr = 1
task = 'cartpole-swingup'

def get_data_task(task, rrs=[1,2,4,8]):
    data = []
    for bs in [32,64,128,256,512]:
        for lr in ['15e-5', '3e-4', '6e-4', '12e-3']:
            for rr in rrs:
                data.append(get_data_(rr, bs, lr, task))
    data = np.stack(data)[:,1:]
    return data

def get_data(TASKS, normalize=True, rrs=[1,2,4,8]):
    data = []
    for task in TASKS:
        data_ = get_data_task(task, rrs)
        if normalize:
            for column in range(data_.shape[-1]):
                #print(column)
                if column == 7:
                    data_[:, column] = (data_[:, column] - data_[:, column].mean()) / (data_[:, column].std() + 1e-8)
        data.append(data_)
    data = np.stack(data)
    data = data.reshape(-1, data.shape[-1])
    import pandas as pd
    data = pd.DataFrame(data)
    data.rename(columns={0: 'n_steps', 1: 'bs', 2: 'lr', 3: 'rr', 4: 'task', 5: 'pnorms', 6: 'overfitting', 7: 'overfitting2'}, inplace=True)
    return data

get_data(TASKS)

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib    
from scipy.optimize import minimize
from rliable import plot_utils


COLORS = ['#BBCC33', '#77AADD', '#44BB99',
           '#EEDD88', '#EE8866', '#FFAABB',
          '#99DDFF', '#44BB99', '#AAAA00',
          '#DDDDDD']
palette = sns.set_palette(COLORS)
plt.rcParams['text.usetex'] = False #Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = r'\usepackage{sansmath} \sansmath' #Force sans-serif math mode (for axes labels)
plt.rcParams['font.family'] = 'sans-serif' # ... for regular text
plt.rcParams['font.sans-serif'] = ['Helveta Nue'] # Choose a nice font here

def graph1():
    sns.set_style("whitegrid")
    
    fig, ax1 = plt.subplots()
    fig.set_size_inches(496.0/192*2, 369.6/192*2)
    ax1.ticklabel_format(useOffset=False, style='plain')
    
    def polynomial(x, a, b):
        return a * x + b
        
    data = get_data(TASKS, True, [1,2,4,8])
    data['n_steps'] = np.log2(data['n_steps'])
    steps = np.log2(data['rr'])

    ax1.scatter(steps, data['overfitting'], color=COLORS[1], s=40.0, alpha=0.8, label='Empirical value')
    # Compute linear regression (trendline)
    x = steps
    y = data['overfitting']
    
    def mae_loss(params):
        a, b = params
        y_pred = polynomial(x, a, b)
        return np.mean(np.abs(y - y_pred))
    
    def mse_loss(params):
        a, b = params
        y_pred = polynomial(x, a, b)
        return np.mean((y - y_pred)**2)
    
    # Initial guess for the parameters
    initial_guess = [1, 1]
    
    # Minimize the MAE loss
    result = minimize(mae_loss, initial_guess, method='Powell')
    #result = minimize(mse_loss, initial_guess, method='Powell')
    a, b = result.x  # Extract the coefficients from the optimization result
    
    # Generate a smooth line for the polynomial
    x_line = np.linspace(x.min(), x.max(), 500)
    y_line = polynomial(x_line, a, b)
    
    # Plot the trendline with a thicker line width and brighter color
    trendline_color = COLORS[1]   
    ax1.plot(x_line, y_line, color=trendline_color, linestyle="--", alpha=0.8, linewidth=3, label='Trend')
    
    ax1.set_xticks([0,1,2,3])
    ax1.set_xticklabels([str(rr) for rr in [1,2,4,8]])

    yticks = np.array([0,2,4])
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticks)
    ax1.legend(prop={'size': 14}, ncol=1, frameon=False)

    plot_utils._annotate_and_decorate_axis(ax1, xlabel='$\sigma$: UTD Ratio',
                                           ylabel='Overfitting',
                                           labelsize='xx-large', ticklabelsize='xx-large',
                                           grid_alpha=0.2, legend=False)

    
    fig.tight_layout()
    fig.savefig(f'analysis_utd_overfitting.eps')
    fig.show()
    
graph1()

def graph1_1():
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots()
    fig.set_size_inches(496.0/192*2, 369.6/192*2)
    ax1.ticklabel_format(useOffset=False, style='plain')
    def polynomial(x, a, b):
        return a * x + b 
    data = get_data(TASKS, True, [1,2,4,8])
    data['n_steps'] = np.log2(data['n_steps'])
    steps = data['bs']
    data['pnorms'] = data['pnorms'] / 1000
    ax1.scatter(steps, data['overfitting'], color=COLORS[1], s=40.0, alpha=0.8, label='Empirical value')
    # Compute linear regression (trendline)
    x = steps
    y = data['overfitting']
    def mae_loss(params):
        a, b = params
        y_pred = polynomial(x, a, b)
        return np.mean(np.abs(y - y_pred))
    def mse_loss(params):
        a, b = params
        y_pred = polynomial(x, a, b)
        return np.mean((y - y_pred)**2)
    # Initial guess for the parameters
    initial_guess = [1, 1]
    # Minimize the MAE loss
    result = minimize(mae_loss, initial_guess, method='Powell')
    #result = minimize(mse_loss, initial_guess, method='Powell')
    a, b = result.x  # Extract the coefficients from the optimization result
    # Generate a smooth line for the polynomial
    x_line = np.linspace(x.min(), x.max(), 500)
    y_line = polynomial(x_line, a, b)
    # Plot the trendline with a thicker line width and brighter color
    trendline_color = COLORS[1]   
    ax1.plot(x_line, y_line, color=trendline_color, linestyle="--", alpha=0.8, linewidth=3, label='Trend')
    ax1.set_xticks([1,2,3,4,5])
    ax1.set_xticklabels(['32', '64', '128', '256', '512'])
    ax1.legend(prop={'size': 14}, ncol=1, frameon=False)

    plot_utils._annotate_and_decorate_axis(ax1, xlabel='$B$: Batch Size',
                                           ylabel='Overfitting',
                                           labelsize='xx-large', ticklabelsize='xx-large',
                                           grid_alpha=0.2, legend=False)
    
    
    yticks = np.array([0,2,4])
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticks)
    
    fig.tight_layout()
    fig.savefig(f'analysis_b_overfitting.pdf')
    fig.show()
    
graph1_1()
    
def graph2():
    sns.set_style("whitegrid")
    
    fig, ax1 = plt.subplots()
    fig.set_size_inches(496.0/192*2, 369.6/192*2)
    ax1.ticklabel_format(useOffset=False, style='plain')
    
    def polynomial(x, a, b):
        return a * x + b
        
    data = get_data(TASKS, True, [1,2,4,8])
    data['n_steps'] = np.log2(data['n_steps'])
    steps = np.log2(data['rr'])
    data['pnorms'] = data['pnorms'] / 1000
    
    ax1.scatter(steps, data['pnorms'], color=COLORS[1], s=40.0, alpha=0.8, label='Empirical value')
    # Compute linear regression (trendline)
    x = steps
    y = data['pnorms']
    
    def mae_loss(params):
        a, b = params
        y_pred = polynomial(x, a, b)
        return np.mean(np.abs(y - y_pred))
    
    def mse_loss(params):
        a, b = params
        y_pred = polynomial(x, a, b)
        return np.mean((y - y_pred)**2)
    
    # Initial guess for the parameters
    initial_guess = [1, 1]
    
    # Minimize the MAE loss
    result = minimize(mae_loss, initial_guess, method='Powell')
    #result = minimize(mse_loss, initial_guess, method='Powell')
    a, b = result.x  # Extract the coefficients from the optimization result
    
    # Generate a smooth line for the polynomial
    x_line = np.linspace(x.min(), x.max(), 500)
    y_line = polynomial(x_line, a, b)
    
    # Plot the trendline with a thicker line width and brighter color
    trendline_color = COLORS[1]   
    ax1.plot(x_line, y_line, color=trendline_color, linestyle="--", alpha=0.8, linewidth=3, label='Trend')
    
    ax1.set_xticks([0,1,2,3])
    ax1.set_xticklabels([str(rr) for rr in [1,2,4,8]], fontsize=25)
    
    yticks = np.array([0,2,4])
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticks)
    ax1.legend(prop={'size': 14}, ncol=1, frameon=False)

    plot_utils._annotate_and_decorate_axis(ax1, xlabel='$\sigma$: UTD Ratio',
                                           ylabel='Parameter Norm',
                                           labelsize='xx-large', ticklabelsize='xx-large',
                                           grid_alpha=0.2, legend=False)
    
    
    fig.tight_layout()
    fig.savefig(f'analysis_utd_pnorms.pdf')
    fig.show()
    
graph2()


def graph2_1():
    sns.set_style("whitegrid")
    
    fig, ax1 = plt.subplots()
    fig.set_size_inches(496.0/192*2, 369.6/192*2)
    ax1.ticklabel_format(useOffset=False, style='plain')
    
    def polynomial(x, a, b):
        return a * x + b
        
    data = get_data(TASKS, True, [1,2,4,8])
    data['n_steps'] = np.log2(data['n_steps'])
    steps = data['lr']
    data['pnorms'] = data['pnorms'] / 1000

    ax1.scatter(steps, data['pnorms'], color=COLORS[1], s=40.0, alpha=0.8, label='Empirical value')
    # Compute linear regression (trendline)
    x = steps
    y = data['pnorms']
    
    def mae_loss(params):
        a, b = params
        y_pred = polynomial(x, a, b)
        return np.mean(np.abs(y - y_pred))
    
    def mse_loss(params):
        a, b = params
        y_pred = polynomial(x, a, b)
        return np.mean((y - y_pred)**2)
    
    # Initial guess for the parameters
    initial_guess = [1, 1]
    
    # Minimize the MAE loss
    result = minimize(mae_loss, initial_guess, method='Powell')
    #result = minimize(mse_loss, initial_guess, method='Powell')
    a, b = result.x  # Extract the coefficients from the optimization result
    
    # Generate a smooth line for the polynomial
    x_line = np.linspace(x.min(), x.max(), 500)
    y_line = polynomial(x_line, a, b)
    
    # Plot the trendline with a thicker line width and brighter color
    trendline_color = COLORS[1]   
    ax1.plot(x_line, y_line, color=trendline_color, linestyle="--", alpha=0.8, linewidth=3, label='Trend')
    '''
    sns.boxplot(x='log_rr', y='pnorms', data=data, ax=ax1, width=0.4, boxprops=dict(facecolor=COLORS[0], linewidth=3, alpha=0.85), 
                medianprops=dict(color='black', linewidth=3), whiskerprops=dict(linewidth=2), 
                capprops=dict(linewidth=2), flierprops=dict(marker='o', markersize=0, linestyle='none', alpha=0.0))
    '''
    
    ax1.set_xticks([1,2,3,4])
    ax1.set_xticklabels(['1.5e-4', '3e-4', '6e-4', '1.2e-3'])
    
    yticks = np.array([0,2,4])
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticks)
    ax1.legend(prop={'size': 14}, ncol=1, frameon=False)
    
    plot_utils._annotate_and_decorate_axis(ax1, xlabel='$\eta$: Learning Rate',
                                           ylabel='Parameter Norm',
                                           labelsize='xx-large', ticklabelsize='xx-large',
                                           grid_alpha=0.2, legend=False)
    
    
    fig.tight_layout()
    
    fig.savefig(f'analysis_lr_pnorms.pdf')
    fig.show()
    
graph2_1()

