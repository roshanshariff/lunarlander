import numpy as np
import itertools

def aggregate_results(results):
    results_sum = results.cumsum(axis=1)
    results_mean = results_sum.mean(axis=0)
    results_stderr = results_sum.std(axis=0) / np.sqrt(float(results.shape[0]))
    return (results_mean, results_stderr)

def plot_results(results, label,  above=False):
    (mean, stderr) = aggregate_results(results)
    fill_between(np.arange(results.shape[1]), mean+2*stderr, mean-2*stderr,
                 color='moccasin', edgecolor='moccasin')
    plot(mean, color='black', lw=2)
    
    xposition = results.shape[1] - 20
    if above:
        yposition = mean[xposition] + 2.5*stderr[xposition]
        va = 'bottom'
    else:
        yposition = mean[xposition] - 2.5*stderr[xposition]
        va = 'top'
    text(xposition, yposition, label, va=va, ha='right', size=10, family='serif')
    
def set_labels(xtext, ytext):
    xlabel(xtext, size=10, family='serif', ha='right', x=1)
    ylabel(ytext, size=10, rotation='horizontal', multialignment='right',
           ha='right', va='top', y=1.0, family='serif')

def set_tick_style():
    xticks(size=9, family="serif")
    yticks(size=9, family="serif")


