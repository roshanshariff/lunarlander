from __future__ import (division, print_function)

import collections
import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
import os
import re
import shutil

def smooth(data, window=100):
    return np.convolve(data, np.ones(window)/window)

def count_results(fileglob):
    counts = collections.defaultdict(int)
    for pathname in glob.iglob (fileglob):
        matches = re.findall(r"(.*)-([0-9]+).txt", os.path.basename(pathname))
        if matches:
            (name, number) = matches[0]
            counts[name] += 1
    for (name,count) in counts.items():
        print("{} {}".format(count,name))

def renumber_results(fileglob, offset):
    for pathname in glob.iglob (fileglob):
        matches = re.findall(r"(.*-)([0-9]+).txt", pathname)
        if matches:
            (name, number) = matches[0]
            shutil.move(pathname, name + str(int(number)+offset) + ".txt")

def load_results(file_name, max_failures=50):
    results = []
    failures = 0
    while failures < max_failures:
        try:
            results.append(np.loadtxt(file_name.format(len(results)+failures)))
        except IOError:
            failures += 1
    min_episodes = min(result.size for result in results)
    max_episodes = max(result.size for result in results)
    if min_episodes != max_episodes:
        print('WARNING: number of episodes varies between {} and {}'.format(min_episodes, max_episodes))
        results = [result[:min_episodes] for result in results]
    print("Loaded {} runs".format(len(results)))
    return np.vstack(results)

def load_param_study(fileglob, param_names, output_file=None):
    pattern = '-'.join(r'{}=(.*)'.format(p) for p in param_names) + '.txt'
    param_values = collections.defaultdict(list)
    param_data = {}
    for pathname in glob.iglob (fileglob):
        matches = re.findall(pattern, os.path.basename(pathname))
        if matches:
            print ('Loading file {}'.format(pathname))
            key = tuple(float(v) for v in matches[0])
            data = np.loadtxt(pathname)
            param_data[key] = data
            for (name, value) in zip(param_names, key): param_values[name].append(value)
    param_values = { name: np.unique(values) for (name, values) in param_values.items() }
    if output_file:
        with open(output_file, 'wb') as out:
            pickle.dump ({'param_names': param_names, 'param_values': param_values, 'data': param_data}, out)
    return (param_names, param_values, param_data)

def load_directory(directory, max_failures=30):
    for pathname in glob.iglob (os.path.join(directory, "*-0.txt")):
        pattern = re.sub (r'^(.*-)0(\.txt)$', r'\1{}\2', pathname)
        (basename,) = re.findall (r'^(.*)-0.txt', os.path.basename(pathname), flags=re.I)
        print ("Loading {}".format(basename))
        yield (basename, load_results(pattern, max_failures))

def aggregate_results(results):
    nresults = np.size(results, axis=0)
    results_mean = results.mean(axis=0)
    results_stderr = results.std(axis=0) / np.sqrt(float(nresults))
    cumresults_stderr = results.cumsum(axis=1).std(axis=0) / np.sqrt(float(nresults))
    return (results_mean, results_stderr, cumresults_stderr)

def summarize_directory (directory, output_directory=None):
    if output_directory is None: output_directory = directory
    for (name, results) in load_directory(directory):
        np.savetxt (os.path.join(output_directory, '{}.txt'.format(name)),
                    np.array(aggregate_results(results)).T,
                    header='{} ({} runs)'.format(name, np.size(results, axis=0)))

def plot_results(mean, stderr, label,  above=False):
    plt.fill_between(np.arange(mean.size), mean+2*stderr, mean-2*stderr,
                     color='moccasin', edgecolor='moccasin')
    plt.plot(mean, color='black', lw=2)

    xposition = mean.size - 20
    if above:
        yposition = mean[xposition] + 2.5*stderr[xposition]
        va = 'bottom'
    else:
        yposition = mean[xposition] - 2.5*stderr[xposition]
        va = 'top'
    plt.text(xposition, yposition, label, va=va, ha='right', size=10, family='serif')

def set_labels(xtext, ytext):
    plt.xlabel(xtext, size=10, family='serif', ha='right', x=1)
    plt.ylabel(ytext, size=10, rotation='horizontal', multialignment='right',
               ha='right', va='top', y=1.0, family='serif')

def set_tick_style():
    plt.xticks(size=9, family="serif")
    plt.yticks(size=9, family="serif")
