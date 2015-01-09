import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict


def _parse(file_path, pattern_iter, pattern_loss, pattern_accuracy):
    with open(file_path, 'r') as f:
        log = f.read()
    iter_list = map(int, pattern_iter.findall(log))
    loss_list = pattern_loss.findall(log)
    loss_kv = defaultdict(list)
    for k, v in loss_list:
        loss_kv[k].append(float(v))
    accuracy_list = pattern_accuracy.findall(log)
    accuracy_kv = defaultdict(list)
    for k, v in accuracy_list:
        accuracy_kv[k].append(float(v))
    for k in loss_kv:
        assert len(loss_kv[k]) == len(iter_list)
    for k in accuracy_kv:
        assert len(accuracy_kv[k]) == len(iter_list)
    return (iter_list, loss_kv, accuracy_kv)


def parse_train(file_path):
    pattern_iter = re.compile(r'Iteration (\d+), lr = ')
    pattern_loss = re.compile(r'Train net output #\d+: (loss.*?) = (.+?) ')
    pattern_accuracy = re.compile(
        r'Train net output #\d+: (accuracy.*?) = (\d\.\d+)')
    return _parse(file_path, pattern_iter, pattern_loss, pattern_accuracy)


def parse_test(file_path):
    pattern_iter = re.compile(r'Iteration (\d+), Testing net')
    pattern_loss = re.compile(r'Test net output #\d+: (loss.*?) = (.+?) ')
    pattern_accuracy = re.compile(
        r'Test net output #\d+: (accuracy.*?) = (\d\.\d+)')
    return _parse(file_path, pattern_iter, pattern_loss, pattern_accuracy)


def config_mpl():
    mpl.rc('lines', linewidth=1.5)
    mpl.rc('font', family='Times New Roman', size=16, monospace='Courier New')
    mpl.rc('legend', fontsize='small', fancybox=False,
           labelspacing=0.1, borderpad=0.1, borderaxespad=0.2)
    mpl.rc('figure', figsize=(12, 10))
    mpl.rc('savefig', dpi=120)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('log', help="Caffe training log")
    parser.add_argument('-o', '--output', help="Save as image if necessary")
    parser.add_argument('--loss', nargs='*',
        help="Name of training loss to be shown. To show loss_xxx, just provide xxx.")
    parser.add_argument('--acc', nargs='*',
        help="Name of training accuracy to be shown. To show accuracy_xxx, just provide xxx.")
    args = parser.parse_args()

    train_iter_list, train_loss_kv, train_accuracy_kv = parse_train(args.log)
    test_iter_list, test_loss_kv, test_accuracy_kv = parse_test(args.log)
    train_loss_filter, train_acc_filter = None, None
    if args.loss is not None:
        train_loss_filter = ['loss_' + name for name in args.loss]
    if args.acc is not None:
        train_acc_filter = ['accuracy_' + name for name in args.acc]

    config_mpl()
    legend_font = mpl.font_manager.FontProperties(family='monospace')
    color_list = ['#FF4136', '#0074D9', '#FFDC00',
                  '#B10DC9', '#7FDBFF', '#2ECC40', '#111111']

    fig = plt.figure()
    title = os.path.splitext(os.path.basename(args.log))[0]
    title = title.replace('_', ' ').title()
    plt.suptitle(title, fontsize=24, fontweight='bold')
    # plot loss
    ax = fig.add_subplot(211)
    legend_list = []
    for title, loss_list in test_loss_kv.iteritems():
        ax.plot(test_iter_list, loss_list, color=color_list[len(legend_list)])
        legend_list.append('test' + title[len('loss'):])
    for title, loss_list in train_loss_kv.iteritems():
        if train_loss_filter is not None and title not in train_loss_filter:
            continue
        ax.plot(train_iter_list, loss_list, color=color_list[len(legend_list)])
        legend_list.append('train' + title[len('loss'):])
    ax.set_xlabel('Iterations', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.legend(legend_list, loc='upper right', prop=legend_font)
    # plot accuracy
    ax = fig.add_subplot(212)
    legend_list = []
    for title, accuracy_list in test_accuracy_kv.iteritems():
        plt.plot(test_iter_list, accuracy_list,
                 color=color_list[len(legend_list)])
        legend_list.append('test' + title[len('accuracy'):])
    for title, accuracy_list in train_accuracy_kv.iteritems():
        if train_acc_filter is not None and title not in train_acc_filter:
            continue
        ax.plot(train_iter_list, accuracy_list,
                color=color_list[len(legend_list)])
        legend_list.append('train' + title[len('accuracy'):])
    ax.set_xlabel('Iterations', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.legend(legend_list, loc='lower right', prop=legend_font)

    if args.output is not None:
        fig.savefig(args.output)
    plt.show()
