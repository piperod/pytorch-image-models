import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

sizes = [160, 192, 227, 270, 321, 382, 454]


def msg_wrapper(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return lines
        print(lines)
        

def acc_wrapper(file_path):
    lines = msg_wrapper(file_path)
    lines = [line for line in lines if line.startswith("Test: [ 195/195]")]
    accs = []
    for line in lines:
        match = re.search(r'Acc@1:\s+\d+\.\d+\s+\(\s*(\d+\.\d+)\)', line)
        if match:
            accs.append(float(match.group(1)))
    return accs

def loss_wrapper(file_path):
    lines = msg_wrapper(file_path)
    losses = []
    for line in lines:
        match = re.search(r'batch\s+\d+\s+loss:\s+(\d+\.\d+)', line)
        if match:
            losses.append(float(match.group(1)))
    return losses

def csv_wrapper(file_path):
    df = pd.read_csv(file_path)
    accs = df['eval_top1'].tolist()
    losses = df['train_loss'].tolist()
    return accs, losses
    
def draw_result_curve(acc=True, csv=True):
    dic = {}
    for size in sizes:
        if csv:
            accs, losses = csv_wrapper(f"alex_size_{size}/summary.csv")
            if acc:
                dic[size] = accs
            else:
                dic[size] = losses
        else:
            if acc:
                dic[size] = acc_wrapper(f"alex_size_{size}.err")
            else:
                dic[size] = loss_wrapper(f"alex_size_{size}.out")
        
    max_len = max([len(x) for x in dic.values()])

    plt.figure(figsize=(10, 6))
    for size in sizes:
        temp = dic[size]
        temp += [None] * (max_len - len(temp))
        plt.plot(range(1, max_len+1), temp, label=str(size))
        
    if acc:
        plt.xlabel('Epoch')
        plt.ylabel('Top1 Acc')
        plt.title('Acc of AlexNet w/ Different Input Sizes')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig('alexnet_acc.png')
    else:
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss of AlexNet w/ Different Input Sizes')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig('alexnet_loss.png')
    plt.close()

draw_result_curve(acc=True)
draw_result_curve(acc=False)

