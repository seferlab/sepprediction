
import subprocess
from sklearn.preprocessing import OneHotEncoder
from Bio import SeqIO
import numpy as np

import os,re,sys
def readfasta(file):
    if not os.path.exists(file):
        print('Error: "' + file + '" does not exist.')
        sys.exit(1)

    with open(file) as f:
        records = f.read()

    if re.search('>', records) is None:
        print('The input file seems not in fasta format.')
        sys.exit(1)

    records = records.split('>')[1:]
    myFasta = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0], array[1].upper()
        myFasta.append([name, sequence])
    return myFasta

# 创建 OneHotEncoder 对象
encoder = OneHotEncoder(sparse=False)

# 构建RNAfold命令行命令
executable = "D:\\downloads\\ViennaRNA Package\\RNAfold.exe"
options = ["--noPS", "--noLP"]
command = [executable] + options

def extract_features(sequence):
    # 在命令行中运行RNAfold
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, error = process.communicate(input=sequence)

    # 提取结果
    lines = output.split('\n')
    structure = lines[1].split()[0]  # 提取二级结构
    print(structure)
    free_energy_str = lines[1].split()[1]  # 提取自由能

    # 使用正则表达式提取有效的浮点数值
    match = re.search(r'\((-?\d+(\.\d+)?)\)', free_energy_str)
    if match:
        free_energy = float(match.group(1))
    else:
        print(f"Skipping sequence: {sequence}. Invalid free energy value: {free_energy_str}")
        return None

    # 独热编码二级结构
    structure_features = list(structure)
    structure_features_array = np.array(structure_features).reshape(-1, 1)
    encoded_structure_features = encoder.fit_transform(structure_features_array)

    # 将独热编码的二级结构特征和自由能合并到特征矩阵中
    features = np.concatenate((encoded_structure_features.flatten(), [free_energy]))

    return features


def get_features(fasta_file, label):
    data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        # length = get_length(seq)
        fs = extract_features(seq)
        # print(fick)
        # print(np.array(fick).shape)

        # row = [length, fick]
        data.append(fs)
    return data


# 读取正样本fasta文件
path_p = 'p.fa'
p_data = get_features(path_p, 1)

# 读取负样本fasta文件
path_n = 'n.fa'
n_data = get_features(path_n, 0)

print(np.array(p_data).shape)

print(np.array(n_data).shape)
