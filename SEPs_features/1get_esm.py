from Bio import SeqIO
import torch
from multiprocessing import cpu_count
torch.set_num_threads(cpu_count())
import numpy as np
import gc
import esm
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
# model = model.cuda()
# model = model

def infer(seq):
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("tmp", seq),
    ]
    # print(data)
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # batch_tokens = batch_tokens.cuda()
    # batch_tokens = batch_tokens

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    token_representations = token_representations.detach().cpu().numpy()
    # print(token_representations.shape)
    token_representations = token_representations[0][1:-1,:]
    # (7, 1280)
    return token_representations

def read_fa(path):
    seqs = []
    lines = SeqIO.parse(path,format="fasta")
    for x in list(lines):
        id = str(x.id)
        seq = str(x.seq)
        seqs.append(seq)
    return seqs


path = "./Pep_NEW_N_Rat.txt"
max_len = 25
seqs = read_fa(path)
ress = []
for i,seq in enumerate(seqs):
    seq = seq[0:max_len]
    print(seq)
    tmp = infer(seq)
    print(tmp.shape)
    res = np.zeros((max_len,tmp.shape[-1]))
    for i,x in enumerate(tmp):
        res[i,:]=x
    ress.append(res)

ress = np.array(ress)
print("ress",ress.shape)
print(len(ress), ress.shape)

np.save("esm_Pep_NEW_N_Rat.npy", ress)

xx = np.load("esm_Pep_NEW_N_Rat.npy", allow_pickle=True)
print(xx)
print(xx.shape)


# path = "./hump_neg12616.txt"
# max_len = 30
# seqs = read_fa(path)
# ress = []
# for i,seq in enumerate(seqs):
#     seq = seq[0:max_len]
#     print(seq)
#     tmp = infer(seq)
#     print(tmp.shape)
#     res = np.zeros((max_len,tmp.shape[-1]))
#     for i,x in enumerate(tmp):
#         res[i,:]=x
#     ress.append(res)
#
# ress = np.array(ress)
# print("ress",ress.shape)
# print(len(ress), ress.shape)
# np.save("./esm_1b/esm_{path}.npy".format(path=path.split("/")[-1].split(".")[0]), ress)
#
#
# path = "/home/ys/work/hemolytic/data/dataset/HemoPI-1/HemoPI-1_val_neg.fa"
# max_len = 50
# seqs = read_fa(path)
# ress = []
# for i,seq in enumerate(seqs):
#     seq = seq[0:max_len]
#     print(seq)
#     tmp = infer(seq)
#     print(tmp.shape)
#     res = np.zeros((max_len,tmp.shape[-1]))
#     for i,x in enumerate(tmp):
#         res[i,:]=x
#     ress.append(res)
#
# ress = np.array(ress)
# print("ress",ress.shape)
# print(len(ress), ress.shape)
# np.save("./esm_{path}.npy".format(path=path.split("/")[-1].split(".")[0]), ress)
#
# path = "/home/ys/work/hemolytic/data/dataset/HemoPI-1/HemoPI-1_val_pos.fa"
# max_len = 50
# seqs = read_fa(path)
# ress = []
# for i,seq in enumerate(seqs):
#     seq = seq[0:max_len]
#     print(seq)
#     tmp = infer(seq)
#     print(tmp.shape)
#     res = np.zeros((max_len,tmp.shape[-1]))
#     for i,x in enumerate(tmp):
#         res[i,:]=x
#     ress.append(res)
#
# ress = np.array(ress)
# print("ress",ress.shape)
# print(len(ress), ress.shape)
# np.save("./esm_{path}.npy".format(path=path.split("/")[-1].split(".")[0]), ress)



exit()
