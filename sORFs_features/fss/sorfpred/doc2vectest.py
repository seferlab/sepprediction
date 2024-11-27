from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from Bio import SeqIO
import numpy as np

def read_data(path, seq_len):
    docs = []
    rx = SeqIO.parse(path, format="fasta")
    for i, x in enumerate(rx):
        seq = str(x.seq)
        for j in range(0, len(seq), seq_len):
            # 将DNA序列按照seq_len长度进行分段，每段长度为seq_len
            sub_seq = seq[j:j+seq_len]
            docs.append(TaggedDocument(words=list(sub_seq), tags=[i]))
    return docs


print("load train")
train_path1 = "horf_pos12616.txt"
train_path2 = "horf_neg12616.txt"


paths = [train_path1,train_path2]

# 将所有训练和测试集中的DNA序列按照seq_len长度进行分段
seq_len = 40
docs = []
for path in paths:
    doc = read_data(path, seq_len)
    docs.extend(doc)

print("begin train")
model = Doc2Vec(documents=docs, vector_size=100, window=10, min_count=2, workers=40, epochs=128, dm=1)
model.save("d2v2.model")


# Load the model
# model = Doc2Vec.load("d2v.model")

# Read the new document
# new_doc_path = "path/to/new/document.fasta"
# new_doc = read_data(new_doc_path)[0]  # Assume there's only one document

# Generate the vector representation of the new document
# new_vector = model.infer_vector(new_doc.words)
