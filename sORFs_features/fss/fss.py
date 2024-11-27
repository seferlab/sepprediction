'''


repRNA
'''

import warnings
warnings.filterwarnings("ignore")

'''
biopython
'''
import pandas as pd
from Bio import SeqIO
import numpy as np
from Bio.Seq import Seq

def read_data(path):
    pseq = []
    nseq = []
    rx  = SeqIO.parse(path,format="fasta")
    for x in list(rx):
        id = str(x.id)
        seq = str(x.seq)
        if "Positive" in id:
            pseq.append(seq)
        if "Negative" in id:
            nseq.append(seq)
    return pseq,nseq

def binary(seq):
    '''
    HUM:
    recall 0.7063333333333334
precise 0.8020439061317184
se 0.7063333333333334
sp 0.8256666666666667
acc 0.766
f1 0.751152073732719
mcc 0.5358289010392546
auc 0.8370923333333334
ap 0.8563034376072827

    MOU:
    recall 0.6915384615384615
precise 0.7523012552301255
se 0.6915384615384615
sp 0.7723076923076924
acc 0.7319230769230769
f1 0.7206412825651302
mcc 0.46536658630534333
auc 0.795112426035503
ap 0.8081969659347983

    RAT:
recall 0.63
precise 0.6528497409326425
se 0.63
sp 0.665
acc 0.6475
f1 0.6412213740458015
mcc 0.2951808536762876
auc 0.69720625
ap 0.6724808157566561


    '''
    std = {"A": np.array([1, 0, 0, 0]),
           "T": np.array([0, 1, 0, 0]),
           "C": np.array([0, 0, 1, 0]),
           "G": np.array([0, 0, 0, 1])
           }
    A = seq.count("A")
    T = seq.count("T")
    C = seq.count("C")
    G = seq.count("G")
    res = A*std["A"] + T*std["T"] + C*std["C"] + G*std["G"]
    res = res/len(seq)
    return res.tolist()

def EIIP(seq):
    '''
    HUM:
    recall 0.7063333333333334
precise 0.8020439061317184
se 0.7063333333333334
sp 0.8256666666666667
acc 0.766
f1 0.751152073732719
mcc 0.5358289010392546
auc 0.8370923333333334
ap 0.8563034376072827

    MOU:
    recall 0.6915384615384615
precise 0.7523012552301255
se 0.6915384615384615
sp 0.7723076923076924
acc 0.7319230769230769
f1 0.7206412825651302
mcc 0.46536658630534333
auc 0.795112426035503
ap 0.8081969659347983

    RAT:
recall 0.63
precise 0.6528497409326425
se 0.63
sp 0.665
acc 0.6475
f1 0.6412213740458015
mcc 0.2951808536762876
auc 0.69720625
ap 0.6724808157566561

    '''

    std = {"A": np.array([0.12601,0,0,0]),
           "T": np.array([0,0.13400,0,0]),
           "C": np.array([0,0,0.08060,0]),
           "G": np.array([0,0,0,0.13350])
           }
    A = seq.count("A")
    T = seq.count("T")
    C = seq.count("C")
    G = seq.count("G")
    res = A*std["A"] + T*std["T"] + C*std["C"] + G*std["G"]
    res = res/len(seq)
    return res.tolist()

def entroy(seq):
    '''
    HUM:
    recall 0.6966666666666667
precise 0.7892749244712991
se 0.6966666666666667
sp 0.814
acc 0.7553333333333333
f1 0.740084985835694
mcc 0.5142185861573462
auc 0.8298262222222221
ap 0.8498622399378348

    MOU:
    recall 0.6861538461538461
precise 0.742714404662781
se 0.6861538461538461
sp 0.7623076923076924
acc 0.7242307692307692
f1 0.713314674130348
mcc 0.4497676279050852
auc 0.7906023668639054
ap 0.8039775387134166

    RAT:
recall 0.6075
precise 0.6497326203208557
se 0.6075
sp 0.6725
acc 0.64
f1 0.627906976744186
mcc 0.2805933809392673
auc 0.6886000000000001
ap 0.6723563764998344

    '''
    a = seq.count("A") / len(seq)
    t = seq.count("T") / len(seq)
    c = seq.count("C") / len(seq)
    g = seq.count("G") / len(seq)

    a = -a * np.log(a)
    t = -t * np.log(t)
    c = -c * np.log(c)
    g = -g * np.log(g)
    return [a,t,c,g]

'''
sum 
'''
def SNCP(seq):
    '''NCP, Nucleotide chemical property
    HUM:
    recall 0.699
precise 0.7792642140468228
se 0.699
sp 0.802
acc 0.7505
f1 0.7369530838165524
mcc 0.5036788886926888
auc 0.8278062222222222
ap 0.8483951546733512

    MOU:
    recall 0.67
precise 0.7192402972749794
se 0.67
sp 0.7384615384615385
acc 0.7042307692307692
f1 0.6937475109518121
mcc 0.4094221425241643
auc 0.7829295857988166
ap 0.792708410028353

    RAT:
recall 0.6575
precise 0.6461916461916462
se 0.6575
sp 0.64
acc 0.64875
f1 0.6517967781908302
mcc 0.2975455651535133
auc 0.6921843750000001
ap 0.6632905998448542

    '''
    std = {"A": np.array([1, 1, 1]),
           "T": np.array([0, 0, 1]),
           "C": np.array([0, 1, 0]),
           "G": np.array([1, 0, 0]),
           }
    A = seq.count("A")
    T = seq.count("T")
    C = seq.count("C")
    G = seq.count("G")
    res = A*std["A"] + T*std["T"] + C*std["C"] + G*std["G"]
    res = res/len(seq)
    return res.tolist()

'''
sum
'''
def SPCP(seq):
    '''
    HUM:
    recall 0.6603333333333333
precise 0.7151624548736462
se 0.6603333333333333
sp 0.737
acc 0.6986666666666667
f1 0.6866551126516464
mcc 0.39850622488884885
auc 0.7681037777777778
ap 0.77901510953442

    MOU:
    recall 0.6207692307692307
precise 0.6787216148023549
se 0.6207692307692307
sp 0.7061538461538461
acc 0.6634615384615384
f1 0.648453194053837
mcc 0.32812135464628417
auc 0.7255476331360947
ap 0.7017224881262398

    RAT:
recall 0.6525
precise 0.6525
se 0.6525
sp 0.6525
acc 0.6525
f1 0.6525
mcc 0.305
auc 0.69565
ap 0.6728133352135643



    '''
    std = {
        "A": np.array([37.03, 83.8, 279.9, 122.7, 14.68]),
        "T": np.array([29.71, 102.7, 251.3, 35.7, 11.77]),
        "C": np.array([27.30, 71.5, 206.3, 69.2, 10.82 ]),
        "G": np.array([35.46, 68.8, 229.6, 124.0, 14.06])
    }
    A = seq.count("A")
    T = seq.count("T")
    C = seq.count("C")
    G = seq.count("G")
    res = A*std["A"] + T*std["T"] + C*std["C"] + G*std["G"]
    res = res/len(seq)
    return res.tolist()

def z_curve(seq):
    '''
        x = (A+G)-(C+T)
        y = (A+C)-(G+T)
        z = (A+T)-(C+G)
    HUM:
    recall 0.699
precise 0.7792642140468228
se 0.699
sp 0.802
acc 0.7505
f1 0.7369530838165524
mcc 0.5036788886926888
auc 0.8278062222222222
ap 0.8483951546733512

    MOU:
    recall 0.67
precise 0.7192402972749794
se 0.67
sp 0.7384615384615385
acc 0.7042307692307692
f1 0.6937475109518121
mcc 0.4094221425241643
auc 0.7829295857988166
ap 0.792708410028353

    RAT:
recall 0.6575
precise 0.6461916461916462
se 0.6575
sp 0.64
acc 0.64875
f1 0.6517967781908302
mcc 0.2975455651535133
auc 0.6921843750000001
ap 0.6632905998448542


    '''
    A = seq.count("A")/len(seq)
    T = seq.count("T")/len(seq)
    C = seq.count("C")/len(seq)
    G = seq.count("G")/len(seq)

    x = (A + G) - (C + T)
    y = (A + C) - (G + T)
    z = (A + T) - (C + G)
    return [x,y,z]


def cumulativeSkew(seq):
    '''
        x = (G-C)/((G+C)+0.1)
        y = (A-T)/((A+T)+0.1)
    HUM:
    recall 0.5966666666666667
precise 0.7920353982300885
se 0.5966666666666667
sp 0.8433333333333334
acc 0.72
f1 0.6806083650190115
mcc 0.4540293317009027
auc 0.7655314999999999
ap 0.7988452493918131

    MOU:
    recall 0.5807692307692308
precise 0.7122641509433962
se 0.5807692307692308
sp 0.7653846153846153
acc 0.6730769230769231
f1 0.6398305084745762
mcc 0.35220800363430516
auc 0.7223215976331361
ap 0.7444507773562292

    RAT:
recall 0.5975
precise 0.5623529411764706
se 0.5975
sp 0.535
acc 0.56625
f1 0.5793939393939395
mcc 0.1327595497100517
auc 0.60369375
ap 0.5736299086624186

    '''
    A = seq.count("A")/len(seq)
    T = seq.count("T")/len(seq)
    C = seq.count("C")/len(seq)
    G = seq.count("G")/len(seq)

    x = (G - C) / (G + C)
    y = (A - T) / (A + T)
    return [x,y]

def gc(seq):
    '''
    HUM:
    recall 0.6563333333333333
precise 0.7736738703339883
se 0.6563333333333333
sp 0.808
acc 0.7321666666666666
f1 0.7101893597835889
mcc 0.46976774837114305
auc 0.8011357777777777
ap 0.8147875495713842

    MOU:
    recall 0.68
precise 0.6553002223869533
se 0.68
sp 0.6423076923076924
acc 0.6611538461538462
f1 0.6674216685541713
mcc 0.32253688944393905
auc 0.7193825443786982
ap 0.7016385602796467

    RAT:
recall 0.675
precise 0.6338028169014085
se 0.675
sp 0.61
acc 0.6425
f1 0.6537530266343826
mcc 0.2856039770274685
auc 0.685471875
ap 0.655451128441752


    '''


    G = seq.count("G")
    C = seq.count("C")
    res = (G+C)/len(seq)
    return [res]

def  get_GC2(mRNA):
    '''
    HUM:
    recall 0.6453333333333333
precise 0.7771979124849459
se 0.6453333333333333
sp 0.815
acc 0.7301666666666666
f1 0.7051538881806592
mcc 0.46710566103438284
auc 0.803607611111111
ap 0.8219576287257142

    MOU:
    recall 0.6846153846153846
precise 0.7131410256410257
se 0.6846153846153846
sp 0.7246153846153847
acc 0.7046153846153846
f1 0.6985871271585556
mcc 0.4095585472322421
auc 0.7723147928994082
ap 0.7866349146359596

    RAT:
recall 0.6725
precise 0.6481927710843374
se 0.6725
sp 0.635
acc 0.65375
f1 0.660122699386503
mcc 0.3077164392400309
auc 0.7110875000000001
ap 0.6697046699038475


    :param mRNA:
    :return:
    '''


    if len(mRNA) < 3:
        numGC = 0
        mRNA = 'ATG'
    else:
        numGC = mRNA[1::3].count("C") + mRNA[1::3].count("G")
    res = numGC * 1.0 / len(mRNA) * 3
    return [res]

def get_GC3(mRNA):
    '''
    HUM:
    recall 0.6483333333333333
precise 0.7472147522089896
se 0.6483333333333333
sp 0.7806666666666666
acc 0.7145
f1 0.6942709262894878
mcc 0.4328064152118756
auc 0.7775322222222221
ap 0.7966680371860582

    MOU:
    recall 0.5961538461538461
precise 0.7129714811407544
se 0.5961538461538461
sp 0.76
acc 0.678076923076923
f1 0.6493506493506493
mcc 0.3610328793125426
auc 0.7224266272189348
ap 0.7265719760370856

    RAT:
recall 0.605
precise 0.587378640776699
se 0.605
sp 0.575
acc 0.59
f1 0.5960591133004925
mcc 0.18008105471603858
auc 0.6318531250000001
ap 0.6521537733073381


    :param mRNA:
    :return:
    '''


    if len(mRNA) < 3:
        numGC = 0
        mRNA = 'ATG'
    else:
        numGC = mRNA[2::3].count("C") + mRNA[2::3].count("G")
    res = numGC * 1.0 / len(mRNA) * 3
    return [res]





import sys
sys.path.append("./repDNA/")
# from psenac import PseKNC, PseDNC
# def psednc(seq):
#     '''recall 0.6988906497622821
#     precise 0.7788079470198676
#     sp 0.8016627078384798
#     acc 0.7502970297029703
#     f1 0.7366882438922531
#     mcc 0.5032282627329334
#     auc 0.8262957476789723
#     ap 0.8420944286683038'''
#     res = PseDNC(lamada=3).make_psednc_vec([seq])[0]
#     return res
#
# def pseknc(seq):
#     '''recall 0.7242472266244057
#     precise 0.8024582967515365
#     sp 0.8218527315914489
#     acc 0.7730693069306931
#     f1 0.7613494377342774
#     mcc 0.5487316607424979
#     auc 0.8420432258865955
#     ap 0.854172048796229'''
#     res = PseKNC(k=3).make_pseknc_vec([seq])[0]
#     return res


def mer2(seq):
    '''
    HUM:
    recall 0.7226666666666667
precise 0.8259047619047619
se 0.7226666666666667
sp 0.8476666666666667
acc 0.7851666666666667
f1 0.7708444444444446
mcc 0.574841967394478
auc 0.8576774444444444
ap 0.8792317164677897

    MOU:
recall 0.6915384615384615
precise 0.7783549783549784
se 0.6915384615384615
sp 0.803076923076923
acc 0.7473076923076923
f1 0.7323828920570264
mcc 0.49772110571662526
auc 0.8230011834319526
ap 0.8431707277649103

    RAT:
recall 0.6925
precise 0.7194805194805195
se 0.6925
sp 0.73
acc 0.71125
f1 0.7057324840764332
mcc 0.4227973839964652
auc 0.7692625
ap 0.7783652355513928

    :param seq:
    :return:
    '''
    mers = []
    for x in "ATCG":
        for y in "ATCG":
            tmp = x+y
            mers.append(tmp)

    res = []
    for mer in mers:
        c = seq.count(mer)/len(seq)
        res.append(c)
    return res


def mer3(seq):
    '''
    HUM:
    recall 0.731
precise 0.8586530931871574
se 0.731
sp 0.8796666666666667
acc 0.8053333333333333
f1 0.7897011163125676
mcc 0.6175290410125113
auc 0.8761355555555557
ap 0.8993352197585296

    MOU:
recall 0.69
precise 0.8367537313432836
se 0.69
sp 0.8653846153846154
acc 0.7776923076923077
f1 0.7563237774030355
mcc 0.564128612695818
auc 0.8446621301775148
ap 0.8721300460174524

    RAT:
recall 0.66
precise 0.7521367521367521
se 0.66
sp 0.7825
acc 0.72125
f1 0.7030625832223701
mcc 0.44585797337226346
auc 0.79010625
ap 0.8179520311270242


    :param seq:
    :return:
    '''
    mers = []
    for x in "ATCGN":
        for y in "ATCGN":
            for z in "ATCGN":
                tmp = x+y+z
                mers.append(tmp)

    res = []
    for mer in mers:
        c = seq.count(mer)/len(seq)
        res.append(c)
    return res

def mer4(seq):
    '''
    HUM:
    recall 0.7283333333333334
precise 0.8615930599369085
se 0.7283333333333334
sp 0.883
acc 0.8056666666666666
f1 0.789378612716763
mcc 0.6187792832197673
auc 0.879751
ap 0.9027357250781467

    MOU:
recall 0.6938461538461539
precise 0.8390697674418605
se 0.6938461538461539
sp 0.8669230769230769
acc 0.7803846153846153
f1 0.759578947368421
mcc 0.5693618636311442
auc 0.8469923076923077
ap 0.8747849124824654

    RAT:
recall 0.6875
precise 0.7575757575757576
se 0.6875
sp 0.78
acc 0.73375
f1 0.7208387942332897
mcc 0.4695129501661913
auc 0.808175
ap 0.8368272713521967


    '''
    mers = []
    for x in "ATCGN":
        for y in "ATCGN":
            for z in "ATCGN":
                for k in "ATCGN":
                    tmp = x+y+z+k
                    mers.append(tmp)

    res = []
    for mer in mers:
        c = seq.count(mer)/len(seq)
        res.append(c)
    return res


# def mers1(seq):
#     '''recall 0.694136291600634
#         precise 0.7735099337748345
#         sp 0.7969121140142518
#         acc 0.7455445544554455
#         f1 0.7316767592399249
#         mcc 0.49367251727127287
#         auc 0.8201300766795534
#         ap 0.830797246553591'''
#     mers = []
#     for x in "ATCG":
#         for y in "ATCG":
#             tmps = []
#             for z in "ATCG":
#                 tmp = x+z+y
#                 tmps.append(tmp)
#             sum = 0
#             for mer in tmps:
#                 sum = sum +seq.count(mer)
#             sum = sum/len(seq)
#             mers.append(sum)
#     return mers
#
# def mers2(seq):
#     '''recall 0.7020602218700476
#     precise 0.7737991266375546
#     sp 0.7949326999208234
#     acc 0.7485148514851485
#     f1 0.736186123805567
#     mcc 0.4991594180507166
#     auc 0.8202401835490927
#     ap 0.8331186952508668'''
#     mers = []
#     for x in "ATCG":
#         for y in "ATCG":
#             tmps = []
#             for z in "ATCG":
#                 for k in "ATCG":
#                     tmp = x+z+k+y
#                     tmps.append(tmp)
#             sum = 0
#             for mer in tmps:
#                 sum = sum +seq.count(mer)
#             sum = sum/len(seq)
#             mers.append(sum)
#     return mers
#
# def mers3(seq):
#     '''recall 0.696513470681458
#     precise 0.7758164165931156
#     sp 0.7988915281076802
#     acc 0.7477227722772277
#     f1 0.7340292275574112
#     mcc 0.49803190397479213
#     auc 0.822509608471265
#     ap 0.8343884298982014'''
#     mers = []
#     for x in "ATCG":
#         for y in "ATCG":
#             tmps = []
#             for z in "ATCG":
#                 for k in "ATCG":
#                     for l in "ATCG":
#                         tmp = x + z+k+l +y
#                         tmps.append(tmp)
#             sum = 0
#             for mer in tmps:
#                 sum = sum +seq.count(mer)
#             sum = sum/len(seq)
#             mers.append(sum)
#     return mers
#
# def mers4(seq):
#     '''recall 0.7060221870047544
#     precise 0.7710947641713544
#     sp 0.7905779889152811
#     acc 0.7483168316831683
#     f1 0.7371251292657705
#     mcc 0.4983933381145191
#     auc 0.822082356174078
#     ap 0.8370525771808196'''
#     mers = []
#     for x in "ATCG":
#         for y in "ATCG":
#             tmps = []
#             for z in "ATCG":
#                 for k in "ATCG":
#                     for l in "ATCG":
#                         for m in "ATCG":
#                             tmp = x+z+k+l+m+y
#                             tmps.append(tmp)
#             sum = 0
#             for mer in tmps:
#                 sum = sum +seq.count(mer)
#             sum = sum/len(seq)
#             mers.append(sum)
#     return mers
#
#
# def mmers1(seq):
#     mers = []
#     for x in "ATCG":
#         for y in "ATCG":
#             for p in "ATCG":
#                 for q in "ATCG":
#                     tmps = []
#                     for z in "ATCG":
#                         tmp = x+y+z+p+q
#                         tmps.append(tmp)
#                     sum = 0
#                     for mer in tmps:
#                         sum = sum +seq.count(mer)
#                     sum = sum/len(seq)
#                     mers.append(sum)
#     return mers
#
# def mmers2(seq):
#     mers = []
#     for x in "ATCG":
#         for y in "ATCG":
#             for p in "ATCG":
#                 for q in "ATCG":
#                     tmps = []
#                     for z in "ATCG":
#                         for k in "ATCG":
#                             tmp = x+y+z+k+p+q
#                             tmps.append(tmp)
#                     sum = 0
#                     for mer in tmps:
#                         sum = sum +seq.count(mer)
#                     sum = sum/len(seq)
#                     mers.append(sum)
#     return mers
#
# def mmers3(seq):
#     mers = []
#     for x in "ATCG":
#         for y in "ATCG":
#             for p in "ATCG":
#                 for q in "ATCG":
#                     tmps = []
#                     for z in "ATCG":
#                         for k in "ATCG":
#                             for l in "ATCG":
#                                 tmp = x+y+z+k+l+p+q
#                                 tmps.append(tmp)
#                     sum = 0
#                     for mer in tmps:
#                         sum = sum +seq.count(mer)
#                     sum = sum/len(seq)
#                     mers.append(sum)
#     return mers
#
# def mmers4(seq):
#     mers = []
#     for x in "ATCG":
#         for y in "ATCG":
#             for p in "ATCG":
#                 for q in "ATCG":
#                     tmps = []
#                     for z in "ATCG":
#                         for k in "ATCG":
#                             for l in "ATCG":
#                                 for m in "ATCG":
#                                     tmp = x+y+z+k+l+m+p+q
#                                     tmps.append(tmp)
#                     sum = 0
#                     for mer in tmps:
#                         sum = sum +seq.count(mer)
#                     sum = sum/len(seq)
#                     mers.append(sum)
#     return mers




def get_fs(seq):
    f1 = binary(seq)
    f2 = EIIP(seq)
    f3 = entroy(seq)
    f4 = SNCP(seq)
    f5 = SPCP(seq)
    f6 = z_curve(seq)
    f7 = cumulativeSkew(seq)
    f8 = gc(seq)
    fg2 = get_GC2(seq)
    fg3 = get_GC3(seq)
    # f9 = psednc(seq)
    # f_9 = pseknc(seq)
    #
    fm2 = mer2(seq)
    fm3 = mer3(seq)
    fm4 = mer4(seq)

    # f11 = mers1(seq)
    # f12 = mers2(seq)
    # f13 = mers3(seq)
    # f14 = mers4(seq)
    #
    # f15 = mer4(seq)
    # f16 = mmers1(seq)
    # f17 = mmers2(seq)
    # f18 = mmers3(seq)
    # f19 = mmers4(seq)
    # res = f9 + f_9 + fg2 + fg3
    res =fm3 + fm4
    # res = f2
    return res



# 读取fasta文件并提取特征
def get_features(fasta_file, label):
    data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        # length = get_length(seq)
        fs = get_fs(seq)
        # print(fick)
        # print(np.array(fick).shape)

        # row = [length, fick]
        data.append(fs)
    return data


# # 读取正样本fasta文件
# path_p = 'p.fa'
# p_data = get_features(path_p, 1)
#
# # 读取负样本fasta文件
# path_n = 'n.fa'
# n_data = get_features(path_n, 0)
#
# print(np.array(p_data).shape)
#
# print(np.array(n_data).shape)
#
def get_data10(pos, neg):
    from sklearn.model_selection import train_test_split

    train_p_data =get_features(path_p_train,1)
    # print(np.array(train_p_data).shape)
    train_n_data = get_features(path_n_train,0)
    print(np.array(train_n_data).shape)
    # exit()

    # 合并所有数据的特征
    all_data = np.concatenate([train_p_data, train_n_data], axis=0)

    # # 保存总特征为CSV文件
    # all_features = pd.DataFrame(all_data)
    # all_features.to_csv("machine_ORF_RAT_features.csv", index=False, header=False)
    # print("csv文件已生成")

    train_p_data, test_p_data = train_test_split(train_p_data, test_size=0.4, random_state=3)
    val_p_data, test_p_data = train_test_split(test_p_data, test_size=0.5, random_state=3)
    train_n_data, test_n_data = train_test_split(train_n_data, test_size=0.4, random_state=3)
    val_n_data, test_n_data = train_test_split(test_n_data, test_size=0.5, random_state=3)
    train_p_data = np.array(train_p_data)
    train_n_data = np.array(train_n_data)
    val_p_data = np.array(val_p_data)
    val_n_data = np.array(val_n_data)
    train_data = np.concatenate([train_p_data, train_n_data], axis=0)
    val_data = np.concatenate([val_p_data, val_n_data], axis=0)
    test_data = np.concatenate([test_p_data, test_n_data], axis=0)
    train_label = [1] * len(train_p_data) + [0] * len(train_n_data)
    val_label = [1] * len(val_p_data) + [0] * len(val_n_data)
    test_label = [1] * len(test_p_data) + [0] * len(test_n_data)
    train_label = np.array(train_label)
    val_label = np.array(val_label)
    test_label = np.array(test_label)
    return train_data, val_data, test_data, train_label, val_label, test_label
# fastas1 = readFasta.readFasta('train_hum+.txt')
# fastas = readFasta.readFasta('train_hum-.txt')
path_p_train = "Orf_NEW_P_Rat.fasta"
path_n_train = "Orf_NEW_N_Rat.fasta"
train_data, val_data, test_data, train_label, val_label, test_label = get_data10(path_p_train, path_n_train)
# print(train_data)
# print(train_data.shape)
# exit()
from catboost import CatBoostClassifier
clf = CatBoostClassifier(learning_rate=0.01,
                             iterations=100000000,
                             depth=8,
                             loss_function="Logloss",
                             early_stopping_rounds=200,
                             # eval_metric="AUC",
                             # eval_metric="MCC",
                             eval_metric="Accuracy",
                             thread_count=50,
                             # od_wait=500,
                             # task_type="CPU",
                             task_type="CPU",
                             # devices='0:1:2:3'
                             devices='0'
                             )
# from sklearn import svm
# svc = svm.SVC(probability=True, kernel='rbf')

# from sklearn.linear_model import LogisticRegression as LR
# l = LR(C=1, penalty='l1', solver='liblinear')


clf.fit(X=train_data,
            y=train_label,
        eval_set=(val_data,
                  val_label))

pred_res = clf.predict_proba(test_data)[:,1]

pred_label = [0 if x < 0.5 else 1 for x in pred_res]

from sklearn import metrics
tn, fp, fn, tp = metrics.confusion_matrix(y_true=test_label, y_pred=pred_label).ravel()

recall = metrics.recall_score(y_pred=pred_label, y_true=test_label)
precise = metrics.precision_score(y_pred=pred_label, y_true=test_label)

se = tp/(tp+fn)
sp = tn/(tn+fp)

acc = metrics.accuracy_score(y_pred=pred_label, y_true=test_label)
f1 = metrics.f1_score(y_pred=pred_label, y_true=test_label)
mcc = metrics.matthews_corrcoef(y_pred=pred_label, y_true=test_label)

auc = metrics.roc_auc_score(y_true=test_label, y_score=pred_res)
ap = metrics.average_precision_score(y_score=pred_res, y_true=test_label)


print("tn", tn,flush=True)
print("tp", tp,flush=True)
print("fp", fp,flush=True)
print("fn", fn,flush=True)

print("recall",recall,flush=True)
print("precise",precise,flush=True)
print("se",  se,flush=True)
print("sp",  sp,flush=True)
print("acc", acc,flush=True)
print("f1",  f1,flush=True)
print("mcc", mcc,flush=True)
print("auc", auc,flush=True)
print("ap",  ap,flush=True)








