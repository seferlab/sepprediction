
'''
 两个都没有名字的文件合并一起去重, 得到还是无名字的无重文件
'''
def tiqu(fi1,fi2,write_fi):
    file=open(fi1,'r')
    files = open(fi2, 'r')
    du=file.readlines()
    du2 = files.readlines()
    print('文件1里有{}个元素'.format(len(du)))
    print('文件2里有{}个元素'.format(len(du2)))
    du3=du2
    du4=[]
    write_dom=open(write_fi,'a')
    s=0


    for i in du:
        # i=i.rstrip()
        # i=i+'\n'
        du4.append(i)
        if i in du3:
            du3.remove(i)
            s+=1
    for i in du3:
        du4.append(i)
    du4.append('\n')
    for i in du4:
        write_dom.write(i)
    print('合并后有{}个元素,一共有{}个元素相同并被删除'.format(len(du4)-1,s))
# tiqu("./综合/Hallbrink-2005/P.txt","./综合/Dobchev-2010/P.txt",'huihe_P_1.txt')
# tiqu('huihe_P_1.txt',"./综合/Hansen-2008/P.txt",'huihe_P_2.txt')
# tiqu('huihe_P_2.txt',"./综合/Sanders-2011c/P.txt",'huihe_P_3.txt')

# tiqu("./综合/Hallbrink-2005/N.txt","./综合/Dobchev-2010/N.txt",'huihe_N_1.txt')
# tiqu('huihe_N_1.txt',"./综合/Hansen-2008/N.txt",'huihe_N_2.txt')
# tiqu('huihe_N_2.txt',"./综合/Sanders-2011c/N.txt",'huihe_N_3.txt')


'''
文件1 无名字, 文件2 有名字 合并成无名字去重的文件
'''


def tiqu2(fi1,fi2,fi3):
    file = open(fi1, 'r')
    du = file.readlines()
    print('初始文件一共有{}个元素'.format(len(du)))
    with open(fi2, "r") as file1_file,open(fi3, "w") as output_file:
        fi2_num=0
        fi_to=0
        for file1_line in file1_file:
        #     print(file1_line)

            if file1_line.startswith(">"):
        #     #     # 如果这一行是 FASTA 格式的头部，提取 ID 号
        #         id = file1_line.strip()[1:]
        #         print(id)
                seq1 = file1_file.readline().strip()
                seq1=seq1+'\n'
                fi2_num += 1

                if seq1 not in du:
                    du.append(seq1)
                else:
                    fi_to+=1
        print('文件2一共有{}个元素,一共有{}个元素是相似'.format(fi2_num,fi_to))
        for i in du:
            output_file.write(i)



#
# tiqu2('huihe_P_3.txt',"./综合/CPP924_P.fasta",'huihe_P_4.txt')
# tiqu2('huihe_N_3.txt',"./综合/CPP924_N.fasta",'huihe_N_4.txt')
# tiqu2('huihe_P_4.txt',"./论文数据集/Mlcpp2.0数据集/Layer1_P.txt",'huihe_P_5.txt')
# tiqu2('huihe_N_4.txt',"./论文数据集/Mlcpp2.0数据集/Layer1_N.txt",'huihe_N_5.txt')
# tiqu2('huihe_P_5.txt',"./论文数据集/Mlcpp2.0数据集/Layer1_P_indepen.txt",'huihe_P_6.txt')
# tiqu2('huihe_N_5.txt',"./论文数据集/Mlcpp2.0数据集/Layer1_N_indepen.txt",'huihe_N_6.txt')

# tiqu('huihe_P_6.txt',"./论文数据集/CPPred-rf数据集/test/P.txt",'huihe_P_7.txt')


'''
# 给所有无名字的数据集自动编号
'''
# names=['>Positive_','>Negative_']

def ming_id(fi2,fi3,names):
    with open(fi2, "r") as files,open(fi3, "w") as output_file:
        du = files.readlines()
        print('一共有{}个元素'.format(len(du)))
        num=0
        for i in du:
            num+=1
            name=names+str(num)+'\n'
            output_file.write(name)
            output_file.write(i)

# ming_id('../dataset/MLCPP_data_N.fasta','../dataset/MLCPP_data_NN.fasta',names='>Negative_')
# ming_id('../dataset/MLCPP_data_P.fasta','../dataset/MLCPP_data_PP.fasta',names='>Positive_')

# 给有名字的保存为无名字的
def tiqu3(fi1,fi2):
    with open(fi1, "r") as files:
        fi2_num=0
        fi_to=0
        s=[]
        for file1_line in files:
            # print(file1_line)

            if file1_line.startswith(">"):
                # print(file1_line)
                seq1 = files.readline().strip()

                seq1=seq1+'\n'
                s.append(seq1)
    with open(fi2, "w") as filess:
        for i in s:
            filess.write(i)

# tiqu3('../dataset/CPP924_NN.fasta','../dataset/CPP924_NN.fasta')
# tiqu3('../dataset/CPP924_PP.fasta','../dataset/CPP924_PP.fasta')



# 去重 (无名的去重)
def tiqu4(fi1,fi2):
    with open(fi1, "r") as files,open(fi2, "w") as output_file:
        du = files.readlines()
        print('去重前一共有{}个元素'.format(len(du)))
        all=[]
        for i in du:
            if i not in all:
                all.append(i)
        print('去重后一共有{}个元素'.format(len(all)))
        for i in all:
            output_file.write(i)


# tiqu4('G:\代码\课题方向\数据集\924_N.txt','924_N1.txt')

