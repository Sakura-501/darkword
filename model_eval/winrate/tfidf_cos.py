from itertools import count
import jieba
import math
import numpy as np

original_text1="以下是一些与卡池相关的黑话和术语：\n\n1. \"抽\" - 在游戏中，指玩家抽取到想要的物品或角色。\n2. \"爆伤\" - 当攻击者对目标造成更高的伤害时，称为“爆伤”效果。\n3. “破防率”- 游戏中描述某个技能或者装备的防御能力如何被突破的程度。\n4.“闪避概率”-游戏里表示一个动作能否成功躲避敌人的攻击的概率。\n5.\"暴击几率\"-在战斗中，如果攻击力高于敌人当前生命值一定比例的情况下，就有可能触发一次额外高额的伤害输出。\n6.“命中命中率”-在射击类游戏中，形容武器击中敌方身体部位多少的机会有多大。\n7.“法术穿透力”-在某些RPG游戏中，指的是魔法的攻击能够直接破坏对方护盾的能力大小。\n8.“物攻加成”-在一些格斗游戏中，是指增加物理攻击力的效果。\n9.“法系抗性”-某些游戏中，用于衡量怪物抵抗不同类型的属性攻击能力的数值指标。\n10.“元素克制性”-有些游戏中的元素相克关系会影响战斗结果的表现方式。例如火克水、土克木等。\n这些是常见的关于卡池的游戏用语和技术名词，希望对你有所帮助！"
original_text2="2019年威胁猎人共监测到与“猫池”相关的变种词或短语共计 5783 万条，没有一个主要类别能包含所有相关信息，以下为部分高频的关键 词段：\n手机号、接码手机、猫池、养号、拆信、信封等。"

# 停用词列表
def stop_words_list():
    stop_words = [line.strip() for line in open("/home/w1nd/darkword/1darkword/model_eval/data/stop_words/hit_stopwords.txt",encoding="utf-8").readlines()]
    tmp=[line.strip() for line in open("/home/w1nd/darkword/1darkword/model_eval/data/stop_words/cn_stopwords.txt",encoding="utf-8").readlines()]
    stop_words.extend(tmp)
    return stop_words

# 去除停用词
def sentence_depart(line):
    sentence_depart = jieba.cut(line.strip())
    stop_words=stop_words_list()
    output=""
    for word in sentence_depart:
        # print(word)
        if word not in stop_words:
            if word != "\t":
                output += word
                output += " "
    return output

# 对去除停用词后的文本进行分词
def get_wipe_stop_words_text_list(text):
    line_depart=""
    lines=text.splitlines()
    for line in lines:
        line_depart+=sentence_depart(line)
        # print(line_depart)
    wipe_text=line_depart.split()
    return wipe_text

# 合并两个文本的词汇矩阵
def merge_two_text(text1,text2):
    vocabulary = []
    vocabulary = list(set(text1+text2))
    return vocabulary

# 分别创建两个文本的向量矩阵：
def create_two_text_vector(text1,text2,vocabulary):
    text1_arr=[]
    text2_arr=[]
    for v in vocabulary:
        c1 = text1.count(v)
        c2 = text2.count(v)
        text1_arr.append(c1)
        text2_arr.append(c2)
    return text1_arr,text2_arr

# TF词频计算：TF=某个词在文章的出现次数/文章的总次数
def cal_tf(arr1,arr2):
    tf1_list=[]
    tf2_list=[]
    length = len(arr1)
    for a,b in zip(arr1,arr2):
        tf1_list.append(a/length)
        tf2_list.append(b/length)
    return tf1_list,tf2_list

# IDF逆文档频率计算：IDF=log(语料库的文档总数/(包含改词的文档数+1))
def cal_idf(text1,text2,vocabulary):
    # 先计算单个词在两个文档中出现的次数
    length = len(vocabulary)
    text1_count_arr = [0]*length
    text2_count_arr = [0]*length
    # 首先是计算文档一的词的出现次数，因为if是先判断是否在文档一中！
    for index in range(length):
        if vocabulary[index] in text1:
            text1_count_arr[index]+=1
            if vocabulary[index] in text2:
                text1_count_arr[index]+=1
    # 然后是计算文档二的
    for index in range(length):
        if vocabulary[index] in text2:
            text2_count_arr[index]+=1
            if vocabulary[index] in text1:
                text2_count_arr[index]+=1
    return text1_count_arr,text2_count_arr
    
# TF-IDF词频-逆文档频率计算：TF-IDF=TF*IDF
def cal_tf_idf(tf,idf):
    tf_idf = []
    for i in tf:
        for j in idf:
            tf_idf.append(i*j)
    return tf_idf

# 余弦相似度计算
def cal_cosine_similarity(x,y,norm = False):
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x==y else float(0)
    # 计算出x*y x*x y*y ，矩阵有三列，后面计算方便
    res = np.array([[x[i]*y[i],x[i]*x[i],y[i]*y[i]]for i in range(len(x))] )
    cos = sum(res[:,0])/(np.sqrt(sum(res[:,1]))*np.sqrt(sum(res[:,2])))
    return 0.5 * cos + 0.5 if norm else cos

    

# # 两个文本去除停用词，并提取成词汇数组
# text1=sentence_depart(original_text1)
# text1=get_wipe_stop_words_text_list(text1)
# print(text1)
# text2=sentence_depart(original_text2)
# text2=get_wipe_stop_words_text_list(original_text2)
# print(text2)

# # 创建词汇列表
# vocabulary = merge_two_text(text1,text2)
# print(vocabulary)

# # 利用词汇列表对两个文本创建向量矩阵
# arr1,arr2 = create_two_text_vector(text1,text2,vocabulary)
# print(arr1,arr2)

# # TF词频计算
# tf1, tf2 = cal_tf(arr1,arr2)
# print(tf1,tf2)

# # IDF逆文档频率计算
# idf1, idf2 = cal_idf(text1,text2,vocabulary)
# print(idf1,idf2)

# # TF-IDF词频-逆文档频率计算
# tf_idf1 = cal_tf_idf(tf1,idf1)
# tf_idf2 = cal_tf_idf(tf2,idf2)
# print(tf_idf1,tf_idf2)

# # 余弦相似度计算
# cos = cal_cosine_similarity(tf_idf1,tf_idf2)
# print(cos)