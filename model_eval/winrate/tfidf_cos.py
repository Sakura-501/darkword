import jieba
import math
import numpy as np

original_text1="以下是一些与卡池相关的黑话和术语：\n\n1. \"抽\" - 在游戏中，指玩家抽取到想要的物品或角色。\n2. \"爆伤\" - 当攻击者对目标造成更高的伤害时，称为“爆伤”效果。\n3. “破防率”- 游戏中描述某个技能或者装备的防御能力如何被突破的程度。\n4.“闪避概率”-游戏里表示一个动作能否成功躲避敌人的攻击的概率。\n5.\"暴击几率\"-在战斗中，如果攻击力高于敌人当前生命值一定比例的情况下，就有可能触发一次额外高额的伤害输出。\n6.“命中命中率”-在射击类游戏中，形容武器击中敌方身体部位多少的机会有多大。\n7.“法术穿透力”-在某些RPG游戏中，指的是魔法的攻击能够直接破坏对方护盾的能力大小。\n8.“物攻加成”-在一些格斗游戏中，是指增加物理攻击力的效果。\n9.“法系抗性”-某些游戏中，用于衡量怪物抵抗不同类型的属性攻击能力的数值指标。\n10.“元素克制性”-有些游戏中的元素相克关系会影响战斗结果的表现方式。例如火克水、土克木等。\n这些是常见的关于卡池的游戏用语和技术名词，希望对你有所帮助！"
original_text2="2019年威胁猎人共监测到与“猫池”相关的变种词或短语共计 5783 万条，没有一个主要类别能包含所有相关信息，以下为部分高频的关键 词段：\n手机号、接码手机、猫池、养号、拆信、信封等。"

# 停用词列表
def stop_words_list():
    stop_words = [line.strip() for line in open("/home/w1nd/darkword/1darkword/model_eval/data/hit_stopwords.txt",encoding="utf-8").readlines()]
    tmp=[line.strip() for line in open("/home/w1nd/darkword/1darkword/model_eval/data/cn_stopwords.txt",encoding="utf-8").readlines()]
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

text1=sentence_depart(original_text1)
text1=get_wipe_stop_words_text_list(text1)
print(text1)
text2=sentence_depart(original_text2)
text2=get_wipe_stop_words_text_list(original_text2)
print(text2)
