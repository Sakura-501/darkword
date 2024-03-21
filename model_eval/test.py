import re
# text="""<s> Human: 解释一下菠菜有几种意思
# </s><s> Assistant:  1. 一种植物，指一种常见的蔬菜。
# 2. 在某些语言中（如英语），“菠菜”也可以用来形容一个人愚蠢或无知的行为或者言论。例如："He is so gullible, always believing in those stupid rumors." (他太容易相信那些荒唐的谣言了！)
# 3. 另外还有一个意思是指某种颜色接近绿色的物体或者是植物的颜色。比如绿色墨水、绿宝石等物品和饰品都是属于这种颜色的范畴之内。
# 4. 还有一些地方会把菠菜当作食物的一种称呼，即用菠菜来代替其他食材进行烹饪制作而成菜色丰富多样的菜品系列。这些菜品通常以菠菜为主要原料制成各种不同的口味与口感的组合搭配而成的菜肴。因此可以说在有些场合下，人们也会使用“菠菜”这个词语来形容某一道美食的味道特点等等方面所具有的独特之处。总之呢，无论是何种含义都有其独特的特点和用途所在哦~!
# </s>"""

text="""<s> Human: 解释一下菠菜有几种意思
</s><s> Assistant: """
pattern = r"Assistant: (.*?)\n</s>"
try:
    matches = re.findall(pattern,text,re.DOTALL)
    print(matches[0])
except:
    print("")
# import json

# def load_eval_data():
#     with open("/home/w1nd/darkword/1darkword/model_eval/data/eval_test.json","r",encoding="utf-8") as jsonfile:
#         eval_data = json.load(jsonfile)
#     jsonfile.close()
#     print("评估数据的数量："+str(len(eval_data)))
#     print(eval_data)
#     return eval_data
# eval_data=load_eval_data()

# for each_conversation in eval_data:
#     # print(each_conversation)
#     question=each_conversation["conversations"][0]["content"]
#     answer=each_conversation["conversations"][1]["content"]
#     print(question)
#     print(answer)