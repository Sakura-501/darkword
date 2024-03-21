from tfidf_cos import sentence_depart,get_wipe_stop_words_text_list,merge_two_text,create_two_text_vector,cal_tf,cal_idf,cal_tf_idf,cal_cosine_similarity
import json

def get_cosine_similarity(original_text1,original_text2):
    # 两个文本去除停用词，并提取成词汇数组
    text1=sentence_depart(original_text1)
    text1=get_wipe_stop_words_text_list(text1)
    # print(text1)
    text2=sentence_depart(original_text2)
    text2=get_wipe_stop_words_text_list(original_text2)
    # print(text2)

    # 创建词汇列表
    vocabulary = merge_two_text(text1,text2)
    # print(vocabulary)

    # 利用词汇列表对两个文本创建向量矩阵
    arr1,arr2 = create_two_text_vector(text1,text2,vocabulary)
    # print(arr1,arr2)

    # TF词频计算
    tf1, tf2 = cal_tf(arr1,arr2)
    # print(tf1,tf2)

    # IDF逆文档频率计算
    idf1, idf2 = cal_idf(text1,text2,vocabulary)
    # print(idf1,idf2)

    # TF-IDF词频-逆文档频率计算
    tf_idf1 = cal_tf_idf(tf1,idf1)
    tf_idf2 = cal_tf_idf(tf2,idf2)
    # print(tf_idf1,tf_idf2)

    # 余弦相似度计算
    cos = cal_cosine_similarity(tf_idf1,tf_idf2)
    print(cos)
    return cos

def lora_atom_eval(base_file,lora_file):
    with open(base_file,"r",encoding="utf-8") as basefile:
        base_data = json.load(basefile)
        print(base_data)
    
if __name__ == "main":
    base_atom_response_file=""
    lora_atom_response_file=""
    lora_baichuan2_response_file=""
    lora_chatglm3_response_file=""
    lora_atom_eval(base_atom_response_file,lora_atom_response_file)
    