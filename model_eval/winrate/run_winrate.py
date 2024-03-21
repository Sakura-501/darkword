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
    # print(cos)
    return cos

def judge_winner(text,cos1,cos2):
    winner_a="model_a"
    winner_b="model_b"
    tie="tie"
    if cos1 > 0.6 or cos2 > 0.6:
        if cos1 > cos2:
            return winner_a
        elif cos2 > cos1:
            return winner_b
        else:
            return tie
    # 如果tf-idf余弦相似度都没有超过0.6，就需要研究者自行判断
    else:
        print("question: \n"+text[0])
        print("standard_answer: \n"+text[1])
        print(f"model_a_answer: {cos1}\n"+text[2])
        print(f"model_b_answer: {cos2}\n"+text[3])
        choice = input("请选择 1.model_a胜出 2.model_b胜出 3.tie平手\n")
        if choice == "1":
            return winner_a
        elif choice == "2":
            return winner_b
        return tie

def base_and_lora_eval(base_file,lora_file,choice,output_file):
    with open(base_file,"r",encoding="utf-8") as basefile:
        base_responses = json.load(basefile)
    basefile.close()
    with open(lora_file,"r",encoding="utf-8") as lorafile:
        lora_responses = json.load(lorafile)
    lorafile.close()
    with open(output_file,"wt",encoding="utf-8") as output:
        for one_base_response,one_lora_response in zip(base_responses,lora_responses):
            # base_vs_atom
            if choice == 1:
                one_result={"model_a":"atom_base","model_b":"atom_lora"}
                text0 = one_base_response["question"]
                text1 = one_base_response["standard_answer"]
                text2 = one_base_response["atom_base_response"]
                text3 = one_lora_response["atom_lora_response"]
                text=[text0,text1,text2,text3]
                cos1 = get_cosine_similarity(text1,text2)
                cos2 = get_cosine_similarity(text1,text3)
                # print(cos1,cos2)
                winner = judge_winner(text,cos1,cos2)
                one_result["winner"] = winner
                output.write(json.dumps(one_result,ensure_ascii=False)+"\n")
            # base_vs_baichuan2
            elif choice == 2:
                one_result={"model_a":"atom_base","model_b":"baichuan2_lora"}
                text0 = one_base_response["question"]
                text1 = one_base_response["standard_answer"]
                text2 = one_base_response["atom_base_response"]
                text3 = one_lora_response["baichuan2_lora_response"]
                text=[text0,text1,text2,text3]
                cos1 = get_cosine_similarity(text1,text2)
                cos2 = get_cosine_similarity(text1,text3)
                # print(cos1,cos2)
                winner = judge_winner(text,cos1,cos2)
                one_result["winner"] = winner
                output.write(json.dumps(one_result,ensure_ascii=False)+"\n")
            # base_vs_chatglm3
            elif choice == 3:
                one_result={"model_a":"atom_base","model_b":"chatglm3_lora"}
                text0 = one_base_response["question"]
                text1 = one_base_response["standard_answer"]
                text2 = one_base_response["atom_base_response"]
                text3 = one_lora_response["chatglm3_lora_response"]
                text=[text0,text1,text2,text3]
                cos1 = get_cosine_similarity(text1,text2)
                cos2 = get_cosine_similarity(text1,text3)
                # print(cos1,cos2)
                winner = judge_winner(text,cos1,cos2)
                one_result["winner"] = winner
                output.write(json.dumps(one_result,ensure_ascii=False)+"\n")
            # baichuan2_vs_chatglm3
            else:
                one_result={"model_a":"baichuan2_lora","model_b":"chatglm3_lora"}
                text0 = one_base_response["question"]
                text1 = one_base_response["standard_answer"]
                text2 = one_base_response["baichuan2_lora_response"]
                text3 = one_lora_response["chatglm3_lora_response"]
                text=[text0,text1,text2,text3]
                cos1 = get_cosine_similarity(text1,text2)
                cos2 = get_cosine_similarity(text1,text3)
                # print(cos1,cos2)
                winner = judge_winner(text,cos1,cos2)
                one_result["winner"] = winner
                output.write(json.dumps(one_result,ensure_ascii=False)+"\n")
    output.close()
        
    
if __name__ == "__main__":
    base_atom_response_file="/home/w1nd/darkword/1darkword/model_eval/data/results/lora_baichuan2_response.json"
    lora_atom_response_file="/home/w1nd/darkword/1darkword/model_eval/data/results/lora_chatglm3_response.json"
    lora_baichuan2_response_file="/home/w1nd/darkword/1darkword/model_eval/data/results/lora_baichuan2_response.json"
    lora_chatglm3_response_file="/home/w1nd/darkword/1darkword/model_eval/data/results/lora_chatglm3_response.json"
    
    base_atom_vs_lora_atom_file="/home/w1nd/darkword/1darkword/model_eval/data/winner/base_atom_vs_lora_atom.json"
    base_atom_vs_lora_baichuan2_file="/home/w1nd/darkword/1darkword/model_eval/data/winner/base_atom_vs_lora_baichuan2.json"
    base_atom_cs_lora_chatglm3_file="/home/w1nd/darkword/1darkword/model_eval/data/winner/base_atom_cs_lora_chatglm3.json"
    test_file="/home/w1nd/darkword/1darkword/model_eval/data/winner/test.json"
    
    
    # 1.base_atom_vs_lora_atom
    # base_and_lora_eval(base_atom_response_file,lora_atom_response_file,1,base_atom_vs_lora_atom_file)
    
    # 2.base_atom_vs_lora_baichuan2
    # base_and_lora_eval(base_atom_response_file,lora_baichuan2_response_file,2,base_atom_vs_lora_baichuan2_file)
    
    # 3.base_atom_vs_lora_chatglm3
    # base_and_lora_eval(base_atom_response_file,lora_chatglm3_response_file,3,base_atom_cs_lora_chatglm3_file)
    
    # 4.lora_baichuan2_vs_lora_chatglm3
    # base_and_lora_eval(lora_baichuan2_response_file,lora_chatglm3_response_file,4,test_file)
    
    