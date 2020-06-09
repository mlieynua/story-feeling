#!/usr/bin/env python
# coding: utf-8


from gensim.models.word2vec import Word2Vec as wv
import gensim
from sklearn.preprocessing import StandardScaler,RobustScaler
import MeCab
import matplotlib.pyplot as plt
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')
class StoryFeeling:
    def __init__(self,text_file_path,model,title):
        self.text_file_path = text_file_path
        self.model = model
        self.title = title
    def process_text(self):
        with open(self.text_file_path,encoding='utf-8') as input_file:
            sentences = input_file.readlines()
            sentences = ''.join(sentences)
            sentence_array = ''
            sentence_list = []
            for sentence in sentences:
                if sentence == '。':
                    sentence_list.append(sentence_array)
                    sentence_array = ''
                else:
                    sentence_array += sentence
        sentence_list = list(map(lambda x:x.strip().replace('\u3000','').replace('\n',''),sentence_list))
        self.sentence_list = sentence_list
    def feeling_analyzer(self,feelingword):
        tagger = MeCab.Tagger('-Ochasen')
        tokenlist = []
        tokenlists = []
        neg_idx_list = []
        for idx ,sentence in enumerate(self.sentence_list):
            node = tagger.parseToNode(sentence)
            while node:
                features = node.feature.split(',')
                if features[0] != 'BOS/EOS':
                    if features[0] in ['助詞','助動詞'] and features[6] in ['ず','ぬ','ん','ない']:
                    #print(features[6])
                        neg_idx_list.append(idx)
                    if features[0] not in ['助詞','助動詞','記号']:
                        token = features[6] if features[6] != '*' else node.surface#*なら見出し語がトークン
                        tokenlist.append(token)
                node = node.next
            tokenlists.append(tokenlist)
            tokenlist = []
        vec_avg_list = []
        strong_flag = 0
        weak_flag = 0
        for sen_num ,tokens in enumerate(tokenlists):
            vec_sum = 0
            for word in tokens:
                #print(word)
                if word in ["非常", "たいへん","極めて","たいそう","かなり","すごく","とても"]:
                        strong_flag += 1
                if word in ["すこし","ちょっと","やや","少々","いくらか","わずかに"]:
                        weak_flag += 1
                try:
                    simile = self.model.wv.similarity(w1=feelingword, w2=word)
                    #print(simile)
                    similetofeeling = self.model.wv.similarity(w1="感情", w2=word)
                    #print(simile)
                    if similetofeeling >= 0.3:
                        simile *= 1.5
                    else:
                        simile *= 0.8
                    vec_sum += simile*100
                    if strong_flag == 1:
                        #print(word)
                        vec_sum *= 1.5
                        strong_flag = 0
                    if weak_flag == 1:
                        #print(word)
                        vec_sum *= 0.7
                        weak_flag = 0
                except KeyError:#学習モデルの中に入っていない語彙
                    pass
                
            vec_avg = vec_sum/len(tokens)
            vec_avg= round(vec_avg,3)
            vec_avg_list.append(vec_avg)
            vec_avg = 0
        #vec_avg_array = np.array(vec_avg_list).reshape(-1,1)
        list_med = np.median(vec_avg_list)
        list_med_0=list(map(lambda x:round(x-list_med,3),vec_avg_list)) 
        dict_mean3 = {}
        for neg_idx in neg_idx_list:
            list_med_0[neg_idx] = -(list_med_0[neg_idx])
        for idx in range(0,len(list_med_0)-2):
            mean3 = round(sum(list_med_0[idx:idx+4])/4,4)
            dict_mean3[idx] = mean3
        sorted_dict_mean3 = sorted(dict_mean3.items(),key=lambda x:x[1],reverse=True)
        counth = 0
        countl = 0
        highsentences_list = []
        lowsentences_list =[]
        for key , _ in sorted_dict_mean3:
            highsentences = self.sentence_list[key:key+4]
            highsentences_list.append(highsentences)
            counth += 1
            if counth == 3:
                break
        for key, _ in reversed(sorted_dict_mean3):
            lowsentences = self.sentence_list[key:key+4]
            lowsentences_list.append(lowsentences)
            countl += 1
            if countl == 3:
                break
    
        self.feel_avg_dict = dict_mean3
        self.high = highsentences_list
        self.low = lowsentences_list
        #print(dict_mean3,highsentences_list,lowsentences_list)
    def make_graph(self):
        fig = plt.figure(figsize=(16,4))
        ax = fig.add_subplot(1,1,1,title='The Little Match Girl')
        dic_to_list = list(self.feel_avg_dict.values())
        ax.plot(dic_to_list)
        
        ax.set_ylim([min(dic_to_list)-10, max(dic_to_list)+10])
        plt.show()
        print("1st high score:\n{}。".format('。'.join(self.high[0])))
        print("2nd high score:\n{}。".format('。'.join(self.high[1])))
        print("3rd high score:\n{}。".format('。'.join(self.high[2])))
        print("1st low score:\n{}。".format('。'.join(self.low[0])))
        print("2nd low score:\n{}。".format('。'.join(self.low[1])))
        print("3rd low score:\n{}。".format('。'.join(self.low[2])))

if __name__ == '__main__':
    model = wv.load('./latest-ja-word2vec-gensim-model/word2vec.gensim.model')
    text_file_path = "machi.txt"
    title = 'The Little Match Girl'
    st = StoryFeeling(text_file_path=text_file_path,model=model,title=title)
    st.process_text()
    st.feeling_analyzer('うれしい')
    st.make_graph()

