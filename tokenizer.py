from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import re

class ChineseTokenizer:
    def __init__(self, stopword_file='/home/ricky42613/news_cluster/stop.txt'):
        self.stopwords = []
        with open(stopword_file,'r') as f:
            self.stopwords = [l[:-1] for l in f.readlines()]
        self.ws_driver  = CkipWordSegmenter(model="bert-base") # use ckip tokenizer to tokenize data
        self.pos_driver = CkipPosTagger(model="bert-base")
        self.ner_driver = CkipNerChunker(model="bert-base")
        self.pos_tags = ['Na', 'Nb', 'Nc', 'VA', 'VC']
        self.ner_tags = ['PERSON', 'GPE','ORG','PRODUCT','FAC','NORP','EVENT','WORK_OF_ART','LAW']
    
    def filter_nonchinese(self, text):
        '''
        return True if token is chinese and not in stop words list
        text: token in sentence
        '''
        pattern = re.compile(r'[^\u4e00-\u9fa50-9]')
        nonchinese = re.sub(pattern,"",text)
        return len(nonchinese) == len(text) and text not in self.stopwords and len(text) > 1

    def tokenize_with_special_token(self, text, pos_tags=None, ner_tags=None, return_tag = False):
        pos_tags = self.pos_tags if pos_tags == None else pos_tags
        ner_tags = self.ner_tags if ner_tags == None else ner_tags
        ws  = self.ws_driver([text], use_delim=True, show_progress=False)
        pos = self.pos_driver(ws, use_delim=True, show_progress=False)
        ner = self.ner_driver([text], use_delim=True, show_progress=False)
        ret = []
        valid_tag = {
            'pos': pos_tags, # ['Na', 'Nb', 'Nc', 'Nd'],
            'ner' : ner_tags
        }
        for item in ner[0]:
            if item.ner in valid_tag['ner'] and len(item.word) > 1:
                if return_tag:
                    ret.append((item.word,item.ner))
                else:
                    ret.append(item.word)
        for i in range(len(pos[0])):
            if pos[0][i] in valid_tag['pos'] and len(ws[0][i]) > 1:
                if return_tag:
                    ret.append('{}_{}'.format(ws[0][i],pos[0][i]))
                else:
                    ret.append(ws[0][i])
        return ret
    def tokenize(self,text, rm_noisy = True):
        if type(text) == str:
            text = [text]
        ws = self.ws_driver(text, batch_size=2048, show_progress= False)
        if rm_noisy:
            for i in range(0, len(ws)):
                ws[i] = filter(self.filter_nonchinese, ws[i])
        return ws

    # def tokenize_verb(self, text):
    #     ws  = self.ws_driver([text], use_delim=True)
    #     pos = self.pos_driver(ws, use_delim=True)
    #     ret = []
    #     verb_tag = ['VA', 'VC']
    #     for i in range(len(pos[0])):
    #         if pos[0][i] in verb_tag and len(ws[0][i]) > 1:
    #             ret.append(ws[0][i])
    #     return ret

    # def tokenize_zh(self,text):
    #     ws  = self.ws_driver([text], use_delim=True)
    #     pos = self.pos_driver(ws, use_delim=True)
    #     ner = self.ner_driver([text], use_delim=True)
    #     ret = []
    #     valid_tag = ['Na', 'Nb', 'Nc', 'VA', 'VC']
    #     valid_entity = ['PERSON', 'GPE','ORG','PRODUCT','FAC','NORP','EVENT','WORK_OF_ART']
    #     for i in range(len(pos[0])):
    #         if pos[0][i] in valid_tag and len(ws[0][i]) > 1:
    #             ret.append(ws[0][i])
    #     for item in ner[0]:
    #         if item.ner in valid_entity and len(item.word) > 1:
    #             ret.append(item.word)
    #     return ret
    
