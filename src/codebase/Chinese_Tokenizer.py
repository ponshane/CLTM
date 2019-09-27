import jieba
import os
import configparser

### init and read config
config = configparser.ConfigParser()
config.read('config.ini')

# 載入使用者自訂字典
# 此字典應該要用相對應的簡轉繁套件轉換
# 否則對應不到，就失去字典的效力了
try:
    userdictPath = config["Jieba"]["Dictionary_Address"]
    if os.path.isfile(userdictPath):
        # load_userdict 會順便 initialize jieba 載入字典
        # jieba 已經做了很好的 OOP 設置不需要在另外寫，因此我這邊只有寫 Tokenizer function
        jieba.load_userdict(userdictPath)
except KeyError:
    print("Can't not use customized dictionary")

def Tokenizer(sentences):
    """ Tokenizer function 用於執行中文斷詞任務

    **注意更改 config.ini 裡頭的 Jieba Section Dictionary Addresss**

    Args:
        language(Chinese): 只有中文給你選
        sentences(list(string)): List [] 每個 element 就是一行，即 Sentence_Segmentation.py Returns

    Returns:
        Nested List [[Sentence1 斷詞結果], [Sentence2 斷詞結果], ...] 每個element為 \
        每行斷詞結果

    """
    # init 結果 sent list
    output_sents = list()
    for each_sentence in sentences:
        words = jieba.lcut(each_sentence, cut_all=False)

        each_output_sent = []
        for word in words:
            # 以防斷出空白字
            if len(word) != 0:
                each_output_sent.append(word)
        # 以防最後有空白句
        if len(each_output_sent) != 0:
            output_sents.append(each_output_sent)
        else:
            output_sents.append(["there_is_no_word"])

    return output_sents

if __name__=='__main__':
    chi_results = Tokenizer("Chinese", ['立法院社昨审查劳动基准法部分条文修正草案，两党立法委员昨都挤在会议室内，会前就已经在主席台上打成一团，\
    场外则有抗议劳团沿着立法院不断用大声公，集体唿喊自己的诉求，希望在里面开会的委员能听到他们的声音', '而晚间民进党搞突袭劳基法闯关成功打包送 \
    院会','原本立院外抗议的群众，昨晚10时许宣布解散，事后有许多民众迟迟不肯离去，今凌晨约20名独派的群众转往凯道集结，席地而坐占用道路表达不满，\
    警方不敢大意，增派警力在凯道严阵以待，最后在警方三次举牌后，将现场民众强制驱离，过程中抗议民众洪德仁，因而嘴角受伤流血被送医治疗，警方现场总\
    共动用2辆警备车，其中7男6女南港展览馆，另有1名男子被送往关渡', '现场留下30多名民众在现场不愿离开，警方最后将民众驱离到台北宾馆前，随即现场民\
    众渐渐散去，但警方还是不敢大意，仍在现场持续待命，防止抗议民众突袭', '突发中心简铭柱台北报导'])
    print("Chinese Token Results:", chi_results)
