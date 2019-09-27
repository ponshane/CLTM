from pycorenlp import StanfordCoreNLP
import configparser

### init and read config
config = configparser.ConfigParser()
config.read('../config.ini')

nlp = StanfordCoreNLP(config["CoreNLP"]["Chinese_URL"])

def POSTagger(sentences):
    """ POSTagger function 用於執行中文標記詞性任務

    **注意檢查 config.ini 裡頭的 CoreNLP Chinese_URL **
    目前採用 58 虛擬機上的 coreNLP server

    Args:
    language(Chinese): 只有中文給你選
    sentences(list(list(string))): Nested List [[Sentence1 斷詞結果], [Sentence2 斷詞結果], ...]
    每個element為每行斷詞結果，每一行都各自為一個 list，因此是一個巢狀的 list

    Returns:
    Nested List [[Sentence1 詞性標記結果], [Sentence2 詞性標記結果], ...] 每個element為
        每行詞性標記結果，每一行都各自為一個 list，因此是一個巢狀的 list
    """

    result = list()

    input_sentence = '\n'.join([' '.join(each_sentence) for each_sentence in sentences])

    output = nlp.annotate(input_sentence, properties={
        'annotators': 'pos',
        'tokenize.language': 'Whitespace', # first property
        'ssplit.eolonly': 'true', # second property
        'outputFormat': 'json'
    })

    try:
        sentence = output['sentences'][0]
    except (IndexError, TypeError):
        return([["IGotAnIndexError"]])

    # 回傳會將所有 tokens 都放在 ['sentences'][0]
    # 如此一來無法區分句，因此撰寫以下程式協助區分
    # 主要運用每句區隔 "\n"
    tempString = []
    for each_token in sentence['tokens']:
        # 每句的最後才會是 \n
        if(each_token['word'] != "\n"):
            tempString.append(each_token['word']+"#"+each_token['pos'])
        # 每句的最後時
        else:
            if(tempString!=[]):
                result.append(tempString)
                tempString = []

    # 最後一句不會有 \n 因此得直接將剩餘的最後一個 tempString 直接放進 result
    result.append(tempString)
    return(result)

if __name__=='__main__':
    chi_results = POSTagger("Chinese", [['立法院', '社昨', '审查', '劳动基准', '法', '部分', '条文', '修正',\
    '草案', '，', '两党', '立法委员', '昨都', '挤', '在', '会议室', '内', '，', '会前', '就', '已经', '在',\
    '主席台', '上', '打成', '一团', '，', ' ', ' ', ' ', ' ', '场外', '则', '有', '抗议', '劳团', '沿着',\
    '立法院', '不断', '用', '大声', '公', '，', '集体', '唿喊', '自己', '的', '诉求', '，', '希望', '在',\
    '里面', '开会', '的', '委员', '能', '听到', '他们', '的', '声音'], ['而', '晚间', '民进党', '搞', '突袭', '劳基法',\
     '闯关', '成功', '打包', '送', ' ', ' ', ' ', ' ', ' ', '院', '会']])
    print("Chinese POSTagger results:", chi_results)
