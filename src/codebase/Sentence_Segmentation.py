# -*- coding: utf-8 -*-
import re,json,os,sys
from hanziconv import HanziConv

cleanr = re.compile('<.*?>')
def clean_html(raw_text):
    # 清除所有 html 的標籤
    cleantext = re.sub(cleanr, '', raw_text)
    return cleantext

def Segmentation_Core(data,tokenizer):
    # 將所有的斷行字元配上 escape 以防失效
    regexPattern = '|'.join(map(re.escape, tokenizer))
    output_list = re.split(regexPattern, data)
    # 過濾空白行
    output_list = list(filter(None, output_list))
    return output_list

def Sentence_Segmentation(article, rep_comma_regexp = "\\n", \
    rep_period_regexp = "(\\n){2,}", keep_digits = True, \
    remove_html = True):
    """ Sentence_Segmentation function 用於執行中文斷行任務

    順序如下：1. 移除 HTML、2. 替換(\\n)逗號以及(\\n{2,})句號、3. 去除特殊符號字元、
    4. 保留數字、5. 依照斷句字元進行斷句全形的分號、逗號、句號、問號以及驚嘆號、
    6. 翻譯為簡體文字、7. 避免空白斷句結果

    Args:
        language(Chinese): 只有中文能使用
        article(string): 欲斷行之文章內容
        rep_comma_regexp(string): 適用於中文PTT文章，將 \n 取代為逗號
        rep_period_regexp(string): 適用於中文PTT文章，將連續兩個 \n 取代為句號
        keep_digits(Boolean): 決定是否保留數字於斷行後的結果，適用於中英文

    Returns:
        List [] 每個 element 就是一行

    """
    if remove_html:
        article = clean_html(article)

    withoutNote_res = []
    final_res = []
    sepical_symbols = '[＂<>:《》+\-=#$%&()*@＃＄％＆＇\(\)\[\]\{\}（）＊＋－／：\
            ＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏・━┿│┷┯．−]+'
    
    # this try block is to remove special characters and change breakline symbol
    try:
        article = re.sub(str(rep_period_regexp),'。',article)
        article = re.sub(str(rep_comma_regexp),'，',article)
        article = re.sub(sepical_symbols,'',article)
    except:
        return "RegExp Error!"

    # determine whether to keep digits
    if(keep_digits):

        """
        # 消除起始為空白的字元, + 代表無論多少個空白起始都消掉
        temp = re.sub("^[ ]+", "", i)
        # 消除結尾為空白之字元, + 代表無論有多少個空白在結尾都消掉
        temp = re.sub("[ ]+$", "", temp)
        # 消除字串間所有空白字元
        temp = re.sub("[ ]+","", temp)
        # 上面三行 = [\s]+
        """
        article = re.sub('[\s]+','', article) #remove space
    else:
        article = re.sub('[\s\d]+','', article) #remove space & digits

    #斷句，中文逗號不算斷句
    #此版本不用冒號作為斷句依據
    segmentation_used_note = (";", "；", "！", "!", "？", "?", "。")
    res = Segmentation_Core(article,segmentation_used_note)

    #將繁體字轉為簡體字為後續 NLP 做準備
    for i in res:
        i = HanziConv.toSimplified(i)
        withoutNote_res.append(i)

    #用來消除整句都是空白的斷句
    for temp in withoutNote_res:
        if temp != "":
            final_res.append(temp)

    return final_res

if __name__=='__main__':
    chi_results = Sentence_Segmentation("    立法院社｠昨審查《勞動基準法》部分條文修正草案，兩黨立法委員昨都擠在會議室內，會前就已經在主席台上打成一團，場外則有抗議勞團沿著立法院不斷用大聲公，集體呼喊自己的訴求，希望在裡面    開會的委員能聽到他們的聲音。而晚$間民＠進黨搞突襲 《勞基法》闖關     成功打包送院會。\n\n原本立院外抗議的群眾，昨晚10時許宣布解散，事後有許多民眾遲遲不肯離去，今凌晨約20名獨派的群眾轉往凱道集結，席地而坐占用道路表達不滿，警方不敢大意     ，增派警力在凱道嚴陣以待，最後在警方三次舉牌後，將現場民眾強制驅離，過程中抗＆議民眾洪德仁，因而嘴角受傷流血被送醫治療，警方現場總共動用2輛警備車，其中7男6女被送往南港展覽館，另有1名男子被送往關渡。     現場留下30多名民眾在現場不願離開，警方最後將民眾驅離到台北賓館前，隨即現場民眾漸漸散去，但警方還是不敢大意，仍在現場持續待命，防止抗議民眾突襲。（突發中心簡銘柱／台北報導）", keep_digits = True)
    print("Chinese Test Case:", chi_results)
    #eng_results = Sentence_Segmentation("English", "That doesn’t mean the GOP tax cuts are popular. They’re not. Both the House and Senate bills are heavily weighted toward business tax cuts, with working- and middle-class voters likely to see just a fraction of the total tax savings Republicans are doling out. Various provisions would harm graduate students, people with high medical expenses and other groups that don’t typically have high-priced lobbyists fighting for their cause in Washington. But Congress can change the bill during the final few weeks of haggling, to make it less damaging and more politically defensible.", remove_english_symbols = False)
    #print("English Test Case:", eng_results)
