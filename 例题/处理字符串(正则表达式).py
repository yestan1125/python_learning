import re
from collections import Counter

def find_most_frequent_word(text):
    # 去除非字母字符并将文本转换为小写
    words = re.findall(r'[a-zA-Z]+', text.lower())
    
    # 统计单词的频率
    word_count = Counter(words)
    
    # 找到频率最高的单词，若有多个，按字典序排序
    max_word, max_count = min(word_count.items(), key=lambda item: (-item[1], item[0]))
    
    # 输出结果
    print(max_word, max_count)

# 测试
input_text = input().strip()  # 获取输入，去掉末尾的回车符
find_most_frequent_word(input_text)

#给定一个由英文字符、数字、空格和英文标点符号组成的字符串，长度不超过5000，请将其切分为单词，
#要求去掉所有的非英文字母，然后将单词全部转换成小写，然后统计每一个词出现的次数，
#输出频次最高的那个词以及它出现的次数。如果有多个词的频次相同，则输出按字典序排列在最前面的那个。
#输入样例
#A character may be part of a Unicode identifier if and only if one of the following statements is true.
#输出样例
#a  2
