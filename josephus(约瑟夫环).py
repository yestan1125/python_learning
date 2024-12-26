def josephus(n, m):
    # 初始化一个列表，表示n个人
    people = list(range(1, n + 1))
    
    # 当前的位置
    index = 0
    
    # 循环直到只剩一个人
    while len(people) > 1:
        # 计算下一个要淘汰的人的索引
        index = (index + m - 1) % len(people)
        # 淘汰掉这个人
        people.pop(index)
    
    # 返回最后一个剩下的人
    return people[0]

# 示例：有10个人，每报到第3个人就淘汰一个，最后剩下的是谁？
n = 10
m = 3
print(f"最后剩下的人是: {josephus(n, m)}")
