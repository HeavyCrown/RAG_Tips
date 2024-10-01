def get_bset_segment(relevance_values: list, max_length: int, overall_max_length: int, minimum_value: float):
    """
    Args:
        relevance_values (list): 每个chunk的相关度得分
        max_length (int): 一个segment的最大长度（单位：chunk）
        overall_max_length (int): 所有segment的最大长度，chunk的总数量
        minimum_value (float): best_segment的下限

    Returns:
        best_segments (list): 元组列表 (start, end) 代表一个best segments在文档中的位置
        scores (list): 一个列表，存储每个best_segment的得分
    """

    best_segments= []
    scores = []
    total_length = 0

    while total_length < overall_max_length:
        best_segment = None
        best_value = -1000
        for start in range(len(relevance_values)):
            # 找到一个起点
            if relevance_values[start] < 0:
                continue
            for end in range(start+1, min(start+max_length+1, len(relevance_values)+1)):
                # 跳过不相关chunk
                if relevance_values[end-1] < 0:
                    continue
                # 跳过和best_segments有重叠的部分
                if any(start < seg_end and end > seg_start for seg_start, seg_end in best_segments):
                    continue
                # 判断是否超过所要找回的最大segment长度
                if total_length + end - start > overall_max_length:
                    continue

                # 计算segment的得分，如果超过best_value则更新
                segment_value = sum(relevance_values[start: end])
                if segment_value > best_value:
                    best_value = segment_value
                    best_segment = (start, end)
        
        # 如果没有找到合理的segment
        if best_segment is None or best_value < minimum_value:
            break
        
        # 否则则把segment加入到best_segments
        best_segments.append(best_segment)
        scores.append(best_value)
        total_length += best_segment[1] - best_segment[0]

    return best_segments, scores

if __name__ == "__main__":
    chunk_values = [] # 相关度得分
    irrelevant_chunk_penalty = 0.2 # 相关度计算后范围为0~1，为了将不相关的chunk筛选掉，需要将整个相关度得分-0.2，用以筛选正相关的内容
    max_length = 20 # 单个segment的最大长度
    overall_max_length = 30 # 找回的最大segment长度
    minimum_value = 0.7 # 判断是否相关的阈值

    # 构造相关度得分
    relenvance_values = [v - irrelevant_chunk_penalty for v in chunk_values]

    # 执行算法
    best_segments, scores = get_bset_segment(relenvance_values, max_length, overall_max_length, minimum_value)

    # 输出结果
    print ("Best segment indices")
    print (best_segments) 
    print ()
    print ("Best segment scores")
    print (scores)
    print ()