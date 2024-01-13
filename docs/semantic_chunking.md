# Code
#### Sources: [Llamaindex - semantic chunking](https://github.com/run-llama/llama_index/blob/main/llama_index/node_parser/text/semantic_splitter.py)
```python
def _build_node_chunks(
        self, sentences: List[SentenceCombination], distances: List[float]
    ) -> List[str]:
    chunks = []
    if len(distances) > 0:
        breakpoint_distance_threshold = np.percentile(distances, 
                                                      self.breakpoint_percentile_threshold)

        indices_above_threshold = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

        # Chunk sentences into semantic groups based on percentile breakpoints
        start_index = 0

        for index in indices_above_threshold:
            end_index = index - 1

            group = sentences[start_index : end_index + 1]
            combined_text = "".join([d["sentence"] for d in group])
            chunks.append(combined_text)

            start_index = index

        if start_index < len(sentences):
            combined_text = "".join([d["sentence"] for d in sentences[start_index:]])
            chunks.append(combined_text)
    else:
        # If, for some reason we didn't get any distances (i.e. very, very small documents) just
        # treat the whole document as a single node
        chunks = [" ".join([s["sentence"] for s in sentences])]

    return chunks
```
#### Idea: 
- Chia ``sentences`` thành các ``groups`` đảm bảo mỗi ``group`` có ``similarity_score`` lớn hơn ``breakpoint_distance_threshold``.
- ``breakpoint_distance_threshold`` càng nhỏ thì sẽ có càng nhiều chunk.
- Nhận xét: 
    ```
    - Đơn giản là ném cả cuốn sách và bắt nó chia bằng 1 ``threshold``.
    - Tốc độ ảnh hưởng bởi tốc độ embedding và sẽ tốn thời gian với lượng input documents có nhiều câu (vì mỗi câu sẽ đều được tạo embedding).
    ```
