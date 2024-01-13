Benchmark: [Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
# Tổng quan 
## Đánh giá cơ bản về các task được show trên leaderboard:
1. `Bitext Mining`: tính toán sự liên quan về mặt ngữ nghĩa của cặp câu gốc-bản dịch của 1 model ngôn ngữ (bỏ qua vì không liên quan đến embedding)
2. `Classification`: Huấn luyện model trên tập train với tối đa 100 lần lặp. Input là 1 câu và trả về output là 1 nhãn có xác suất cao nhất.
3. `Clustering`: phân cụm dữ liệu
4. `Pair Classification`: Đưa vào cặp câu và trả về 1 nhãn thường được biết đến như "duplicate" hoặc "paraphrase" thể hiện mối quan hệ của 2 câu.
5. `Retrieval`: Cho 1 truy vấn và tìm văn bản có độ tương đồng cao nhất trong corpus.
6. `Reranking`: Cho 1 truy vấn và 1 tập văn bản, sắp xếp lại tập văn bản theo độ tương đồng với corpus. 
7. `STS` (Semantic Textual Similarity): Cho 2 câu và tính độ tương quan của chúng. 
8. `Summarization`

Các task cần chú trọng sẽ tập trung vào **Classification, Clustering, Retrieval, Reranking, STS**