## Xem xét các phương pháp trên MTEB leaderboard:
1. e5
2. instructor embedding
3. sentence-t5, sentences-transformer mpnet

3 phương pháp trên đứng top cao trong 5 task trên nên chỉ tập trung vào 3 phương pháp này, 1 số phương pháp khác như SGPT, GTR sẽ được nghiên cứu sau.

# Đánh giá chi tiết về từng phương pháp:
## [E5(EmbEddings from bidirEctional Encoder rEpresentations)](https://arxiv.org/pdf/2212.03533.pdf)
E5 tập trung vào việc finetune model để tập trung vào các task STS, Reranking. Tập trung chủ yếu vào việc thu thập, filter data cho 2 bước finetune model.
1. Data pair unlabeled: Mỗi sample data gồm 1 cặp data có độ tương đồng cao (title - post, post - comment).
```python
{
    'query' : 'big little lies season 2 how many episodes',
    'relevant' : 'Big Little Lies (TV series) series garnered several accolades. It received 16 Emmy Award nominations and won eight, including Outstanding Limited Series and acting awards for Kidman, Skarsgård, and Dern. The trio also won Golden Globe Awards in addition to a Golden Globe Award for Best Miniseries or Television Film win for the series. Kidman and Skarsgård also received Screen Actors Guild Awards for their performances. Despite originally being billed as a miniseries, HBO renewed the series for a second season. Production on the second season began in March 2018 and is set to premiere in 2019. All seven episodes are being written by Kelley'
}
```
- Handcraft filter: Lọc đi các sample data quá dài hoặc quá ngắn, lọc các sample có perplexity quá cao.
- Consistency-based filter: Thực hiện training một model trên toàn bộ pair data bằng Contrastive loss. Duyệt qua tất cả sample pair data có dạng [query, relevant], với mỗi sample thì kiểm tra xem relevant có đứng top 1 trong 1 triệu sample được dùng để thực hiện xếp hạng độ tương quan với query không. Các sample mà relevant đứng top 1 trong kết quả xếp hạng thì sẽ được giữ lại.

Thực hiện finetune model dựa trên Contrastive loss.

2. Labeled data: Mỗi sample data gồm bộ 3 [query, pos, neg], trong đó pos là câu có độ tương đồng cao với query còn neg là câu đánh giá là hard negative(câu được đánh giá là độ tương đồng rất cao nhưng không phải kết quả cuối).
```python
{
    'query': 'big little lies season 2 how many episodes?',
    'pos':  'Big Little Lies (TV series) series garnered several accolades. It received 16 Emmy Award nominations and won eight, including Outstanding Limited Series and acting awards for Kidman, Skarsgård, and Dern. The trio also won Golden Globe Awards in addition to a Golden Globe Award for Best Miniseries or Television Film win for the series. Kidman and Skarsgård also received Screen Actors Guild Awards for their performances. Despite originally being billed as a miniseries, HBO renewed the series for a second season. Production on the second season began in March 2018 and is set to premiere in 2019. All seven episodes are being written by Kelley',
    'neg':  'Little People, Big World final minutes of the season two-A finale, "Farm Overload". A crowd had gathered around Jacob, who was lying on the ground near the trebuchet. The first two episodes of season two-B focus on the accident, and how the local media reacted to it. The first season of "Little People, Big World" generated solid ratings for TLC (especially in the important 18–49 demographic), leading to the show\'s renewal for a second season. Critical reviews of the series have been generally positive, citing the show\'s positive portrayal of little people. Conversely, other reviews have claimed that the show has a voyeuristic bend'
}
```
Finetune model trên **Labeled Data** sử dụng Triplet loss. 3 dataset được sử dụng là **NLI (Natural Language Inference)**, **MS-MARCO passage ranking dataset**, **NQ (Natural Questions) dataset**

## [Instructor embedding](https://github.com/HKUNLP/instructor-embedding)
Cải tiến dựa trên Labeled data trong đó mỗi phần query, pos, neg, đều được thêm một prompt để tạo context cho các câu sau.
```python
{
    'query': ['Represent the Wikipedia question for retrieving relevant documents;',
              'big little lies season 2 how many episodes'],
    'pos':   ['Represent the Wikipedia document for retrieval;',
              'Big Little Lies (TV series) series garnered several accolades. It received 16 Emmy Award nominations and won eight, including Outstanding Limited Series and acting awards for Kidman, Skarsgård, and Dern. The trio also won Golden Globe Awards in addition to a Golden Globe Award for Best Miniseries or Television Film win for the series. Kidman and Skarsgård also received Screen Actors Guild Awards for their performances. Despite originally being billed as a miniseries, HBO renewed the series for a second season. Production on the second season began in March 2018 and is set to premiere in 2019. All seven episodes are being written by Kelley'],
    'neg':   ['Represent the Wikipedia document for retrieval;',
              'Little People, Big World final minutes of the season two-A finale, "Farm Overload". A crowd had gathered around Jacob, who was lying on the ground near the trebuchet. The first two episodes of season two-B focus on the accident, and how the local media reacted to it. The first season of "Little People, Big World" generated solid ratings for TLC (especially in the important 18–49 demographic), leading to the show\'s renewal for a second season. Critical reviews of the series have been generally positive, citing the show\'s positive portrayal of little people. Conversely, other reviews have claimed that the show has a voyeuristic bend']
}
```

## sentence-t5, sentences-transformer mpnet 
Đều sử dụng Contrastive loss và SimCSE trên 1 lượng data lớn (khoảng 1 tỷ sample). [Tham khảo](https://huggingface.co/sentence-transformers/all-mpnet-base-v2#background). Ngoài ra không có điểm đặc sắc.

# Một số đánh giá:
1. Muốn đạt top 1 thì cần cả data unlabeled và labeled data. Hiện nguồn data unlabeled có thể xây dựng từ 2 nguồn chính là data CC, data crawl (báo chí, forums) theo cặp header-content, title-post, post-comment.
2. Có thể áp dụng instruction data cho cả bước unlabeled.
3. Có thể sử dụng ChatGPT để xây dựng labeled data hay không?

# Thông tin về data
E5 data: 
![Thông tin](https://cdn.discordapp.com/attachments/1109339491807809656/1195727243171868713/image.png?ex=65b50ac6&is=65a295c6&hm=93d36da2deae5f2d473d34d8464250b2100e249acca48d7cb621dac17eb354a1&)
Instructor data:
![Thông tin](https://instructor-embedding.github.io/static/images/instruction_examples.png)

