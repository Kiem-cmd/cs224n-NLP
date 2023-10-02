# Slide - Video

--------------------------------

## Lecture plan

1. A brief not on subword modeling 
2. Motivating model pretraining from word embeddings 
3. Model pretraining three ways 
    1. Decoders
    2. Encoders 
    3. Encoder - Decoder 
4. Interlude: what do we think pretraining is teaching ? 
5. Very large models and in-context learning 

----------------------
### 1. A brief not on subword modeling 

#### Slide 1 : Word structure and subword models 
Giả sử chúng ta có một model word embedding: 

* Với các từ bình thường : 
       * hat  > [1,0,23,42,...] 
       * learn > [...,...,......] 

*  Với các từ bị viêt sai: 

       * laern > UNK 
       * taaaasty > UNK

Rõ ràng nếu là con người thì chúng ta có thể hiểu được, vì thế nếu model ko thể hiểu được những từ bị viết sai thì sẽ mất đi rất nhiều thông tin 

Vì thế cần phải mapping nó đến một cái gì đó, và đây là một vấn đề lớn trong ngôn ngữ vì có một số nước mà ngôn ngữ của họ rất phức tạp
    
VD: Bali có tới 300 chia cho một động từ ...và nếu mỗi một cách chia lại mapping đến một vector riêng thì thật lãng phí vì chúng có nhiều điểm chung

Tôi phải có một vocab khổng lồ để lưu toàn bộ cách chia động từ ==> Sai lầm 


#### Slide 2: The byte-pair encoding algorithms 

Subword modeling in NLP bao gồm một loạt các phương pháp để suy luận về cấu trúc dưới của word level( Parts of words, character, bytes , ...) 

Mô hình phổ biến nhất hiện nay là học vocab theo parts of words. Trong quá trình training sẽ chia sequence thành các subword 

<b>Byte-pair encoding</b> là một thuật toán đơn giản hiệu quả  để định nghĩa subword vocab
*    Thuật toán bắt đầu với chỉ các character 
*  Sử dụng corpus , tìm các cặp từ phổ biến mà liền nhau "a,b" thì thêm "ab" - new subword
* Thay thế character pair bởi new subword
 và tiếp tục như thế 
* Cuối cùng ta có đc một vocab mà ở đó các subword xuất hiện rất phổ biến nhờ đó bạn có thể tạo nên các từ 

Kết quả : 
* Với common word:  nó xuất hiện đủ nhiều để bản thân nó là một subword 

       * hat  -> [......] 
       * learn -> [..........] 
* Với những từ viết sai: xuất hiện ít mà chỉ những subword của nó mới xuất hiện trong các từ khác 

       * taaaaaasty -> taa ### aaa### sty ### -> [......] [.....] [......] 

       * tương tự với các từ khác 

Từ một từ sai chúng ta học được vector từ một số subword trong từ sai đó thay vì chỉ nhận đc UNK
 
------------------------
### 2. Motivating model pretraining from word embeddings 

#### Slide 3: Motivating word meaning and context

Nghĩa của từ được xác định hoặc một phần dựa trên các từ có xu hướng xung quanh nó (context)  và không thể nghiên cứu nghĩa của từ khi đặt nó ngoài context

VD: I **record** the **record** 

Trong Word2Vec 2 từ **record** chỉ được thể hiện bởi một vector, vì thế nó là sự kết hợp giữa hai ý nghĩa của hai từ **record**  --> dễ bị fail 

Chúng ta có thể sử dụng RNN hoặc Transformer để xây dựng ý nghĩa dựa trên context một cách tốt hơn

#### Slide 4: Where we were: pretrained word embeddings 

-------------------------
# Lecture Note
-----------------------