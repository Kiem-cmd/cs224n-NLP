### 1. Introduction to NLP 
NLP là mảng của khoa học tập trung phát triển các hệ thống tự động để hiểu và sinh ra ngôn ngữ nhiên. 
#### 1.1 Humans and language 

#### 1.2 Language and machines 
#### 1.3 A few uses of NLP 
* Machine translation 
* Question Answering and information retrieval 
* Summarization and analysis of text 
* ... 
### 2. Representing words
#### 2.1  Signifier and signified 
Signifier là kí hiệu, từ ngữ , .... còn signified là ý nghĩa mà kí hiệu hay từ ngữ đó đại diện.
- Nghĩa của một từ là vô cùng phức tạp, phát sinh từ nhu cầu giao tiếp và đạt được mục tiêu của con người trong cuộc sống.
- Ngôn ngữ cho phép con người truyền đạt những ý trưởng phức tạp qua các ký hiệu ngôn ngữ.
#### 2.2 Independent words, independent vectors 
- Có lẽ cách biểu diễn đơn giản nhất là xét các từ độc lập với nhau, có thể coi tập các từ là một set: [..., tea, ..., coffee, ..., the, ...] 

    -> Thông thường vector biểu diễn các thành phần độc lập là set_of_one_hot. Ví dụ:  
    $$v_{tea} = [0, 1 , 0 , 0 .....]$$
- Tại sao chúng ta phải biểu diễn word thành các vector ??? 

    -> Để có thể tính toán tốt hơn với chúng ??? Tuy nhiên khi tính toán với các vector_one_hot thì các từ khác nhau là khác nhau, không có khái niệm gì  về sự tương đồng hay các mối quan hệ giữa các từ:
    $$v_{tea}^Tv_{coffee} = v_{tea}^Tv_{the} = 0$$
- Vì vậy chúng ta cần có một số giải pháp !!!!
#### 2.3 Vector from annotated discrete properties 
Vậy câu hỏi đặt ra là chúng ta có nên biểu diễn từ ngữ là một tập các đặc điểm và mối quan hệ với các từ khác không ?? Nếu nên thì liệu có tài nguyên để làm việc này không ??? 

Có nhiều nguồn tài nguyên như WordNet cung cấp các thông tin như từ đồng nghĩa, trái nghĩa, mối quan hệ. Vậy có thể xây dựng các word vector như thế này dựa trên chúng ko ? 
$$v_{tea} = [0,1,0,..,1,...,1,..0,...]^T$$
* Thiếu vốn từ, rất tốn kém 
* High dimension 
### 3. Distribution semantics and Word2Vec
Một ý tưởng từ rất lâu trong ngôn ngữ học - _"You shall know a word by the company it keeps"_ hay " Bạn có thể hiểu một từ bằng các từ xung quanh nó" 

Dựa trên ý tưởng này, có thể hiểu distribution of words xung quanh một từ sẽ là cách biểu diễn từ đó. 

Chi tiết hơn thì như nào là một từ ở gần từ khác ? ngay cạnh? hay cùng một tài liệu? và làm như nào để  học nó ??
#### 3.1 Co-occurrence matrices and document contexts 
#### 3.2 Word2Vec model and objective 
#### 3.3
#### 3.4
#### 3.5 
### Appendix A continuous Bag-of-words 
### Appendix B Singular Value Decomposition 
