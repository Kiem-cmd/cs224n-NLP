

### 1. Dependency Grammar và Dependency Structure  

* Parse trees trong NLP được dùng để phân tích cấu trúc ngữ pháp của câu. Có hai loại cấu trúc chính được sử dụng: 
    
    * Constituency Structures: sử dụng cụm từ để tổ chức các thành phần lồng nhau. Sẽ đề cập ở phần sau.
    * Dependency Structures: Cho biết từ nào phụ thuộc(modify/arguments) vào từ khác. Sự phụ thuộc này được miêu tả bằng mũi tên đi từ từ này đến từ khác. Và sự phụ thuộc này tạo thành một cây cấu trúc(tree structure)

#### 1.1 Dependency Parsing 

* Dependency Parsing là nhiệm vụ phân tích cấu trúc cú pháp phụ thuộc của một câu đầu vào S. Đầu ra là một cây phụ thuộc trong đó các từ của câu đầu vào được kết nối bằng quan hệ phụ thuộc (dependency relations). 

    * Input: $S = \{w_0,w_1,....\}$ trong đó $w_0$ là từ đầu tiên
    * Output: Graph $G$ - biểu đồ dependency tree
    
* Nhiều biến thể khác nhau của phương pháp dependency - based  trong những năm gần đây ( bao gồm cả mạng Neural) sẽ được trình bày sau. 

* Có hai vấn đề con trong Denpendency parsing: 
    1. _Learning_ :  Cho training set $D$ gồm các câu được chú thích bằng dependency-graphs, tạo ra model $M$ có thể được sử dụng để phân tích câu mới 
    
    2. _Parsing_: Cho mô hình $M$ và một câu $S$ hãy suy ra optimial dependence- graph $D$ for $S$ theo $M$

#### 1.2 Transition-Based Dependency Parsing

Transition(biến chuyển) -Based Dependency Parsing  dựa trên state machine xác định các transition có thể có để tạo mapping từ câu đầu vào đến dependency tree. - Có thể hiểu là thực hiện một loạt các chuyển đổi, mỗi chuyển đổi sẽ thay đổi cấu trúc của chuỗi

* _The learning problem_: Tạo ra một mô hình có thể dự đón quá trình transition tiếp theo của state machine dựa vào transition history. 
* _The parsing problem_: Xây dựng trình tự chuyển tiếp tối ưu cho câu đầu vào, được cho bởi model cho trước. Hầu hết các transition-based không sử dụng formal grammar  - tập hợp quy tắc xác định cấu trúc hợp lệ của một ngôn ngữ 
 
#### 1.3 Greedy Deterministic Transition - Based Parsing ( Chuyển đổi xác định tham lam) 

* Hệ thống dược Nivre giới thiệu vào năm 2003 và hoàn toàn khác biệt với các phương pháp khác tại thời điểm đó. 

* Hệ thống này là một state machine, bao gồm các state và transitions giữa các state đó. Mô hình tạo ra một chuỗi các transition từ state 1 đến 1 trong nhiều state cuối. 

* <b>States:</b> 
    
    * Với mỗi câu: $S = \{w_0,w_1, ... \}$ một state: $c = (\sigma,\beta,A)$
        * a stack $\sigma$ của $w_i$
        * a buffer $\beta$ của $w_i$ 
        * $A$ có form $(w_i,r,w_j)$ trong đó $w_i,w_j \in S$ và $r$ mô tả mối quan hệ phụ thuộc  
    
* Với mọi câu $S = \{w_0,w_1,...\}$ 
    1. State ban đầu $c_0$ có dạng ($[w_0]_\sigma,[w_1,,...,w_n]_\beta,\varnothing)$
    chỉ có ROOT($w_0$) nằm trên stack $\sigma$ còn tất cả nằm trong buffer $\beta$ và vì là state ban đầu nên chưa có transition.

    2. State cuối cùng có dạng $(\sigma,[ ]_\beta,A)$
* Có ba loại transitions giữa các state:
    
    * SHIFT: chuyển từ đầu tiên từ buffer lên stack(với điều kiện buffer ko rỗng)
    * LEFT-ARC: Thêm một arc $(w_j,r,w_i)$ vào tập $A$ trong đó $w_i$ là từ thứ hai từ trên xuống của stack và $w_j$ là từ đầu tiên của stack.. Xóa $w_i$ khỏi stack
    * RIGHT-ARC:   Thêm một arc $(w_i,r,w_j)$ vào tập $A$ trong đó $w_i$ là từ thứ hai từ trên xuống của stack và $w_j$ là từ đầu tiên của stack.. Xóa $w_j$ khỏi stack


    * ![Alt text](https://github.com/Kiem-cmd/cs224n-NLP/blob/main/Lecture%204%20Dependency%20Parsing/image/image.png?raw=True)

* Example:
    * <b>Input</b>: $S = $ "Book me the morning flight" 
    * <b>Label</b>: SHIFT, LEFT-ARC, SHIFT, SHIFT, SHIFT 
    * State 1:
        * buffer: book|me|the|morning|flight
        * stack: ROOT 
    * State 2: 
        * buffer: me|the|morning|flight 
        * stack: book|ROOT 
    * State 3: 
        * buffer: me|the|morning|flight 
        * stack: ROOT 
        * A: ROOT -> book

    => Cứ thế đến hết thì được một tập A đó là dependency-tree

#### 1.4 Neural Dependency Parsing 

* Mặc dù có nhiều mô hình DL để phân tính dependency parsers nhưng mô hình này tập trung cụ thể vào các trình phân tích neural dependency parsers dựa trên quá trình transition. Loại mô hình này đã chứng minh hiệu suất tương đương và hiệu quả tốt hơn đáng kể so với mô hình tranditional-based discriminative dependency parsers. Sự khác biệt chính so với mô hình trước đó là Dense và Sparse 

* Mục đích của model là dự đoán trình tự transition từ state ban đầu đến state cuối cùng. 
Tức là có cấu hình của state $c = (\sigma,\beta,A) $ chúng ta sẽ dự đoán $T \in (SHIFT, LEFT-arc, RIGHT- arc)$ 

* <b>Lựa chọn tính năng đầu vào(feature selection)</b>:
    * $S_{word}$ = Vector biểu diễn của từ của một vài từ ở đầu của stack và buffer và 

    * $S_{tag}$ = POS tag của cho các từ trong $S_{word}$ - Ví dụ = like = động từ(verb), word = danh từ (n) 

    * $S_{label}$ là các nhãn arc tương ứng của các từ - Ví dụ : aux, conj, nmod, ... 

    * ![Alt text](https://github.com/Kiem-cmd/cs224n-NLP/blob/main/Lecture%204%20Dependency%20Parsing/image/image-2.png?raw=True)

* <b>FeedForward Neural Network</b>: 

    * Các feature $[x^w, x^t, x^l]$ sẽ được nhân với các ma trận trọng số $[W^w, W^t, W^l]$ tương ứng. 
    * ![Alt text](https://github.com/Kiem-cmd/cs224n-NLP/blob/main/Lecture%204%20Dependency%20Parsing/image/image-3.png?raw=True)