# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603798
- **Họ và tên:** Nguyễn Đức Thắng *(cần kiểm tra lại theo hồ sơ nộp bài)*
- **SEED:** 2133
- **Notebook nguồn:** [Colab Day 01](https://colab.research.google.com/drive/1UPJFVjZgOuu78ZWCwhNG1FL8_oA9YiEa)
- **Tình trạng:** Bản nháp có số liệu đã chạy đến Phần 3; chưa phải báo cáo nộp hoàn chỉnh.

## 1. Bằng chứng

| Tệp | Có? | Ghi chú |
| --- | --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ | Notebook trên Colab có output đến Phần 3; chưa xác minh đã xuất tệp đúng tên. |
| [`lab1_report.json`](lab1_report.json) — do ô “Báo cáo cuối” sinh ra | ☐ | Output ô báo cáo ghi đã tạo tệp JSON, nhưng chưa xác minh tệp đã được lưu vào thư mục nộp. |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ | Chưa thấy ảnh tính tay; cần tự tính trên giấy, chụp ảnh và nộp. |

**Lưu ý bảo mật:** Trước khi nộp, kiểm tra notebook không chứa API key trong code, prompt, kết quả in hoặc output đã lưu. Nếu key từng lộ trong ảnh, cần thu hồi và thay key mới.

## 2. Kết quả chính

Dữ liệu sau khi bỏ trùng có **5.169 tin** (653 spam, 4.516 ham), chia thành **3.619 train**, **775 validation** và **775 test**. Tập SUB gồm 40 tin cân bằng (20 spam, 20 ham). Số liệu dưới đây lấy từ các output đã lưu trong Colab, không phải số liệu mô phỏng.

| Mô hình | F1 test (775 tin) | F1 SUB (40 tin) | p95 ms/tin | USD/1.000 tin | Train (giây) | Tin/giây cả lô |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + Logistic Regression | 0.899 | 0.919 | 0.07 | 0 | 3.7 | 37.223 |
| 2. BoW + MLP | 0.933 | 0.947 | 0.31 | 0 | 14.0 | 18.728 |
| 3a. MiniLM + kNN | 0.886 | 1.000 | 127.28 | 0 | 103.0* | 21 |
| 3b. MiniLM + Logistic Regression | 0.942 | 0.974 | 37.80 | 0 | 103.8* | 57 |
| 3c. MiniLM + MLP head | 0.942 | 0.974 | 34.66 | 0 | 104.1* | 57 |
| 4a/4b/4c. LLM zero-shot/few-shot | — | — | — | — | — | — |

*Thời gian ở các hàng 3a–3c bao gồm khoảng 103 giây encode tập train; encoder MiniLM được đóng băng, không được huấn luyện lại trong thí nghiệm. Chi phí API bằng 0 đối với mô hình chạy cục bộ trong notebook **không** có nghĩa chi phí vận hành CPU/GPU bằng 0. Số đo p95 phụ thuộc môi trường Colab và cần đo lại ở môi trường triển khai.*

**Nhận xét:** MiniLM + LogReg và MiniLM + MLP đạt F1 test cao nhất (0.942), nhưng chỉ hơn BoW + MLP 0.009 điểm F1, trong khi độ trễ lớn hơn nhiều. kNN đạt F1 = 1.000 trên SUB nhưng chỉ 0.886 trên toàn bộ test, cho thấy không nên xếp hạng mô hình chỉ bằng 40 mẫu. **LLM chưa có số đo:** đã kiểm tra cấu hình Groq và một lần gọi chào hỏi, nhưng chưa có kết quả phân loại SUB được lưu.

## 3. Dự đoán và thực tế

**Dự đoán 1 — F1 của BoW + LogReg (`ml_f1`).** Tôi dự đoán F1 khoảng **0.85** và Precision cao hơn Recall do dữ liệu mất cân bằng. Kết quả đạt **F1 = 0.899**, **Precision = 0.973**, **Recall = 0.835**. Dự đoán về tương quan Precision/Recall đúng, nhưng F1 cao hơn khoảng **0.049** so với tôi ước lượng. Điều này cho thấy một bộ phân loại tuyến tính dùng từ khóa vẫn có thể làm khá tốt trên bộ SMS test thông thường.

**Dự đoán 2 — Tin nhắn “ngụy trang” (`emb_tricky`).** Tôi dự đoán **Encoder + LogReg sẽ đúng nhiều hơn BoW + LogReg** vì embedding biểu diễn ý nghĩa câu tốt hơn. Thực tế **cả hai đều đúng 5/6 tin** và cùng bỏ sót tin Spam: “Last chance! Your account rewards expire tonight, confirm details to keep them”. Dự đoán chưa khớp vì hiểu ngữ nghĩa tốt hơn không đồng nghĩa bộ phân loại luôn nhận đúng mọi kiểu lừa đảo; hơn nữa sáu mẫu quá ít để kết luận mô hình nào mạnh hơn.

**Dự đoán 3 — Learning rate (`ml_lr`).** Tôi dự đoán `lr=0.001` làm loss giảm đều, còn `lr=50` sẽ “giảm mạnh hoặc tăng mạnh”. Kết quả trên biểu đồ cho thấy `lr=0.001` giảm rất chậm và có **loss cuối ≈ 0.628**; `lr=0.5` có **loss cuối ≈ 0.080**; riêng `lr=50` có một đỉnh loss rất cao ở giai đoạn đầu rồi giảm nhanh, **loss cuối ≈ 0.005**. Dự đoán ban đầu về `lr=50` quá mơ hồ: bước cập nhật lớn gây dao động ban đầu nhưng trong lần chạy cụ thể này vẫn giảm loss mạnh về cuối. Không nên kết luận mọi learning rate lớn đều làm mô hình không hội tụ.

**Giới hạn nhật ký:** Hai ô `dl_curve` và `dl_relu` từng ghi dự đoán chỉ bằng “2” và “6”, nên không đủ thông tin để đối chiếu ý nghĩa. Các lần sau tôi cần ghi cả kết quả dự kiến lẫn lý do trước khi chạy.

## 4. Phân tích 50 case khó

**Trạng thái: CHƯA CÓ KẾT QUẢ THỰC NGHIỆM.** Trong phiên bản notebook hiện có, các ô `predict("hard_eval", ...)`, chạy 50 case và đọc lỗi chưa được thực thi. Vì vậy, chưa thể xác định trung thực model nào tụt F1 nhiều nhất hoặc khẳng định hai case dưới đây đã bị dự đoán sai.

**Các tình huống cần kiểm tra sau khi chạy:**

- **Case #1 — `ham_looks_spam` (nhãn ham):** “Congrats!!! You WON the bet lol, I owe you a beer. Call me later”. Một bộ lọc BoW có thể nhận nhầm thành Spam do các từ “Congrats”, “WON”, “Call” dù ngữ cảnh chỉ là bạn bè nói chuyện. Cần đối chiếu cột dự đoán thực tế của từng model.
- **Case #29 — `obfuscation` (nhãn spam):** “C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m”. Các từ bị biến đổi bằng chữ số làm token khác với từ trong từ điển train, nên có thể khiến BoW bỏ sót Spam. Cần đối chiếu kết quả thực tế để xác nhận.

**Bổ sung bắt buộc sau khi chạy stress test:**

- F1 50 case của từng model: **[điền từ output `summary`]**.
- Model tụt nhiều nhất so với F1 test thông thường và mức giảm: **[chưa có số liệu]**.
- Nhóm lỗi khó nhất của BoW, Embedding và LLM: **[chưa có số liệu]**.
- Hai case *thực sự dự đoán sai*, tên model và nhãn đoán: **[thay hai ví dụ giả thuyết ở trên bằng kết quả đã kiểm chứng]**.

Bộ 50 case được viết có chủ đích để kiểm tra điểm yếu của mô hình; nó hữu ích cho stress test nhưng không thay thế được tập dữ liệu đại diện lưu lượng SMS thực tế.

## 5. Thử thách cá nhân

Với **SEED = 2133**, đề được gán theo `SEED % 6 = 3`, tương ứng **Thử thách #4**:

> **Embedding:** Chạy `knn_predict` với **k = 1, 5, 25**. Báo F1. Vì sao k quá nhỏ hoặc quá lớn đều có hại?

**Trạng thái: Chưa thực hiện.** Ô code thử thách trong notebook hiện còn trống. Phần này được giảng viên quy định **không dùng AI để làm thay**, vì vậy tôi sẽ tự viết và chạy code, sau đó ghi lại:

| k | F1 test | Nhận xét của tôi |
| ---: | ---: | --- |
| 1 | [tự đo] | [tự điền] |
| 5 | [tự đo] | [tự điền] |
| 25 | [tự đo] | [tự điền] |

**Kết luận cá nhân sau thí nghiệm:** [tự phân tích sự khác nhau và giải thích bằng lời của mình].

## 6. Phương án cho sản phẩm thật

Nhà mạng cần xử lý **10 triệu SMS/ngày**, tương đương khoảng **116 tin/giây** trung bình, với SLA **p95 dưới 50 ms/tin**. Theo phép đo hiện tại:

- **Đạt ngưỡng p95:** BoW + LogReg (**0.07 ms**), BoW + MLP (**0.31 ms**), MiniLM + LogReg (**37.80 ms**), MiniLM + MLP (**34.66 ms**).
- **Không đạt ngưỡng:** MiniLM + kNN (**127.28 ms**).
- **Chưa thể đánh giá:** các mô hình LLM vì chưa có kết quả độ trễ hoặc chi phí.

**Tôi chọn BoW + MLP làm mô hình xử lý chính** vì F1 test = **0.933**, p95 = **0.31 ms/tin** và thông lượng thử nghiệm cả lô khoảng **18.728 tin/giây**. Mô hình này có độ trễ rất thấp, đủ dư địa so với yêu cầu trung bình 116 tin/giây và không cần API trả tiền cho mỗi tin. Đánh đổi là chất lượng kém hơn MiniLM + LogReg khoảng **0.009 F1** trên test, đồng thời BoW dễ bị ảnh hưởng bởi cách viết né từ khóa và thiếu ngữ cảnh.

**Phương án lai (router) đề xuất:** BoW + MLP xử lý phần lớn tin nhắn. Với những trường hợp xác suất Spam gần ngưỡng quyết định, ví dụ **0.3–0.7**, chuyển sang MiniLM + LogReg để đánh giá lại. Những trường hợp vẫn không chắc chắn hoặc có rủi ro chặn nhầm tin OTP nên được xử lý thận trọng, có thể yêu cầu kiểm tra bổ sung. Router cần được benchmark riêng; không thể suy ra p95 toàn hệ thống chỉ từ p95 của từng mô hình thành phần. Nếu dùng MiniLM + LogReg một mình, thông lượng đo được **57 tin/giây mỗi tiến trình** còn thấp hơn nhu cầu trung bình, nên phải tối ưu hoặc triển khai nhiều worker, có dự phòng giờ cao điểm.

**Chi phí LLM:** Chưa có USD/1.000 tin trong notebook nên **chưa thể ước tính chi phí tháng bằng số thực nghiệm**. Nếu đo được mức giá **C USD/1.000 tin**, chi phí cho 10 triệu tin/ngày trong 30 ngày xấp xỉ **300.000 × C USD/tháng**, chưa tính hạ tầng, thuế và các chi phí khác. Không được giả định LLM là miễn phí chỉ vì cột giá chưa được điền.

## 7. AI log và điều đã học

| Phần | Tôi hỏi AI gì | AI hỗ trợ gì | Tôi đã đối chiếu / còn cần xác minh gì |
| --- | --- | --- | --- |
| Kiến thức nền & Phần 0 | Data leakage khi tin nhắn trùng giữa train/test; cách vector hóa text và tạo ma trận 0 | Giải thích nguyên lý rò rỉ dữ liệu, Binary BoW, `np.zeros` và cách chia train/test | Đối chiếu với phần loại trùng của notebook và kích thước train/val/test. |
| Phần 1 | Gradient Descent; cách trả lời Thí nghiệm 1.A, 1.B; cách phân tích tin Spam bị đoán thành Ham | Gợi ý hướng giải thích trọng số của từ `free`, Learning Rate, Precision/Recall/F1 và giới hạn BoW | Xem output F1 = 0.899, tin `thesmszone.com...` bị bỏ sót và biểu đồ loss của ba learning rate; không thay số đo. |
| Phần 2 | Cách đọc biểu đồ Train Loss/Validation Loss ở Thí nghiệm 2.A | Giải thích Overfitting, điểm có Validation Loss thấp nhất và cách so sánh MLP với LogReg | Đối chiếu epoch index 6, F1 MLP = 0.933; cần hoàn chỉnh câu trả lời bằng lời riêng. |
| Phần 3 | Cách dự đoán ở Thí nghiệm 3.B; câu hỏi về Embedding, Cosine Similarity, kNN và Bắt lỗi 2 | Giải thích tại sao Encoder có thể phân loại theo nghĩa và tại sao top-k phải dùng cosine, sắp xếp giảm dần | Đối chiếu output 3.B: hai mô hình cùng đúng 5/6; F1 của 3b và 3c cùng 0.942. |
| Phần 4 | Cách cấu hình Groq key qua Secrets và triển khai các hàm gọi API, prompt, đọc nhãn | Hướng dẫn bảo mật API key và cung cấp ví dụ code tham khảo cho `call_openai`, `build_classification_prompt`, `parse_label`, `llm_classify_batch` | Đã có lần gọi chào hỏi thành công trong Colab; **chưa có test PASS và kết quả thí nghiệm 4.A/4.B được lưu**. Phải tự kiểm tra trước khi nộp. |
| Báo cáo | Nhờ AI điền báo cáo Markdown theo kết quả Colab | Hỗ trợ tổng hợp số liệu, tổ chức phân tích và đánh dấu phần chưa đủ bằng chứng | Tôi cần kiểm tra lại tên, bổ sung kết quả stress test, tự hoàn thành thử thách cá nhân và xác nhận các tệp đính kèm. |

**Điều tôi học được:** Trước đây tôi nghĩ một mô hình có Accuracy/F1 cao trên test đã đủ tốt, nhưng nay tôi hiểu rằng dữ liệu trùng lặp, lệch lớp và các tin nhắn ngụy trang có thể khiến đánh giá trở nên quá lạc quan. Tôi cũng thấy rằng Embedding có thể cải thiện chất lượng biểu diễn nhưng không nhất thiết làm dự đoán nhanh hơn, còn một hệ thống thực tế phải cân nhắc cùng lúc F1, p95, thông lượng và chi phí. Quan trọng nhất, cần phân biệt **dự đoán trước thí nghiệm** với **nhận xét sau khi đã xem kết quả**, và chỉ báo cáo những số liệu đã đo.

---

**Cần hoàn tất trước khi nộp:** (1) chạy Phần 4 và điền kết quả LLM; (2) chạy 50 case khó, thay giả thuyết bằng những lỗi thực tế; (3) tự làm thử thách #4; (4) hoàn thành tính tay và chụp `hand_calc.jpg`; (5) xuất `day01_lab.ipynb`, `lab1_report.json`; (6) kiểm tra họ tên chính thức và xóa mọi API key lộ trong output.
