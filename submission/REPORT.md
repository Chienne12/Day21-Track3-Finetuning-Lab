# Lab 21 — Evaluation Report

**Họ tên**: Phùng Trọng Chiến  **MSSV**: 2A202602430  **Ngày**: 07/10/2026
**Tier**: T4  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4, 14.6 GB, fp16

---

## 1. Setup

| Thiết lập | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON gồm intent, urgency, product, sentiment |
| Train / val | 225 / 25, seed 42 |
| `max_length` | 1024 — p95=98, độ dài gợi ý=256 |
| `MASK_MODE` | assistant-only |
| Epochs / max_steps | 2 epochs / ngân sách 30 optimizer steps cho mỗi run |
| Batch hiệu dụng | 1 × 16 = 16 |
| Eval đầy đủ | 50 ticket target / 15 câu regression; smoke_mode=false |

Giữ model và dataset mặc định để chạy lần đầu. max_length giữ 1024 theo tier dù NB1 gợi ý 256.

**Template có giữ khối `<think>` không?** 
Có. Trong kết quả NB1, phần suy luận bên trong thẻ `<think>` vẫn được giữ lại.

---

## 2. Mask proof (NB1)

| Kiểm tra | Kết quả |
|---|---|
| `supervised_fraction` | 0.4149, tương ứng 39/94 token ở mẫu proof |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Đoạn được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn tính loss chứa câu trả lời, không chứa ticket. Chế độ everything tính loss cả trên ticket, nên cấu hình train dùng assistant-only.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

Baseline 8 mẫu được đo trước train. Bảng dưới là lượt đánh giá đầy đủ 50 target / 15 regression, chạy lại sau train.

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3462.7 |
| (b) base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 1063.6 |
| (c) LoRA fine-tune | 0.9700 | 0.4556 | 1.0000 | 1442.7 |

**(b) có thật sự mạnh hơn (a) không?** Có. Target tăng từ 0 lên 76.5%, format tăng từ 0 lên 1. Vì vậy, fine-tune cần so với (b), không chỉ với prompt đơn giản.

**Bạn có sửa OPTIMIZED_PROMPT không?** 
Nếu cần cải thiện OPTIMIZED_PROMPT, tôi sẽ thêm các quy tắc để xác định mức độ cấp bách .
Xác định mức độ cấp bách dựa trên yêu cầu về thời gian của khách hàng:
- “Lúc nào cũng được” hoặc “chỉ hỏi thăm” → thấp.
- “Khẩn cấp” hoặc “cần ngay lập tức” → cao.
Không nâng mức độ cấp bách chỉ vì sản phẩm bị lỗi hoặc đang chờ hoàn tiền.

Việc này sẽ giúp cải thiện prompt giúp xác định rõ mức độ khẩn cấp rõ ràng hơn. Nó sẽ giúp phân biệt giữa một yêu cầu khẩn cấp và một lời phàn nàn .
---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | target (NB5 §4) | s | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| correct | text-linear | 16 | 32464896 | 1e-4 | 0.6259 | 0.97 | 427.1 | 8.78 |
| attn_only | q,v | 283 | 32456704 | 1e-4 | 0.5386 | 0.97 | 280.3 | 8.79 |
| wrong_lr | text-linear | 16 | 32464896 | 1e-5 | 1.5702 | 0.00 | 415.9 | 8.78 |
| qlora | text-linear, base 4-bit | 16 | 32464896 | 1e-4 | 0.7058 | 0.94 | 490.1 | 3.86 |

Cả bốn run cùng ngân sách 30 steps.

**4.1 — attn_only có cùng số tham số huấn luyện với correct. Trên tập target nó thắng, thua, hay hòa? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về rank so với vị trí gắn adapter?**

Hai run hòa ở target=0.97, dù attn_only có loss thấp hơn: 0.5386 so với 0.6259. Rank 283 của attn_only được code tính để khớp ngân sách, chênh khoảng 0.025% tham số. Vì vậy, loss thấp hơn chưa chứng minh phân loại tốt hơn. Trong lượt này, đổi vị trí và tăng rank để khớp số tham số không làm điểm target cao hơn.

**4.2 — wrong_lr chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

wrong_lr giảm LR từ 1e-4 xuống 1e-5. Loss tại các mốc log giảm từ 2.163 xuống 1.119, còn correct giảm từ 2.163 xuống 0.02761 ở mốc cuối. Tuy loss vẫn giảm, wrong_lr có target và format bằng 0. Nếu xem loss giảm là bằng chứng model đã làm được bài toán thì sẽ kết luận sai.

**4.3 — qlora tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo có ủng hộ khuyến nghị “không dùng QLoRA cho dòng model này” không?**

VRAM giảm 8.78 → 3.86 GB, tiết kiệm 4.92 GB, khoảng 56%. Đổi lại, target giảm 97% → 94% và thời gian train tăng 427.1 → 490.1 giây. Số đo này ủng hộ việc dùng LoRA 16-bit ở cấu hình đang chạy nếu GPU đủ bộ nhớ.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: FAILED

`target Δ = +0.2050` · `regression Δ ≈ -0.3356` · `valid_trace_rate = 0.0000`

Nhìn riêng target, fine-tune tốt hơn: 97% so với 76.5%, tăng 20.5 điểm phần trăm. Nhưng regression giảm từ 79.11% xuống 45.56%, tức khoảng 33.56 điểm phần trăm, vượt mức cho phép là 2 điểm. Vì vậy, code trả FAILED dù format vẫn bằng 1. Fine-tune cũng chậm hơn baseline tối ưu: 1442.7 so với 1063.6 ms/mẫu. Với tiêu chí của lab, tăng điểm phân loại chưa đủ để chọn bản fine-tune.

---

## 6. Định tính — bắt buộc có cả ca THUA

Nhãn viết theo thứ tự **intent / urgency / product / sentiment**. Log chưa in câu trả lời baseline theo từng ticket.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1, index 3 | VN804124: bình giữ nhiệt, chưa thấy tiền, khi nào tiện | hoan_tien / thap / bình giữ nhiệt / tich_cuc | Log không in output (b) | urgency=trung_binh; điểm 0.75 | Sai urgency |
| 2, index 5 | DH249548: nồi chiên không dầu, thiếu phụ kiện, khi nào tiện | san_pham_loi / thap / nồi chiên không dầu / trung_tinh | Log không in output (b) | urgency=trung_binh; điểm 0.75 | Sai urgency |
| 3, index 12 | VN613097: áo khoác gió, bị lỗi, khi nào tiện | san_pham_loi / thap / áo khoác gió / tich_cuc | Log không in output (b) | urgency=trung_binh; điểm 0.75 | Sai urgency |
| 4, index 47 | DH936478: ốp lưng điện thoại, shipper không gọi, hỏi cho biết | van_chuyen / thap / ốp lưng điện thoại / tich_cuc | Log không in output (b) | Điểm 1.00, đúng bốn trường | FT đúng toàn bộ |
| 5, index 48 | DH734695: ốp lưng điện thoại, giá bao nhiêu, mong phản hồi | hoi_thong_tin / trung_binh / ốp lưng điện thoại / trung_tinh | Log không in output (b) | Điểm 1.00, đúng bốn trường | FT đúng toàn bộ |

**Có mẫu chung nào ở các ca FT thua không?** Ba ca FT sai đều có câu “Khi nào tiện”: nhãn urgency=thap nhưng model trả trung_binh. Chưa xác nhận được ca thua baseline vì thiếu câu trả lời (b) của cùng ticket.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Từ output đã chạy, bản fine-tune chưa đạt tiêu chí để thay base model có prompt tốt. Điểm phân loại tăng từ 76.5% lên 97%, nhưng regression giảm quá ngưỡng và thời gian trả lời cũng tăng. Nếu chỉ nhìn target thì dễ bỏ qua phần giảm này. Mốc prompt tối ưu cũng cho thấy có thể cải thiện kết quả đáng kể trước khi cần train.

Lab giúp hiểu rõ thứ tự kiểm tra: đọc mask trước để biết loss tính trên đoạn nào, đọc baseline để biết mốc cần vượt, rồi xem target và regression cùng nhau. Trong các đối chứng, LR tạo khác biệt rõ ở ngân sách train đã chạy: wrong_lr có loss giảm nhưng target vẫn bằng 0. Vị trí adapter chưa tạo khác biệt về target giữa correct và attn_only khi số tham số gần bằng nhau. QLoRA giảm bộ nhớ nhưng điểm target thấp hơn. Nếu thử tiếp, có thể kiểm tra các câu regression sai rồi bổ sung dữ liệu phổ thông và đo lại trên cùng eval.

**Ba điều tôi học được:**

1. Đoạn loss phải đọc được: everything chứa cả ticket, assistant-only che câu hỏi thành công.
2. Prompt tốt tự đạt 76.5% target; cần dùng nó làm mốc so sánh.
3. Loss giảm chưa đủ: wrong_lr vẫn có target bằng 0.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** lấy output từng mẫu để so với baseline, phân tích các câu regression sai, rồi thử replay dữ liệu phổ thông với cùng ngân sách train.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (data/CUSTOM_DATASET.md)
- [ ] B3 reasoning-trace collapse, hai MASK_MODE và valid_trace_rate
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — chưa có link
