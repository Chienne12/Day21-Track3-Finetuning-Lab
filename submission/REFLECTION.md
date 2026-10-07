# Reflection — Lab 21

**1. Điều gì làm bạn ngạc nhiên nhất?**

Chỉ cần viết prompt rõ hơn, base model đã đạt 76.5% target mà chưa cần train. Tôi thấy đây là bước đơn giản nên thử trước khi bỏ thời gian fine-tune.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

NB4 mất khoảng 22.6 phút vì phải train ba cấu hình đối chứng. Tôi chưa tính hết thời gian cho phần so sánh này, ban đầu chỉ chú ý đến run train chính.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng nghĩ loss giảm là model đã tốt hơn. Nhưng run wrong_lr có loss giảm mà target vẫn bằng 0, nên giờ tôi muốn kiểm tra câu trả lời thực tế trước khi kết luận.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI để hiểu thứ tự notebook, đọc log và viết lại report. Có lúc AI hướng dẫn đo lại baseline sau train mà chưa nhắc rõ giới hạn của cách làm này; mốc đầy đủ đó không thể được gọi là đã đóng băng trước train.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Tôi sẽ lấy vài ví dụ thật, thống nhất thế nào là trả lời đúng rồi thử base model với prompt rõ ràng. Nếu kết quả đã đủ tốt, tôi có thể giải quyết bài toán mà chưa cần fine-tune.
