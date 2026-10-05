# BÁO CÁO THỰC HÀNH: SENSOR REALITY SPRINT (TOPIC T1)
**Họ và tên:** Nguyễn Minh Quyền  
**MSSV:** 2A202602438  
**Chủ đề:** T1 - Camera Degradation Health Score  
**Nền tảng mục tiêu:** Camera trước trên xe tự hành ADAS (Front-facing Automotive Camera)  
**Repository nhóm:** `K4-Track4-Day04-BOT-Sensor-Reality-Sprint` ([GitHub Link](https://github.com/ngminhquyen2710-sketch/K4-Track4-Day04-BOT-Sensor-Reality-Sprint))

---

## 1. Problem (Bài toán & Hiện tượng lỗi thực tế)
* **Hệ thống & Cảm biến:** Hệ thống nhận diện chướng ngại vật phía trước (Forward Object Detection) sử dụng Camera quang học chuẩn ô tô.
* **Failure thực tế:** Trong điều kiện vận hành thực địa, camera thường xuyên gặp các suy thoái quang học nghiêm trọng:
  * Ống kính mờ sương, đọng nước mưa, hoặc rung lắc động cơ gây **Defocus & Motion Blur**.
  * Góc chiếu mặt trời trực diện lúc bình minh/hoàng hôn gây lóa trắng cảm biến (**Sun Glare / Severe Overexposure**).
* **Vấn đề kỹ thuật:** Các mạng nơ-ron sâu (CNN/Transformer) mặc định coi mọi pixel đầu vào là chính xác. Khi cảm biến suy giảm, mô hình bị sụt giảm tự tin và bỏ sót vật thể nghiêm trọng (False Negatives) mà hệ thống không có cảnh báo nào về việc dữ liệu đầu vào đã mất độ tin cậy.

---

## 2. Method (Phương pháp & Cơ chế đánh giá)
* **Nghiên cứu đối sánh:** Tham khảo benchmark *nuScenes-C (Dong et al., CVPR 2023 - [arXiv:2301.12239](https://arxiv.org/abs/2301.12239), [GitHub: nuScenes-C](https://github.com/SysCV/nuScenes-C))* về khả năng chống chịu nhiễu cảm biến và các phương pháp đánh giá chất lượng ảnh không tham chiếu (*No-Reference Image Quality Assessment - NR-IQA*).
* **Thuật toán đề xuất:** Xây dựng module tiền xử lý tính toán **Camera Health Score (0 - 100%)**:
  * **Input:** Khung hình ảnh thô (RGB) kích thước $800 \times 600$ từ camera xe.
  * **Blur Metric:** Đạo hàm bậc 2 rời rạc Laplacian để đo tần số không gian cạnh biên:
    $$\text{Blur Score} = \text{Var}(\nabla^2 I)$$
  * **Glare Metric:** Tỷ lệ pixel bão hòa trắng ($> 240/255$) trên toàn diện tích cảm biến:
    $$\text{Glare Ratio (\%)} = \frac{\sum [I(x,y) > 240]}{W \times H} \times 100\%$$
  * **Entropy:** Đo lường mật độ phân bổ thông tin thang độ xám (Shannon Entropy).
  * **Output:** Chỉ số `Camera Health Score` định lượng độ tin cậy của cảm biến trước khi truyền vào mạng nơ-ron nhận diện.

---

## 3. Benchmark (Kết quả đo đạc đối chứng tại lớp)

### A. Thí nghiệm đối chứng Synthetic Traffic Scene (Kiểm soát 100% biến số):
| Cấp độ thử nghiệm (Level) | Naive LapVar | Robust LapVar | Glare Ratio (%) | Camera Health (%) | Detections (Vật thể) | Avg Confidence | Trạng thái cảm biến |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **L0: Baseline (Clear)** | **342.9** | **278.7** | **0.3%** | **97.5%** | **2** | **0.859** | Hoạt động tối ưu |
| **L1: Mild Blur ($k=9$)** | 3.8 | 4.7 | 0.1% | 31.1% | 2 | 0.672 | Suy giảm nhẹ |
| **L2: Moderate Blur ($k=21$)** | 0.7 | 0.8 | 0.0% | 30.2% | 1 | 0.547 | Cảnh báo suy giảm |
| **L3: Heavy Blur + Rain** | 1837.4 *(bị lừa)* | 22.1 | 0.0% | 20.3% | 3 *(báo giả)* | 0.545 | Nguy hiểm (Nhiễu hạt) |
| **L4: Severe Blur + Glare** | **0.7** | **0.7** | **19.1%** | **1.6%** | **0** | **0.000** | **Cảm biến tê liệt (Failure)** |

### B. Kiểm chứng mở rộng trên ảnh thật Camera ADAS Highway (`assets/real_adas_highway.jpg`):
* **Baseline (L0 - Clear):** Robust LapVar = $148.8$, Glare = $0.7\%$, **Health Score = 56.7%**, phát hiện **5 vật thể**, Avg Confidence = **0.656**.
* **Failure Case (L4 - Chói nắng 21.0%):** **Health Score chạm đáy 0.4%**, số vật thể nhận diện rơi về **0**, Avg Confidence rơi về **0.000**.
* **Đồng nhất quy luật:** Xác nhận cả trên ảnh mô phỏng lẫn ảnh đời thực, chói nắng và mờ đục đều triệt tiêu gradient tương phản và làm sụp đổ hoàn toàn tính năng nhận diện.

### C. Đối sánh rõ ràng giữa Paper và Kết quả tự đo tại lớp:
* **Paper nuScenes-C (Dong et al., 2023) cho biết:** Trên 1000 video thực tế, các mô hình học sâu 3D (FCOS3D, BEVFormer) bị sụt giảm mAP từ 30% đến hơn 50% khi gặp suy thoái quang học cấp 4–5. Model hoạt động như "hộp đen", không có khả năng tự chẩn đoán suy giảm.
* **Nhóm quan sát và tự đo được tại lớp:** Tại L4, Health Score rơi xuống 1.6% (synthetic) và 0.4% (real image), kéo theo Confidence tụt từ 0.859 về 0.000 và mất sạch 100% đối tượng. Nhóm đã bù đắp khoảng trống của paper bằng module Health Score thời gian thực để kích hoạt cảnh báo an toàn.
---

## 4. Failure Case & Đột phá phân tích lỗi (2 Failure Modes)
* **Failure Case 1 (Cảm biến quang học sụp đổ):**
  * Tại **Level 4 (Severe Blur + Glare $19.1\%$)**: Sức khỏe camera rơi xuống mức tê liệt **1.6%**. 
  * Số lượng đối tượng nhận diện được rơi về **0**, Average Confidence rơi về **0.000**. Toàn bộ xe và người đi bộ bị xóa sạch do vùng lóa mặt trời triệt tiêu hoàn toàn gradient tương phản biên.
* **Failure Case 2 (Lỗ hổng của thuật toán đo độ nét truyền thống):**
  * Tại **Level 3 (Heavy Blur + Rain Noise)**: Nếu chỉ dùng phương sai Laplacian thông thường (Naive LapVar), chỉ số này nhảy vọt bất thường lên **1829.6** (cao gấp 5 lần ảnh gốc) vì nhiễu hạt mưa tạo ra các gradient nhân tạo tần số cực cao.
  * **Giải pháp cải tiến của nhóm:** Đưa bộ lọc trung vị (Median Filter $5\times 5$) vào tiền xử lý để bóc tách lớp nhiễu cao tần, giúp đo độ nét thực chất của vật thể (Robust LapVar chỉ còn $21.9$), đưa điểm Health Score về đúng giá trị thực tế là **20.2%**.

---

## 5. Engineering Decision (Quyết định kỹ thuật & Fallback)
1. **Thiết lập cơ chế Fallback thích ứng (Adaptive Sensor Weighting):**
   * **Health Score $\ge 70\%$:** Trạng thái bình thường; phân bổ trọng số Camera $100\%$ cho bộ lọc Sensor Fusion (EKF).
   * **$40\% \le \text{Health Score} < 70\%$:** Trạng thái suy giảm; giảm trọng số camera xuống $40\%$, tăng trọng số Radar/LiDAR lên $60\%$ và bật sấy kính/gạt nước.
   * **Health Score $< 40\%$:** Ngắt luồng dữ liệu camera khỏi hệ thống ra quyết định phanh khẩn cấp (AEB); kích hoạt cảnh báo *"Front Camera Blind"* cho tài xế và chuyển xe về chế độ duy trì khoảng cách an toàn dựa trên Radar.
2. **Data Loop & Rare-Case Mining:** Gắn flag tự động lưu lại các đoạn video ngắn có `Health Score < 40%` để đưa vào pipeline gán nhãn active learning cho vòng huấn luyện tiếp theo.
