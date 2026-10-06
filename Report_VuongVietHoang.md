# BÁO CÁO THỰC HÀNH: SENSOR REALITY SPRINT (TOPIC T1)
**Họ và tên:** Vương Việt Hoàng  
**MSSV:** 2A202602528  
**Chủ đề:** T1 - Camera Degradation Health Score  
**Nền tảng mục tiêu:** Xe tự hành / Robot tự hành ngoài trời (Outdoor Autonomous Mobility)  
**Repository nhóm:** `K4-Track4-Day04-QuyenHoang-Sensor-Reality-Sprint`

---

## 1. Problem (Bối cảnh thực tế & Giới hạn vật lý của cảm biến)
* **Bối cảnh:** Trong các hệ thống ADAS thương mại cấp độ L2+/L3, camera là cảm biến cốt lõi chịu trách nhiệm phân loại đối tượng, nhận diện làn đường và biển báo.
* **Bản chất vật lý của Failure:**
  * Khác với các cảm biến chủ động như LiDAR hay Radar, camera là cảm biến thụ động (passive sensor), phụ thuộc hoàn toàn vào quang thông môi trường và độ trong suốt của thấu kính.
  * Hiện tượng **Lens Soiling** (bụi bẩn, bùn đất bám thấu kính), **Defocus/Motion Blur** (rung cơ học, lấy nét sai), và **Direct Sunlight Glare** (chói lóa vượt quá dải động Dynamic Range của cảm biến CMOS) làm biến dạng hoàn toàn tín hiệu quang học đầu vào.
* **Rủi ro:** Mô hình nhận diện phía sau không thể phân biệt giữa "không có vật thể" và "camera không thể nhìn thấy vật thể", dẫn đến nguy cơ tai nạn nghiêm trọng.

---

## 2. Method (Thuật toán giám sát sức khỏe cảm biến)
* **Nguồn tham khảo:** 
  * Nghiên cứu của *Dong et al. (CVPR 2023) - nuScenes-C Corruption Robustness* ([arXiv:2301.12239](https://arxiv.org/abs/2301.12239), [GitHub: nuScenes-C](https://github.com/SysCV/nuScenes-C)).
  * Các tiêu chuẩn công nghiệp về giám sát chất lượng cảm biến (Sensor Self-Diagnostic & Health Assessment).
* **Kiến trúc giải pháp:**
  * Xây dựng cơ chế **No-Reference Sensor Health Assessment** chạy ở tần số cao (30 - 60 FPS) trước mạng nơ-ron nhận diện:
    1. **Variance of Laplacian ($\text{Var}(\nabla^2 I)$):** Đánh giá độ sắc nét tần số cao của khung hình.
    2. **Pixel Saturation Analysis:** Quét tỷ lệ các pixel bị bão hòa trắng ($> 240$) để đo lường mức độ lóa sáng (Glare).
    3. **Shannon Entropy:** Đo lường entropy thông tin thang xám, phát hiện sự suy giảm tương phản tổng thể.
  * **Hàm chuẩn hóa sức khỏe (Health Function):** Kết hợp có trọng số giữa điểm nét và mức phạt chói lóa:
    $$\text{Health Score} = 0.7 \times \text{Sharpness} + 0.3 \times (100 - \text{Glare Penalty})$$

---

## 3. Benchmark & Kết quả thực nghiệm

### A. Thí nghiệm đối chứng Synthetic Traffic Scene (Kiểm soát 100% biến số):
* **Baseline (L0):** Naive LapVar = $342.9$, Robust LapVar = $278.7$, Glare = $0.3\%$, **Health Score = 97.5%**, Phát hiện đủ $2/2$ đối tượng chính với Avg Confidence = $0.859$.
* **L1 (Mild Blur, $k=9$):** Robust LapVar giảm xuống $4.7$ (giảm $98.3\%$), Health Score giảm xuống $31.1\%$, Confidence giảm còn $0.672$.
* **L2 (Moderate Blur, $k=21$):** Robust LapVar = $0.8$, Health Score = $30.2\%$, số lượng đối tượng giảm còn $1$, Confidence = $0.547$.
* **L3 (Rain & Heavy Blur):** Naive LapVar bị đánh lừa lên $1837.4$ do nhiễu hạt mưa; nhờ Median Filter bóc tách, Robust LapVar chỉ còn $22.1$ và Health Score thực tế giữ ở mức **20.3%**.
* **L4 (Failure Case: Severe Blur + Glare $19.1\%$):** Health Score chạm đáy **1.6%**, số lượng đối tượng nhận diện rơi về **0**, Average Confidence rơi về **0.000** (Mất hoàn toàn đối tượng, cảm biến mù lòa).

### B. Kiểm chứng mở rộng trên ảnh thực tế Camera ADAS Highway (`assets/real_adas_highway.jpg`):
* Chạy trực tiếp trên ảnh chụp thực tế đường cao tốc từ Udacity Self-Driving Car Dataset:
  - **Baseline L0:** Robust LapVar = $148.8$, Health Score = $56.7\%$, nhận diện đủ 5 xe/vật thể trên đường, Avg Confidence = $0.656$.
  - **Failure Case L4:** Chói nắng $21.0\%$ làm Health Score tụt xuống **0.4%**, số lượng vật thể biến mất hoàn toàn ($0$ detections, Confidence = $0.000$).
* **Kết luận:** Quy luật suy thoái và cơ chế sụp đổ đặc trưng (feature collapse) xảy ra đồng nhất trên cả ảnh mô phỏng lẫn ảnh đời thực.

### C. Đối sánh rạch ròi Paper vs Nhóm tự đo:
* **Paper nuScenes-C (Dong et al., 2023) cho biết:** Khi mức suy thoái đạt severity 4–5, mAP của các mô hình nhận diện 3D sụt giảm mạnh từ 30% đến 50%.
* **Nhóm quan sát và tự đo được tại lớp:** Tại L4, Health Score rơi xuống $1.6\%$ và Confidence tụt về $0.000$. Nhóm đã phát triển thành công module Camera Health Score chạy thời gian thực để lấp đầy khoảng trống thiếu cơ chế tự chẩn đoán của paper.
---

## 4. Failure Case (Phân tích chi tiết trường hợp sụp đổ)
* **Ca kiểm tra cụ thể:** Level 4 (Severe Blur $k=29$ kết hợp Sun Glare vùng sáng $19.1\%$).
* **Phân tích bản chất vật lý & giải thuật:**
  * Khi bị lóa nắng trực diện kết hợp tán xạ mờ đục, các pixel chuyển thành giá trị bão hòa trắng ($> 240/255$).
  * Hiện tượng này làm triệt tiêu hoàn toàn gradient không gian giữa mép biên đối tượng và mặt đường. Mạng nơ-ron nhận diện mất hoàn toàn các đặc trưng cục bộ (local edge features) $\to$ **Toàn bộ xe và người đi bộ biến mất hoàn toàn**.
* **Đóng góp cải tiến:** Phát hiện lỗ hổng của metric đo nét truyền thống khi gặp mưa/nhiễu muối tiêu (Naive Laplacian bị nhiễu đánh lừa), và cải tiến thành công bằng Robust Median Prefiltering.

---

## 5. Engineering Trade-off & Quyết định kỹ thuật
* **Khi nào NÊN dùng Camera Health Score:**
  * Trong các hệ thống Multi-Sensor Fusion: Dùng điểm Health Score làm hệ số tin cậy động (Dynamic Covariance $\mathbf{R}_k$) trong bộ lọc Extended Kalman Filter (EKF). Càng mờ, ma trận hiệp phương sai sai số đo đạc càng tăng để hệ thống tự động tin vào Radar/LiDAR nhiều hơn.
  * Trong Data Loop / MLOps: Tự động gom các frame có `Health < 40%` để bổ sung vào tập huấn luyện chống chịu thời tiết xấu.
* **Khi nào KHÔNG NÊN quá phụ thuộc vào chỉ số này:**
  * Ban đêm (Nighttime): Cảnh đêm tự nhiên có ít cạnh sắc và entropy thấp, dễ gây báo động giả (False Positive) cho bộ đo Laplacian. Cần kết hợp cảm biến ánh sáng môi trường (Ambient Light Sensor) để chuyển đổi ngưỡng linh hoạt.
* **Chiến lược an toàn (Operational Safety Envelope):**
  * Tự động kích hoạt cơ chế gạt nước / sấy thấu kính khi phát hiện mờ liên tục $> 3$ giây.
  * Nếu cả Camera và Radar đều suy giảm, cảnh báo người lái chiếm quyền kiểm soát (Take Over Request) trong vòng 5 giây trước khi thực hiện Minimum Risk Maneuver (MRM).
