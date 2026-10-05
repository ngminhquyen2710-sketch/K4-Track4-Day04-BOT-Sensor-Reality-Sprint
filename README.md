# SENSOR REALITY SPRINT - DAY 04
## Topic T1: Camera Degradation Health Score in ADAS Perception

**Nhóm thực hiện:** BOT
**Thành viên:**
1. Nguyễn Minh Quyền - MSSV: `2A202602438`
2. Vương Việt Hoàng - MSSV: `2A202602528`

---

## 1. Giới thiệu dự án
Dự án tập trung vào việc nghiên cứu và thực nghiệm đánh giá mức độ suy giảm chất lượng của cảm biến Camera quang học trên xe tự hành ADAS khi gặp các điều kiện thời tiết và môi trường bất lợi (Defocus Blur, Motion Blur, Mưa, và Chói nắng / Sun Glare). 

Nhóm đã phát triển module **Camera Health Score** nhằm định lượng độ tin cậy của khung hình cảm biến trước khi truyền vào các mạng nơ-ron sâu nhận diện đối tượng.

---

## 2. Cấu trúc thư mục Repository
```
K4-Track4-Day04-QuyenHoang-Sensor-Reality-Sprint/
├── TEAMMATES.md                  # Danh sách và phân công 2 thành viên
├── README.md                     # Hướng dẫn và tổng quan dự án
├── src/
│   └── benchmark_t1.py           # Mã nguồn Python chạy benchmark đối chứng 5 cấp độ
├── assets/
│   ├── benchmark_plot.png        # Đồ thị suy giảm Health Score & Độ tin cậy
│   └── failure_case_comparison.png # Ảnh minh chứng đối chứng Baseline vs Failure Case
├── logs/
│   ├── benchmark.log             # Log thực thi chi tiết
│   └── benchmark_results.csv     # Bảng số liệu định lượng phục vụ báo cáo
└── reports/
    ├── Report_NguyenMinhQuyen.md # Báo cáo cá nhân của Nguyễn Minh Quyền
    └── Report_VuongVietHoang.md   # Báo cáo cá nhân của Vương Việt Hoàng
```

---

## 3. Hướng dẫn chạy lại mã nguồn (Reproducibility)
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install opencv-python numpy matplotlib
   ```
2. Chạy bài benchmark:
   ```bash
   cd K4-Track4-Day04-QuyenHoang-Sensor-Reality-Sprint
3. Chạy bài benchmark mặc định (Synthetic Scene đối chứng chuẩn):
   ```bash
   python src/benchmark_t1.py
   ```
4. Chạy kiểm chứng mở rộng trên ảnh giao thông thực tế đời thực (Real-world Image):
   ```bash
   python src/benchmark_t1.py --image assets/real_traffic.jpg
   ```
   *(Bạn cũng có thể thay bằng đường dẫn tới bất kỳ ảnh camera giao thông thực tế nào).*
5. Kết quả bảng số liệu và đồ thị sẽ tự động được cập nhật vào thư mục `logs/` và `assets/`.

---

## 4. Tài liệu tham khảo học thuật (Academic References)
* **nuScenes-C Benchmark:** Dong et al., *"Benchmarking and Analyzing Robustness in 3D Object Detection"*, CVPR 2023. ([arXiv:2301.12239](https://arxiv.org/abs/2301.12239) / [GitHub: nuScenes-C](https://github.com/SysCV/nuScenes-C))
* **No-Reference Image Quality Assessment (NR-IQA):** Pech-Pacheco et al., *"Diatom autofocusing in brightfield microscopy: a comparative study"*, ICPR 2000 (Laplacian Variance).
