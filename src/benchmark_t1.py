import os
import sys
import math

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
import matplotlib.pyplot as plt

def ensure_dirs():
    os.makedirs("assets", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def generate_or_load_sample_image():
    """Tạo hoặc nạp ảnh kịch bản giao thông (đường phố có xe, người đi bộ, làn đường)"""
    ensure_dirs()
    img_path = "assets/sample_traffic.jpg"
    
    # Tạo một ảnh giao thông trực quan bằng synthetic graphics nếu chưa có ảnh
    h, w = 600, 800
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 1. Bầu trời và cảnh nền
    canvas[:300, :] = [210, 180, 140]  # Sky (light blue/grey)
    # Tòa nhà và cây cối xa
    cv2.rectangle(canvas, (50, 150), (220, 300), (90, 90, 90), -1)
    cv2.rectangle(canvas, (260, 120), (450, 300), (120, 120, 120), -1)
    cv2.rectangle(canvas, (500, 170), (750, 300), (80, 80, 80), -1)
    
    # 2. Mặt đường
    pts_road = np.array([[100, 600], [350, 300], [450, 300], [700, 600]], np.int32)
    cv2.fillPoly(canvas, [pts_road], (60, 60, 60))
    # Vạch kẻ đường
    cv2.line(canvas, (400, 300), (400, 600), (255, 255, 255), 4)
    for y in range(320, 580, 60):
        cv2.line(canvas, (400, y), (400, y + 35), (0, 255, 255), 5)
        
    # Vỉa hè / Lề đường
    cv2.rectangle(canvas, (0, 300), (100, 600), (100, 130, 90), -1)
    cv2.rectangle(canvas, (700, 300), (800, 600), (100, 130, 90), -1)

    # 3. Xe hơi phía trước (Car 1 - xe gần)
    cv2.rectangle(canvas, (320, 420), (480, 530), (0, 0, 180), -1)  # Thân xe đỏ
    cv2.rectangle(canvas, (340, 430), (460, 470), (40, 40, 40), -1)   # Kính sau
    cv2.circle(canvas, (345, 525), 18, (20, 20, 20), -1)             # Bánh xe trái
    cv2.circle(canvas, (455, 525), 18, (20, 20, 20), -1)             # Bánh xe phải
    cv2.rectangle(canvas, (380, 490), (420, 505), (255, 255, 255), -1) # Biển số
    cv2.putText(canvas, "29A-999", (382, 501), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    # 4. Xe thứ hai ở xa (Car 2)
    cv2.rectangle(canvas, (375, 320), (425, 360), (180, 100, 30), -1) # Xe xanh ở xa
    cv2.rectangle(canvas, (382, 325), (418, 340), (30, 30, 30), -1)

    # 5. Người đi bộ bên lề phải (Pedestrian)
    cv2.circle(canvas, (720, 410), 12, (200, 180, 150), -1)          # Đầu
    cv2.rectangle(canvas, (712, 422), (728, 470), (200, 50, 50), -1) # Áo
    cv2.line(canvas, (716, 470), (716, 520), (50, 50, 150), 3)       # Chân trái
    cv2.line(canvas, (724, 470), (724, 520), (50, 50, 150), 3)       # Chân phải

    # Biển báo giao thông
    cv2.circle(canvas, (150, 260), 22, (0, 0, 220), -1)
    cv2.circle(canvas, (150, 260), 18, (255, 255, 255), -1)
    cv2.putText(canvas, "50", (140, 267), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.line(canvas, (150, 282), (150, 420), (160, 160, 160), 4)

    cv2.imwrite(img_path, canvas)
    return canvas

def compute_entropy(gray):
    """Tính độ hỗn loạn thông tin (Shannon Entropy)"""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / (hist.sum() + 1e-7)
    non_zero = hist[hist > 0]
    return float(-np.sum(non_zero * np.log2(non_zero)))

def calculate_sensor_health(img):
    """
    Tính các chỉ số đo lường chất lượng quang học của Camera ADAS
    - Naive Laplacian Variance (dễ bị nhiễu hạt mưa đánh lừa)
    - Robust Laplacian Variance (sau khi tách nhiễu thông thấp)
    - Glare / Overexposure Ratio (% diện tích bị bão hòa trắng)
    - Shannon Entropy (lượng thông tin/tương phản)
    - Health Score (Chỉ số sức khỏe tổng hợp 0 - 100%)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Naive Blur Metric
    naive_lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # 2. Robust Blur Metric (Dùng Median Filter loại bỏ nhiễu hạt mưa/muối tiêu cao tần)
    prefiltered = cv2.medianBlur(gray, 5)
    robust_lap_var = float(cv2.Laplacian(prefiltered, cv2.CV_64F).var())
    
    # Đo mức độ nhiễu hạt cao tần (Noise residual)
    noise_residual = float(np.std(gray.astype(np.float32) - prefiltered.astype(np.float32)))
    
    # 3. Overexposure / Glare: Tỷ lệ pixel > 240
    overexp_ratio = float((np.sum(gray > 240) / gray.size) * 100.0)
    
    # 4. Entropy
    entropy_val = compute_entropy(gray)
    
    # 5. Robust Composite Camera Health Score
    # Baseline tham chiếu: RobLap ~ 265.0 là tối ưu (100%)
    sharpness_score = min(100.0, (robust_lap_var / 265.0) * 100.0)
    # Phạt nếu bị lóa sáng quá 1%
    glare_penalty = min(100.0, overexp_ratio * 5.0)
    # Phạt nếu nhiễu hạt mưa/tuyết vượt ngưỡng (> 3.0)
    noise_penalty = min(100.0, max(0.0, (noise_residual - 3.0) * 6.0))
    
    composite_health = max(0.0, min(100.0, (sharpness_score * 0.7) + (100.0 - glare_penalty) * 0.3 - noise_penalty * 0.4))
    
    return round(naive_lap_var, 1), round(robust_lap_var, 1), round(overexp_ratio, 1), round(entropy_val, 2), round(composite_health, 1)

def apply_corruption(img, level):
    """Tạo 5 cấp độ suy giảm môi trường theo nuScenes-C standard"""
    h, w = img.shape[:2]
    
    if level == 0:
        return img.copy(), "L0: Baseline (Clear)"
    elif level == 1:
        # Nhòe nhẹ do rung nhẹ / vận tốc nhỏ
        k = 9
        res = cv2.GaussianBlur(img, (k, k), 0)
        return res, "L1: Mild Blur (k=9)"
    elif level == 2:
        # Nhòe vừa do sương mù / lens mờ
        k = 21
        res = cv2.GaussianBlur(img, (k, k), 0)
        return res, "L2: Moderate Blur (k=21)"
    elif level == 3:
        # Nhòe nặng + nhiễu hạt mưa (Rain/Noise)
        k = 33
        blurred = cv2.GaussianBlur(img, (k, k), 0)
        noise = np.random.normal(0, 15, blurred.shape).astype(np.float32)
        res = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return res, "L3: Heavy Blur + Rain Noise"
    elif level == 4:
        # Failure Case nghiêm trọng: Blur nặng + Chói nắng trực diện (Sun Glare)
        k = 29
        blurred = cv2.GaussianBlur(img, (k, k), 0)
        # Mô phỏng đốm lóa sáng cực mạnh (Sun glare) chiếu thẳng vào camera ADAS
        glare = np.zeros((h, w, 3), dtype=np.float32)
        cv2.circle(glare, (w // 2, int(h * 0.4)), 180, (255, 255, 255), -1)
        glare = cv2.GaussianBlur(glare, (121, 121), 0)
        res = np.clip(blurred.astype(np.float32) + glare * 0.9, 0, 255).astype(np.uint8)
        return res, "L4: Severe Blur + Glare (Failure Case)"

def run_detection_proxy(img):
    """
    Bộ nhận diện proxy kết hợp Contour Saliency + Edge Confidence
    Mô phỏng khả năng trích xuất vật thể (Xe, người) của mạng nơ-ron nhận diện.
    Trả về: số lượng đối tượng nhận diện được, confidence trung bình, và ảnh vẽ bounding box.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Canny edge detector
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated = img.copy()
    valid_boxes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 450:  # Lọc nhiễu quá nhỏ
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Tỷ lệ khung hình hợp lý cho xe hoặc người
            aspect = bh / float(bw)
            if 0.3 < aspect < 3.5:
                # Tính gradient intensity trong box làm proxy cho confidence
                box_gray = gray[y:y+bh, x:x+bw]
                if box_gray.size > 0:
                    conf = min(0.98, float(np.std(box_gray) / 75.0))
                    if conf > 0.25:
                        valid_boxes.append((x, y, bw, bh, conf))
    
    # NMS đơn giản
    valid_boxes = sorted(valid_boxes, key=lambda b: b[4], reverse=True)[:5]
    
    avg_conf = float(np.mean([b[4] for b in valid_boxes])) if len(valid_boxes) > 0 else 0.0
    det_count = len(valid_boxes)
    
    for (x, y, bw, bh, conf) in valid_boxes:
        color = (0, 255, 0) if conf > 0.6 else (0, 165, 255)
        cv2.rectangle(annotated, (x, y), (x+bw, y+bh), color, 2)
        cv2.putText(annotated, f"Obj: {conf:.2f}", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    
    return det_count, round(avg_conf, 3), annotated

def main():
    print("=" * 80)
    print("SENSOR REALITY SPRINT - TOPIC T1: CAMERA DEGRADATION HEALTH SCORE")
    print("Nhóm: Nguyễn Minh Quyền (2A202602438) & Vương Việt Hoàng (2A202602528)")
    print("=" * 80)
    
    import argparse
    parser = argparse.ArgumentParser(description="ADAS Camera Health Score Benchmark")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to real custom traffic image (e.g. assets/real_traffic.jpg)")
    args = parser.parse_args()

    if args.image and os.path.exists(args.image):
        print(f"-> Nạp ảnh thực tế từ file: {args.image}")
        img_base = cv2.imread(args.image)
        img_base = cv2.resize(img_base, (800, 600))
    else:
        img_base = generate_or_load_sample_image()
    
    levels = [0, 1, 2, 3, 4]
    benchmark_records = []
    saved_frames = []
    
    header = f"{'Degradation Level':<32} | {'NaiveLap':<9} | {'RobLap':<8} | {'Glare%':<7} | {'Health%':<8} | {'Dets':<5} | {'AvgConf':<7}"
    print(header)
    print("-" * len(header))
    
    for lvl in levels:
        deg_img, lbl = apply_corruption(img_base, lvl)
        naive_lap, robust_lap, overexp, entropy_val, health_score = calculate_sensor_health(deg_img)
        det_cnt, avg_conf, annotated = run_detection_proxy(deg_img)
        
        # Overlay Health Score lên ảnh minh chứng
        status_color = (0, 255, 0) if health_score >= 60 else ((0, 165, 255) if health_score >= 35 else (0, 0, 255))
        cv2.putText(annotated, f"Health Score: {health_score}% ({'OK' if health_score>=60 else 'DEGRADED'})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(annotated, f"RobLap: {robust_lap} | Glare: {overexp}% | Dets: {det_cnt}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        saved_frames.append((lbl, annotated))
        
        record = {
            "Level": lbl,
            "Naive_Lap": naive_lap,
            "Robust_Lap": robust_lap,
            "Entropy": entropy_val,
            "Glare_Ratio": overexp,
            "Health_Score": health_score,
            "Detections": det_cnt,
            "Avg_Confidence": avg_conf
        }
        benchmark_records.append(record)
        
        row = f"{lbl:<32} | {naive_lap:<9.1f} | {robust_lap:<8.1f} | {overexp:<7.1f} | {health_score:<8.1f} | {det_cnt:<5} | {avg_conf:<7.3f}"
        print(row)
        
    print("=" * 80)
    
    # 1. Lưu file log kết quả
    with open("logs/benchmark_results.csv", "w", encoding="utf-8") as f:
        f.write("Level,Naive_Lap,Robust_Lap,Entropy,Glare_Ratio,Health_Score,Detections,Avg_Confidence\n")
        for r in benchmark_records:
            f.write(f'"{r["Level"]}",{r["Naive_Lap"]},{r["Robust_Lap"]},{r["Entropy"]},{r["Glare_Ratio"]},{r["Health_Score"]},{r["Detections"]},{r["Avg_Confidence"]}\n')
            
    with open("logs/benchmark.log", "w", encoding="utf-8") as f:
        f.write("SENSOR REALITY SPRINT LOG\n")
        f.write("Topic T1: Camera degradation health score\n")
        f.write("Teammates: Nguyen Minh Quyen (2A202602438), Vuong Viet Hoang (2A202602528)\n\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in benchmark_records:
            f.write(f"{r['Level']:<32} | {r['Naive_Lap']:<9.1f} | {r['Robust_Lap']:<8.1f} | {r['Glare_Ratio']:<7.1f} | {r['Health_Score']:<8.1f} | {r['Detections']:<5} | {r['Avg_Confidence']:<7.3f}\n")
    print("-> Đã lưu kết quả vào 'logs/benchmark_results.csv' và 'logs/benchmark.log'.")

    # 2. Tạo biểu đồ trực quan (Plot)
    labels = ["L0\nBaseline", "L1\nMild", "L2\nModerate", "L3\nRainNoise", "L4\nSevereGlare"]
    healths = [r["Health_Score"] for r in benchmark_records]
    confs = [r["Avg_Confidence"] for r in benchmark_records]
    lap_vars = [r["Robust_Lap"] for r in benchmark_records]
    dets = [r["Detections"] for r in benchmark_records]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Đồ thị 1: Health Score vs Detector Confidence
    color1 = 'tab:red'
    ax1.set_xlabel('Mức độ suy giảm (Degradation Level)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Camera Health Score (%)', color=color1, fontsize=11, fontweight='bold')
    line1 = ax1.plot(labels, healths, color=color1, marker='o', linewidth=2.5, label='Camera Health Score (%)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_ylim(-5, 110)
    
    ax1_twin = ax1.twinx()
    color2 = 'tab:blue'
    ax1_twin.set_ylabel('Detector Avg Confidence', color=color2, fontsize=11, fontweight='bold')
    line2 = ax1_twin.plot(labels, confs, color=color2, marker='s', linewidth=2.5, linestyle='--', label='Detector Confidence')
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    ax1_twin.set_ylim(-0.05, 1.05)
    
    # Đặt ngưỡng fallback
    ax1.axhline(y=50, color='darkorange', linestyle=':', linewidth=2, label='Ngưỡng cảnh báo Fallback (50%)')
    ax1.set_title('Tác động suy giảm: Sức khỏe Camera vs Độ tin cậy Detector', fontsize=12, fontweight='bold')
    
    # Gom legend
    lines = line1 + line2 + [plt.Line2D([0], [0], color='darkorange', linestyle=':', linewidth=2)]
    legend_labels = ['Camera Health Score (%)', 'Detector Confidence', 'Ngưỡng Fallback (50%)']
    ax1.legend(lines, legend_labels, loc='lower left')

    # Đồ thị 2: Số vật thể phát hiện & Độ mờ Laplacian
    x_indices = np.arange(len(labels))
    width = 0.35
    ax2.bar(x_indices, dets, width=width, color='teal', edgecolor='black', alpha=0.8, label='Số lượng đối tượng phát hiện')
    ax2.set_ylabel('Số lượng đối tượng (Detections)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Mức độ suy giảm', fontsize=11, fontweight='bold')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(labels)
    ax2.set_title('Mất mát đối tượng nhận diện theo cấp độ suy giảm', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, max(dets) + 2)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plot_path = "assets/benchmark_plot.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"-> Đã xuất đồ thị benchmark tại: {plot_path}")

    # 3. Tạo ảnh so sánh đối chứng Failure Case (Level 0 vs Level 4)
    frame_clean = saved_frames[0][1]
    frame_failed = saved_frames[4][1]
    comparison = np.hstack([frame_clean, frame_failed])
    comp_path = "assets/failure_case_comparison.png"
    cv2.imwrite(comp_path, comparison)
    print(f"-> Đã xuất ảnh so sánh Failure Case tại: {comp_path}")
    print("\nBENCHMARK HOÀN TẤT THÀNH CÔNG!")

if __name__ == "__main__":
    main()
