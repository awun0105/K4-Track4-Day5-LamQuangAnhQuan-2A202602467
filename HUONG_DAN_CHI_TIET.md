# CẨM NANG TOÀN DIỆN: BỘ LỌC KALMAN & HỢP NHẤT XÁC SUẤT ĐA CẢM BIẾN
## Track 4 · Ngày 5 — Kalman Filter & Probabilistic Fusion Pilot (Lynx-07)

> **Tác giả:** Lâm Quang Anh Quân — MSSV: `2A202602467`  
> **Repository:** `K4-Track4-Day5-LamQuangAnhQuan-2A202602467`  
> **Mục tiêu học thuật:** Xây dựng từ nguyên lý gốc (First-Principles Thinking) toàn bộ chuỗi thuật toán định vị và hợp nhất cảm biến từ không gian trạng thái 1 chiều đến hệ thống hợp nhất đa cảm biến bất đồng bộ (LiDAR, Radar, GPS, UWB, Camera) và bộ lọc mở rộng phi tuyến (EKF).  
> **Điểm nhấn đặc biệt:** Mọi công thức toán học đều đi kèm bảng giải phẫu từng thành phần (kích thước ma trận, đơn vị vật lý, ý nghĩa) cùng cách phát biểu / diễn giải trực giác bằng lời nói đời thường.

---

## MỤC LỤC
1. [Nguyên Lý Gốc: Tại Sao Cần Bộ Lọc Xác Suất?](#1-nguyên-lý-gốc-tại-sao-cần-bộ-lọc-xác-suất)
2. [Biểu Diễn Niềm Tin Bằng Phân Phối Gaussian & Hợp Nhất Bayes](#2-biểu-diễn-niềm-tin-bằng-phân-phối-gaussian--hợp-nhất-bayes)
3. [Bộ Lọc Kalman Tuyến Tính 1 Chiều (Scalar KF)](#3-bộ-lọc-kalman-tuyến-tính-1-chiều-scalar-kf)
4. [Chỉ Số Giám Sát Sức Khỏe NIS (Normalized Innovation Squared)](#4-chỉ-số-giám-sát-sức-khỏe-nis-normalized-innovation-squared)
5. [Không Gian Trạng Thái & Bộ Lọc Kalman Dạng Ma Trận (Part 5)](#5-không-gian-trạng-thái--bộ-lọc-kalman-dạng-ma-trận-part-5)
6. [Hợp Nhất Đa Cảm Biến Bất Đồng Bộ Thời Gian Thực (Part 6)](#6-hợp-nhất-đa-cảm-biến-bất-đồng-bộ-thời-gian-thực-part-6)
7. [Loại Trừ Dữ Liệu Ngoại Lai Bằng Cổng Kiểm Định Chi-Bình Phương (Part 7)](#7-loại-trừ-dữ-liệu-ngoại-lai-bằng-cổng-kiểm-định-chi-bình-phương-part-7)
8. [Nhiệm Vụ Chẩn Đoán Cảm Biến Xe Tự Hành Lynx-07 (Part 9)](#8-nhiệm-vụ-chẩn-đoán-cảm-biến-xe-tự-hành-lynx-07-part-9)
9. [Bonus Nâng Cao: Extended Kalman Filter (EKF - Part 8)](#9-bonus-nâng-cao-extended-kalman-filter-ekf---part-8)
10. [Playbook Triển Khai Thực Tế Trong Xe Tự Hành & Robot Công Nghiệp](#10-playbook-triển-khai-thực-tế-trong-xe-tự-hành--robot-công-nghiệp)

---

## 1. NGUYÊN LÝ GỐC: TẠI SAO CẦN BỘ LỌC XÁC SUẤT?

### 1.1 Thách thức của thế giới thực: Nhiễu và Trễ pha
Trong kỹ thuật điều khiển và xe tự hành, việc đo đạc vị trí từ bất kỳ cảm biến vật lý nào (như GPS, UWB, LiDAR) luôn đi kèm với hai rào cản tự nhiên:
1. **Nhiễu đo lường ngẫu nhiên (Measurement Noise):** Cảm biến bị ảnh hưởng bởi nhiệt độ, tán xạ sóng, nhiễu điện tử, khúc xạ khí quyển dẫn đến số đo rung lắc liên tục quanh giá trị thực.
2. **Hiện tượng trễ pha (Phase Lag / Delay) của các bộ lọc cổ điển:**
   - Phương pháp trực quan nhất để làm mượt dữ liệu là **Trung bình trượt nhân quả (Causal Moving Average)** với cửa sổ $k$:

$$\bar{z}_t = \frac{1}{k} \sum_{i=0}^{k-1} z_{t-i}$$

#### 🔍 Giải phẫu thành phần công thức:
| Thành phần | Tên gọi | Kiểu / Kích cỡ | Đơn vị | Ý nghĩa vật lý |
| :---: | :--- | :---: | :---: | :--- |
| $\bar{z}_t$ | Giá trị ước lượng trung bình trượt | Vô hướng | $\text{m}$ | Vị trí ước tính sau khi làm mượt tại thời điểm $t$. |
| $k$ | Kích thước cửa sổ (Window size) | Số nguyên dương | Không thứ nguyên | Số lượng điểm đo trong quá khứ được gom lại để tính trung bình. |
| $z_{t-i}$ | Phép đo tại thời điểm quá khứ $t-i$ | Vô hướng | $\text{m}$ | Tọa độ cảm biến đo được lùi về trước $i$ bước thời gian. |
| $\sum_{i=0}^{k-1}$ | Tổng tích lũy | Phép toán | - | Cộng dồn $k$ giá trị đo gần nhất tính từ thời điểm hiện tại. |

#### 🗣️ Cách phát biểu & Diễn giải bằng lời:
> **Cách đọc:** *"Vị trí ước tính tại thời điểm $t$ bằng trung bình cộng số học của $k$ phép đo gần nhất tính từ hiện tại trở về quá khứ."*  
> **Nghịch lý trực giác:** Nếu chọn $k$ nhỏ, dữ liệu không đủ độ mượt (nhiễu vẫn còn lớn). Nếu chọn $k$ lớn để triệt tiêu nhiễu, bộ lọc sẽ tích lũy một độ trễ thời gian $\Delta t_{\text{lag}} \approx \frac{k-1}{2} \cdot \Delta t_s$. Khi xe tự hành phanh gấp hoặc rẽ ngoặt, giá trị trung bình trượt sẽ "ngủ quên" ở quá khứ, khiến xe tiếp tục nghĩ mình đang đi thẳng và gây tai nạn.

```
[Số đo thô rung lắc] ──> [Moving Average] ──> Giảm nhiễu NHƯNG trễ pha nghiêm trọng!
[Số đo thô rung lắc] ──> [Kalman Filter]  ──> Giảm nhiễu VÀ không trễ pha (nhờ dự đoán vật lý)!
```

### 1.2 Ý tưởng đột phá của Rudolf Kalman: Mô hình động học kết hợp đo đạc
Bộ lọc Kalman giải quyết mâu thuẫn trên bằng cách không chỉ nhìn về quá khứ mà còn **dự phóng tương lai** dựa trên quy luật vật lý:
- Chúng ta có phương trình chuyển động của xe (ví dụ: vận tốc nhân thời gian cho ra quãng đường).
- Chúng ta có các phép đo mới từ cảm biến kèm theo ước lượng độ tin cậy.
- Bộ lọc Kalman kết hợp hai nguồn thông tin độc lập này theo tỷ lệ tối ưu toán học (độ lợi Kalman $K$) để giảm phương sai sai số xuống mức thấp nhất có thể.

---

## 2. BIỂU DIỄN NIỀM TIN BẰNG PHÂN PHỐI GAUSSIAN & HỢP NHẤT BAYES

### 2.1 Hàm mật độ xác suất Gaussian (Gaussian Belief)
Trong lý thuyết ước lượng tối ưu, trạng thái của một đại lượng không được biểu diễn bằng một con số vô hướng cố định mà bằng một biến ngẫu nhiên tuân theo phân phối chuẩn $\mathcal{N}(\mu, \sigma^2)$:

$$p(x) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

#### 🔍 Giải phẫu thành phần công thức:
| Thành phần | Tên gọi | Kiểu / Kích cỡ | Đơn vị | Ý nghĩa vật lý |
| :---: | :--- | :---: | :---: | :--- |
| $p(x)$ | Mật độ xác suất (Probability density) | Vô hướng $\ge 0$ | $1/\text{m}$ | Mật độ khả năng trạng thái thực của xe nằm tại tọa độ $x$. |
| $x$ | Tọa độ khảo sát | Vô hướng | $\text{m}$ | Vị trí không gian đang được đánh giá xác suất. |
| $\mu$ | Kỳ vọng toán học (Mean) | Vô hướng | $\text{m}$ | Điểm trung tâm có mật độ xác suất cao nhất — vị trí khả dĩ nhất của xe. |
| $\sigma^2$ | Phương sai (Variance) | Vô hướng $> 0$ | $\text{m}^2$ | Thước đo độ bất định (Uncertainty). Càng nhỏ, đỉnh chuông càng nhọn $\Rightarrow$ niềm tin càng chắc chắn. |
| $\sigma$ | Độ lệch chuẩn (Standard deviation) | Vô hướng $> 0$ | $\text{m}$ | Biên độ phân tán sai số quanh giá trị trung bình ($1\sigma$ ứng với 68.3% độ tin cậy). |
| $\exp(\dots)$ | Hàm mũ cơ số tự nhiên $e$ | Phép toán | - | Tạo hình dạng đường cong hình chuông đối xứng đặc trưng. |

#### 🗣️ Cách phát biểu & Diễn giải bằng lời:
> **Cách đọc:** *"Hàm mật độ xác suất của tọa độ $x$ tỉ lệ nghịch với căn bậc hai của phương sai nhân hai pi, và tỉ lệ thuận với hàm mũ của âm sai số bình phương chia cho hai lần phương sai."*  
> **Diễn giải trực giác:** Càng xa giá trị kỳ vọng $\mu$, xác suất xe có mặt ở đó càng suy giảm theo hàm mũ nghịch đảo. Phương sai $\sigma^2$ đóng vai trò là "chiều rộng chiếc ô hoài nghi": ô càng hẹp, ta càng nắm chắc vị trí của xe.

---

### 2.2 Công thức hợp nhất hai nguồn thông tin Gaussian độc lập
Giả sử ta có hai cảm biến độc lập cùng đo một vị trí $x$:
- Cảm biến 1 cho phân phối $\mathcal{N}(\mu_1, \sigma_1^2)$.
- Cảm biến 2 cho phân phối $\mathcal{N}(\mu_2, \sigma_2^2)$.

Theo định lý Bayes, hàm mật độ xác suất hậu nghiệm tỉ lệ thuận với tích của hai hàm mật độ xác suất độc lập:

$$p(x \mid z_1, z_2) \propto p(z_1 \mid x) \cdot p(z_2 \mid x)$$

Khi nhân hai đường cong chuông Gaussian, kết quả thu được **luôn là một đường cong chuông Gaussian mới** $\mathcal{N}(\mu, \sigma^2)$ với công thức hợp nhất:

$$\sigma^2 = \frac{1}{\frac{1}{\sigma_1^2} + \frac{1}{\sigma_2^2}} = \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2 + \sigma_2^2}$$

$$\mu = \sigma^2 \left(\frac{\mu_1}{\sigma_1^2} + \frac{\mu_2}{\sigma_2^2}\right) = \mu_1 + \frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2} (\mu_2 - \mu_1)$$

#### 🔍 Giải phẫu thành phần công thức hợp nhất:
| Thành phần | Tên gọi | Kiểu / Kích cỡ | Đơn vị | Ý nghĩa vật lý |
| :---: | :--- | :---: | :---: | :--- |
| $\sigma^2$ | Phương sai hợp nhất (Fused Variance) | Vô hướng | $\text{m}^2$ | Độ bất định của trạng thái sau khi đã gộp bằng chứng từ cả hai cảm biến. |
| $\frac{1}{\sigma^2}$ | Độ chính xác (Precision / Information) | Vô hướng | $1/\text{m}^2$ | Lượng thông tin tri thức. Độ chính xác hợp nhất bằng **tổng độ chính xác** của từng cảm biến. |
| $\mu_1, \mu_2$ | Số đo trung bình của cảm biến 1 và 2 | Vô hướng | $\text{m}$ | Vị trí mà cảm biến 1 và cảm biến 2 báo cáo. |
| $\sigma_1^2, \sigma_2^2$ | Phương sai sai số của cảm biến 1 và 2 | Vô hướng | $\text{m}^2$ | Mức độ nhiễu hay độ bất định cố hữu của từng cảm biến. |
| $\frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2}$ | Trọng số điều chỉnh (Kalman Gain 1D) | Vô hướng $[0, 1]$ | Không thứ nguyên | Tỉ lệ nhường bước: cảm biến 1 nhường bao nhiêu phần đường về phía cảm biến 2. |
| $\mu_2 - \mu_1$ | Độ lệch đo lường (Measurement Discrepancy) | Vô hướng | $\text{m}$ | Khoảng cách bất đồng giữa hai cảm biến. |

#### 🗣️ Cách phát biểu & Diễn giải bằng lời:
> **Cách đọc phương sai:** *"Nghịch đảo phương sai hợp nhất bằng tổng các nghịch đảo phương sai thành phần."*  
> **Cách đọc kỳ vọng:** *"Vị trí hợp nhất bằng trung bình có trọng số của hai vị trí, trong đó trọng số tỉ lệ thuận với độ tin cậy (nghịch đảo phương sai) của mỗi cảm biến."*  
> **Định luật vàng:** $\sigma^2 < \min(\sigma_1^2, \sigma_2^2)$ — **Phương sai sau hợp nhất luôn nhỏ hơn phương sai của cảm biến tốt nhất**. Việc tiếp nhận thêm thông tin độc lập không bao giờ làm ta hoài nghi hơn, mà luôn làm tăng độ tin cậy!

---

## 3. BỘ LỌC KALMAN TUYẾN TÍNH 1 CHIỀU (SCALAR KF)

Bộ lọc Kalman 1 chiều là trường hợp đặc biệt mà ở đó phép cập nhật Bayes được lặp đi lặp lại xen kẽ với bước ngoại suy thời gian.

```
                  ┌────────────────────────────────────────┐
                  │          Trạng thái ban đầu            │
                  │              (x₀, P₀)                  │
                  └──────────────────┬─────────────────────┘
                                     │
            ┌────────────────────────▼────────────────────────┐
            │               1. DỰ ĐOÁN (PREDICT)              │
            │          x⁻ = x̂                                 │
            │          P⁻ = P + Q  (Độ bất định tăng lên)     │
            └────────────────────────┬────────────────────────┘
                                     │
                                     │ [Có phép đo mới z từ cảm biến]
                                     ▼
            ┌─────────────────────────────────────────────────┐
            │              2. CẬP NHẬT (UPDATE)               │
            │  Độ mới (Innovation):    y = z - x⁻             │
            │  Phương sai độ mới:      S = P⁻ + R             │
            │  Độ lợi Kalman:          K = P⁻ / S             │
            │  Cập nhật trạng thái:    x̂ = x⁻ + K·y           │
            │  Cập nhật phương sai:    P = (1 - K)·P⁻         │
            └────────────────────────┬────────────────────────┘
                                     │
                                     └─────> Lặp lại chu kỳ kế tiếp
```

### 3.1 Nhóm công thức Dự đoán (Predict Phase)

$$\hat{x}^- = \hat{x}_{k-1}$$

$$P^- = P_{k-1} + Q$$

#### 🔍 Giải phẫu thành phần:
| Thành phần | Tên gọi | Kiểu | Đơn vị | Ý nghĩa vật lý |
| :---: | :--- | :---: | :---: | :--- |
| $\hat{x}^-$ | Ước lượng tiên nghiệm (A priori state) | Vô hướng | $\text{m}$ | Dự đoán vị trí ở thời điểm hiện tại trước khi nhìn thấy số đo cảm biến. |
| $\hat{x}_{k-1}$ | Ước lượng hậu nghiệm bước trước | Vô hướng | $\text{m}$ | Vị trí tốt nhất đã chốt ở bước thời gian trước. |
| $P^-$ | Phương sai tiên nghiệm | Vô hướng | $\text{m}^2$ | Độ bất định của dự đoán trước khi cập nhật cảm biến. |
| $P_{k-1}$ | Phương sai bước trước | Vô hướng | $\text{m}^2$ | Độ bất định đã biết ở bước trước. |
| $Q$ | Phương sai nhiễu quá trình (Process Noise) | Vô hướng | $\text{m}^2$ | Mức độ hỗn loạn của môi trường (gió, mặt đường, trượt bánh) làm ta mất dần niềm tin theo thời gian. |

#### 🗣️ Cách phát biểu & Diễn giải bằng lời:
> **Phát biểu:** *"Khi thời gian trôi đi mà chưa có phép đo mới, vị trí được bảo toàn theo quán tính, còn độ bất định $P^-$ bắt buộc phải tăng thêm một lượng $Q$ vì thế giới luôn biến động ngẫu nhiên."*

---

### 3.2 Nhóm công thức Cập nhật (Update Phase)

$$y = z - \hat{x}^-$$

$$S = P^- + R$$

$$K = \frac{P^-}{S} = \frac{P^-}{P^- + R}$$

$$\hat{x} = \hat{x}^- + K \cdot y$$

$$P = (1 - K) \cdot P^-$$

#### 🔍 Giải phẫu thành phần:
| Thành phần | Tên gọi | Kiểu | Đơn vị | Ý nghĩa vật lý |
| :---: | :--- | :---: | :---: | :--- |
| $z$ | Phép đo thực tế (Measurement) | Vô hướng | $\text{m}$ | Con số thực tế mà phần cứng cảm biến bắn về. |
| $y$ | Độ mới / Phần dư (Innovation / Residual) | Vô hướng | $\text{m}$ | Khoảng cách bất ngờ: Cảm biến đo được khác bao nhiêu so với điều ta dự đoán. |
| $R$ | Phương sai nhiễu đo lường (Sensor Noise) | Vô hướng | $\text{m}^2$ | Độ nhiễu cố hữu của cảm biến theo bảng thông số kỹ thuật (Datasheet). |
| $S$ | Phương sai của độ mới (Innovation Variance) | Vô hướng | $\text{m}^2$ | Tổng độ bất định gộp của cả dự đoán ($P^-$) lẫn cảm biến ($R$). |
| $K$ | Độ lợi Kalman (Kalman Gain) | Vô hướng $[0, 1]$ | Không thứ nguyên | Trọng số quyết định mức độ tin vào cảm biến so với mô hình dự đoán. |
| $\hat{x}$ | Ước lượng hậu nghiệm (A posteriori state) | Vô hướng | $\text{m}$ | Kết quả vị trí cuối cùng được chốt lại sau khi dung hòa dự đoán và số đo. |
| $P$ | Phương sai hậu nghiệm | Vô hướng | $\text{m}^2$ | Độ bất định đã được co lại sau khi hấp thụ thông tin từ cảm biến. |

#### 🗣️ Cách phát biểu & Diễn giải bằng lời:
> **Độ mới $y$:** *"Độ mới bằng số đo cảm biến trừ đi dự đoán tiên nghiệm. Nó phản ánh phần thông tin mới tinh khôi mà mô hình chưa lường trước được."*  
> **Độ lợi Kalman $K$:** *"Độ lợi Kalman là tỷ số giữa độ bất định của dự đoán trên tổng độ bất định toàn hệ thống. Nếu cảm biến cực chuẩn ($R \to 0$), $K \to 1$ (tin 100% vào cảm biến). Nếu cảm biến quá nhiễu ($R \gg P^-$), $K \to 0$ (bỏ qua cảm biến, giữ nguyên dự đoán)."*  
> **Cập nhật trạng thái $\hat{x}$:** *"Ước lượng mới bằng dự đoán cũ cộng thêm một phần của độ mới, được điều tiết thông qua độ lợi Kalman."*  
> **Cập nhật phương sai $P$:** *"Phương sai mới bằng phương sai tiên nghiệm nhân với thừa số $(1 - K)$, chứng minh rằng độ bất định luôn luôn giảm đi sau mỗi lần cập nhật."*

---

## 4. CHỈ SỐ GIÁM SÁT SỨC KHỎE NIS (NORMALIZED INNOVATION SQUARED)

### 4.1 Định nghĩa toán học & Giải phẫu thành phần
Để giám sát xem bộ lọc Kalman có đang hoạt động chuẩn xác hay bị hỏng hóc trong thực tế mà **không có tọa độ thực (Ground Truth)**, ta dùng chỉ số **NIS**:

$$\text{NIS} = \mathbf{y}^T \mathbf{S}^{-1} \mathbf{y}$$

Trong không gian 1 chiều:
$$\text{NIS} = \frac{y^2}{S} = \frac{(z - \hat{x}^-)^2}{P^- + R}$$

#### 🔍 Giải phẫu thành phần:
| Thành phần | Tên gọi | Kiểu / Kích cỡ | Đơn vị | Ý nghĩa vật lý |
| :---: | :--- | :---: | :---: | :--- |
| $\text{NIS}$ | Khoảng cách độ mới chuẩn hóa bình phương | Vô hướng $\ge 0$ | Không thứ nguyên | Thước đo thống kê chuẩn hóa mức độ sai lệch giữa cảm biến và dự đoán. |
| $\mathbf{y}$ | Vector phần dư độ mới (Innovation vector) | Vector $k \times 1$ | $\text{m}$ | Độ lệch giữa phép đo thực tế và dự đoán của cảm biến. |
| $\mathbf{y}^T$ | Chuyển vị của vector độ mới | Vector hàng $1 \times k$ | $\text{m}$ | Dùng để thực hiện phép nhân vô hướng dạng toàn phương. |
| $\mathbf{S}$ | Ma trận hiệp phương sai độ mới | Ma trận $k \times k$ | $\text{m}^2$ | Thước đo dung sai thống kê mà bộ lọc dự kiến cho phép $\mathbf{y}$ dao động. |
| $\mathbf{S}^{-1}$ | Ma trận nghịch đảo của $\mathbf{S}$ | Ma trận $k \times k$ | $1/\text{m}^2$ | Ma trận trọng số chuẩn hóa trong không gian metric Mahalanobis. |

#### 🗣️ Cách phát biểu & Diễn giải bằng lời:
> **Cách đọc:** *"NIS bằng bình phương độ mới chia cho phương sai độ mới (hoặc dạng toàn phương của vector độ mới qua nghịch đảo ma trận hiệp phương sai độ mới)."*  
> **Diễn giải trực giác:** NIS trả lời câu hỏi: *"Khoảng lệch $y$ này có bình thường không so với dung sai dự kiến $S$?"*. Nếu $y = 3\text{m}$ nhưng cảm biến vốn dĩ có dung sai $S = 9\text{m}^2$, thì $\text{NIS} = 9/9 = 1.0$ (hoàn toàn bình thường). Nhưng nếu dung sai chỉ là $S = 0.01\text{m}^2$ mà lệch $3\text{m}$, thì $\text{NIS} = 900$ (báo động đỏ: cảm biến hoặc mô hình đã hỏng!).

---

### 4.2 Tính chất phân phối Chi-bình phương & Kỳ vọng lý thuyết

$$\text{NIS} \sim \chi^2_k \implies \mathbb{E}[\text{NIS}] = k$$

- Với cảm biến vị trí 2D ($x, y$), số chiều đo $k = 2$:
  - Phân phối lý thuyết: $\text{NIS} \sim \chi^2_2$.
  - **Kỳ vọng vàng:** $\mathbb{E}[\text{NIS}] = 2.0$.
  - Khi lấy trung bình trên hàng trăm bước đo, một bộ lọc khỏe mạnh luôn có **Pooled Mean NIS xấp xỉ 2.0** (luôn nhỏ hơn ngưỡng kiểm định $8.0$).

---

## 5. KHÔNG GIAN TRẠNG THÁI & BỘ LỌC KALMAN DẠNG MA TRẬN (PART 5)

### 5.1 Vector trạng thái và Mô hình động học 2D
Để bám vết một xe tự hành di chuyển trên mặt phẳng 2D, vector trạng thái cần gom cả vị trí và vận tốc:

$$\mathbf{x} = \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix} \in \mathbb{R}^4$$

Mô hình chuyển động vận tốc không đổi (Constant Velocity - CV) trong chu kỳ lấy mẫu $\Delta t$:

$$\mathbf{x}_k = \mathbf{F}(\Delta t) \mathbf{x}_{k-1} + \mathbf{w}_{k-1}$$

$$\mathbf{F}(\Delta t) = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 4}$$

#### 🔍 Giải phẫu ma trận chuyển trạng thái $\mathbf{F}(\Delta t)$ (Exercise 5.1):
| Vị trí hàng, cột | Giá trị | Ý nghĩa vật lý |
| :---: | :---: | :--- |
| $\mathbf{F}[0, 0] = 1$ | 1 | Vị trí $x$ mới thừa kế 100% vị trí $x$ cũ. |
| $\mathbf{F}[0, 2] = \Delta t$ | $\Delta t$ | Vị trí $x$ dịch chuyển một đoạn bằng vận tốc $v_x \times \Delta t$. |
| $\mathbf{F}[1, 1] = 1$ | 1 | Vị trí $y$ mới thừa kế 100% vị trí $y$ cũ. |
| $\mathbf{F}[1, 3] = \Delta t$ | $\Delta t$ | Vị trí $y$ dịch chuyển một đoạn bằng vận tốc $v_y \times \Delta t$. |
| $\mathbf{F}[2, 2] = 1$ | 1 | Giả định vận tốc $v_x$ không đổi qua khoảng thời gian $\Delta t$. |
| $\mathbf{F}[3, 3] = 1$ | 1 | Giả định vận tốc $v_y$ không đổi qua khoảng thời gian $\Delta t$. |

#### 🗣️ Cách phát biểu & Diễn giải bằng lời:
> **Cách đọc:** *"Ma trận chuyển trạng thái $\mathbf{F}$ là ma trận đơn vị $4\times 4$ với hai phần tử ngoài đường chéo tại vị trí $(0,2)$ và $(1,3)$ được điền bằng khoảng thời gian $\Delta t$."*  
> **Diễn giải trực giác:** Phép nhân $\mathbf{F}\mathbf{x}$ chính là dạng đại số tuyến tính cô đọng của hệ phương trình chuyển động thẳng đều Newton: $x_{mới} = x_{cũ} + v_x \Delta t$ và $v_{mới} = v_{cũ}$.

---

### 5.2 Mô hình đo lường & Ma trận quan sát $\mathbf{H}$ (Exercise 5.1)

$$\mathbf{z} = \mathbf{H} \mathbf{x} + \mathbf{v}, \quad \mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 4}$$

#### 🔍 Giải phẫu thành phần:
| Thành phần | Kích thước | Đơn vị | Ý nghĩa vật lý |
| :---: | :---: | :---: | :--- |
| $\mathbf{z} = [z_x, z_y]^T$ | Vector $2 \times 1$ | $\text{m}$ | Tọa độ vị trí thực tế thu được từ thiết bị đo (GPS hoặc UWB). |
| $\mathbf{H}$ | Ma trận $2 \times 4$ | Không thứ nguyên | Ma trận chiếu: trích xuất 2 phần tử vị trí $[x, y]$ và che giấu 2 phần tử vận tốc $[v_x, v_y]$. |
| $\mathbf{v}$ | Vector $2 \times 1$ | $\text{m}$ | Nhiễu đo lường ngẫu nhiên của cảm biến với hiệp phương sai $\mathbf{R} \in \mathbb{R}^{2 \times 2}$. |

#### 🗣️ Cách phát biểu & Diễn giải bằng lời:
> **Cách đọc:** *"Ma trận quan sát $\mathbf{H}$ kích thước $2 \times 4$ gồm một khối ma trận đơn vị $2 \times 2$ ở bên trái và một khối ma trận không $2 \times 2$ ở bên phải."*  
> **Diễn giải trực giác:** Phép nhân $\mathbf{H}\mathbf{x}$ biến vector trạng thái 4 chiều $[x, y, v_x, v_y]^T$ thành vector 2 chiều $[x, y]^T$, tương ứng với khả năng phần cứng chỉ đo được tọa độ mà không đo được vận tốc.

---

### 5.3 Ma trận hiệp phương sai nhiễu quá trình $\mathbf{Q}(\Delta t, q)$
Khi xe tăng tốc hoặc phanh ngẫu nhiên với mật độ phổ công suất gia tốc $q$ (đơn vị $(\text{m}/\text{s}^2)^2/\text{Hz}$):

$$\mathbf{Q}(\Delta t, q) = \begin{bmatrix} \mathbf{Q}_{\text{block}} & \mathbf{0} \\ \mathbf{0} & \mathbf{Q}_{\text{block}} \end{bmatrix}_{\text{theo } (x, v_x) \text{ và } (y, v_y)}$$

$$\mathbf{Q}_{\text{block}} = q \begin{bmatrix} \frac{\Delta t^3}{3} & \frac{\Delta t^2}{2} \\ \frac{\Delta t^2}{2} & \Delta t \end{bmatrix}$$

#### 🔍 Giải phẫu thành phần:
- Thành phần vị trí - vị trí: $q \frac{\Delta t^3}{3}$ (tích phân bậc 2 của gia tốc ngẫu nhiên lên vị trí).
- Thành phần vị trí - vận tốc (hiệp phương sai chéo): $q \frac{\Delta t^2}{2}$.
- Thành phần vận tốc - vận tốc: $q \Delta t$ (tích phân bậc 1 của gia tốc lên vận tốc).

---

### 5.4 Chu trình lọc Kalman ma trận đầy đủ (Exercise 5.2)

#### Pha 1: Dự đoán (Predict)
$$\mathbf{x}^- = \mathbf{F} \mathbf{x}$$
$$\mathbf{P}^- = \mathbf{F} \mathbf{P} \mathbf{F}^T + \mathbf{Q}$$

#### Pha 2: Cập nhật (Update)
$$\mathbf{y} = \mathbf{z} - \mathbf{H} \mathbf{x}^-$$
$$\mathbf{S} = \mathbf{H} \mathbf{P}^- \mathbf{H}^T + \mathbf{R}$$
$$\mathbf{K} = \mathbf{P}^- \mathbf{H}^T \mathbf{S}^{-1}$$
$$\mathbf{x} = \mathbf{x}^- + \mathbf{K} \mathbf{y}$$
$$\mathbf{P} = (\mathbf{I} - \mathbf{K} \mathbf{H}) \mathbf{P}^-$$

#### 🔍 Bảng tra cứu kích thước ma trận trong không gian 2D:
| Ký hiệu | Kích cỡ ma trận | Ý nghĩa đại số |
| :---: | :---: | :--- |
| $\mathbf{x}, \mathbf{x}^-$ | $4 \times 1$ | Vector trạng thái (vị trí 2D, vận tốc 2D). |
| $\mathbf{P}, \mathbf{P}^-$ | $4 \times 4$ | Ma trận hiệp phương sai sai số ước lượng (đối xứng, xác định dương). |
| $\mathbf{F}$ | $4 \times 4$ | Ma trận chuyển trạng thái theo mô hình động học. |
| $\mathbf{Q}$ | $4 \times 4$ | Ma trận hiệp phương sai nhiễu gia tốc ngẫu nhiên. |
| $\mathbf{z}, \mathbf{y}$ | $2 \times 1$ | Vector phép đo thực tế và vector độ mới. |
| $\mathbf{H}$ | $2 \times 4$ | Ma trận quan sát chiếu từ không gian trạng thái sang không gian đo. |
| $\mathbf{R}$ | $2 \times 2$ | Ma trận hiệp phương sai nhiễu cảm biến phần cứng. |
| $\mathbf{S}$ | $2 \times 2$ | Ma trận hiệp phương sai độ mới trong không gian đo. |
| $\mathbf{K}$ | $4 \times 2$ | Ma trận độ lợi Kalman (ánh xạ từ phần dư 2 chiều sang điều chỉnh 4 chiều). |
| $\mathbf{I}$ | $4 \times 4$ | Ma trận đơn vị cùng kích cỡ với trạng thái. |

#### 🗣️ Cách phát biểu & Diễn giải bằng lời:
> **$\mathbf{P}^- = \mathbf{F} \mathbf{P} \mathbf{F}^T + \mathbf{Q}$:** *"Ma trận hiệp phương sai tiên nghiệm bằng ma trận động học $\mathbf{F}$ nhân hiệp phương sai cũ nhân $\mathbf{F}$ chuyển vị, rồi cộng thêm ma trận nhiễu quá trình $\mathbf{Q}$."*  
> **$\mathbf{K} = \mathbf{P}^- \mathbf{H}^T \mathbf{S}^{-1}$:** *"Ma trận độ lợi Kalman bằng hiệp phương sai tiên nghiệm nhân $\mathbf{H}$ chuyển vị rồi nhân với nghịch đảo ma trận hiệp phương sai độ mới $\mathbf{S}$."*  
> **Kỳ diệu của vận tốc:** Dù cảm biến chỉ đo vị trí $(x, y)$, cột độ lợi Kalman tương ứng với hàng vận tốc:
> $$K_{v_x} = \frac{P_{x, v_x}^-}{S_{xx}} \neq 0$$
> Nhờ đó, mỗi khi vị trí bị lệch $\Delta x$, bộ lọc **tự động tính ra vận tốc $v_x$ chính xác** mà không cần cảm biến đo tốc độ!

---

## 6. HỢP NHẤT ĐA CẢM BIẾN BẤT ĐỒNG BỘ THỜI GIAN THỰC (PART 6)

### 6.1 Cơ chế vòng lặp sự kiện bất đồng bộ (Asynchronous Event-Driven Loop)
Trên xe tự hành thực tế, các cảm biến gửi dữ liệu về bộ xử lý với chu kỳ và độ trễ khác nhau:
- **LiDAR:** $10\,\text{Hz}$ ($\Delta t = 0.1\,\text{s}$).
- **Radar:** $20\,\text{Hz}$ ($\Delta t = 0.05\,\text{s}$).
- **GPS:** $5\,\text{Hz}$ ($\Delta t = 0.2\,\text{s}$).

#### Thuật toán điều phối (Exercise 6.1):
```python
def run_fusion(meas, q, x0, P0, t0=0.0):
    kf = KalmanFilter(x0, P0)
    t_prev = t0
    log = []
    for ts, name, z, H, R in meas:
        # Bước 1: Tính khoảng thời gian trôi qua từ lần cập nhật trước
        dt = ts - t_prev
        # Bước 2: Nếu thời gian đã trôi qua, dự phóng trạng thái đến thời điểm ts
        if dt > 0:
            kf.predict(make_F(dt), make_Q(dt, q))
        # Bước 3: Cập nhật với đúng ma trận H và R của cảm biến vừa bắn tín hiệu
        kf.update(z, H, R)
        # Bước 4: Lưu mốc thời gian và ghi nhận log
        t_prev = ts
        log.append((ts, kf.x.copy(), kf.P.copy()))
    return log
```

#### 🗣️ Diễn giải nguyên lý vận hành:
> Nếu hai cảm biến bắn tín hiệu cùng một thời điểm ($\Delta t = 0$), bộ lọc chỉ thực hiện `predict` một lần duy nhất, sau đó lần lượt gọi `update` cho từng cảm biến. Tính giao hoán của phân phối chuẩn bảo đảm: **Cập nhật cảm biến nào trước hay sau đều cho ra cùng một ước lượng tối ưu**.

---

## 7. LOẠI TRỪ DỮ LIỆU NGOẠI LAI BẰNG CỔNG KIỂM ĐỊNH CHI-BÌNH PHƯƠNG (PART 7)

### 7.1 Khoảng cách Mahalanobis & Cổng lọc Chi-Squared (Exercise 7.1)
Khi sóng vô tuyến bị phản xạ đa đường (Multipath) hoặc cảm biến bị lóa sáng, số đo có thể nhảy vọt hàng chục mét (Outlier).

$$d^2 = \mathbf{y}^T \mathbf{S}^{-1} \mathbf{y}$$

$$\text{Điều kiện chấp nhận: } d^2 \le \gamma = \chi^2_{\text{ppf}}(p, k)$$

#### 🔍 Giải phẫu thành phần:
| Thành phần | Tên gọi | Đơn vị | Ý nghĩa vật lý |
| :---: | :--- | :---: | :--- |
| $d^2$ | Khoảng cách Mahalanobis bình phương | Không thứ nguyên | Đo lường độ xa thống kê giữa số đo thực tế và hình elip dung sai của bộ lọc. |
| $p$ | Mức xác suất tin cậy (Gate probability) | Tỷ lệ (thường $0.99$) | Tỷ lệ phần trăm các phép đo hợp lệ được kỳ vọng nằm lọt vào trong cổng (ở đây là 99%). |
| $k$ | Bậc tự do | Số nguyên ($k = \dim(\mathbf{z}) = 2$) | Số chiều của không gian đo lường. |
| $\gamma$ | Ngưỡng cắt Chi-bình phương (Threshold) | Vô hướng ($9.21$ khi $p=0.99, k=2$) | Giá trị giới hạn tối đa của khoảng cách Mahalanobis. |

```python
def gated_update(kf, z, H, R, p=0.99):
    y = z - H @ kf.x
    S = H @ kf.P @ H.T + R
    # Giải hệ phương trình S * x = y thay vì nghịch đảo ma trận
    d2 = float(y.T @ np.linalg.solve(S, y))
    # So sánh với ngưỡng phân vị Chi-bình phương
    if d2 > chi2.ppf(p, df=len(z)):
        return False   # Ngoại lai -> Loại bỏ, giữ nguyên trạng thái
    kf.update(z, H, R)
    return True        # Hợp lệ -> Tiến hành cập nhật
```

#### 🗣️ Cách phát biểu & Diễn giải bằng lời:
> **Cách đọc:** *"Khoảng cách Mahalanobis bình phương $d^2$ được tính bằng dạng toàn phương của vector phần dư qua nghịch đảo ma trận hiệp phương sai phần dư. Nếu $d^2$ vượt quá giá trị phân vị $\chi^2$ tại mức $99\%$ với 2 bậc tự do ($d^2 > 9.21$), phép đo bị coi là dị biệt và bị loại bỏ ngay lập tức."*

---

## 8. NHIỆM VỤ CHẨN ĐOÁN CẢM BIẾN XE TỰ HÀNH LYNX-07 (PART 9)

### 8.1 Dấu vân tay của 3 loại lỗi cảm biến (Fault Signatures)

```
[BỆNH 1: BIAS]             [BỆNH 2: UNDERRATED NOISE]     [BỆNH 3: OUTLIER BURST]
- Mean residual LỆCH XA 0  - Mean residual GẦN 0          - Mean residual GẦN 0
- NIS cao ĐỀU ĐẶN          - Cả Mean & Median NIS ĐỀU CAO - Mean NIS CAO, Median NIS BÌNH THƯỜNG
- Cách sửa: Trừ vector bias - Cách sửa: Phóng đại R        - Cách sửa: Đặt cổng Gating
```

### 8.2 Dữ liệu thực nghiệm của Học viên `2A202602467`
Khởi tạo từ mã sinh viên `STUDENT_ID = "2A202602467"` (`seed = 3897423559`):
- **Cảm biến GPS ($n = 450$):** $\text{mean(NIS)} = 2.88$, $\text{median(NIS)} = 1.40$, $\text{mean residual} = [-0.01, -0.02]\,\text{m}$ $\implies$ **Khỏe mạnh hoàn toàn**.
- **Cảm biến UWB ($n = 900$):**
  - $\text{mean(NIS)} = 12.69$ (bất thường rất lớn).
  - $\text{median(NIS)} = 1.66$ (hoàn toàn bình thường trong hơn 90% thời gian).
  - $\max(\text{NIS}) = 198.65$ (bùng nổ xung nhiễu cực mạnh từ $t = 53.84\,\text{s}$ đến $61.93\,\text{s}$).
  - $\text{mean residual} = [0.07, 0.01]\,\text{m}$ (xấp xỉ 0 $\implies$ không có độ lệch bias hằng số).
- **Chẩn đoán:** Cảm biến **`UWB`** bị lỗi **`outlier_burst`**.

### 8.3 Cấu hình sửa lỗi & Kết quả sau khắc phục:
```python
MY_DIAGNOSIS_SENSOR = "UWB"
MY_DIAGNOSIS_TYPE   = "outlier_burst"
FIX_SENSOR = "UWB"
FIX_METHOD = "gate"
```
- **Pooled Mean NIS sau sửa:** Giảm từ $9.42$ xuống **$1.98$** (đạt chuẩn vàng lý thuyết $\mathbb{E}[\text{NIS}] \approx 2.0$, thỏa mãn điều kiện $< 8.0$).
- **Số phép đo hợp lệ được nhận:** $1267 / 1350$ (loại bỏ chính xác $83$ điểm nhiễu ngoại lai của UWB).
- **Độ bất định vị trí $1\sigma$ cuối cùng:**

$$\sigma_{\text{pos}} = \sqrt{P_{xx} + P_{yy}} = \sqrt{0.0673 + 0.0664} \approx \mathbf{0.366}\,\text{m}$$

- **Bán kính tin cậy 95%:**

$$R_{95\%} \approx 2 \cdot \sigma_{\text{pos}} \approx \mathbf{0.732}\,\text{m}$$

---

## 9. BONUS NÂNG CAO: EXTENDED KALMAN FILTER (EKF - PART 8)

### 9.1 Mô hình đo phi tuyến Range-Bearing (Radar / Trạm mặt đất)
Cảm biến Radar đặt tại mốc cố định $\mathbf{s} = [s_x, s_y]^T$ đo cự ly $r$ và góc phương vị $\theta$:

$$\mathbf{z} = \begin{bmatrix} r \\ \theta \end{bmatrix} = h(\mathbf{x}, \mathbf{s}) + \mathbf{v}$$

$$h(\mathbf{x}, \mathbf{s}) = \begin{bmatrix} \sqrt{(x - s_x)^2 + (y - s_y)^2} \\ \operatorname{atan2}(y - s_y, x - s_x) \end{bmatrix}$$

#### 🔍 Giải phẫu thành phần:
| Thành phần | Đơn vị | Ý nghĩa |
| :---: | :---: | :--- |
| $dx = x - s_x$ | $\text{m}$ | Khoảng cách theo phương $x$ từ trạm radar đến xe. |
| $dy = y - s_y$ | $\text{m}$ | Khoảng cách theo phương $y$ từ trạm radar đến xe. |
| $r = \sqrt{dx^2 + dy^2}$ | $\text{m}$ | Cự ly xuyên tâm (Range / Euclidian distance). |
| $\theta = \operatorname{atan2}(dy, dx)$ | $\text{rad}$ | Góc phương vị (Bearing angle) trong hệ tọa độ cực. |

---

### 9.2 Tuyến tính hóa Taylor bậc 1 & Ma trận Jacobian $\mathbf{H}_{\text{rb}}$ (Exercise 8.1)
Do hàm $h(\mathbf{x})$ phi tuyến, ma trận quan sát $\mathbf{H}$ được thay bằng ma trận đạo hàm riêng Jacobian tính tại điểm ước lượng $\mathbf{x}^-$:

$$\mathbf{H}_{\text{rb}} = \left. \frac{\partial h}{\partial \mathbf{x}} \right|_{\mathbf{x}^-} = \begin{bmatrix} \frac{\partial r}{\partial x} & \frac{\partial r}{\partial y} & \frac{\partial r}{\partial v_x} & \frac{\partial r}{\partial v_y} \\ \frac{\partial \theta}{\partial x} & \frac{\partial \theta}{\partial y} & \frac{\partial \theta}{\partial v_x} & \frac{\partial \theta}{\partial v_y} \end{bmatrix} = \begin{bmatrix} \frac{dx}{r} & \frac{dy}{r} & 0 & 0 \\ -\frac{dy}{r^2} & \frac{dx}{r^2} & 0 & 0 \end{bmatrix}$$

#### 🔍 Giải phẫu từng đạo hàm riêng:
- $\frac{\partial r}{\partial x} = \frac{dx}{\sqrt{dx^2 + dy^2}} = \frac{dx}{r} = \cos\theta$: Tốc độ thay đổi cự ly theo vị trí $x$.
- $\frac{\partial r}{\partial y} = \frac{dy}{r} = \sin\theta$: Tốc độ thay đổi cự ly theo vị trí $y$.
- $\frac{\partial \theta}{\partial x} = \frac{-dy}{dx^2 + dy^2} = -\frac{dy}{r^2}$: Tốc độ thay đổi góc phương vị theo vị trí $x$.
- $\frac{\partial \theta}{\partial y} = \frac{dx}{r^2}$: Tốc độ thay đổi góc phương vị theo vị trí $y$.
- Các cột vận tốc bằng 0 vì radar chỉ đo đạc tọa độ vị trí.

#### 🗣️ Cách phát biểu & Diễn giải bằng lời:
> **Cách đọc:** *"Ma trận Jacobian $\mathbf{H}_{\text{rb}}$ kích thước $2\times 4$ gồm hàng 1 là đạo hàm của cự ly $(dx/r, dy/r, 0, 0)$ và hàng 2 là đạo hàm của góc phương vị $(-dy/r^2, dx/r^2, 0, 0)$."*  
> **Xử lý quay vòng góc (Angle Wrapping):** Sai số góc $y_\theta = z_\theta - \hat{\theta}$ phải luôn được chuẩn hóa về $[-\pi, \pi]$ bằng công thức:  
> $$y_\theta = (y_\theta + \pi) \pmod{2\pi} - \pi$$  
> nhằm tránh việc sai số $2^\circ$ bị hiểu nhầm thành $358^\circ$ tại ranh giới nhảy pha.

---

## 10. PLAYBOOK TRIỂN KHAI THỰC TẾ TRONG XE TỰ HÀNH & ROBOT CÔNG NGHIỆP

### 10.1 Đảm bảo ổn định số học qua dạng Joseph (Joseph Form Covariance)
Trong tính toán số học dấu phẩy động hữu hạn, phép trừ ma trận $\mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}^-$ có thể làm mất tính đối xứng hoặc khiến các giá trị riêng trên đường chéo trở thành số âm, dẫn tới sụp đổ bộ lọc (Filter Divergence).

Dạng Joseph chuẩn bảo đảm ma trận luôn luôn đối xứng và xác định dương trong mọi hoàn cảnh:

$$\mathbf{P} = (\mathbf{I} - \mathbf{K} \mathbf{H}) \mathbf{P}^- (\mathbf{I} - \mathbf{K} \mathbf{H})^T + \mathbf{K} \mathbf{R} \mathbf{K}^T$$

#### 🔍 Giải phẫu thành phần:
- Thành phần 1: $(\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}^-(\mathbf{I} - \mathbf{K}\mathbf{H})^T$ có cấu trúc $A P^- A^T$ luôn bán xác định dương.
- Thành phần 2: $\mathbf{K}\mathbf{R}\mathbf{K}^T$ phản ánh lượng nhiễu cảm biến được đưa vào, luôn xác định dương khi $\mathbf{R} > 0$.
- **Thực hành công nghiệp:** Sau mỗi bước cập nhật, luôn cưỡng bức đối xứng:
  $$\mathbf{P} \leftarrow \frac{\mathbf{P} + \mathbf{P}^T}{2}$$

---

### 10.2 Bù trễ dữ liệu đo đến muộn (Out-of-Sequence Measurements - OOSM)
Trong hệ thống mạng CAN-bus hoặc ROS2, tín hiệu camera và GPS thường bị trễ thời gian $\tau$ so với tín hiệu IMU:
1. **Chiến lược Ring-Buffer & Replay:** Lưu lịch sử các trạng thái ước lượng trong một hàng đợi vòng tròn độ dài $\tau_{\max}$. Khi gói tin trễ thời điểm $t - \tau$ cập bến, bộ lọc quay ngược thời gian về mốc đó, chèn phép đo vào bước cập nhật, sau đó chạy lại (fast-forward replay) các bước `predict` và `update` đến thời điểm hiện tại.
2. **Chiến lược Cổng Gating thích ứng:** Tăng ngưỡng cổng $d^2$ theo hàm số của khoảng thời gian trễ $\tau$ để tránh việc dự đoán bị trôi làm loại bỏ nhầm các phép đo chính xác.

---

*Tài liệu thuộc khuôn khổ chương trình đào tạo K4 Track 4 — AI20K. Hoàn thành và nghiệm thu bởi Lâm Quang Anh Quân (MSSV: `2A202602467`).*
