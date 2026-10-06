# CẨM NANG TOÀN DIỆN: BỘ LỌC KALMAN & HỢP NHẤT XÁC SUẤT ĐA CẢM BIẾN
## Track 4 · Ngày 5 — Kalman Filter & Probabilistic Fusion Pilot (Lynx-07)

> **Tác giả:** Lâm Quang Anh Quân — MSSV: `2A202602467`  
> **Repository:** `K4-Track4-Day5-LamQuangAnhQuan-2A202602467`  
> **Mục tiêu học thuật:** Xây dựng từ nguyên lý gốc (First-Principles Thinking) toàn bộ chuỗi thuật toán định vị và hợp nhất cảm biến từ không gian trạng thái 1 chiều đến hệ thống hợp nhất đa cảm biến bất đồng bộ (LiDAR, Radar, GPS, UWB, Camera) và bộ lọc mở rộng phi tuyến (EKF).

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
   - *Mâu thuẫn cơ bản:* Nếu chọn $k$ nhỏ, dữ liệu không đủ độ mượt (nhiễu vẫn còn lớn). Nếu chọn $k$ lớn để triệt tiêu nhiễu, bộ lọc sẽ tích lũy một độ trễ thời gian $\Delta t \approx \frac{k-1}{2}$. Khi xe tự hành phanh gấp hoặc rẽ ngoặt, giá trị trung bình trượt sẽ "ngủ quên" ở quá khứ, khiến xe tiếp tục nghĩ mình đang đi thẳng và gây tai nạn.

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
- **Kỳ vọng $\mu$ (Mean):** Vị trí khả dĩ nhất của vật thể.
- **Phương sai $\sigma^2$ (Variance):** Độ bất định (Uncertainty) hay mức độ hoài nghi về vị trí đó. Phương sai càng nhỏ, niềm tin càng sắc nhọn và đáng tin cậy.

### 2.2 Công thức hợp nhất hai nguồn thông tin Gaussian độc lập
Giả sử ta có hai cảm biến độc lập cùng đo một vị trí $x$:
- Cảm biến 1 cho phân phối $\mathcal{N}(\mu_1, \sigma_1^2)$.
- Cảm biến 2 cho phân phối $\mathcal{N}(\mu_2, \sigma_2^2)$.

Theo định lý Bayes, phân phối kết hợp tỉ lệ thuận với tích của hai hàm mật độ xác suất:
$$p(x \mid z_1, z_2) \propto p(z_1 \mid x) \cdot p(z_2 \mid x) = \frac{1}{2\pi \sigma_1 \sigma_2} \exp\left(-\frac{(x - \mu_1)^2}{2\sigma_1^2} - \frac{(x - \mu_2)^2}{2\sigma_2^2}\right)$$

Bằng cách hoàn thành bình phương ở số mũ, ta thu được một phân phối chuẩn mới $\mathcal{N}(\mu, \sigma^2)$ với:
$$\sigma^2 = \frac{1}{\frac{1}{\sigma_1^2} + \frac{1}{\sigma_2^2}} = \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2 + \sigma_2^2}$$
$$\mu = \sigma^2 \left(\frac{\mu_1}{\sigma_1^2} + \frac{\mu_2}{\sigma_2^2}\right) = \mu_1 + \frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2} (\mu_2 - \mu_1)$$

#### Tính chất quan trọng:
1. **Phương sai sau luôn nhỏ hơn cả hai phương sai thành phần:**
   $$\sigma^2 < \min(\sigma_1^2, \sigma_2^2)$$
   Hợp nhất hai nguồn thông tin độc lập luôn làm tăng độ tự tin, không bao giờ làm giảm độ tự tin.
2. **Tính giao hoán (Commutative):** Thứ tự đưa cảm biến vào hợp nhất hoàn toàn không ảnh hưởng đến kết quả cuối cùng: $\text{fuse}(A, B) = \text{fuse}(B, A)$.

---

## 3. BỘ LỌC KALMAN TUYẾN TÍNH 1 CHIỀU (SCALAR KF)

Mỗi chu kỳ của bộ lọc Kalman gồm hai pha nối tiếp nhau: **Dự đoán (Predict)** và **Cập nhật (Update)**.

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

### 3.1 Ý nghĩa các tham số
- **$Q$ (Process Noise Covariance):** Độ không chắc chắn của mô hình chuyển động (ví dụ: gió thổi, mặt đường mấp mô khiến xe không đi theo đường thẳng tắp).
- **$R$ (Measurement Noise Covariance):** Phương sai sai số của thiết bị đo phần cứng.
- **$P$ (Estimation Error Covariance):** Độ bất định hiện tại của ước lượng trạng thái.
- **$K$ (Kalman Gain):** Tỉ lệ trọng số giao thoa giữa mô hình và cảm biến ($0 \le K \le 1$).

### 3.2 Hành vi của độ lợi Kalman ổn định (Steady-State Gain)
Khi hệ thống chạy qua nhiều chu kỳ, $P$ và $K$ sẽ hội tụ về một giá trị hằng số $K_{\text{ss}}$:
$$K_{\text{ss}} = \frac{\sqrt{Q}}{\sqrt{Q} + \sqrt{R}}$$
- Nếu $Q \gg R$ (mô hình vật lý rất kém hoặc cảm biến cực kỳ chính xác): $K \to 1 \Rightarrow x \approx z$ (tin hoàn toàn vào cảm biến).
- Nếu $Q \ll R$ (mô hình vật lý chuẩn xác tuyệt đối nhưng cảm biến quá nhiễu): $K \to 0 \Rightarrow x \approx x^-$ (bỏ qua cảm biến, tin hoàn toàn vào mô hình động học).

---

## 4. CHỈ SỐ GIÁM SÁT SỨC KHỎE NIS (NORMALIZED INNOVATION SQUARED)

### 4.1 Định nghĩa toán học
Làm thế nào để một chiếc xe tự hành biết được một cảm biến đang bị trôi số liệu hoặc hỏng hóc khi **không có Ground Truth (tọa độ thực tế)** trên đường phố?

Lời giải nằm ở chỉ số **NIS (Normalized Innovation Squared)**:
$$\text{NIS} = \mathbf{y}^T \mathbf{S}^{-1} \mathbf{y}$$
Trong trường hợp 1 chiều:
$$\text{NIS} = \frac{y^2}{S} = \frac{(z - \hat{x}^-)^2}{P^- + R}$$

### 4.2 Ý nghĩa thống kê & Kiểm định giả thuyết
Nếu bộ lọc được cấu hình trung thực (ma trận $Q$ và ma trận $R$ phản ánh đúng thực tế khách quan):
- Độ mới $y \sim \mathcal{N}(0, S)$.
- Khi chuẩn hóa bằng phương sai $S$, đại lượng $\frac{y}{\sqrt{S}}$ tuân theo phân phối chuẩn tắc $\mathcal{N}(0, 1)$.
- Do đó, bình phương của nó tuân theo **phân phối Chi-bình phương với bậc tự do bằng số chiều của phép đo $k$**:
  $$\text{NIS} \sim \chi^2_k$$
  $$\mathbb{E}[\text{NIS}] = k$$

Với cảm biến 2D (đo tọa độ $[x, y]$):
- Bậc tự do $k = 2$.
- Kỳ vọng lý thuyết: $\mathbb{E}[\text{NIS}] = 2$.
- Giá trị trung bình gom cụm (Pooled Mean NIS) trên toàn bộ hành trình của một bộ lọc khỏe mạnh luôn nằm trong khoảng $1.5 \le \text{NIS} \le 3.0$ (chắc chắn $< 8.0$).

---

## 5. KHÔNG GIAN TRẠNG THÁI & BỘ LỌC KALMAN DẠNG MA TRẬN (PART 5)

### 5.1 Vector trạng thái và Mô hình động học 2D
Để theo dõi một vật thể chuyển động trên mặt phẳng $2D$, vector trạng thái bao gồm cả tọa độ và vận tốc:
$$\mathbf{x} = \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix} \in \mathbb{R}^4$$

Giả sử mô hình vận tốc không đổi (Constant Velocity - CV) trong khoảng thời gian $\Delta t$:
$$x(t + \Delta t) = x(t) + v_x(t) \cdot \Delta t$$
$$y(t + \Delta t) = y(t) + v_y(t) \cdot \Delta t$$
$$v_x(t + \Delta t) = v_x(t)$$
$$v_y(t + \Delta t) = v_y(t)$$

#### Ma trận chuyển trạng thái $\mathbf{F}(\Delta t)$ (Exercise 5.1):
$$\mathbf{F}(\Delta t) = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

#### Ma trận quan sát $\mathbf{H}$ (Exercise 5.1):
Cảm biến định vị như GPS hoặc UWB chỉ đo tọa độ vị trí $[x, y]^T$, không đo được trực tiếp vận tốc:
$$\mathbf{z} = \begin{bmatrix} z_x \\ z_y \end{bmatrix} = \mathbf{H} \mathbf{x} + \mathbf{v}$$
$$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 4}$$

### 5.2 Ma trận hiệp phương sai nhiễu quá trình $\mathbf{Q}(\Delta t, q)$
Theo mô hình gia tốc trắng liên tục (Continuous White Noise Acceleration), gia tốc ngẫu nhiên được mô hình hóa bằng nhiễu trắng với mật độ phổ công suất $q$:
$$\mathbf{Q}(\Delta t, q) = \int_0^{\Delta t} \mathbf{F}(\tau) \mathbf{G} q \mathbf{G}^T \mathbf{F}(\tau)^T d\tau$$
Cho mỗi trục độc lập $(x, v_x)$ và $(y, v_y)$:
$$\mathbf{Q}_{\text{block}} = q \begin{bmatrix} \frac{\Delta t^3}{3} & \frac{\Delta t^2}{2} \\ \frac{\Delta t^2}{2} & \Delta t \end{bmatrix}$$

```python
def make_F(dt):
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    return F

def make_H():
    H = np.zeros((2, 4))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    return H
```

### 5.3 Cài đặt Lớp `KalmanFilter` nhiều chiều (Exercise 5.2)

```python
class KalmanFilter:
    def __init__(self, x0, P0):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)

    def predict(self, F, Q):
        # x⁻ = F @ x
        self.x = F @ self.x
        # P⁻ = F @ P @ F.T + Q
        self.P = F @ self.P @ F.T + Q

    def update(self, z, H, R):
        # Innovation (độ mới): y = z - H @ x⁻
        y = z - H @ self.x
        # Hiệp phương sai độ mới: S = H @ P⁻ @ H.T + R
        S = H @ self.P @ H.T + R
        # Độ lợi Kalman: K = P⁻ @ H.T @ S⁻¹
        # (Dùng np.linalg.solve thay vì inv để ổn định số học)
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(len(z)))
        # Cập nhật trạng thái: x = x⁻ + K @ y
        self.x = self.x + K @ y
        # Cập nhật hiệp phương sai: P = (I - K @ H) @ P⁻
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P
        return y, S, K
```

#### Điểm mấu chốt: Vận tốc xuất hiện từ đâu khi chỉ đo vị trí?
Dù cảm biến chỉ đo vị trí $(x, y)$, thành phần ma trận $K$ ở hàng vận tốc sẽ tự động có giá trị khác 0:
$$K_{v_x} = \frac{P_{x, v_x}^-}{S}$$
Nhờ sự tương quan (Covariance) giữa vị trí và vận tốc được lan truyền trong bước `predict`, mỗi khi vị trí thay đổi một khoảng bất thường $\Delta x$, bộ lọc Kalman suy diễn ra một sự thay đổi tương ứng về vận tốc $v_x$.

---

## 6. HỢP NHẤT ĐA CẢM BIẾN BẤT ĐỒNG BỘ THỜI GIAN THỰC (PART 6)

### 6.1 Cơ chế xử lý luồng bất đồng bộ (Asynchronous Fusion Loop)
Trong xe tự hành, các cảm biến có tần số lấy mẫu và độ trễ khác nhau:
- **LiDAR:** $10\,\text{Hz}$ ($\Delta t = 0.1\,\text{s}$), độ chính xác cao.
- **Radar:** $20\,\text{Hz}$ ($\Delta t = 0.05\,\text{s}$), đo trực tiếp vận tốc Doppler.
- **GPS-RTK:** $5\,\text{Hz}$ ($\Delta t = 0.2\,\text{s}$), tọa độ toàn cầu nhưng độ trễ lớn.

```
Thời gian t:   0.00    0.05    0.10    0.15    0.20    0.25 (s)
Radar (20Hz):   ●       ●       ●       ●       ●       ●
LiDAR (10Hz):   ■               ■               ■
GPS (5Hz):      ▲                               ▲
```

#### Quy tắc điều phối:
1. Sắp xếp toàn bộ dữ liệu hỗn hợp theo mốc thời gian tăng dần `ts`.
2. Giữ biến `t_prev` lưu mốc thời gian của bước cập nhật trước.
3. Khi nhận được một phép đo bất kỳ tại thời điểm $t_k$:
   - Tính bước thời gian: $\Delta t = t_k - t_{k-1}$.
   - Nếu $\Delta t > 0$: chạy hàm `predict(make_F(dt), make_Q(dt, q))` để dịch chuyển ước lượng trạng thái tới đúng mốc thời gian hiện tại.
   - Nếu $\Delta t = 0$ (hai cảm biến ghi nhận đồng thời): **bỏ qua bước predict, thực hiện update trực tiếp**.
   - Gọi `update(z, H, R)` với ma trận quan sát $\mathbf{H}$ và ma trận nhiễu $\mathbf{R}$ của riêng cảm biến đó.
   - Cập nhật $t_{k-1} \leftarrow t_k$ và ghi nhận trạng thái vào log.

```python
def run_fusion(meas, q, x0, P0, t0=0.0):
    kf = KalmanFilter(x0, P0)
    t_prev = t0
    log = []
    for ts, name, z, H, R in meas:
        dt = ts - t_prev
        if dt > 0:
            kf.predict(make_F(dt), make_Q(dt, q))
        kf.update(z, H, R)
        t_prev = ts
        log.append((ts, kf.x.copy(), kf.P.copy()))
    return log
```

---

## 7. LOẠI TRỪ DỮ LIỆU NGOẠI LAI BẰNG CỔNG KIỂM ĐỊNH CHI-BÌNH PHƯƠNG (PART 7)

### 7.1 Khoảng cách Mahalanobis & Cổng lọc Chi-Squared
Khi một cảm biến bị lỗi hoặc gặp phản xạ đa đường (Multipath reflection), nó có thể bắn ra một giá trị bất thường cách xa vị trí thực hàng chục mét (Outlier). Nếu đưa giá trị này vào cập nhật Kalman, toàn bộ trạng thái của xe sẽ bị giật lệch nghiêm trọng.

Để bảo vệ bộ lọc, ta sử dụng **Cổng lọc kiểm định Chi-bình phương (Chi-squared Validation Gate)**:
1. Tính khoảng cách Mahalanobis bình phương của phần dư:
   $$d^2 = \mathbf{y}^T \mathbf{S}^{-1} \mathbf{y}$$
2. So sánh $d^2$ với ngưỡng phân vị xác suất $p$ (thường chọn $p = 0.99$ hoặc $0.999$) của phân phối $\chi^2$ với bậc tự do $k = \dim(\mathbf{z})$:
   $$\text{Ngưỡng cắt } \gamma = \chi^2_{\text{ppf}}(p, k)$$
3. **Quy tắc quyết định:**
   - Nếu $d^2 \le \gamma$: Phép đo nằm trong phạm vi dung sai thống kê hợp lý $\Rightarrow$ **Chấp nhận và cập nhật**.
   - Nếu $d^2 > \gamma$: Phép đo là ngoại lai (outlier) $\Rightarrow$ **Từ chối và giữ nguyên trạng thái bộ lọc**.

```python
def gated_update(kf, z, H, R, p=0.99):
    y = z - H @ kf.x
    S = H @ kf.P @ H.T + R
    d2 = y @ np.linalg.solve(S, y)
    if d2 > chi2.ppf(p, df=len(z)):
        return False
    kf.update(z, H, R)
    return True
```

---

## 8. NHIỆM VỤ CHẨN ĐOÁN CẢM BIẾN XE TỰ HÀNH LYNX-07 (PART 9)

### 8.1 Dấu vân tay của 3 loại lỗi cảm biến (Fault Fingerprints)

| Loại lỗi | Dấu vân tay NIS | Dấu vân tay Residual | Cơ chế phát sinh thực tế | Phương pháp khắc phục tối ưu |
| :--- | :--- | :--- | :--- | :--- |
| **Lệch hằng số (Constant Bias)** | NIS cao đều đặn trên toàn hành trình | Vector mean residual lệch xa mốc $[0, 0]$ (có offset hằng số $\ge 1\,\text{m}$) | Anten GPS bị lệch tâm hoặc gắn sai góc so với trục xe | `bias`: Trừ vector offset trung bình khỏi số đo |
| **Nhiễu đánh giá thấp (Underrated Noise)** | NIS cao spiky, cả mean và median NIS đều tăng cao | Mean residual xấp xỉ $[0, 0]$, nhưng độ lệch chuẩn phân tán rộng | Nhà sản xuất công bố $\sigma_{\text{spec}} = 0.5\,\text{m}$ nhưng thực tế nhiễu lên tới $2\,\text{m}$ | `inflate_R`: Nhân ma trận $\mathbf{R}$ với hệ số phóng đại $\frac{\text{mean(NIS)}}{2}$ |
| **Chùm ngoại lai đột biến (Outlier Burst)** | **Mean NIS rất cao ($\ge 10$), nhưng Median NIS vẫn ở mức bình thường ($\approx 1.5 - 2.0$)** | Mean residual gần $[0, 0]$, các cú vọt NIS (spike) tập trung gom cụm trong một khoảng thời gian | Tín hiệu UWB bị che khuất tạm thời hoặc phản xạ qua kết cấu kim loại | `gate`: Kích hoạt cổng kiểm định Chi-squared `gated_update` |

### 8.2 Kết quả chẩn đoán số liệu cho Học viên `2A202602467`
Từ dữ liệu sinh ra bởi hạt giống SHA-256 (`seed = 3897423559`):
- **Cảm biến GPS ($n = 450$):**
  - $\text{mean(NIS)} = 2.88$
  - $\text{median(NIS)} = 1.40$
  - $\text{mean residual} = [-0.01, -0.02]\,\text{m}$
  - $\Rightarrow$ Hoàn toàn khỏe mạnh và khớp chuẩn với thông số kỹ thuật.
- **Cảm biến UWB ($n = 900$):**
  - $\text{mean(NIS)} = 12.69$ (bất thường rất lớn)
  - $\text{median(NIS)} = 1.66$ (hoàn toàn bình thường trong phần lớn thời gian)
  - $\text{max(NIS)} = 198.65$ (các cú nhảy cực mạnh từ $t = 53.84\,\text{s}$ đến $61.93\,\text{s}$)
  - $\text{mean residual} = [0.07, 0.01]\,\text{m}$ (không có độ lệch hằng số $\Rightarrow$ loại trừ giả thuyết `bias`).
  - Median không bị kéo cao $\Rightarrow$ loại trừ giả thuyết `underrated_noise`.

#### Kết luận khai báo:
```python
MY_DIAGNOSIS_SENSOR = "UWB"
MY_DIAGNOSIS_TYPE   = "outlier_burst"
FIX_SENSOR = "UWB"
FIX_METHOD = "gate"
```

### 8.3 Kết quả sau khắc phục:
- **Pooled Mean NIS:** Giảm ngoạn mục từ $12.69$ xuống **$1.98$** (đạt chuẩn tuyệt đối $< 8.0$, xấp xỉ kỳ vọng lý thuyết $2.0$).
- **Pooled Median NIS:** **$1.34$**.
- **Số lượng phép đo được chấp nhận:** $1267 / 1350$ (loại bỏ chính xác 83 điểm đo ngoại lai bất thường của UWB).
- **Độ bất định vị trí $1\sigma$ cuối cùng:** $\sqrt{P_{xx} + P_{yy}} = \sqrt{0.066976 + 0.066976} \approx \mathbf{0.366}\,\text{m}$.
- **Bán kính tin cậy 95%:** $\approx \mathbf{0.732}\,\text{m}$.

---

## 9. BONUS NÂNG CAO: EXTENDED KALMAN FILTER (EKF - PART 8)

### 9.1 Mô hình đo phi tuyến Range-Bearing
Cảm biến Radar hoặc Camera đặt tại vị trí $\mathbf{s} = [s_x, s_y]^T$ đo khoảng cách $r$ và góc phương vị $\theta$:
$$\mathbf{z} = \begin{bmatrix} r \\ \theta \end{bmatrix} = h(\mathbf{x}, \mathbf{s}) + \mathbf{v}$$
$$h(\mathbf{x}, \mathbf{s}) = \begin{bmatrix} \sqrt{(x - s_x)^2 + (y - s_y)^2} \\ \operatorname{atan2}(y - s_y, x - s_x) \end{bmatrix}$$

### 9.2 Ma trận Jacobian $\mathbf{H}_{\text{rb}}$
Vì hàm $h(\mathbf{x})$ phi tuyến tính, ta phải lấy xấp xỉ Taylor bậc 1 (tính ma trận Jacobian) tại điểm ước lượng hiện tại $\mathbf{x}^-$:
$$dx = x - s_x, \quad dy = y - s_y, \quad r^2 = dx^2 + dy^2, \quad r = \sqrt{r^2}$$
$$\mathbf{H}_{\text{rb}} = \frac{\partial h}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial r}{\partial x} & \frac{\partial r}{\partial y} & \frac{\partial r}{\partial v_x} & \frac{\partial r}{\partial v_y} \\ \frac{\partial \theta}{\partial x} & \frac{\partial \theta}{\partial y} & \frac{\partial \theta}{\partial v_x} & \frac{\partial \theta}{\partial v_y} \end{bmatrix} = \begin{bmatrix} \frac{dx}{r} & \frac{dy}{r} & 0 & 0 \\ -\frac{dy}{r^2} & \frac{dx}{r^2} & 0 & 0 \end{bmatrix}$$

```python
def h_rb(x, s):
    dx, dy = x[0] - s[0], x[1] - s[1]
    return np.array([np.hypot(dx, dy), np.arctan2(dy, dx)])

def H_rb(x, s):
    dx, dy = x[0] - s[0], x[1] - s[1]
    r2 = dx**2 + dy**2
    r = np.sqrt(r2)
    H = np.zeros((2, 4))
    H[0, 0] = dx / r
    H[0, 1] = dy / r
    H[1, 0] = -dy / r2
    H[1, 1] = dx / r2
    return H
```

### 9.3 Xử lý quay vòng góc pha (Angle Wrapping)
Phần dư của góc phương vị $y_\theta = z_\theta - \hat{\theta}$ có thể bị nhảy pha tại ranh giới $[-\pi, \pi]$ (ví dụ: $-179^\circ - (+179^\circ) = -358^\circ$, thực chất chỉ lệch $2^\circ$). Ta bắt buộc phải chuẩn hóa góc về khoảng $[-\pi, \pi]$:
$$y_\theta = (y_\theta + \pi) \pmod{2\pi} - \pi$$

---

## 10. PLAYBOOK TRIỂN KHAI THỰC TẾ TRONG XE TỰ HÀNH & ROBOT CÔNG NGHIỆP

### 10.1 Phòng chống thoái hóa ma trận hiệp phương sai (Numerical Stability)
Trong tính toán dấu phẩy động 32-bit hoặc 64-bit, phép trừ ma trận $\mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}^-$ có thể làm mất tính đối xứng hoặc xuất hiện trị riêng âm do sai số làm tròn.
- **Dạng Joseph (Joseph Form):** Luôn bảo toàn tính đối xứng và xác định dương:
  $$\mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H}) \mathbf{P}^- (\mathbf{I} - \mathbf{K}\mathbf{H})^T + \mathbf{K} \mathbf{R} \mathbf{K}^T$$
- **Cưỡng bức đối xứng:** Sau mỗi bước cập nhật:
  $$\mathbf{P} = \frac{\mathbf{P} + \mathbf{P}^T}{2}$$

### 10.2 Bù trễ phần cứng (Hardware Latency & Out-of-Sequence Measurements - OOSM)
Nếu một cảm biến có độ trễ truyền thông $\tau$ (dữ liệu đo ở thời điểm $t - \tau$ nhưng tới bộ xử lý ở thời điểm $t$):
- *Cách 1 (Buffer & Rewind):* Lưu trữ bộ đệm lịch sử trạng thái trong $\tau_{\max}$ giây. Khi nhận được gói tin trễ, quay ngược bộ lọc về thời điểm $t - \tau$, chèn phép đo vào và tái chạy (replay) các bước predict/update tới hiện tại.
- *Cách 2 (Mở rộng trạng thái):* Đưa độ trễ vào trạng thái giả định để cập nhật trực tiếp.

---
*Bản quyền tài liệu thuộc về khóa đào tạo K4 Track 4 — AI20K.*
