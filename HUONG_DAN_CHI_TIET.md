# HƯỚNG DẪN CHI TIẾT: BỘ LỌC KALMAN VÀ HỢP NHẤT DỮ LIỆU ĐA CẢM BIẾN
## Bài Thực Hành K4 · Track 4 · Ngày 5 — Kalman Filter & Sensor Fusion (Xe Tự Hành Lynx-07)

> **Người thực hiện:** Lâm Quang Anh Quân — MSSV: `2A202602467`  
> **Kho mã nguồn (Repo):** `K4-Track4-Day5-LamQuangAnhQuan-2A202602467`  
> **Mục tiêu bài lab:** Tự tay xây dựng từ đầu bộ lọc Kalman bằng Python và thư viện NumPy. Hiểu bản chất cách kết hợp số liệu từ nhiều cảm biến (GPS, UWB, LiDAR, Radar, Camera) để định vị xe tự hành chính xác, loại bỏ số đo rác (outlier) và chẩn đoán lỗi phần cứng cảm biến khi đang chạy trên đường.

---

## MỤC LỤC
1. [Bản Chất Vấn Đề: Tại Sao Phải Dùng Bộ Lọc Xác Suất?](#1-bản-chất-vấn-đề-tại-sao-phải-dùng-bộ-lọc-xác-suất)
2. [Đo Lường Không Chắc Chắn: Biểu Diễn Bằng Phân Phối Chuẩn (Gaussian)](#2-đo-lường-không-chắc-chắn-biểu-diễn-bằng-phân-phối-chuẩn-gaussian)
3. [Bộ Lọc Kalman 1 Chiều: Hai Bước Lặp Dự Đoán & Hiệu Chỉnh](#3-bộ-lọc-kalman-1-chiều-hai-bước-lặp-dự-đoán--hiệu-chỉnh)
4. [Theo Dõi Sức Khỏe Bộ Lọc Bằng Chỉ Số NIS](#4-theo-dõi-sức-khỏe-bộ-lọc-bằng-chỉ-số-nis)
5. [Bộ Lọc Kalman Dạng Ma Trận Cho Xe Chạy Trên Mặt Phẳng 2D (Part 5)](#5-bộ-lọc-kalman-dạng-ma-trận-cho-xe-chạy-trên-mặt-phẳng-2d-part-5)
6. [Hợp Nhất Nhiều Cảm Biến Chạy Lệch Tần Số (Part 6)](#6-hợp-nhất-nhiều-cảm-biến-chạy-lệch-tần-số-part-6)
7. [Lọc Bỏ Điểm Đo Bất Thường Bằng Cổng Chi-Bình Phương (Part 7)](#7-lọc-bỏ-điểm-đo-bất-thường-bằng-cổng-chi-bình-phương-part-7)
8. [Nhiệm Vụ Thực Tế: Bắt Bệnh Cảm Biến Trên Xe Lynx-07 (Part 9)](#8-nhiệm-vụ-thực-tế-bắt-bệnh-cảm-biến-trên-xe-lynx-07-part-9)
9. [Mở Rộng Cho Cảm Biến Phi Tuyến: Bộ Lọc EKF (Part 8 Bonus)](#9-mở-rộng-cho-cảm-biến-phi-tuyến-bộ-lọc-ekf-part-8-bonus)
10. [Kinh Nghiệm Thực Tế Khi Triển Khai Cho Xe Tự Hành & Robot](#10-kinh-nghiệm-thực-tế-khi-triển-khai-cho-xe-tự-hành--robot)

---

## 1. BẢN CHẤT VẤN ĐỀ: TẠI SAO PHẢI DÙNG BỘ LỌC XÁC SUẤT?

### 1.1 Thách thức thực tế: Cảm biến luôn có nhiễu và bộ lọc cổ điển gây trễ
Khi lập trình xe tự hành hoặc robot, việc đọc dữ liệu từ cảm biến phần cứng (như GPS, UWB, LiDAR) gặp phải hai khó khăn lớn:
1. **Nhiễu đo lường:** Sóng vô tuyến bị dội tường, thiết bị nóng lên, hay thời tiết xấu làm cho con số cảm biến trả về luôn rung lắc liên tục quanh vị trí thật.
2. **Cái bẫy trễ pha khi lấy trung bình cộng:**
   - Cách tự nhiên nhất mà mọi người hay nghĩ đến là **lấy trung bình trượt** (Moving Average) qua $k$ mẫu gần nhất:

$$\bar{z}_t = \frac{1}{k} \sum_{i=0}^{k-1} z_{t-i}$$

#### 🔍 Chi tiết từng thành phần trong công thức:
| Ký hiệu | Tên gọi | Kiểu giá trị | Đơn vị | Ý nghĩa thực tế |
| :---: | :--- | :---: | :---: | :--- |
| $\bar{z}_t$ | Vị trí sau khi làm mượt | Số thực | mét ($\text{m}$) | Con số vị trí đã được làm êm tại thời điểm hiện tại $t$. |
| $k$ | Kích thước cửa sổ | Số nguyên dương | - | Số lượng mẫu đo trong quá khứ được gom lại để chia trung bình. |
| $z_{t-i}$ | Mẫu đo quá khứ | Số thực | mét ($\text{m}$) | Con số cảm biến đã đo được cách đây $i$ bước thời gian. |
| $\sum$ | Dấu tổng | Phép toán | - | Cộng dồn $k$ mẫu đo gần nhất lại với nhau. |

#### 🗣️ Cách đọc và bản chất vấn đề:
> **Cách đọc:** *"Vị trí làm mượt ở thời điểm hiện tại bằng tổng của $k$ giá trị đo gần nhất chia đều cho $k$."*  
> **Hạn chế chết người:** Nếu chọn $k$ nhỏ (ví dụ 3 mẫu), đường đi vẫn rung lắc mạnh. Nếu chọn $k$ lớn (ví dụ 20 mẫu) để đường đi êm ái, xe sẽ bị **trễ thời gian** khoảng $\frac{k-1}{2}$ bước. Khi xe phanh gấp hoặc bẻ lái gấp, bộ lọc trung bình vẫn đang bận "ngủ quên" với các số liệu cũ trong quá khứ, khiến hệ thống tưởng xe vẫn đang chạy thẳng và dẫn tới đâm va.

```
[Dữ liệu cảm biến rung lắc] ──> [Trung bình trượt] ──> Êm hơn NHƯNG phản xạ cực kỳ chậm trễ!
[Dữ liệu cảm biến rung lắc] ──> [Bộ lọc Kalman]   ──> Êm ái VÀ phản xạ tức thì (nhờ dự đoán vật lý)!
```

### 1.2 Cách giải quyết của Rudolf Kalman: Kết hợp vật lý và đo đạc
Thay vì chỉ thụ động nhìn lại các số liệu quá khứ, bộ lọc Kalman làm hai việc cùng lúc:
- **Dùng quy luật chuyển động vật lý để dự đoán trước:** Xe đang ở đâu, chạy vận tốc bao nhiêu thì sau một khoảng thời gian $\Delta t$ nó sẽ di chuyển tới đâu.
- **Dùng số đo mới của cảm biến để hiệu chỉnh lại:** Khi cảm biến gửi dữ liệu về, bộ lọc so sánh xem thực tế lệch với dự đoán bao nhiêu, từ đó tìm điểm dung hòa tối ưu nhất.

---

## 2. ĐO LƯỜNG KHÔNG CHẮC CHẮN: BIỂU DIỄN BẰNG PHÂN PHỐI CHUẨN (GAUSSIAN)

### 2.1 Biểu diễn vị trí bằng hình chuông xác suất
Trong bộ lọc Kalman, ta không bao giờ khẳng định *"Xe đang ở chính xác mốc 5.0 mét"*. Thay vào đó, ta nói *"Vị trí xe khả dĩ nhất là 5.0 mét, nhưng có độ chênh lệch sai số khoảng 0.2 mét"*. Ta biểu diễn điều này bằng phân phối chuẩn $\mathcal{N}(\mu, \sigma^2)$:

$$p(x) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

#### 🔍 Chi tiết từng thành phần:
| Ký hiệu | Tên gọi | Kiểu | Đơn vị | Ý nghĩa thực tế |
| :---: | :--- | :---: | :---: | :--- |
| $p(x)$ | Mật độ xác suất | Số thực $\ge 0$ | $1/\text{m}$ | Mức độ tin cậy rằng xe thực sự đang đứng ở tọa độ $x$. |
| $x$ | Tọa độ đang xét | Số thực | mét ($\text{m}$) | Vị trí bất kỳ trên đường mà ta muốn kiểm tra. |
| $\mu$ | Giá trị trung bình | Số thực | mét ($\text{m}$) | Điểm đỉnh của hình chuông — vị trí có khả năng đúng cao nhất. |
| $\sigma^2$ | Phương sai | Số thực dương | $\text{m}^2$ | Thước đo độ bất định (độ nghi ngờ). Càng nhỏ nghĩa là ta càng chắc chắn về vị trí của xe. |
| $\sigma$ | Độ lệch chuẩn | Số thực dương | mét ($\text{m}$) | Độ rộng sai số. Khoảng $\pm 1\sigma$ bao quát 68% khả năng, còn $\pm 2\sigma$ bao quát khoảng 95% khả năng. |

#### 🗣️ Cách đọc và ý nghĩa trực quan:
> **Cách đọc:** *"Mật độ xác suất tại vị trí $x$ giảm dần theo hàm mũ của khoảng cách từ $x$ tới tâm $\mu$, chia cho hai lần phương sai."*  
> **Hình dung đời thường:** $\mu$ là điểm ta nhắm tới, còn $\sigma$ là độ rộng của vòng ngắm. Vòng ngắm càng hẹp thì tay súng càng tự tin và bắn càng chuẩn.

---

### 2.2 Khi hai cảm biến cùng đo một vị trí thì gộp lại thế nào?
Giả sử trên xe có hai cảm biến độc lập cùng đo vị trí $x$:
- Cảm biến 1 báo: vị trí $\mu_1$, độ sai số phương sai $\sigma_1^2$.
- Cảm biến 2 báo: vị trí $\mu_2$, độ sai số phương sai $\sigma_2^2$.

Theo quy tắc xác suất Bayes, khi gộp hai nguồn thông tin độc lập này lại, ta nhân hai đường cong hình chuông với nhau. Kết quả thu được **vẫn là một hình chuông chuẩn mới** với tâm $\mu$ và phương sai $\sigma^2$:

$$\sigma^2 = \frac{\sigma_1^2 \cdot \sigma_2^2}{\sigma_1^2 + \sigma_2^2} = \frac{1}{\frac{1}{\sigma_1^2} + \frac{1}{\sigma_2^2}}$$

$$\mu = \mu_1 + \frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2} (\mu_2 - \mu_1)$$

#### 🔍 Chi tiết từng thành phần:
| Ký hiệu | Tên gọi | Kiểu | Đơn vị | Ý nghĩa thực tế |
| :---: | :--- | :---: | :---: | :--- |
| $\sigma^2$ | Phương sai gộp | Số thực | $\text{m}^2$ | Mức độ bất định sau khi đã lắng nghe cả 2 cảm biến. |
| $\frac{1}{\sigma^2}$ | Độ chính xác (Precision) | Số thực | $1/\text{m}^2$ | Thước đo lượng thông tin. Độ chính xác sau khi gộp bằng **tổng độ chính xác** của từng cảm biến. |
| $\mu$ | Vị trí chốt cuối cùng | Số thực | mét ($\text{m}$) | Vị trí dung hòa hợp lý nhất giữa hai cảm biến. |
| $\frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2}$ | Tỷ lệ nhường bước | Khoảng $[0, 1]$ | Không thứ nguyên | Trọng số quyết định: cảm biến 1 sẽ nhường bao nhiêu bước về phía cảm biến 2. |
| $\mu_2 - \mu_1$ | Độ lệch giữa hai cảm biến | Số thực | mét ($\text{m}$) | Khoảng cách vênh nhau giữa hai số đo. |

#### 🗣️ Cách phát biểu và nguyên lý cốt lõi:
> **Cách đọc:** *"Vị trí kết hợp bằng vị trí cảm biến 1 cộng thêm một phần độ lệch giữa hai cảm biến, trong đó phần điều chỉnh này phụ thuộc vào tỷ lệ phương sai của cảm biến 1 trên tổng phương sai cả hai."*  
> **Quy tắc vàng:** $\sigma^2 < \min(\sigma_1^2, \sigma_2^2)$ — **Phương sai sau khi kết hợp luôn luôn nhỏ hơn phương sai của cảm biến xịn nhất**. Nói một cách dễ hiểu: Lắng nghe thêm một nguồn tin độc lập (dù cảm biến đó có hơi nhiễu) luôn làm ta chắc chắn hơn là chỉ nghe một bên đơn độc!

---

## 3. BỘ LỌC KALMAN 1 CHIỀU: HAI BƯỚC LẶP DỰ ĐOÁN & HIỆU CHỈNH

Một chu kỳ hoạt động của bộ lọc Kalman lặp đi lặp lại đúng 2 giai đoạn:
1. **Dự đoán (Predict):** Dùng quán tính vật lý đẩy trạng thái tới thời điểm hiện tại.
2. **Cập nhật (Update):** Khi có số đo mới từ cảm biến gửi về, điều chỉnh lại vị trí.

```
                      ┌────────────────────────────────────────┐
                      │           Trạng thái ban đầu           │
                      │               (x₀, P₀)                 │
                      └──────────────────┬─────────────────────┘
                                         │
                ┌────────────────────────▼────────────────────────┐
                │             1. BƯỚC DỰ ĐOÁN (PREDICT)           │
                │    Vị trí dự tính:    x⁻ = x̂                    │
                │    Độ nghi ngờ tăng:  P⁻ = P + Q                │
                └────────────────────────┬────────────────────────┘
                                         │
                                         │ [Nhận được số đo z từ cảm biến]
                                         ▼
                ┌─────────────────────────────────────────────────┐
                │             2. BƯỚC CẬP NHẬT (UPDATE)           │
                │    Độ lệch thực tế:   y = z - x⁻                │
                │    Độ nghi ngờ gộp:   S = P⁻ + R                │
                │    Trọng số Kalman:   K = P⁻ / S                │
                │    Chốt vị trí mới:   x̂ = x⁻ + K·y              │
                │    Co hẹp độ nghi ngờ: P = (1 - K)·P⁻           │
                └────────────────────────┬────────────────────────┘
                                         │
                                         └─────> Quay lại bước 1 cho chu kỳ kế tiếp
```

### 3.1 Giai đoạn 1: Dự đoán (Predict Phase)

$$\hat{x}^- = \hat{x}_{k-1}$$

$$P^- = P_{k-1} + Q$$

#### 🔍 Chi tiết từng biến số:
| Ký hiệu | Tên gọi | Đơn vị | Ý nghĩa thực tế |
| :---: | :--- | :---: | :--- |
| $\hat{x}^-$ | Vị trí dự tính trước | mét ($\text{m}$) | Vị trí ta đoán xe đang đứng trước khi đọc số đo cảm biến. |
| $\hat{x}_{k-1}$ | Vị trí đã chốt ở bước trước | mét ($\text{m}$) | Vị trí tin cậy nhất ở chu kỳ liền trước. |
| $P^-$ | Độ bất định dự tính | $\text{m}^2$ | Mức độ nghi ngờ về vị trí dự đoán vừa tính ra. |
| $P_{k-1}$ | Độ bất định ở bước trước | $\text{m}^2$ | Mức độ nghi ngờ đã biết ở bước trước. |
| $Q$ | Nhiễu quá trình (Process Noise) | $\text{m}^2$ | Mức độ biến động ngẫu nhiên của môi trường (gió tạt, trơn trượt bánh, đường mấp mô). |

#### 🗣️ Phát biểu bằng lời:
> *"Khi thời gian trôi đi mà chưa có số đo mới, ta tạm giữ nguyên vị trí theo quán tính, nhưng độ nghi ngờ $P^-$ bắt buộc phải cộng thêm một lượng $Q$, vì thế giới bên ngoài luôn có những tác động ngẫu nhiên làm ta bớt chắc chắn đi."*

---

### 3.2 Giai đoạn 2: Cập nhật theo số đo cảm biến (Update Phase)

$$y = z - \hat{x}^-$$

$$S = P^- + R$$

$$K = \frac{P^-}{S} = \frac{P^-}{P^- + R}$$

$$\hat{x} = \hat{x}^- + K \cdot y$$

$$P = (1 - K) \cdot P^-$$

#### 🔍 Chi tiết từng biến số:
| Ký hiệu | Tên gọi | Đơn vị | Ý nghĩa thực tế |
| :---: | :--- | :---: | :--- |
| $z$ | Số đo thực tế | mét ($\text{m}$) | Giá trị thực tế mà cảm biến vừa đo được và gửi về vi điều khiển. |
| $y$ | Độ lệch đo lường (Residual) | mét ($\text{m}$) | Cảm biến đo được lệch bao nhiêu so với vị trí ta vừa dự đoán trước đó. |
| $R$ | Nhiễu cảm biến | $\text{m}^2$ | Sai số cố hữu của cảm biến phần cứng (ghi trên thông số kỹ thuật của nhà sản xuất). |
| $S$ | Tổng độ bất định gộp | $\text{m}^2$ | Tổng cộng cả độ nghi ngờ của dự đoán ($P^-$) lẫn độ nhiễu của cảm biến ($R$). |
| $K$ | Trọng số Kalman (Kalman Gain) | Không thứ nguyên ($0 \le K \le 1$) | Tỷ lệ chia phần: ta tin vào cảm biến bao nhiêu phần, tin vào dự đoán bao nhiêu phần. |
| $\hat{x}$ | Vị trí chốt cuối cùng | mét ($\text{m}$) | Vị trí chính xác nhất sau khi đã kết hợp cả dự đoán và cảm biến. |
| $P$ | Độ bất định chốt cuối | $\text{m}^2$ | Mức độ nghi ngờ mới, luôn nhỏ hơn mức ban đầu nhờ có thêm số đo cảm biến. |

#### 🗣️ Phát biểu bằng lời và trực giác đời thường:
> - **Độ lệch $y$:** *"Lấy số đo cảm biến trừ đi số dự đoán để biết thực tế có điều gì bất ngờ xảy ra."*  
> - **Trọng số Kalman $K$:** *"Là tỷ số giữa độ nghi ngờ của dự đoán trên tổng độ nghi ngờ của cả hệ thống.  
>   + Nếu cảm biến cực kỳ xịn ($R \to 0$), $K \to 1$: ta tin 100% vào cảm biến và kéo thẳng vị trí về số đo cảm biến.  
>   + Nếu cảm biến quá lởm khởm, rung lắc mạnh ($R$ rất lớn), $K \to 0$: ta hầu như bỏ qua cảm biến và tin vào mô hình tính toán."*  
> - **Chốt vị trí $\hat{x}$:** *"Lấy vị trí dự đoán cộng thêm một phần độ lệch theo tỷ lệ $K$."*  
> - **Co hẹp phương sai $P$:** *"Nhân phương sai cũ với $(1 - K)$, chứng minh độ nghi ngờ luôn được thu hẹp lại sau mỗi lần có cảm biến hỗ trợ."*

---

## 4. THEO DÕI SỨC KHỎE BỘ LỌC BẰNG CHỈ SỐ NIS

### 4.1 Chỉ số NIS là gì và dùng để làm gì?
Khi xe chạy ngoài phố, ta **không hề có tọa độ chuẩn (Ground Truth)** để biết xe đang chạy đúng hay sai. Làm sao để máy tính tự phát hiện cảm biến đang bị lỏng dây, chập mạch, hoặc mô hình tính toán đang bị sai?

Người ta dùng thước đo **NIS (Normalized Innovation Squared)**:

$$\text{NIS} = \mathbf{y}^T \mathbf{S}^{-1} \mathbf{y} \quad \left(\text{ở hệ 1 chiều: } \text{NIS} = \frac{y^2}{S} = \frac{(z - \hat{x}^-)^2}{P^- + R}\right)$$

#### 🔍 Chi tiết từng thành phần:
| Ký hiệu | Tên gọi | Kiểu | Đơn vị | Ý nghĩa thực tế |
| :---: | :--- | :---: | :---: | :--- |
| $\text{NIS}$ | Độ lệch chuẩn hóa bình phương | Số thực $\ge 0$ | Không thứ nguyên | Thước đo đánh giá xem độ lệch thực tế có vượt quá mức dung sai cho phép hay không. |
| $y$ | Độ lệch đo lường | Số thực | mét ($\text{m}$) | Khoảng chênh lệch giữa số đo cảm biến và dự đoán. |
| $S$ | Dung sai cho phép | Số thực | $\text{m}^2$ | Tổng sai số mà hệ thống dự kiến chấp nhận được. |

#### 🗣️ Cách phát biểu và ý nghĩa trực giác:
> **Ý nghĩa thực tế:** NIS trả lời câu hỏi: *"Độ lệch $y$ này có bình thường không so với mức rung lắc dự kiến $S$?"*.  
> - Ví dụ: Cảm biến lệch $y = 3\text{m}$. Nếu đây là cảm biến GPS giá rẻ có sai số $S = 9\text{m}^2$, thì $\text{NIS} = 3^2 / 9 = 1.0 \Rightarrow$ Hoàn toàn bình thường, không có gì lạ.  
> - Nhưng nếu đây là cảm biến UWB độ chính xác cao vốn chỉ được phép rung lắc $S = 0.01\text{m}^2$ mà lại lệch tới $3\text{m}$, thì $\text{NIS} = 3^2 / 0.01 = 900 \Rightarrow$ Báo động khẩn cấp: cảm biến hoặc đường truyền đang gặp sự cố nghiêm trọng!

---

### 4.2 Con số kỳ vọng chuẩn trong thực tế
Về mặt toán học xác suất, NIS tuân theo phân phối Chi-bình phương với số bậc tự do bằng số chiều đo đạc $k$:

$$\text{NIS} \sim \chi^2_k \implies \mathbb{E}[\text{NIS}] = k$$

- Với cảm biến đo mặt phẳng 2D (tọa độ $x$ và $y$), số chiều $k = 2$.
- **Kỳ vọng chuẩn của một bộ lọc khỏe mạnh:** $\text{mean}(\text{NIS}) \approx 2.0$.
- Trong bài lab, ngưỡng an toàn được quy định là **$\text{mean}(\text{NIS}) < 8.0$**. Nếu vượt qua ngưỡng này, hệ thống bị coi là bất thường.

---

## 5. BỘ LỌC KALMAN DẠNG MA TRẬN CHO XE CHẠY TRÊN MẶT PHẲNG 2D (PART 5)

### 5.1 Vector trạng thái và Mô hình động học 2D
Khi xe chạy trên mặt phẳng, ta cần theo dõi 4 đại lượng cùng lúc gồm 2 tọa độ vị trí và 2 thành phần vận tốc:

$$\mathbf{x} = \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix} \in \mathbb{R}^4$$

Giả sử trong khoảng thời gian lấy mẫu rất ngắn $\Delta t$, xe chạy gần như thẳng đều:

$$\mathbf{x}_k = \mathbf{F}(\Delta t) \mathbf{x}_{k-1} + \mathbf{w}_{k-1}$$

$$\mathbf{F}(\Delta t) = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 4}$$

#### 🔍 Chi tiết các vị trí trong ma trận chuyển trạng thái $\mathbf{F}(\Delta t)$ (Exercise 5.1):
- Hàng 0: $x_{mới} = 1 \cdot x + \Delta t \cdot v_x$ (vị trí bằng vị trí cũ cộng vận tốc nhân thời gian).
- Hàng 1: $y_{mới} = 1 \cdot y + \Delta t \cdot v_y$.
- Hàng 2: $v_{x, mới} = 1 \cdot v_x$ (giả định vận tốc được giữ nguyên theo quán tính).
- Hàng 3: $v_{y, mới} = 1 \cdot v_y$.

```python
def make_F(dt):
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    return F
```

---

### 5.2 Ma trận quan sát $\mathbf{H}$ (Exercise 5.1)
Cảm biến định vị như GPS hoặc UWB chỉ đo được tọa độ $[x, y]$, không đo được trực tiếp vận tốc:

$$\mathbf{z} = \begin{bmatrix} z_x \\ z_y \end{bmatrix} = \mathbf{H} \mathbf{x} + \mathbf{v}, \quad \mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 4}$$

#### 🔍 Chi tiết thành phần:
- $\mathbf{z}$ ($2\times 1$): Cặp số đo tọa độ thực tế thu về từ cảm biến.
- $\mathbf{H}$ ($2\times 4$): Ma trận đóng vai trò chiếc "kính lọc", trích lấy 2 phần tử vị trí $[x, y]$ và che đi 2 phần tử vận tốc $[v_x, v_y]$.

```python
def make_H():
    H = np.zeros((2, 4))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    return H
```

---

### 5.3 Ma trận nhiễu quá trình $\mathbf{Q}(\Delta t, q)$
Khi xe tăng ga hoặc đạp phanh, gia tốc biến động ngẫu nhiên với cường độ $q$ (đơn vị $(\text{m}/\text{s}^2)^2/\text{Hz}$):

$$\mathbf{Q}_{\text{block}} = q \begin{bmatrix} \frac{\Delta t^3}{3} & \frac{\Delta t^2}{2} \\ \frac{\Delta t^2}{2} & \Delta t \end{bmatrix}$$

Khối ma trận này được xếp trên đường chéo cho từng cặp trục $(x, v_x)$ và $(y, v_y)$.

---

### 5.4 Toàn bộ chu trình tính toán ma trận (Exercise 5.2)

#### Pha 1: Dự đoán (Predict)
- Cập nhật vị trí và vận tốc: $\mathbf{x}^- = \mathbf{F} \mathbf{x}$
- Cập nhật ma trận hiệp phương sai: $\mathbf{P}^- = \mathbf{F} \mathbf{P} \mathbf{F}^T + \mathbf{Q}$

#### Pha 2: Cập nhật (Update)
- Độ lệch đo lường: $\mathbf{y} = \mathbf{z} - \mathbf{H} \mathbf{x}^-$
- Hiệp phương sai độ lệch: $\mathbf{S} = \mathbf{H} \mathbf{P}^- \mathbf{H}^T + \mathbf{R}$
- Trọng số Kalman ma trận: $\mathbf{K} = \mathbf{P}^- \mathbf{H}^T \mathbf{S}^{-1}$
- Chốt trạng thái mới: $\mathbf{x} = \mathbf{x}^- + \mathbf{K} \mathbf{y}$
- Co hẹp ma trận phương sai: $\mathbf{P} = (\mathbf{I} - \mathbf{K} \mathbf{H}) \mathbf{P}^-$

#### 🔍 Bảng tra kích thước ma trận trong bài toán 2D:
| Ký hiệu | Kích cỡ | Ý nghĩa đại số |
| :---: | :---: | :--- |
| $\mathbf{x}$ | $4 \times 1$ | Vector trạng thái gồm 2 vị trí và 2 vận tốc. |
| $\mathbf{P}$ | $4 \times 4$ | Ma trận chứa độ bất định và tương quan sai số giữa các biến. |
| $\mathbf{z}, \mathbf{y}$ | $2 \times 1$ | Vector 2 chiều chứa số đo và độ lệch vị trí. |
| $\mathbf{R}, \mathbf{S}$ | $2 \times 2$ | Ma trận nhiễu cảm biến và ma trận hiệp phương sai độ lệch. |
| $\mathbf{K}$ | $4 \times 2$ | Ma trận trọng số Kalman, biến độ lệch vị trí 2 chiều thành lượng điều chỉnh cho cả 4 biến trạng thái. |

#### 💡 Bí mật thú vị: Vận tốc tự động hiện ra từ đâu khi cảm biến chỉ đo vị trí?
Dù cảm biến chỉ đo tọa độ $(x, y)$, cột trọng số Kalman ở hàng vận tốc không hề bằng 0:
$$K_{v_x} = \frac{P_{x, v_x}^-}{S_{xx}} \neq 0$$
Nhờ sự tương quan chéo giữa vị trí và vận tốc được tích lũy trong ma trận $\mathbf{P}$, mỗi khi thấy vị trí bị lệch đi một đoạn $\Delta x$, bộ lọc **tự động tính ra luôn vận tốc của xe** mà không cần cảm biến đo tốc độ riêng biệt!

---

## 6. HỢP NHẤT NHIỀU CẢM BIẾN CHẠY LỆCH TẦN SỐ (PART 6)

### 6.1 Vấn đề lệch pha thời gian trong xe tự hành thực tế
Trong xe tự hành, các cảm biến hoạt động hoàn toàn độc lập với chu kỳ lấy mẫu khác nhau:
- **Radar:** $20\,\text{Hz}$ (cứ $0.05\,\text{s}$ gửi 1 lần).
- **LiDAR:** $10\,\text{Hz}$ (cứ $0.1\,\text{s}$ gửi 1 lần).
- **GPS:** $5\,\text{Hz}$ (cứ $0.2\,\text{s}$ gửi 1 lần).

### 6.2 Thuật toán vòng lặp sự kiện bất đồng bộ (Exercise 6.1)
Thay vì chờ đợi các cảm biến cùng gửi dữ liệu một lúc (điều không bao giờ xảy ra), ta sắp xếp tất cả các mẫu đo theo trục thời gian tăng dần và xử lý theo từng sự kiện:

```python
def run_fusion(meas, q, x0, P0, t0=0.0):
    kf = KalmanFilter(x0, P0)
    t_prev = t0
    log = []
    for ts, name, z, H, R in meas:
        # Bước 1: Tính khoảng thời gian trôi qua từ lần đo gần nhất
        dt = ts - t_prev
        # Bước 2: Nếu thời gian có trôi đi, chạy predict để đẩy mô hình tới mốc ts
        if dt > 0:
            kf.predict(make_F(dt), make_Q(dt, q))
        # Bước 3: Cập nhật đúng ma trận H và R của cảm biến vừa bắn dữ liệu về
        kf.update(z, H, R)
        # Bước 4: Cập nhật lại mốc thời gian và lưu vết
        t_prev = ts
        log.append((ts, kf.x.copy(), kf.P.copy()))
    return log
```

#### 🗣️ Diễn giải bằng lời:
> Khi một cảm biến gửi số liệu tới, ta kiểm tra xem đã qua bao nhiêu giây. Nếu thời gian đã trôi qua ($\Delta t > 0$), ta chạy `predict` để đẩy mô hình đến đúng thời điểm đó, rồi chạy `update` bằng thông số riêng của cảm biến đó. Nếu hai cảm biến gửi dữ liệu cùng một tích tắc ($\Delta t = 0$), ta không cần predict lại mà update liên tiếp luôn. Nhờ tính chất của xác suất, cập nhật cảm biến nào trước hay sau đều cho ra cùng một kết quả tối ưu.

---

## 7. LỌC BỎ ĐIỂM ĐO BẤT THƯỜNG BẰNG CỔNG CHI-BÌNH PHƯƠNG (PART 7)

### 7.1 Tại sao phải đặt cổng lọc (Outlier Gating)?
Khi sóng GPS hoặc UWB bị dội vào tường nhà cao tầng hoặc kết cấu kim loại (hiện tượng phản xạ đa đường), số đo có thể nhảy vọt hàng chục mét. Nếu đưa số đo sai lệch nghiêm trọng này vào bộ lọc, xe sẽ bị giật lái đột ngột và gây tai nạn.

Ta thiết lập một **Cổng lọc kiểm định Chi-bình phương (Exercise 7.1)**:

$$d^2 = \mathbf{y}^T \mathbf{S}^{-1} \mathbf{y}$$

$$\text{Quy tắc: Chấp nhận số đo nếu } d^2 \le \gamma = \chi^2_{\text{ppf}}(p, k)$$

#### 🔍 Chi tiết từng thành phần:
| Ký hiệu | Tên gọi | Giá trị | Ý nghĩa thực tế |
| :---: | :--- | :---: | :--- |
| $d^2$ | Khoảng cách Mahalanobis bình phương | Số thực $\ge 0$ | Thước đo khoảng cách thống kê từ số đo thực tế đến hình elip tin cậy của bộ lọc. |
| $p$ | Mức độ tin cậy của cổng | $0.99$ (99%) | Ta chấp nhận 99% các số đo bình thường nằm lọt vào cổng. |
| $k$ | Bậc tự do | $2$ | Số chiều của phép đo (tọa độ 2D gồm $x$ và $y$). |
| $\gamma$ | Ngưỡng cắt | $9.21$ | Giá trị tra bảng Chi-bình phương với $p=0.99, k=2$. |

```python
def gated_update(kf, z, H, R, p=0.99):
    y = z - H @ kf.x
    S = H @ kf.P @ H.T + R
    # Dùng np.linalg.solve thay vì nghịch đảo ma trận để tránh lỗi làm tròn số
    d2 = float(y.T @ np.linalg.solve(S, y))
    # So sánh khoảng cách d2 với ngưỡng chuẩn 9.21
    if d2 > chi2.ppf(p, df=len(z)):
        return False   # Điểm đo bất thường -> Vứt bỏ, giữ nguyên trạng thái xe
    kf.update(z, H, R)
    return True        # Điểm đo hợp lệ -> Cập nhật bình thường
```

#### 🗣️ Phát biểu bằng lời:
> *"Nếu khoảng cách thống kê bình phương $d^2$ giữa số đo thực tế và dự đoán vượt quá ngưỡng 9.21 (tức xác suất xảy ra ngẫu nhiên nhỏ hơn 1%), ta kết luận đây là số đo rác (outlier). Ta lập tức vứt bỏ điểm đo này và giữ nguyên trạng thái xe để tránh bị giật lệch quỹ đạo."*

---

## 8. NHIỆM VỤ THỰC TẾ: BẮT BỆNH CẢM BIẾN TRÊN XE LYNX-07 (PART 9)

### 8.1 Dấu hiệu nhận biết 3 loại bệnh phổ biến của cảm biến

```
[BỆNH 1: LỆCH GỐC - BIAS]        [BỆNH 2: KHAI BÁO NHIỄU QUÁ NHỎ]  [BỆNH 3: BÙNG PHÁT NGOẠI LAI - BURST]
- Độ lệch trung bình LỆCH XA 0  - Độ lệch trung bình XẤP XỈ 0     - Độ lệch trung bình XẤP XỈ 0
- NIS cao ĐỀU ĐẶN toàn hành trình - Cả Mean và Median NIS ĐỀU CAO   - Mean NIS RẤT CAO nhưng Median BÌNH THƯỜNG
- Nguyên nhân: Lắp lệch vị trí  - Nguyên nhân: Thông số khai báo điêu - Nguyên nhân: Sóng bị che khuất tạm thời
- Cách sửa: Trừ vector lệch     - Cách sửa: Nhân phóng to ma trận R - Cách sửa: Dùng cổng lọc Chi-bình phương
```

---

### 8.2 Phân tích số liệu thực tế của Học viên `2A202602467`
Từ mã học viên `STUDENT_ID = "2A202602467"` (hạt giống SHA-256 `seed = 3897423559`), ta thu được số liệu chẩn đoán ban đầu:
- **Cảm biến GPS (450 mẫu đo):**
  - $\text{mean(NIS)} = 2.88$, $\text{median(NIS)} = 1.40$, độ lệch trung bình = $[-0.01, -0.02]\,\text{m}$.
  - Nhận xét: Số liệu hoàn toàn đẹp, tiệm cận giá trị kỳ vọng lý thuyết $\mathbb{E}[\text{NIS}] = 2.0$. Cảm biến GPS khỏe mạnh 100%.
- **Cảm biến UWB (900 mẫu đo):**
  - $\text{mean(NIS)} = 12.69$ (vọt lên rất cao bất thường).
  - $\text{median(NIS)} = 1.66$ (trong hơn 90% thời gian, số đo hoàn toàn chuẩn xác).
  - $\max(\text{NIS}) = 198.65$ (xuất hiện một đợt bùng nổ xung nhiễu dữ dội trong khoảng $t = 53.84\,\text{s}$ đến $61.93\,\text{s}$).
  - Độ lệch trung bình = $[0.07, 0.01]\,\text{m}$ (gần như bằng 0 $\Rightarrow$ loại trừ khả năng bị lệch gốc `bias`).

#### 🎯 Kết luận chẩn đoán:
Cảm biến **`UWB`** bị lỗi **`outlier_burst`** (chùm xung nhiễu ngoại lai bùng phát tạm thời).

---

### 8.3 Cấu hình sửa lỗi & Kết quả sau khắc phục:
```python
MY_DIAGNOSIS_SENSOR = "UWB"
MY_DIAGNOSIS_TYPE   = "outlier_burst"
FIX_SENSOR = "UWB"
FIX_METHOD = "gate"
```

#### 📊 Kết quả thực nghiệm sau khi sửa:
- **Chỉ số Pooled Mean NIS:** Giảm mạnh từ $9.42$ xuống còn **$1.98$** (đạt mức hoàn hảo, tiệm cận giá trị lý thuyết $2.0$ và vượt xa yêu cầu $< 8.0$).
- **Số mẫu đo được chấp nhận:** $1267 / 1350$ mẫu (hệ thống đã loại bỏ chính xác $83$ mẫu ngoại lai nguy hiểm của UWB trong đợt bùng phát nhiễu).
- **Độ bất định vị trí $1\sigma$ cuối hành trình:**

$$\sigma_{\text{vị trí}} = \sqrt{P_{xx} + P_{yy}} = \sqrt{0.0673 + 0.0664} \approx \mathbf{0.366}\,\text{m}$$

- **Bán kính tin cậy 95%:**

$$R_{95\%} \approx 2 \cdot \sigma_{\text{vị trí}} \approx \mathbf{0.732}\,\text{m}$$

---

### 8.4 Tại sao hai cách sửa kia (`bias` và `inflate_R`) lại sai?
1. **Tại sao không dùng `bias`?**  
   Độ lệch trung bình của UWB chỉ là $[0.07, 0.01]\,\text{m}$ (gần như bằng 0). Nếu ta tự tiện trừ đi một vector cố định, trong suốt 90 giây hoạt động bình thường còn lại, quỹ đạo của xe sẽ bị kéo lệch sai hoàn toàn so với thực tế.
2. **Tại sao không dùng `inflate_R`?**  
   Đợt nhiễu chỉ kéo dài cục bộ trong 8 giây. Nếu ta nhân phóng to ma trận sai số $\mathbf{R}$ cho toàn bộ hành trình, bộ lọc sẽ mất lòng tin vào UWB trong suốt cả chuyến đi, làm giảm độ chính xác bám đường ngay cả khi UWB đang chạy rất tốt.
3. **Một tình huống mà cách sửa `gate` sẽ thất bại:**  
   Khi xe bất ngờ bẻ lái cực gắt hoặc phanh cháy đường vượt quá khả năng dự đoán của mô hình vật lý, vị trí dự đoán sẽ bị trễ so với thực tế. Lúc này, dù cảm biến đo đúng nhưng độ lệch $y$ lại rất lớn, khiến bộ lọc hiểu nhầm là số đo rác và loại bỏ liên tiếp. Xe sẽ mất phương hướng hoàn toàn (hiện tượng phân kỳ bộ lọc).

---

## 9. MỞ RỘNG CHO CẢM BIẾN PHI TUYẾN: BỘ LỌC EKF (PART 8 BONUS)

### 9.1 Khi cảm biến đo cự ly và góc thay vì tọa độ Descartes
Radar hoặc Camera đặt tại trạm quan sát $\mathbf{s} = [s_x, s_y]^T$ đo khoảng cách $r$ và góc phương vị $\theta$:

$$\mathbf{z} = \begin{bmatrix} r \\ \theta \end{bmatrix} = h(\mathbf{x}, \mathbf{s}) + \mathbf{v}$$

$$h(\mathbf{x}, \mathbf{s}) = \begin{bmatrix} \sqrt{(x - s_x)^2 + (y - s_y)^2} \\ \operatorname{atan2}(y - s_y, x - s_x) \end{bmatrix}$$

Vì hàm số $h(\mathbf{x})$ chứa căn bậc hai và hàm lượng giác $\text{atan2}$ (phi tuyến tính), ta không thể dùng ma trận tuyến tính $\mathbf{H}$ thông thường mà phải dùng **ma trận Jacobian $\mathbf{H}_{\text{rb}}$** để xấp xỉ tuyến tính tại điểm xe đang đứng.

---

### 9.2 Ma trận đạo hàm riêng Jacobian $\mathbf{H}_{\text{rb}}$ (Exercise 8.1)
Đặt $dx = x - s_x$, $dy = y - s_y$, khoảng cách $r = \sqrt{dx^2 + dy^2}$:

$$\mathbf{H}_{\text{rb}} = \begin{bmatrix} \frac{\partial r}{\partial x} & \frac{\partial r}{\partial y} & 0 & 0 \\ \frac{\partial \theta}{\partial x} & \frac{\partial \theta}{\partial y} & 0 & 0 \end{bmatrix} = \begin{bmatrix} \frac{dx}{r} & \frac{dy}{r} & 0 & 0 \\ -\frac{dy}{r^2} & \frac{dx}{r^2} & 0 & 0 \end{bmatrix}$$

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

#### 💡 Lưu ý sống còn: Hiện tượng nhảy pha góc (Angle Wrapping)
Góc phương vị nằm trong khoảng $[-\pi, \pi]$. Nếu xe đi từ $-179^\circ$ sang $+179^\circ$, độ lệch hình học thực tế chỉ là $2^\circ$, nhưng phép trừ số học thông thường cho ra:
$$-179^\circ - (+179^\circ) = -358^\circ$$
Nếu đưa con số $-358^\circ$ này vào bộ lọc, xe sẽ bị giật lái vòng tròn! Ta bắt buộc phải chuẩn hóa góc về khoảng $[-\pi, \pi]$:

$$y_\theta = (y_\theta + \pi) \pmod{2\pi} - \pi$$

---

## 10. KINH NGHIỆM THỰC TẾ KHI TRIỂN KHAI CHO XE TỰ HÀNH & ROBOT

### 10.1 Tránh lỗi sập bộ lọc do sai số làm tròn số học (Dạng Joseph)
Trong máy tính, các phép trừ số thực $\mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}^-$ có thể làm cho ma trận $\mathbf{P}$ bị mất tính đối xứng hoặc xuất hiện phương sai âm trên đường chéo, khiến chương trình bị dừng đột ngột.

Để khắc phục, trong các hệ thống xe tự hành thực tế, người ta áp dụng công thức **Dạng Joseph (Joseph Form)**:

$$\mathbf{P} = (\mathbf{I} - \mathbf{K} \mathbf{H}) \mathbf{P}^- (\mathbf{I} - \mathbf{K} \mathbf{H})^T + \mathbf{K} \mathbf{R} \mathbf{K}^T$$

Và luôn cưỡng bức đối xứng sau mỗi chu kỳ:
```python
P = 0.5 * (P + P.T)
```

---

### 10.2 Xử lý dữ liệu cảm biến bị trễ (Out-of-Sequence Measurements)
Trong thực tế, dữ liệu từ Camera xử lý AI thường mất $50\,\text{ms}$ mới ra kết quả, trong khi IMU gửi dữ liệu tức thì:
- **Giải pháp bộ đệm quay ngược thời gian (Ring Buffer & Replay):** Máy tính lưu lại lịch sử ước lượng trong khoảng 1 giây gần nhất. Khi gói tin camera gửi về ghi nhãn thời gian $t - 50\,\text{ms}$, hệ thống quay ngược trạng thái về mốc đó, chèn phép đo camera vào để cập nhật, rồi tính toán nhanh lại các bước tiếp theo cho tới hiện tại.

---

*Tài liệu hoàn thành theo chuẩn học thuật và kỹ thuật của chương trình đào tạo K4 Track 4 — AI20K.*  
*Học viên: Lâm Quang Anh Quân (MSSV: `2A202602467`).*
