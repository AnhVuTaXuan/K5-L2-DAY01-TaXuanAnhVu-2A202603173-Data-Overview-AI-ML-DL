# BÁO CÁO BÀI THỰC HÀNH NGÀY 1 – ĐỌC PREDICTION YOLO11 ĐỂ HIỂU ĐỊNH DẠNG NHÃN

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ).

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi.

> **Cách sử dụng mẫu này:**
> - Copy và điền thông tin trên máy, lưu thành `REPORT.md`.
> - Upload file vào panel Files của Colab (notebook không tự tạo file này).
> - Ô cuối Colab sẽ đóng ZIP cùng thư mục `day1_lab_outputs/`.
> - Giải nén và đưa vào thư mục `report/` trên GitHub.
> - Nộp **link repository** trên LMS.
>
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output. Họ tên/MSSV chỉ nằm trong tên repository.

---

## Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
|---|---|
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng. |
| Ground truth | Nhãn do người gán theo quy tắc. |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (ví dụ: ImageNet-1K, COCO-80). |
| Checkpoint | File model `.pt` đã huấn luyện sẵn. |
| Guideline | Quy tắc gán nhãn do người viết. |
| Annotator | Người gán nhãn. |
| Reviewer / QC | Người kiểm tra nhãn. |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa. |
| Model score | Số "tin" của model; không phải điểm chất lượng nhãn người. |

---

## 1. Phân loại ảnh (Image Classification)

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.

Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

### Câu 1. Chép record hạng 1

- `class_id`: 468
- `class_name`: `cab`
- `rank`: 1
- `score`: 0.510915
- `taxonomy_name`: `ImageNet-1K`

Rank thể hiện thứ hạng của lớp dự đoán dựa trên score mà model tính toán. Rank 1 là lớp có score cao nhất trong danh sách các lớp dự đoán (Top-K). Các lớp có rank tiếp theo có score thấp hơn.

Tuy nhiên, rank chỉ thể hiện thứ tự dự đoán của model, không chứng minh lớp đó là Ground Truth.

### Câu 2. Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao?

Đây là kết quả phân loại cho toàn bộ bức ảnh (Image Classification), không phải kết quả nhận diện một vật thể riêng biệt.

Model phân tích nội dung tổng thể của ảnh và đưa ra một lớp đại diện là `cab`. Mặc dù trong ảnh có nhiều phương tiện giao thông như xe buýt, ô tô và xe tải, model không tạo ra nhãn riêng cho từng phương tiện mà đưa ra một kết quả phân loại chung cho ảnh.

### Câu 3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra hay đi kèm taxonomy/checkpoint?

Danh sách tên lớp không phải do model tự nghĩ ra. Nó được xác định bởi taxonomy (hệ thống phân loại lớp) mà model được huấn luyện theo.

Trong trường hợp này, checkpoint `yolo11n-cls.pt` sử dụng taxonomy ImageNet-1K, bao gồm danh sách các lớp đã được định nghĩa trước.

Khi dự đoán, model tính toán điểm số cho các lớp có sẵn trong taxonomy và lựa chọn những lớp có score cao nhất.

### Câu 4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

Cần lưu cả ba trường để đảm bảo khả năng định danh, đối chiếu và quản lý nhãn một cách nhất quán.

- `class_id`: Mã định danh của lớp, giúp hệ thống tham chiếu đến lớp cụ thể mà không phụ thuộc vào cách viết tên.
- `class_name`: Tên lớp giúp con người đọc và hiểu nội dung dự đoán.
- `taxonomy_name`: Xác định hệ thống phân loại mà lớp đó thuộc về, tránh nhầm lẫn khi các taxonomy khác nhau có thể sử dụng cùng tên hoặc cách đánh số lớp khác nhau.

### Câu 5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?

Guideline cần quy định rõ tiêu chí xác định chủ đề hoặc đối tượng chính của bức ảnh, chẳng hạn như vật thể chiếm ưu thế, đối tượng được tập trung thể hiện hoặc mục đích chính của bức ảnh.

Ngoài ra, cần quy định thứ tự ưu tiên khi có nhiều đối tượng cùng xuất hiện, cách xử lý trường hợp không xác định được chủ thể chính và trường hợp ảnh không phù hợp với bất kỳ lớp nào trong taxonomy.

Nếu trường hợp không rõ ràng hoặc nằm ngoài phạm vi hướng dẫn, annotator cần thực hiện escalation thay vì tự ý lựa chọn một lớp.

### Câu 6. Vì sao không được lấy `score` của model làm Ground Truth?

Score là mức độ tin cậy của model đối với kết quả dự đoán, không phải nhãn đúng được xác nhận bởi con người.

Model có thể dự đoán sai nhưng vẫn đưa ra score cao do những hạn chế trong dữ liệu huấn luyện, đặc điểm hình ảnh hoặc khả năng nhận diện.

Ground Truth phải được xác định dựa trên hình ảnh thực tế và guideline đã thống nhất, sau đó được kiểm tra bởi reviewer/QC khi cần thiết.

Vì vậy, score chỉ nên được sử dụng như thông tin tham khảo để hỗ trợ quá trình review, không được sử dụng trực tiếp thay thế Ground Truth.

---

## 2. Phát hiện vật thể (Object Detection)

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.

Chỉ dùng ảnh `sample_id = kitchen`. Chọn một record.

### Câu 1. Chép một record

- `class_name`: `person`
- `score`: 0.912625
- `bbox_xyxy`: `[385.33, 69.24, 498.92, 348.92]`
- `bbox_width`: 113.58
- `bbox_height`: 279.68

### Câu 2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

Bounding box được biểu diễn bằng bốn tọa độ `[x1, y1, x2, y2]`, trong đó `(x1, y1)` là góc trên bên trái và `(x2, y2)` là góc dưới bên phải của box. Tọa độ được tính theo pixel, với gốc tọa độ nằm ở góc trên bên trái ảnh.

Đây là bounding box của người đứng ở khu vực giữa, hơi lệch về bên phải ảnh, không phải bounding box của toàn bộ cảnh nhà bếp.

Với record đã chọn, góc trên bên trái có tọa độ `(385.33, 69.24)`, góc dưới bên phải có tọa độ `(498.92, 348.92)`. Box có chiều rộng khoảng 113.58 pixel và chiều cao 279.68 pixel, bao quanh người đứng ở giữa ảnh.

### Câu 3. Chọn hai ngưỡng (ví dụ: 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

**Ảnh hưởng của Detection Score Threshold**

Với ảnh `kitchen`, số lượng predictions thay đổi theo ngưỡng confidence như sau:

- Threshold = `0.20`: Model đưa ra 17 predictions.
- Threshold = `0.35`: Model đưa ra 11 predictions.
- Threshold = `0.60`: Model đưa ra 6 predictions.

Khi threshold giảm từ 0.60 xuống 0.20, số lượng predictions tăng từ 6 lên 17. Điều này cho thấy model giữ lại nhiều bounding box có confidence thấp hơn, bao gồm một số lớp như `spoon`, `potted plant`, `dining table` và `bottle`.

Ngược lại, khi tăng threshold, những predictions có confidence thấp bị loại bỏ, giúp giảm số lượng bounding box cần reviewer kiểm tra. Tuy nhiên, threshold quá cao có thể loại bỏ những vật thể thực sự tồn tại trong ảnh. Vì vậy, việc lựa chọn threshold cần cân bằng giữa khả năng giữ lại vật thể và khối lượng công việc QC.

**Lưu ý:** Confidence score và threshold chỉ phản ánh mức độ tin cậy của model và điều kiện lọc predictions, không xác nhận prediction là Ground Truth.

Threshold là ngưỡng confidence dùng để lọc prediction. Ngưỡng càng cao thì càng nhiều prediction có score thấp bị loại bỏ. Tuy nhiên, số lượng prediction giảm không đồng nghĩa tất cả prediction còn lại đều chính xác hơn về Ground Truth.

### Câu 4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?

Khi hạ threshold, số lượng prediction được giữ lại tăng lên, bao gồm cả những dự đoán có score thấp và những trường hợp có khả năng nhận diện sai cao hơn.

Vì vậy, reviewer cần dành thêm thời gian kiểm tra từng prediction để xác định:

- Class có chính xác không.
- Bounding box có bao quanh đúng vật thể không.
- Có prediction trùng lặp hoặc nhận diện nhầm không.
- Có vật thể thực tế bị model bỏ sót không.

Reviewer cũng cần yêu cầu chỉnh sửa hoặc gán nhãn bổ sung khi cần thiết.

### Câu 5. Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

Annotator cần vẽ bounding box bao quanh toàn bộ phần vật thể có thể quan sát được trong ảnh, đảm bảo box sát với ranh giới vật thể nhất có thể và không bao gồm quá nhiều vùng nền xung quanh.

Mỗi vật thể riêng biệt phải có một bounding box riêng, kể cả khi nhiều vật thể nằm gần hoặc chồng lấn nhau.

Đối với vật thể bị che khuất hoặc bị cắt khỏi mép ảnh, cần tuân thủ quy tắc xử lý đã được quy định trong guideline.

### Câu 6. Trên ảnh `kitchen`, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Tôi chọn trường hợp người xuất hiện một phần ở mép trái ảnh. Đây là đối tượng bị cắt bởi ranh giới ảnh, chỉ có một phần cơ thể được quan sát.

Theo tôi, trường hợp này có thể xử lý bằng guideline nếu dự án đã quy định rõ cách gán nhãn đối tượng bị cắt mép.

Annotator chỉ vẽ bounding box bao quanh phần vật thể thực sự nhìn thấy trong ảnh, không tự suy đoán phần nằm ngoài khung hình.

Nếu không thể xác định chắc chắn đối tượng hoặc guideline chưa quy định cách xử lý, annotator cần escalation để hỏi Lab Coach thay vì tự quyết định.

---

## 3. Phân đoạn theo từng đối tượng (Instance Segmentation)

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.

Chỉ dùng ảnh `sample_id = kitchen`. Chọn một record.

### Câu 1. Chép một record

- `instance_id`: `kitchen-001`
- `class_name`: `person`
- `score`: 0.899318
- `polygon_point_count`: 348
- `polygon_xy` (5 điểm đầu):

```json
[
  [446.0, 70.0],
  [445.0, 71.0],
  [444.0, 71.0],
  [443.0, 72.0],
  [442.0, 72.0]
]
```

**Giải thích kết quả:**

Instance Segmentation là nhiệm vụ xác định vùng pixel thuộc về từng đối tượng riêng biệt trong hình ảnh, đồng thời xác định class của từng đối tượng.

Trong record `kitchen`, model dự đoán một instance thuộc class `person`, có `instance_id=kitchen-001` và score `0.899318`. Vùng đối tượng được biểu diễn bằng polygon gồm 348 điểm tọa độ.

Khác với Bounding Box chỉ bao quanh đối tượng bằng hình chữ nhật, Mask biểu diễn vùng pixel thuộc về đối tượng, còn Polygon mô tả đường biên của vùng đó thông qua các điểm tọa độ. Cách biểu diễn này giúp mô tả hình dạng đối tượng chi tiết hơn và hạn chế đưa background vào vùng annotation.

`instance_id` dùng để phân biệt từng đối tượng riêng biệt trong phạm vi output. Nó không phải `class_id` và cũng không phải `tracking_id`.

Ví dụ, trong một ảnh có ba người, cả ba có cùng class `person` nhưng cần được gán thành ba instance riêng biệt. `instance_id` trong output không mặc nhiên là ID duy nhất xuyên suốt nhiều ảnh hoặc nhiều lần chạy.

**Vấn đề cần lưu ý:**

Trong ảnh `kitchen`, một số đối tượng ở gần nhau hoặc bị che khuất, chẳng hạn người đứng gần các dụng cụ và vật thể phía sau. Điều này có thể khiến việc xác định đường biên chính xác trở nên khó khăn.

- **Annotator:** Tạo mask riêng cho từng đối tượng, bám sát phần biên có thể quan sát được và tuân theo guideline về vùng bị che khuất; không đưa background hoặc đối tượng khác vào mask.
- **Reviewer/QC:** Kiểm tra độ chính xác của đường biên, vùng background bị đưa nhầm vào mask, các phần đối tượng bị bỏ sót và tình trạng nhiều instance bị gộp chung.

Nếu ranh giới giữa hai đối tượng hoặc cách xử lý vùng bị che khuất không thể xác định rõ theo thông tin hiện có, cần tham khảo guideline hoặc escalation thay vì tự suy đoán.

### Câu 2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

Bounding Box chỉ xác định vị trí và phạm vi của vật thể bằng một hình chữ nhật, trong khi polygon sử dụng nhiều điểm tọa độ nối với nhau để mô tả đường biên và hình dạng thực tế của vật thể.

Ví dụ, đối với người đứng giữa ảnh, polygon có thể mô tả đường biên xung quanh đầu, cánh tay, thân và chân, thay vì chỉ bao quanh toàn bộ người bằng một hình chữ nhật.

Nhờ đó, có thể phân biệt rõ hơn vùng thuộc về vật thể và vùng background xung quanh.

### Câu 3. `instance_id` dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào?

`instance_id` dùng để định danh và phân biệt từng đối tượng riêng biệt (instance) trong kết quả phân đoạn, kể cả khi chúng thuộc cùng một class.

Ví dụ, nếu trong ảnh có ba người thì cả ba đều có cùng `class_id = 0` và `class_name = person`, nhưng mỗi người sẽ có một `instance_id` khác nhau.

`instance_id` không phải `class_id` vì nó không đại diện cho loại đối tượng mà đại diện cho từng đối tượng cụ thể.

Nó cũng không phải `tracking_id` vì bài toán Instance Segmentation ở đây xử lý ảnh đơn, không theo dõi và liên kết đối tượng qua nhiều frame của video.

### Câu 4. Viết một quy tắc vẽ biên mask để người gán nhãn làm theo.

Annotator phải vẽ đường biên mask sát với ranh giới thực tế của từng vật thể, chỉ bao gồm những vùng có thể xác định là thuộc về vật thể và không bao gồm background hoặc các vật thể khác.

Đối với các vật thể riêng biệt nằm gần hoặc tiếp xúc nhau, cần tạo mask riêng cho từng instance.

Khi vật thể bị che khuất, chỉ gán nhãn phần có thể quan sát được theo quy định của dự án, không tự suy đoán phần bị che khuất.

### Câu 5. Trên ảnh `kitchen`, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

Tôi chọn trường hợp người đứng giữa ảnh và chiếc chảo treo phía sau, gần vùng vai và đầu người. Một phần đường biên giữa hai đối tượng bị che khuất nên không thể quan sát trực tiếp toàn bộ ranh giới thực tế.

Tôi chọn escalation nếu guideline chưa có quy tắc xử lý rõ ràng.

Annotator có thể xác định phần cơ thể nhìn thấy nhưng không nên tự suy đoán đường biên ở vùng bị che khuất. Việc hỏi Lab Coach giúp thống nhất cách xử lý, tránh tạo mask không chính xác và ảnh hưởng đến chất lượng Ground Truth.

---

## 4. Vòng đời và kiểm tra chất lượng (Lifecycle & QC)

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây mô tả vòng đời production để liên kết các khái niệm đã học:

```text
Ảnh thô → Guideline → Ground Truth → Huấn luyện → Prediction → QC/Rework
```

Bảng dưới đây mô tả Ground Truth của từng tác vụ, những khó khăn quan sát được từ kết quả inference và trách nhiệm của Annotator và Reviewer.

| Tác vụ | Ground Truth (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định từ ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|
| **Classification** | Một class ID và class name đại diện cho toàn bộ ảnh (nếu dự án sử dụng single-label classification). | Ảnh `traffic` có nhiều loại phương tiện nhưng model dự đoán `cab` đứng đầu với score `0.510915`. Khó quyết định lớp đại diện cho toàn bộ ảnh nếu có nhiều chủ thể. | Quan sát toàn bộ ảnh và chọn lớp theo tiêu chí trong guideline. Không dựa hoàn toàn vào prediction của model. | Kiểm tra class có phù hợp với nội dung ảnh và quy tắc phân loại không; đặc biệt chú ý ảnh có nhiều chủ thể hoặc chủ đề. |
| **Object Detection** | Class ID, class name và bounding box cho từng vật thể, biểu diễn bằng tọa độ `[x1, y1, x2, y2]`. | Ảnh `kitchen` có nhiều vật thể nằm gần hoặc chồng lấn nhau như bát, đĩa và dụng cụ nhà bếp. Người ở mép ảnh bị cắt một phần, cần xác định phạm vi bounding box phù hợp. | Xác định từng vật thể, chọn class phù hợp và vẽ bounding box bao quanh phần vật thể nhìn thấy theo guideline. | Kiểm tra class, vị trí và kích thước box; phát hiện box sai, box trùng lặp, vật thể bị bỏ sót hoặc xử lý không đúng với vật thể bị cắt mép. |
| **Instance Segmentation** | Class ID, class name và một mask riêng cho từng instance, biểu diễn bằng polygon hoặc vùng pixel. | Ảnh `kitchen` có người đứng gần các dụng cụ nhà bếp treo phía sau. Các vật dụng trên bàn như bát, đĩa và cốc nằm gần nhau, khiến việc xác định đường biên mask khó khăn. | Xác định từng instance và vẽ mask sát với đường biên có thể quan sát được, phân biệt rõ từng vật thể và background. | Kiểm tra mask có bám sát đường biên không, có chứa background, bỏ sót vùng nhìn thấy hoặc gộp nhiều instance thành một mask không. |

### Nhận xét

Ba tác vụ có đơn vị Ground Truth khác nhau nên tiêu chí kiểm tra chất lượng (QC) cũng khác nhau:

- **Classification:** Tập trung vào tính chính xác của class đại diện cho toàn bộ ảnh.
- **Object Detection:** Tập trung vào class, vị trí và kích thước bounding box của từng vật thể.
- **Instance Segmentation:** Tập trung vào class, khả năng phân biệt từng instance và độ chính xác của đường biên mask.

### 4.1. Guideline, QC và quy trình xử lý lỗi

Mỗi loại annotation cần có guideline riêng để đảm bảo dữ liệu được gán nhãn thống nhất giữa các Annotator.

| Loại annotation | Quy tắc đề xuất | Điểm Reviewer/QC kiểm tra |
|---|---|---|
| **Classification** | Chọn nhãn đại diện cho nội dung chính của ảnh theo taxonomy được cung cấp và quy tắc single-label của bài lab. | Nhãn có phản ánh đúng nội dung ảnh, taxonomy và guideline không? |
| **Object Detection** | Tạo một bounding box riêng cho mỗi đối tượng, bao sát phần đối tượng nhìn thấy được và tuân thủ quy tắc xử lý vật thể bị cắt ở mép ảnh. | Class, vị trí, kích thước, box trùng lặp và đối tượng bị bỏ sót. |
| **Instance Segmentation** | Tạo mask riêng cho từng instance, bám sát đường biên có thể quan sát được và không gộp nhiều đối tượng vào một mask. | Độ chính xác đường biên, background bị đưa nhầm vào mask, phần đối tượng bị bỏ sót và instance bị gộp. |

#### Các trường hợp cần chú ý

- **Classification:** Ảnh có nhiều loại phương tiện nhưng model chỉ đưa ra một nhãn đứng đầu. Annotator cần căn cứ vào guideline để xác định nhãn phù hợp, không mặc định prediction đứng đầu là Ground Truth.
- **Detection:** Đối tượng nằm sát mép ảnh, bị cắt hoặc các bounding box ở gần nhau có thể gây khó khăn khi xác định vị trí và kích thước.
- **Segmentation:** Các đối tượng ở gần hoặc che khuất nhau có thể gây khó khăn trong việc xác định ranh giới mask và phân biệt từng instance.

#### Quy trình xử lý lỗi và Escalation

Cần phân biệt giữa lỗi annotation đã xác định được theo guideline và trường hợp chưa đủ thông tin để kết luận.

- **Annotator:** Tuân thủ guideline, chủ động sửa các lỗi đã xác định rõ và không tự suy đoán khi gặp trường hợp chưa được quy định.
- **Reviewer/QC:** Kiểm tra annotation, ghi nhận lỗi cụ thể, phản hồi cho Annotator và yêu cầu rework khi phát hiện sai sót.
- **Escalation:** Nếu guideline chưa đề cập đến trường hợp cụ thể hoặc có nhiều cách diễn giải hợp lý, cần chuyển Lab Coach hoặc người phụ trách để thống nhất cách xử lý trước khi tiếp tục annotation.

Sau khi Annotator hoàn thành rework, Reviewer cần kiểm tra lại kết quả để xác nhận lỗi đã được xử lý. Annotation chỉ được chấp nhận khi đáp ứng các tiêu chí chất lượng theo guideline.

**Lưu ý về phạm vi thực hiện:**

Lab hôm nay chỉ thực hiện inference và quan sát prediction, chưa trực tiếp thực hiện toàn bộ quy trình annotation, training và QC production. Các nội dung về Annotator, Reviewer, Rework và Escalation được trình bày nhằm liên hệ kết quả thực hành với quy trình production thực tế.

---

## 5. An toàn dữ liệu (Data Safety)

### Câu 1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (Viết 1 câu)

**Quy tắc bảo vệ dữ liệu:**

Tôi chỉ sử dụng dữ liệu được cung cấp cho mục đích thực hiện bài lab, không tự ý sao chép, chia sẻ hoặc tải dữ liệu lên các nền tảng bên ngoài khi chưa được cho phép.

### Câu 2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

**Xử lý dữ liệu không đúng phạm vi:**

Tôi sẽ dừng xử lý dữ liệu đó, không tự ý chỉnh sửa hoặc chia sẻ, đồng thời báo cho Lab Coach hoặc người phụ trách dự án để được hướng dẫn xử lý.

---
## 6. Reproducibility (Khả năng tái lập)

### 6.1. Execution Environment

Thông tin môi trường thực thi được ghi nhận từ notebook:

| Component | Version / Configuration |
|---|---|
| Execution date | 2026-11-06 |
| Operating System | Linux 6.6.159+ x86_64 |
| Python | 3.13.18 |
| Ultralytics | 8.4.145 |
| PyTorch | 2.11.0+cpu |
| Torchvision | 0.26.0+cpu |
| NumPy | 2.4.6 |
| Execution device | CPU |
| CUDA available | False |

**Execution note:** Notebook được thực thi trên CPU. CUDA không khả dụng trong môi trường này, do đó không sử dụng GPU để thực hiện inference.

### 6.2. Model Checkpoints

Ba checkpoint tương ứng với ba tác vụ được sử dụng trong Lab:

| Task | Checkpoint | File size (bytes) |
|---|---|---:|
| Classification | `yolo11n-cls.pt` | 5,790,624 |
| Object Detection | `yolo11n.pt` | 5,613,764 |
| Instance Segmentation | `yolo11n-seg.pt` | 6,182,636 |

Để kiểm tra tính nhất quán và nhận diện chính xác các checkpoint, sử dụng thuật toán SHA256:

| Checkpoint | SHA256 |
|---|---|
| `yolo11n-cls.pt` | `c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7` |
| `yolo11n.pt` | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| `yolo11n-seg.pt` | `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152` |

SHA256 giúp xác định checkpoint có bị thay đổi giữa các lần sử dụng hay không.

### 6.3. Inference Configuration

Các checkpoint và confidence threshold được sử dụng:

| Task | Checkpoint | Confidence threshold |
|---|---|---:|
| Classification | `yolo11n-cls.pt` | Theo cấu hình notebook gốc |
| Object Detection | `yolo11n.pt` | `0.35` |
| Instance Segmentation | `yolo11n-seg.pt` | `0.35` |

Đối với Object Detection, thực hiện khảo sát thêm các mức confidence threshold:

- `0.20`: quan sát số lượng prediction khi sử dụng ngưỡng thấp.
- `0.35`: ngưỡng chuẩn được sử dụng trong Lab.
- `0.60`: quan sát số lượng prediction khi sử dụng ngưỡng cao.

Kết quả khảo sát Detection:

| Confidence threshold | Number of predictions |
|---|---:|
| 0.20 | 17 |
| 0.35 | 11 |
| 0.60 | 6 |

Khi confidence threshold tăng, số lượng prediction được giữ lại giảm do các prediction có confidence thấp hơn ngưỡng bị loại bỏ.

Các tham số inference khác được giữ nguyên theo notebook gốc, không tự ý thay đổi trong quá trình thực hiện Lab.

### 6.4. Input Samples

Các sample được sử dụng để quan sát prediction:

| Task | Sample ID |
|---|---|
| Classification | `traffic` |
| Object Detection | `kitchen` |
| Instance Segmentation | `kitchen` |

### 6.5. Notebook Modification

- Notebook: sử dụng notebook gốc được cung cấp cho Lab.
- Thay đổi so với notebook gốc: Không chỉnh sửa.
- Các cấu hình và quy trình thực thi được giữ nguyên theo notebook được cung cấp.

### 6.6. Reproducibility Notes

Để hỗ trợ tái lập kết quả prediction, cần duy trì tính nhất quán của các yếu tố sau:

1. Sử dụng đúng notebook gốc.
2. Sử dụng đúng checkpoint tương ứng với từng task.
3. Đảm bảo checkpoint có cùng SHA256.
4. Sử dụng cùng dữ liệu đầu vào và sample ID.
5. Duy trì confidence threshold và các cấu hình inference tương ứng.
6. Ghi nhận phiên bản thư viện và thiết bị thực thi.

**Limitation:** Kết quả inference có thể khác nhau nếu thay đổi checkpoint, dữ liệu đầu vào, cấu hình inference hoặc môi trường thực thi.
## 6. Checklist trước khi nộp link repository

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền).
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
- [ ] Đã nộp **link repository** trên LMS.

---

**Kết thúc báo cáo thực hành Ngày 1.**
