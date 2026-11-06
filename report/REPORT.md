# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468; `class_name`= cab ; `rank`=1 ; `score`=	0.510915 ; `taxonomy_name`= ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Đây là kết quả phân loại cho toàn bộ bức ảnh (Image Classification), không phải kết quả nhận diện một vật thể riêng biệt.

Model phân tích nội dung tổng thể của ảnh và đưa ra một lớp đại diện là cab. Mặc dù trong ảnh có nhiều phương tiện giao thông như xe buýt, ô tô và xe tải, model không tạo ra nhãn riêng cho từng phương tiện mà đưa ra một kết quả phân loại chung cho ảnh.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp không phải do model tự nghĩ ra. Nó được xác định bởi taxonomy (hệ thống phân loại lớp) mà model được huấn luyện theo.

Trong trường hợp này, checkpoint yolo11n-cls.pt sử dụng taxonomy ImageNet-1K, bao gồm danh sách các lớp đã được định nghĩa trước.

Khi dự đoán, model tính toán điểm số cho các lớp có sẵn trong taxonomy và lựa chọn những lớp có score cao nhất.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Cần lưu cả ba trường để đảm bảo khả năng định danh, đối chiếu và quản lý nhãn một cách nhất quán.
- class_id: Mã định danh của lớp, giúp hệ thống tham chiếu đến lớp cụ thể mà không phụ thuộc vào cách viết tên.
- class_name: Tên lớp giúp con người đọc và hiểu nội dung dự đoán.
- taxonomy_name: Xác định hệ thống phân loại mà lớp đó thuộc về, tránh nhầm lẫn khi các taxonomy khác nhau có thể sử dụng cùng tên hoặc cách đánh số lớp khác nhau.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định rõ tiêu chí xác định chủ đề hoặc đối tượng chính của bức ảnh, chẳng hạn như vật thể chiếm ưu thế, đối tượng được tập trung thể hiện hoặc mục đích chính của bức ảnh.

Ngoài ra, cần quy định thứ tự ưu tiên khi có nhiều đối tượng cùng xuất hiện, cách xử lý trường hợp không xác định được chủ thể chính và trường hợp ảnh không phù hợp với bất kỳ lớp nào trong taxonomy.

Nếu trường hợp không rõ ràng hoặc nằm ngoài phạm vi hướng dẫn, annotator cần thực hiện escalation thay vì tự ý lựa chọn một lớp.

6. Vì sao không được lấy `score` của model làm ground truth?
Score là mức độ tin cậy của model đối với kết quả dự đoán, không phải nhãn đúng được xác nhận bởi con người.

Model có thể dự đoán sai nhưng vẫn đưa ra score cao do những hạn chế trong dữ liệu huấn luyện, đặc điểm hình ảnh hoặc khả năng nhận diện.

Ground truth phải được xác định dựa trên hình ảnh thực tế và guideline đã thống nhất, sau đó được kiểm tra bởi reviewer/QC khi cần thiết.

Vì vậy, score chỉ nên được sử dụng như thông tin tham khảo để hỗ trợ quá trình review, không được sử dụng trực tiếp thay thế ground truth.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Bounding box được biểu diễn bằng bốn tọa độ [x1, y1, x2, y2], trong đó (x1, y1) là góc trên bên trái và (x2, y2) là góc dưới bên phải của box. Tọa độ được tính theo pixel, với gốc tọa độ nằm ở góc trên bên trái ảnh.
Đây là bounding box của người đứng ở khu vực giữa, hơi lệch về bên phải ảnh, không phải bounding box của toàn bộ cảnh nhà bếp.
Với record đã chọn, góc trên bên trái có tọa độ (385.33, 69.24), góc dưới bên phải có tọa độ (498.92, 348.92). Box có chiều rộng khoảng 113.58 pixel và chiều cao 279.68 pixel, bao quanh người đứng ở giữa ảnh.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Threshold 0.35: 11 prediction.

Threshold 0.60: 6 prediction.

Khi tăng threshold từ 0.35 lên 0.60, số lượng prediction giảm từ 11 xuống 6 vì các dự đoán có score thấp hơn 0.60 bị loại bỏ.
Threshold là ngưỡng confidence dùng để lọc prediction. Ngưỡng càng cao thì càng nhiều prediction có score thấp bị loại bỏ. Tuy nhiên, số lượng prediction giảm không đồng nghĩa tất cả prediction còn lại đều chính xác hơn về ground truth.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ threshold, số lượng prediction được giữ lại tăng lên, bao gồm cả những dự đoán có score thấp và những trường hợp có khả năng nhận diện sai cao hơn. Vì vậy, reviewer cần dành thêm thời gian kiểm tra từng prediction để xác định class có chính xác không, bounding box có bao quanh đúng vật thể không, có prediction trùng lặp hoặc nhận diện nhầm không. Reviewer cũng cần chú ý những vật thể thực tế bị model bỏ sót và yêu cầu chỉnh sửa hoặc gán nhãn bổ sung khi cần thiết.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Annotator cần vẽ bounding box bao quanh toàn bộ phần vật thể có thể quan sát được trong ảnh, đảm bảo box sát với ranh giới vật thể nhất có thể và không bao gồm quá nhiều vùng nền xung quanh. Mỗi vật thể riêng biệt phải có một bounding box riêng, kể cả khi nhiều vật thể nằm gần hoặc chồng lấn nhau. Đối với vật thể bị che khuất hoặc bị cắt khỏi mép ảnh, cần tuân thủ quy tắc xử lý đã được quy định trong guideline.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Tôi chọn trường hợp người xuất hiện một phần ở mép trái ảnh. Đây là đối tượng bị cắt bởi ranh giới ảnh, chỉ có một phần cơ thể được quan sát.

Theo tôi, trường hợp này có thể xử lý bằng guideline nếu dự án đã quy định rõ cách gán nhãn đối tượng bị cắt mép. Annotator chỉ vẽ bounding box bao quanh phần vật thể thực sự nhìn thấy trong ảnh, không tự suy đoán phần nằm ngoài khung hình. Nếu không thể xác định chắc chắn đối tượng hoặc guideline chưa quy định cách xử lý, annotator cần escalation để hỏi Lab Coach thay vì tự quyết định.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
instance_id = kitchen-001

class_name = person

score = 0.899318

polygon_point_count = 348

polygon_xy (5 điểm đầu) = [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Bounding Box chỉ xác định vị trí và phạm vi của vật thể bằng một hình chữ nhật, trong khi polygon sử dụng nhiều điểm tọa độ nối với nhau để mô tả đường biên và hình dạng thực tế của vật thể.

Ví dụ, đối với người đứng giữa ảnh, polygon có thể mô tả đường biên xung quanh đầu, cánh tay, thân và chân, thay vì chỉ bao quanh toàn bộ người bằng một hình chữ nhật. Nhờ đó, có thể phân biệt rõ hơn vùng thuộc về vật thể và vùng background xung quanh.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để phân biệt từng đối tượng riêng biệt trong cùng một ảnh, kể cả khi chúng thuộc cùng một class.

Ví dụ, nếu trong ảnh có ba người thì cả ba đều có cùng class_id và class_name = person, nhưng mỗi người sẽ có một instance_id khác nhau.

instance_id không phải class ID vì nó không đại diện cho loại đối tượng. Nó cũng không phải tracking ID vì bài toán này chỉ phân đoạn đối tượng trên một ảnh đơn, không theo dõi đối tượng qua nhiều frame hoặc video.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Annotator phải vẽ đường biên mask sát với ranh giới thực tế của từng vật thể, chỉ bao gồm những vùng có thể xác định là thuộc về vật thể và không bao gồm background hoặc các vật thể khác. Đối với các vật thể riêng biệt nằm gần hoặc tiếp xúc nhau, cần tạo mask riêng cho từng instance. Khi vật thể bị che khuất, chỉ gán nhãn phần có thể quan sát được theo quy định của dự án, không tự suy đoán phần bị che khuất.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Tôi chọn trường hợp người đứng giữa ảnh và chiếc chảo treo phía sau, gần vùng vai và đầu người. Một phần đường biên giữa hai đối tượng bị che khuất nên không thể quan sát trực tiếp toàn bộ ranh giới thực tế.

Tôi chọn escalation nếu guideline chưa có quy tắc xử lý rõ ràng. Annotator có thể xác định phần cơ thể nhìn thấy nhưng không nên tự suy đoán đường biên ở vùng bị che khuất. Việc hỏi Lab Coach giúp thống nhất cách xử lý, tránh tạo mask không chính xác và ảnh hưởng đến chất lượng ground truth.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|
| Phân loại ảnh | Một class ID và class name đại diện cho toàn bộ ảnh (nếu dự án sử dụng single-label classification). | Ảnh `traffic` có nhiều loại phương tiện nhưng model dự đoán `cab` đứng đầu với score 0.510915. Khó quyết định lớp đại diện cho toàn bộ ảnh nếu có nhiều chủ thể. | Quan sát toàn bộ ảnh và chọn lớp theo tiêu chí trong guideline. Không dựa hoàn toàn vào prediction của model. | Kiểm tra class có phù hợp với nội dung ảnh và quy tắc phân loại không; đặc biệt chú ý các ảnh có nhiều chủ thể hoặc chủ đề. |
| Phát hiện vật thể | Class ID, class name và bounding box cho từng vật thể, biểu diễn bằng tọa độ `[x1, y1, x2, y2]`. | Ảnh kitchen có nhiều vật thể nằm gần hoặc chồng lấn nhau như bát, đĩa và dụng cụ nhà bếp. Ngoài ra, người ở mép trái ảnh bị cắt một phần, cần xác định phạm vi bounding box phù hợp. | Xác định từng vật thể, chọn class phù hợp và vẽ bounding box bao quanh phần vật thể nhìn thấy theo guideline. | Kiểm tra class, vị trí và kích thước box; phát hiện box sai, box trùng lặp, vật thể bị bỏ sót hoặc xử lý không đúng với vật thể bị cắt mép. |
| Instance segmentation | Class ID, class name và một mask riêng cho từng instance, biểu diễn bằng polygon hoặc vùng pixel. | Ảnh `kitchen` có người đứng gần các dụng cụ nhà bếp treo phía sau. Các vật dụng trên bàn như bát, đĩa và cốc nằm gần nhau, khiến việc xác định đường biên mask khó khăn. | Xác định từng instance và vẽ mask sát với đường biên thực tế của vật thể, phân biệt rõ các vật thể riêng biệt và phần background. | Kiểm tra mask có bám sát đường biên không, có chứa background, bỏ sót vùng nhìn thấy hoặc gộp nhiều instance thành một mask không. |

**Nhận xét:**

Ba tác vụ có đơn vị Ground Truth khác nhau nên tiêu chí kiểm tra chất lượng (QC) cũng khác nhau:
- Classification: Tập trung vào tính chính xác của class đại diện cho toàn bộ ảnh.
- Object Detection: Tập trung vào class và vị trí, kích thước của bounding box cho từng vật thể.
- Instance Segmentation: Tập trung vào class, khả năng phân biệt từng instance và độ chính xác của đường biên mask.
## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
1. Quy tắc bảo vệ dữ liệu:

Tôi chỉ sử dụng dữ liệu được cung cấp cho mục đích thực hiện bài lab, không tự ý sao chép, chia sẻ hoặc tải dữ liệu lên các nền tảng bên ngoài khi chưa được cho phép.

2. Xử lý dữ liệu không đúng phạm vi:

Tôi sẽ dừng xử lý dữ liệu đó, không tự ý chỉnh sửa hoặc chia sẻ, đồng thời báo cho Lab Coach hoặc người phụ trách dự án để được hướng dẫn xử lý.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
