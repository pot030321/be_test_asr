# Backend ASR QA

FastAPI backend nhận audio, trả transcript và số liệu thời gian. Trọng số model không nằm trong repo; mỗi máy chạy BE cần có checkpoint trên ổ đĩa. Backend không tự tải model và không lưu audio hoặc transcript.

## Chạy BE trên máy local

### 1. Clone repo và cài môi trường Python

~~~bash
git clone https://github.com/pot030321/be_test_asr.git
cd be_test_asr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade huggingface_hub
~~~

Nếu máy dùng NVIDIA GPU trên Linux, cài CUDA libraries tương thích với NVIDIA driver và CTranslate2. Hướng dẫn hiện tại của faster-whisper dùng CUDA 12 và cuDNN 9:

~~~bash
python -m pip install nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*' nvidia-cuda-nvrtc-cu12
~~~

Máy không có NVIDIA GPU có thể chạy CPU để kiểm tra chức năng. Trong file **.env**, đặt **ASR_DEVICE=cpu**, **ASR_COMPUTE_TYPE=int8**, **ASR_MODEL_WORKERS=1** và **ASR_MAX_INFLIGHT=1**. Nhận dạng bằng model lớn trên CPU sẽ chậm hơn đáng kể.

### 2. Lấy model về máy

Model không được tải tự động khi API khởi động. Nếu đã có checkpoint, dùng đường dẫn hiện có ở **ASR_MODEL_PATH** hoặc **ASR_MODEL_CACHE**. Nếu chưa có, model upstream chiếm khoảng 3.1 GB; kiểm tra ổ đĩa trước khi tải. Có thể dùng repo model baseline riêng để tải qua Git LFS: [model_ASR_testing](https://github.com/pot030321/model_ASR_testing).

Cách tải trực tiếp vào cache riêng của repo BE:

~~~bash
df -h .
hf download Systran/faster-whisper-large-v3 --cache-dir ./models --dry-run
# Chỉ chạy lệnh tải sau khi đã kiểm tra dung lượng dự kiến.
hf download Systran/faster-whisper-large-v3 --cache-dir ./models
~~~

Lệnh trên tạo thư mục cache mà **ASR_MODEL_CACHE=./models** bên dưới có thể đọc. Trang model upstream khai báo license MIT và mô tả đây là checkpoint CTranslate2. Khi dùng GitHub LFS, người tải cần được cấp quyền vào repo model và cài Git LFS.

### 3. Tạo token và cấu hình BE

~~~bash
cp .env.example .env
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
~~~

Dán token vừa tạo vào **ASR_API_TOKEN** trong **.env**. Không commit file **.env** và không gửi token vào chat nhóm công khai. Mặc định file mẫu dùng cache model **./models**, lắng nghe tại **127.0.0.1:8767**, và cho phép FE local ở **http://127.0.0.1:8000**.

Khởi động trong thư mục repo:

~~~bash
set -a
source .env
set +a
export ASR_PYTHON="$PWD/.venv/bin/python"
./run.sh
~~~

Giữ terminal này mở; nhấn Ctrl+C để dừng. Khi chạy model từ repo riêng, đặt **ASR_MODEL_PATH** thành đường dẫn tuyệt đối tới thư mục sau khi ghép model, chẳng hạn **/path/to/model_ASR_testing/weights**. Thư mục đó phải có **model.bin** và **config.json**.

## Chạy FE local và kết nối

Mở terminal thứ hai:

~~~bash
git clone https://github.com/pot030321/fe_test_asr.git
cd fe_test_asr
python3 -m http.server 8000 --bind 127.0.0.1
~~~

Mở **http://127.0.0.1:8000**. Trên giao diện, nhập Backend API URL là **http://127.0.0.1:8767**, nhập token từ file **.env**, chọn file audio hoặc ghi âm rồi chạy nhận dạng. Nếu chọn host/port FE khác, thêm origin chính xác đó vào **ASR_ALLOWED_ORIGINS** rồi khởi động lại BE. **localhost** và **127.0.0.1** là hai origin khác nhau; dùng đúng địa chỉ đã mở trên trình duyệt.

Để tester ở máy khác gọi BE, backend phải lắng nghe trên địa chỉ mạng phù hợp và có đường kết nối mà trình duyệt truy cập được. Khi truy cập qua Internet, đặt BE sau HTTPS reverse proxy hoặc mạng VPN có kiểm soát; không mở raw HTTP trực tiếp ra Internet. Thêm origin FE vào **ASR_ALLOWED_ORIGINS**. Vercel preview có thể được cho phép bằng **ASR_ALLOWED_ORIGIN_REGEX**, nhưng chỉ bật nếu team thật sự cần.

## Endpoint và số liệu

| Method và path | Xác thực | Mô tả |
| --- | --- | --- |
| GET /healthz | Không | Kiểm tra backend sẵn sàng. |
| POST /api/transcribe | Header X-ASR-Token | Multipart gồm file và language: auto, vi hoặc en. Trả transcript, ngôn ngữ, thời lượng, queue, inference, request và RTF. |
| GET /api/metrics | Header X-ASR-Token | Active requests, slots, queue, tổng thành công/lỗi và tối đa 100 request gần nhất. |

RTF = inference_s / audio_s; chỉ tính inference, không tính queue, upload hoặc mạng. request_s được đo trong API và gồm thời gian chờ queue. Client E2E trên FE còn gồm upload và mạng. Metrics chỉ lưu metadata trong RAM, không có audio hoặc transcript; lịch sử mất khi BE restart. Giao diện gửi bản ghi sau khi bấm Dừng ghi, không phải nhận dạng streaming từng phần.

## Test và đo tải

Unit test dùng model giả nên không cần tải model hoặc chiếm GPU:

~~~bash
cd be_test_asr
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
~~~

Đo tải HTTP thật từ máy có file audio và truy cập được backend:

~~~bash
export ASR_API_TOKEN='token-do-quan-ly-cap'
python3 scripts/ccu_test.py --url http://127.0.0.1:8767 --audio ./sample.wav --language vi --ccu 1 4 8 12 16 20 --requests-per-client 5
~~~

Script gửi inference thật và tối đa 20 client mỗi stage. Bắt đầu CCU thấp, sau đó tăng dần; theo dõi error rate, p95, queue, GPU memory và GPU utilization. Không chạy load test lúc team khác đang dùng chung GPU. Không commit audio mẫu vào repo.

## Mặc định và lỗi thường gặp

- API chạy một Uvicorn worker để chỉ nạp một bản model.
- Mặc định GPU CUDA/float16, 4 model workers, tối đa 2 inference đồng thời, queue timeout 60 giây và file audio tối đa 100 MB.
- Không tìm thấy checkpoint: kiểm tra **ASR_MODEL_PATH** hoặc **ASR_MODEL_CACHE**. BE không tự tải model.
- Lỗi CUDA: dùng môi trường Python có CUDA/cuDNN tương thích hoặc chuyển sang CPU bằng các biến cấu hình ở trên.
- HTTP 401: kiểm tra token ở BE và FE.
- Lỗi CORS: thêm đúng origin của FE vào **ASR_ALLOWED_ORIGINS**, rồi restart BE.
- HTTP 429 hoặc queue cao: giảm CCU hoặc điều chỉnh **ASR_MAX_INFLIGHT** và **ASR_QUEUE_TIMEOUT_S**.

Tài liệu tham khảo: [cài đặt faster-whisper trên GPU](https://github.com/SYSTRAN/faster-whisper#gpu), [tải model bằng Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
