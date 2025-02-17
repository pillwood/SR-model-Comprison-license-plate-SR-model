from flask import Flask, request, render_template, send_file
import torch
import cv2
import numpy as np
import io
import base64
from PIL import Image
from drct.archs.DRCT_arch import DRCT  # 모델 아키텍처 임포트

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드 함수 정의
def load_model(model_path):
    model = DRCT(
        upscale=4, in_chans=3, img_size=64, window_size=16, 
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01, 
        overlap_ratio=0.5, img_range=1., depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], gc=32,
        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
    )
    state_dict = torch.load(model_path)['params']
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model

# 모델 선택 옵션
MODEL_PATHS = {
    "DRCT_x4": "weights/DRCT_X4.pth",
    "DRCT_x4_CN": "weights/DRCT_X4_CN.pth"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 파일과 모델 타입이 제대로 전송되었는지 확인
        if "file" not in request.files or "model_type" not in request.form:
            return "No file or model type selected"

        file = request.files["file"]
        model_type = request.form["model_type"]

        if file.filename == "" or model_type not in MODEL_PATHS:
            return "Invalid input"

        # 선택한 모델 로드
        model = load_model(MODEL_PATHS[model_type])

        # 업로드된 이미지 처리 (PIL 이미지로 변환)
        img_pil = Image.open(file.stream).convert("RGB")
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float().unsqueeze(0).to(device)

        # 모델 추론
        with torch.no_grad():
            output = model(img_tensor)
        output = output.squeeze().cpu().clamp(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).astype(np.uint8)

        # 초해상도 이미지: OpenCV로 PNG 인코딩 후 base64 인코딩
        _, img_encoded = cv2.imencode('.png', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        processed_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        # 원본 이미지: PIL 이미지를 BytesIO에 저장 후 base64 인코딩
        buf_orig = io.BytesIO()
        img_pil.save(buf_orig, format='PNG')
        buf_orig.seek(0)
        original_base64 = base64.b64encode(buf_orig.getvalue()).decode('utf-8')

        # 결과 페이지로 렌더링 (좌측: 원본, 우측: SR 이미지)
        return render_template(
            "result.html",
            original_base64=original_base64,
            processed_base64=processed_base64
        )

    # GET 요청일 때 초기 화면(index.html) 렌더링
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
