from flask import Flask, request, send_file, render_template
import torch
import cv2
import numpy as np
import io
from PIL import Image
from drct.archs.DRCT_arch import DRCT  # 모델 아키텍처 임포트

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드 함수 정의
def load_model(model_path):
    model = DRCT(upscale=4, in_chans=3, img_size=64, window_size=16, 
                 compress_ratio=3, squeeze_factor=30, conv_scale=0.01, 
                 overlap_ratio=0.5, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                 embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], gc=32,
                 mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
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
        if "file" not in request.files or "model_type" not in request.form:
            return "No file or model type selected"

        file = request.files["file"]
        model_type = request.form["model_type"]

        if file.filename == "" or model_type not in MODEL_PATHS:
            return "Invalid input"

        # 선택한 모델 로드
        model = load_model(MODEL_PATHS[model_type])

        # 업로드된 이미지 처리
        img_pil = Image.open(file.stream).convert("RGB")
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float().unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
        output = output.squeeze().cpu().clamp(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).astype(np.uint8)

        _, img_encoded = cv2.imencode('.png', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/png')

    return render_template("index.html")  # HTML 템플릿 렌더링

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)