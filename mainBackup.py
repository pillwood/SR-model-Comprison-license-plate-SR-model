from flask import Flask, request, send_file, render_template_string
import torch
import cv2
import numpy as np
import os
import io
from PIL import Image
from drct.archs.DRCT_arch import DRCT  # 모델 아키텍처 임포트
import torch.nn.functional as F

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로딩 (한번 서버가 켜질 때 로딩)
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

# 예제: DRCTx4 모델 사용 (원하는 경로의 가중치를 지정)
MODEL_PATH = "weights/DRCT_X4.pth"
model = load_model(MODEL_PATH)

def test_inference(img_lq, model, scale=4, window_size=16):
    _, _, h_old, w_old = img_lq.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    img_pad = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
    img_pad = torch.cat([img_pad, torch.flip(img_pad, [3])], 3)[:, :, :, :w_old + w_pad]
    
    with torch.no_grad():
        output = model(img_pad)
        output = output[..., :h_old * scale, :w_old * scale]
    return output

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        # 이미지를 PIL Image로 읽기
        img_pil = Image.open(file.stream).convert("RGB")
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        # BGR 변환 없이 RGB 그대로 사용 (모델에 맞게 조정 필요하면 변경)
        img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float().unsqueeze(0)
        img_tensor = img_tensor.to(device)
        
        # 초해상도 처리
        output = test_inference(img_tensor, model)
        output = output.squeeze().cpu().clamp(0, 1).numpy()
        # 채널 순서 조정: (C,H,W) -> (H,W,C)
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).astype(np.uint8)
        
        # 결과 이미지를 메모리 버퍼에 저장하여 반환
        _, img_encoded = cv2.imencode('.png', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/png')
    
    # 간단한 업로드 폼 HTML
    html = '''
    <!doctype html>
    <title>Super Resolution Service</title>
    <h1>Upload an image for Super Resolution</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
    return render_template_string(html)

if __name__ == "__main__":
    # 기본 포트 5000번으로 실행
    app.run(host="0.0.0.0", port=5000)
