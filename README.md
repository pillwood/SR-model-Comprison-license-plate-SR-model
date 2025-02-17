# SR-model-Comprison-license-plate-SR-model
## 4종류의 SR model의 비교 분석 미니 프로젝트

<br/>
학습 진행 모델 : 
<br/>
<br/>

1. FSRCNN
https://github.com/yjn870/FSRCNN-pytorch

<br/>
<br/>

2. ESRGAN
https://github.com/xinntao/ESRGAN

<br/>
<br/>

3. SwinIR
https://github.com/JingyunLiang/SwinIR

<br/>
<br/>

4. DRCT
https://github.com/ming053l/DRCT

<br/>
<br/>

DF2K모델을 통해 학습을 진행 : 
<br/>

![image](https://github.com/user-attachments/assets/0ae41a7b-3d2d-41a4-899c-32c4c4aca287)



<br/><br/>
![image](https://github.com/user-attachments/assets/0986fd23-940a-417e-a282-f5bca1a5c860)
<br/><br/>
![image](https://github.com/user-attachments/assets/675729b9-870f-445d-8c1f-324bf1d2f416)

<br/><br/><br/>
결과적으로 DRCT모델이 학습 시간 대비 가장 성능이 좋다.


## License Plate Model
<br/><br/><br/>
그럼 차량 번호판 데이터를 학습하며 L2 loss를 사용해 노이즈와 아티팩트를 제거하지 않고
L1 loss만을 사용하여 학습한다면 번호판 복원에 대해 더 좋은 성능을 낼 수 있지 않을까?
<br/><br/>

학습 진행 데이터 : 
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=172
<br/>
총 30epoch 학습 (iter 계산해서 넣어줌)
<br/>
이미지 사이즈가 너무 작은 관계로 종 횡 방향 크기가 64미만인 이미지는 제거후 사용. 
<br/>
(script 폴더 내부에 이미지 전처리 스크립트 작성)
<br/>
<br/>
학습된 모델은 <br/>
https://drive.google.com/file/d/1s9SDNT34rCJJoocrjyreM_eAqDzKYDAk/view?usp=sharing
<br/>에서 다운로드 가능합니다.
<br/>
python 3.12.7
pytorch 2.5.1 + cu118버전 사용
<br/>
<br/>
||pip install -r requirements.txt
<br/>
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
<br/>
<br/>
이미지 출처 : https://www.ohmynews.com/NWS_Web/Mobile/img_pg.aspx?CNTN_CD=IE003048022


<br/>

![LR image](https://github.com/user-attachments/assets/85c1aaa6-ff30-40a0-8836-15239e914587)

<br/>
<원본 저해상도 이미지>

<br/>

![DRCT](https://github.com/user-attachments/assets/e66ca762-aee3-4369-82ff-11268f8134fe)

<br/>

<DRCT로 복원한 이미지>

<br/>

![DRCT_LP](https://github.com/user-attachments/assets/242da43f-0f03-41f8-b33f-d8ed04757e90)

<br/>

<DRCT_LP로 복원한 이미지>
