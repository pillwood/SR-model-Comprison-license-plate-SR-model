# Super-Resolution-model-Comparison
4종류의 SR model의 비교 분석 미니 프로젝트


학습 진행 모델 : 

1. FSRCNN
https://github.com/yjn870/FSRCNN-pytorch
2. ESRGAN
https://github.com/xinntao/ESRGAN
3. SwinIR
https://github.com/JingyunLiang/SwinIR
4. DRCT
https://github.com/ming053l/DRCT

DF2K모델을 통해 학습을 진행 : 


![image](https://github.com/user-attachments/assets/0ae41a7b-3d2d-41a4-899c-32c4c4aca287)




![image](https://github.com/user-attachments/assets/0986fd23-940a-417e-a282-f5bca1a5c860)

![image](https://github.com/user-attachments/assets/675729b9-870f-445d-8c1f-324bf1d2f416)


결과적으로 DRCT모델이 학습 시간 대비 가장 성능이 좋다.


그럼 차량 번호판 데이터를 학습하며 L2 loss를 사용해 노이즈와 아티팩트를 제거하지 않고
L1 loss만을 사용하여 학습한다면 번호판 복원에 대해 더 좋은 성능을 낼 수 있지 않을까


학습 진행 데이터 : 
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=172

총 30epoch 학습 (iter 계산해서 넣어줌)

이미지 사이즈가 너무 작은 관계로 종 횡 방향 크기가 64미만인 이미지는 제거후 사용. 
(script 폴더 내부에 이미지 전처리 스크립트 작성)



이미지 출처 : https://www.ohmynews.com/NWS_Web/Mobile/img_pg.aspx?CNTN_CD=IE003048022



![LR image](https://github.com/user-attachments/assets/85c1aaa6-ff30-40a0-8836-15239e914587)


<원본 저해상도 이미지>


![DRCT](https://github.com/user-attachments/assets/e66ca762-aee3-4369-82ff-11268f8134fe)


<DRCT로 복원한 이미지>


![DRCT_LP](https://github.com/user-attachments/assets/242da43f-0f03-41f8-b33f-d8ed04757e90)


<DRCT_LP로 복원한 이미지>
