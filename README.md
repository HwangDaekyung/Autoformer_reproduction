# Autoformer Reproduction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HwangDaekyung/Autoformer_reproduction/blob/main/notebooks/Autoformer_from_scratch.ipynb)

**Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting** (Wu et al., NeurIPS 2021)를 직접 구현하고 논문 결과를 재현한 저장소입니다.

- 논문: https://arxiv.org/abs/2106.13008
- 공식 코드: https://github.com/thuml/Autoformer

공식 코드를 import하지 않고 **데이터 로더, 모델, 학습 루프를 모두 직접 구현**했습니다. 구현이 맞는지는 공식 코드와 같은 가중치를 넣었을 때 출력이 일치하는지로 검증했습니다.

---

## 1. 구성

```
Autoformer_reproduction/
├── README.md
├── notebooks/
│   └── Autoformer_from_scratch.ipynb   # 직접 구현 + 실험 노트북 (Colab)
└── results/                            # 실험 결과 (CSV, 그림)
```

노트북 구성은 다음과 같습니다.

| 섹션 | 내용 | 논문 대응 |
|---|---|---|
| 1 | 데이터 로더 (분할, 정규화, 시간 특징, 슬라이딩 윈도우) | 4장 Datasets |
| 2 | Series Decomposition 블록 | 3.1절, 식 (1) |
| 3 | Auto-Correlation 메커니즘 (FFT, Top-k 지연, Time Delay Aggregation) | 3.2절, 식 (5)~(8) |
| 4 | 임베딩 (위치 임베딩 없음) | Appendix G.2 |
| 5 | 인코더 · 디코더 | 식 (3), (4) |
| 6 | 전체 모델 (디코더 입력 초기화) | 식 (2), Figure 1 |
| 7 | 공식 코드와 출력 일치 검증 | |
| 8 | 학습 · 평가 루프 | 4장 Implementation details |
| 9 | 실험 실행 | Table 1, Table 10 |
| 10 | 논문 수치와 비교 | Table 10 |
| 11 | 시각화 (예측, 분해, 학습된 주기) | Figure 4, 6, 8 |

## 2. 실행 방법

1. **데이터 준비**: 공식 저장소 README의 Google Drive 링크에서 6개 데이터셋을 받아 Google Drive의 `WORK_DIR/Autoformer/dataset/` 아래에 압축을 풉니다.
   ```
   dataset/
   ├── ETT-small/ETTm2.csv
   ├── electricity/electricity.csv
   ├── exchange_rate/exchange_rate.csv
   ├── traffic/traffic.csv
   ├── weather/weather.csv
   └── illness/national_illness.csv
   ```
2. 위의 **Open In Colab** 배지로 노트북을 열고, 런타임을 GPU로 설정합니다.
3. 섹션 0의 `WORK_DIR`를 본인 Drive 경로로 바꿉니다.
4. `QUICK_TEST = True`로 전체 실행해 오류가 없는지 확인합니다.
5. `QUICK_TEST = False`로 바꾸고 섹션 9의 `run_experiment('ETTm2')` 등을 실행합니다.
   - 결과는 실험마다 `WORK_DIR/my_autoformer/results.csv`에 저장되며, 연결이 끊겨도 다시 실행하면 끝난 실험은 건너뜁니다.
6. 섹션 7(공식 코드 비교)을 실행하려면 공식 저장소를 `WORK_DIR/Autoformer`에 clone해 둡니다.

## 3. 실험 설정

| 항목 | 값 |
|---|---|
| 입력 길이 / 디코더 과거 길이 | 96 / 48 (ILI: 36 / 18) |
| 예측 길이 | 96, 192, 336, 720 (ILI: 24, 36, 48, 60) |
| 모델 | d_model 512, head 8, 인코더 2층, 디코더 1층, d_ff 2048, 이동평균 커널 25, dropout 0.05 |
| 학습 | MSE 손실, Adam (lr 1e-4), batch 32, 최대 10 epoch, early stopping patience 3 |
| 학습률 스케줄 | epoch마다 절반 (공식 코드 `lradj=type1`) |
| 반복 | 3회 (시드 2021로 시작) |

데이터셋별 설정은 공식 `scripts/`를 따랐습니다.

| 데이터셋 | 변수 수 | factor (c) | freq | 비고 |
|---|---|---|---|---|
| ETTm2 | 7 | 1 | t (분) | |
| Electricity | 321 | 3 | h | |
| Exchange | 8 | 3 | h | 예측 192, 336은 1 epoch |
| Traffic | 862 | 3 | h | 3 epoch |
| Weather | 21 | 3 | h | 예측 96은 2 epoch |
| ILI | 7 | 3 | h | 입력 36 |

## 4. 결과

> 실험 진행 중입니다. 완료되면 아래 표를 채웁니다.

### ETTm2 (다변량, input-96)

| 예측 길이 | 논문 MSE | 공식 코드 MSE | 내 구현 MSE | 논문 MAE | 공식 코드 MAE | 내 구현 MAE |
|---|---|---|---|---|---|---|
| 96  | 0.255 ± 0.020 | | | 0.339 ± 0.020 | | |
| 192 | 0.281 ± 0.027 | | | 0.340 ± 0.025 | | |
| 336 | 0.339 ± 0.018 | | | 0.372 ± 0.015 | | |
| 720 | 0.422 ± 0.015 | | | 0.419 ± 0.010 | | |

(논문 값은 Table 10의 평균 ± 표준편차입니다.)

## 5. 구현 검증

모듈 이름을 공식 코드와 같게 구성해, 공식 모델의 가중치를 직접 구현한 모델에 그대로 불러올 수 있게 했습니다.
같은 가중치와 같은 입력에서 두 모델의 출력을 비교한 결과는 다음과 같습니다.

| 비교 항목 | 결과 |
|---|---|
| 추론 모드 출력 (샘플별 지연 선택) | 최대 오차 0.0 |
| 학습 모드 출력 (배치 공통 지연 선택, dropout 0) | 최대 오차 0.0 |
| 데이터 로더 (ETTm2 train / val / test) | 샘플 수 34369 / 11425 / 11425, 값 일치 |

## 6. 논문과 공식 코드의 차이

구현 과정에서 확인한 차이입니다. 직접 구현은 모두 공식 코드를 따랐습니다.

| # | 논문 | 공식 코드 |
|---|---|---|
| 1 | 식 (2): 입력의 뒤쪽 절반만 분해 | 입력 전체를 분해한 뒤 마지막 `label_len`개 사용 |
| 2 | 식 (4): 추세마다 사영 W 세 개 | 세 추세를 더한 뒤 사영 하나 (Conv1d, kernel 3) |
| 3 | Algorithm 4: 추론 시 배치 평균 지연 | 추론 시 샘플별 지연 |
| 4 | 학습률 스케줄 언급 없음 | epoch마다 절반 |
| 5 | 데이터셋별 학습 epoch 언급 없음 | Weather, Traffic, Exchange 일부는 1~3 epoch |
| 6 | ETT 6:2:2 분할 | 12개월 / 4개월 / 4개월 (한 달 30일) |

## 7. 재현 시 겪은 이슈 (공식 코드 실행 환경)

| 이슈 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: reformer_pytorch` | 비교 모델(Reformer)용 라이브러리 미설치, `requirements.txt`에 버전 고정 없음 | `pip install reformer_pytorch` |
| `AttributeError: np.Inf` | NumPy 2.0에서 `np.Inf` 삭제 | `utils/tools.py`의 `np.Inf` → `np.inf` |
| `freq='t'` 해석 오류 가능성 | 최신 pandas에서 `'t'` 표기 제거 | 직접 구현에서는 시간 특징을 pandas 없이 직접 계산 |

## 8. 실행 환경

- Google Colab, NVIDIA A100 40GB
- Python 3, PyTorch 2.11.0 (CUDA 13.0)

## 참고

```bibtex
@inproceedings{wu2021autoformer,
  title={Autoformer: Decomposition Transformers with {Auto-Correlation} for Long-Term Series Forecasting},
  author={Haixu Wu and Jiehui Xu and Jianmin Wang and Mingsheng Long},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
