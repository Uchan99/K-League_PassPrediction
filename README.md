# K리그 패스 좌표 예측 AI 모델

**Track1 알고리즘 부문 : K리그–서울시립대 공개 AI 경진대회**

K리그 경기 내 주어진 플레이 시퀀스의 마지막 패스 도착 좌표 `(end_x, end_y)`를 예측하는 멀티모달 딥러닝 모델입니다. 좌표는 FIFA 권장 규격인 105 × 68 그리드에 매핑된 상대 좌표이며, 평가는 정답 좌표와의 **유클리드 거리**로 이뤄집니다.

## 최종 성적

| 구분 | 점수 |
|---|---|
| Baseline Public | 13.9166288138 |
| **Final Public** | **13.1983558111** |
| **Final Private (리더보드)** | **13.22176** |
| 최종 순위 | **76위 / 937팀** |

최종 제출본: [`submissions/ensemble_last.csv`](submissions/ensemble_last.csv)

## 팀 정보

- **팀명**: 전북현대Unicorns
- **팀원** (3인): 허유찬, 조연서, 조시현
- **프로젝트 기간**: 2025-12-01 ~ 2026-01-26
  - 2025-12-29 이전: 팀원별로 전처리·모델링을 독립 진행
  - 2025-12-29 이후: `Baseline/Multimodal.ipynb`를 공동 베이스라인으로 확정 후 고도화 진행

## 저장소 구조

```
K-League_PassPrediction/
├── README.md
├── LICENSE
├── Baseline/                              # 팀 공동 베이스라인 (12/29 확정본)
│   ├── Multimodal.ipynb
│   └── preprocessing_multimodal.py
├── notebooks/                             # 고도화 모델 노트북 (앙상블 구성원)
│   ├── Multimodal_v7_Tuned_H96.ipynb      # ① h96 (가중치 0.35)
│   ├── Multimodal_v7_Polar_H96.ipynb      # ② polar (가중치 0.30)
│   ├── Final_env_v2_4 fixed.ipynb         # ③ env (가중치 0.35)
│   └── Multimodal_v8_Team_H96.ipynb       # 최종 앙상블 실행 노트북
├── src/                                   # 전처리 · 모델 · 손실 함수 모듈
│   ├── preprocessing_multimodal_team.py
│   ├── preprocessing_polar.py
│   ├── Preprocessing_final_v2_fixed.py
│   ├── models_multimodal_v8_team.py
│   ├── polar_loss.py
│   └── polar_utils.py
└── submissions/                           # 추론 결과 및 최종 제출본
    ├── ensemble_last.csv                  # ★ 최종 제출 (Public 13.198 / Private 13.222)
    ├── ensemble_final.csv                 # 비교용 (w_h96=0.5, w_polar=0, w_env=0.5)
    ├── multimodal_v7_h96_tta_corrected.csv
    ├── submission_polar_h96_optimized.csv
    └── Final_env_v2_4.csv
```

## 모델 아키텍처 개요

모든 고도화 모델은 **Multi-modal (이미지 + 시퀀스 + 카테고리)** 구조를 공유합니다.

- **이미지 브랜치**: 플레이 시퀀스를 68×105 그라운드 위에 점·선으로 렌더링(시간 가중 decay 포함) → CNN + Spatial Attention.
- **시퀀스 브랜치**: 연속 피처 + 임베딩(`type`, `result`, `team` 등) → **Split GRU** (Forward / Backward를 분리하여 미래 정보 누수 방지) + Attention Pooling.
- **융합 후 회귀 헤드**: 두 브랜치를 concat → FC → `(end_x, end_y)` 예측.
- **공통 학습 셋업**: 10-Fold `GroupKFold` (game_id 기준) + Fold별 시드 다양화(42~51), Y-Flip Augmentation, ReduceLROnPlateau, Early Stopping(patience=15), TTA(Y-Flip 평균).

### 앙상블 구성원 (notebooks/)

| 노트북 | 핵심 차별점 | 출력 CSV | 앙상블 가중치 |
|---|---|---|---|
| `Multimodal_v7_Tuned_H96.ipynb` | GRU hidden=96, **TTA-Corrected** (Raw→Flip→Scale 순서 버그 수정) | `multimodal_v7_h96_tta_corrected.csv` | **0.35** (h96) |
| `Multimodal_v7_Polar_H96.ipynb` | **Polar 좌표 회귀** — `(distance, sin θ, cos θ)`로 분해 예측 후 좌표 복원, Huber + Cosine Loss | `submission_polar_h96_optimized.csv` | **0.30** (polar) |
| `Final_env_v2_4 fixed.ipynb` | **Cross-Attention Fusion** — GRU(Query) × CNN Feature Map(Key/Value)로 시공간 상호작용 학습 | `Final_env_v2_4_last.csv` | **0.35** (env) |
| `Multimodal_v8_Team_H96.ipynb` | **Team ID 임베딩(dim=8)** 추가 모델 + **최종 앙상블 실행 셀(Cell 19)** 포함 | — (앙상블 러너 역할) | — |

### 최종 앙상블 (`Multimodal_v8_Team_H96.ipynb` Cell 19)

```python
make_submission("ensemble_last", w_h96=0.35, w_polar=0.30, w_env=0.35)
# end_x = 0.35*h96 + 0.30*polar + 0.35*env  (좌표별 가중 평균)
# end_y = 0.35*h96 + 0.30*polar + 0.35*env
# clip: [0, 105] × [0, 68]
```

> ℹ️ V8-Team 모델 자체는 실험적으로 학습·평가했으나, **최종 제출(`ensemble_last.csv`)에는 포함되지 않았습니다.** v8 노트북은 앙상블 러너 역할을 했습니다.

## Reproduction

1. **데이터 배치** — 대회 제공 파일(`train.csv`, `test.csv`, `sample_submission.csv`)을 작업 디렉토리에 배치합니다. `Final_env_v2_4 fixed.ipynb`는 `./data/` 하위 경로를 기대합니다.
2. **개별 모델 학습 & 추론**
   - `notebooks/Multimodal_v7_Tuned_H96.ipynb` → `submission_multimodal/multimodal_v7_h96_tta_corrected.csv`
   - `notebooks/Multimodal_v7_Polar_H96.ipynb` → `submission_polar_h96_optimized.csv`
   - `notebooks/Final_env_v2_4 fixed.ipynb` → `Final_env_v2_4.csv` (앙상블 입력 시 `Final_env_v2_4_last.csv`로 사용)
3. **최종 앙상블 실행**
   - `notebooks/Multimodal_v8_Team_H96.ipynb`의 **Cell 19**를 실행하면 `submission_ensemble/ensemble_last.csv`가 생성됩니다.

> 각 노트북은 학습 단계에서 `models/`, `submission_multimodal/`, `submission_ensemble/` 디렉토리를 자체적으로 생성합니다. 학습 시 fold별 `.pth`(가중치)와 `.pkl`(전처리기)이 `models/`에 저장됩니다.

## 평가 지표

- 정답 좌표 `(x*, y*)`와 예측 좌표 `(x̂, ŷ)` 간의 **유클리드 거리** `√((x*-x̂)² + (y*-ŷ)²)`.
- 그라운드 스케일: 105 × 68 (FIFA 권장).

## 주요 개선 트랙

| 단계 | 기법 | 효과 |
|---|---|---|
| 베이스라인 | Multimodal (CNN + Bi-LSTM) | Public 13.917 |
| 시퀀스 안정화 | LSTM → **Split GRU** (미래 정보 누수 차단) | 안정성 ↑ |
| 데이터 증강 | **Random Y-Flip** (경기장 좌우 대칭 활용) | 일반화 ↑ |
| 추론 안정화 | **TTA (Y-Flip 평균)** + Raw→Flip→Scale 순서 수정 | 분산 ↓ |
| 표현 학습 | **Polar 좌표 회귀** (거리·방향 분해) | 각도 불연속 해소 |
| 상호작용 학습 | **Cross-Attention Fusion** (GRU Query × CNN Map) | 시공간 정합 ↑ |
| 다양성 확보 | **Fold별 시드 다양화** + 다중 모델 가중 평균 | 분산 ↓ |
| **최종** | h96 / polar / env **(0.35 : 0.30 : 0.35) 앙상블** | **Public 13.198** |

## 라이선스

[MIT License](LICENSE)
