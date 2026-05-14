# K리그 패스 좌표 예측 AI 모델

**Track1 알고리즘 부문 : K리그–서울시립대 공개 AI 경진대회**

K리그 경기 내 주어진 플레이 시퀀스의 마지막 패스 도착 좌표 `(end_x, end_y)`를 예측하는 멀티모달 딥러닝 모델입니다. 좌표는 FIFA 권장 규격인 105 × 68 그리드에 매핑된 상대 좌표이며, 평가는 정답 좌표와의 **유클리드 거리**로 이뤄집니다.

## 최종 성적

| 구분 | Public Score (Euclidean) | 비고 |
|---|---:|---|
| 대회 제공 베이스라인 (LSTM) | 17.6204388616 | 출발점 |
| 팀 공동 베이스라인 (12/29 Multimodal) | 13.9166288138 | 약 −4.70 개선 |
| **본 팀 최종 Public** | **13.1983558111** | 베이스라인 대비 약 −0.72 |
| **본 팀 최종 Private (리더보드)** | **13.22176** | 최종 76위 / 937팀 |
| (참고) 수상권 경계 (15위 / 상위 1%) | 12.83347 | 본 팀과 약 +0.39 차이 |

최종 제출본: [`submissions/ensemble_last.csv`](submissions/ensemble_last.csv)

> 평가 지표는 정답 좌표와 예측 좌표 간 **유클리드 거리(낮을수록 우수)** 입니다. 본 팀은 대회 제공 LSTM 베이스라인 대비 **−4.40 (≈25%) 개선**, 수상권 컷오프와는 **0.39**의 격차로 마무리했습니다.

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

## 데이터 전처리

원본 이벤트 시퀀스는 `(game_id, game_episode, time_seconds, type_name, result_name, team_id, start_x/y, end_x/y, …)` 형식입니다. 모든 좌표는 105×68 그리드에 매핑된 상대 좌표이며, 다음 단계로 모델 입력으로 변환합니다.

### 1) 연속 피처 (10차원)

각 시점마다 다음 10개 피처를 생성합니다 ([`src/preprocessing_multimodal_team.py`](src/preprocessing_multimodal_team.py)).

| # | 피처 | 의미 |
|---|---|---|
| 0–1 | `start_x_norm`, `start_y_norm` | 현재 이벤트 시작점을 105/68로 정규화 |
| 2–3 | `end_x_prev_norm`, `end_y_prev_norm` | 직전 이벤트의 끝점 (직전 행 `shift(1)`, fallback = 현재 start) |
| 4–5 | `dx_prev_norm`, `dy_prev_norm` | **직전 이벤트의 변위 벡터** (x/y 분리, 각도 불연속 회피) |
| 6 | `speed_prev_log` | 직전 평균 속도의 `log1p` — 장거리 패스의 outlier 완화 |
| 7 | `time_delta` | 이전 이벤트와의 시간 간격 (최소 0.01s 클램프) |
| 8 | `dist_to_goal_norm` | 시작점 → 상대 골대 중앙 `(105, 34)` 까지 거리 |
| 9 | `is_continuous` | 직전 종점과 현재 시작점의 간격이 **2 m 미만**이면 1, 아니면 0 |

마지막에 `StandardScaler`로 표준화합니다. 스케일러와 라벨 인코더는 **Fold 내부에서 fit** → leakage 방지.

### 2) Continuity Masking (안티-리키지/디노이징)

`is_continuous = 0` (직전 종점과 현재 시작점이 2 m 이상 떨어진 경우 — 즉, 시퀀스가 끊긴 경우) 인 행은 `speed_prev_log`, `dx_prev_norm`, `dy_prev_norm` 을 **0으로 마스킹** 하여 의미 없는 물리량이 학습 신호로 들어가지 않도록 합니다.

### 3) 카테고리 임베딩

`type_name`, `result_name`, `team_id` 를 LabelEncoder로 정수 인덱스화 → 모델 내부에서 임베딩.
- `result_name` 결측치는 `"{type_name}_Implicit"` 로 치환 (Implicit 이벤트 보존).
- 미관측 값은 모두 `"Unknown"` 토큰으로 처리.

### 4) 이미지 채널 생성 (시퀀스 → 2-채널 픽셀맵)

플레이 시퀀스를 **(2, 68, 105) 텐서**로 렌더링하여 CNN 입력으로 사용합니다.

- **채널 0 (점)**: 각 이벤트의 좌표 위치에 `1.0` 점 표시.
- **채널 1 (선)**: 연속한 이벤트 사이를 `cv2.line` 으로 잇되, **시간 가중치** `(t+1)/seq_len` 을 색 강도로 부여 → 시퀀스의 시간 흐름을 공간적으로 인코딩.
- 좌표는 `round → clip(0, W−1)` 로 정수 인덱싱.

### 5) Polar 라벨 변환 (Polar 모델 전용)

`Multimodal_v7_Polar_H96.ipynb` 에서는 타깃을 `(end_x, end_y)` 대신 `(distance, sinθ, cosθ)` 로 분해해 회귀하고, 추론 시 다시 좌표로 복원합니다 ([`src/preprocessing_polar.py`](src/preprocessing_polar.py)).
- **이유**: 패스의 물리적 본질("얼마나 멀리, 어느 방향")과 일치하며, `(sinθ, cosθ)` 사용으로 −π/π 경계의 각도 불연속이 사라집니다.
- 라벨은 **학습 데이터에서만** 생성 → 테스트 시 leak 방지.

### 6) 데이터 증강 — Random Y-Flip

축구장이 가로 중심선(Y축)에 대해 대칭임을 활용. 학습 시 **50% 확률**로 이미지·연속 피처·타깃을 동시에 Y축 반전 → 사실상 데이터 2배 효과. 추론 시에는 **TTA(Original + Y-Flip 평균)** 으로 분산을 추가로 줄입니다.

### 7) Fold 분할

`GroupKFold(n_splits=10)` 를 **`game_id` 기준**으로 적용 → 같은 경기의 이벤트가 학습/검증에 동시에 들어가는 누수를 방지합니다. Fold마다 시드를 42→51로 다양화하여 앙상블 분산도 확보했습니다.

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

| 단계 | 기법 | Public Score |
|---|---|---:|
| 대회 제공 LSTM | 단일 시퀀스 모델 | 17.620 |
| 팀 베이스라인 | Multimodal (CNN + Bi-LSTM) | 13.917 |
| 시퀀스 안정화 | LSTM → **Split GRU** (미래 정보 누수 차단) | — |
| 데이터 증강 | **Random Y-Flip** (경기장 좌우 대칭 활용) | — |
| 추론 안정화 | **TTA (Y-Flip 평균)** + Raw→Flip→Scale 순서 수정 | — |
| 표현 학습 | **Polar 좌표 회귀** (거리·방향 분해) | — |
| 상호작용 학습 | **Cross-Attention Fusion** (GRU Query × CNN Map) | — |
| 다양성 확보 | **Fold별 시드 다양화 (42–51)** | — |
| **최종 앙상블** | h96 / polar / env **(0.35 : 0.30 : 0.35) 가중 평균** | **13.198** |
| (참고) 수상권 컷오프 | 15위 | 12.833 |

> 중간 단계의 단일 모델 점수는 별도 리더보드 제출 없이 내부 CV로만 확인했기 때문에 비워뒀습니다. 본 표는 **최종 앙상블이 어떤 누적 개선의 결과인가**를 보여주기 위함입니다.

## 라이선스

[MIT License](LICENSE)
