# MD2Short

마크다운 문서 한 장을 입력하면 한국어 9:16 숏폼 영상(1080x1920 MP4)을 자동으로 만들어 내는 로컬 CLI 제작 도구다. 일반 본문이나 `### N컷` 스토리보드 형식의 마크다운을 받아 한국어 내레이션 대본을 만들고(OpenAI), ElevenLabs TTS로 음성을 합성하고, ElevenLabs Scribe의 단어 타임스탬프(`words.json`)를 단일 시간 기준으로 삼아 자막·호랑이 이모티콘 cue·숫자 카드·오프닝·CTA를 Remotion으로 합성해 MP4 한 편을 렌더링한다. 모든 중간 산출물과 API 로그, QC 리포트는 `jobs/<job_id>/` 아래에 남아 재현·디버깅·비용 추적이 가능하다. 제품 요구사항은 [`docs/PRD.md`](docs/PRD.md), 시스템 설계는 [`docs/architecture.md`](docs/architecture.md), 문서 색인은 [`docs/README.md`](docs/README.md)를 따른다.

## 데모

마크다운 시황 리포트 한 장으로 만든 결과 영상의 앞 24초다. 전체 영상(48초, 음성 포함)은 [`docs/assets/demo.mp4`](docs/assets/demo.mp4)에 있다.

<p align="center">
  <img src="docs/assets/demo.gif" alt="MD2Short 데모: 오프닝 브리핑, 자막, 호랑이 이모티콘 cue, 숫자 카드" width="300">
</p>

## 1. 사전 요구사항

| 도구 | 용도 | 설치 |
| --- | --- | --- |
| [`uv`](https://docs.astral.sh/uv/) | Python 3.12+ 환경 및 의존성 관리, `md2short` 실행 | `brew install uv` 또는 `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Node.js + `npx` | Remotion 렌더러 실행 (`npx remotion render`) | `brew install node` |
| `ffmpeg` / `ffprobe` | duration 측정 전용 보조 (QC의 `sync_last_word_vs_audio`, `final_mp4_vs_audio`) | `brew install ffmpeg` |

설치 확인:

```bash
uv --version
node --version && npx --version
ffmpeg -version | head -1 && ffprobe -version | head -1
```

Remotion 의존성은 한 번 받아 두어야 한다. `remotion/node_modules`가 없으면 pre-flight가 바로 실패한다.

```bash
cd remotion && npm install
```

`ffmpeg`/`ffprobe`/`npx`/`remotion/node_modules`는 모두 파이프라인 시작 전 pre-flight에서 검사한다(`render.ensure_ffmpeg_available`, `render.ensure_remotion_available`) — API 크레딧을 쓰기 전에 실패한다.

## 2. `.env` 설정

시크릿은 저장소 루트 `.env`에만 둔다(커밋 금지). [`.env.example`](.env.example)을 복사해서 시작한다.

```bash
cp .env.example .env
```

실제로 읽는 변수는 [`src/md2short/config.py`](src/md2short/config.py)가 유일한 기준이다.

| 변수 | 필수 | 설명 |
| --- | --- | --- |
| `ELEVENLABS_API_KEY` | 필수 | ElevenLabs API 키. TTS(음성 합성)와 Scribe(STT 단어 타임스탬프)에 공통으로 쓰인다. |
| `ELEVNLABS_API` | 선택(legacy) | 오타가 섞인 구 이름. 하위 호환을 위해 **아직 지원된다** — `config.get_elevenlabs_api_key()`가 `ELEVENLABS_API_KEY`를 먼저 보고, 없을 때만 이 이름으로 fallback한다. 둘 중 하나만 있으면 된다. |
| `OPENAI_API_KEY` | 필수 | 대본 생성(`gpt-5-mini`, Responses API)과 `--visual-mode photo`의 이미지 생성(`gpt-image-2`)에 쓰인다. 없으면 대본 단계가 `fallback_extract`(마크다운 평문 추출)로 내려가고 photo mode는 emoticon mode로 떨어진다. |
| `ELEVENLABS_VOICE_ID` | 선택 | 기본 voice id 오버라이드. `schema.DEFAULT_VOICES["ko"]`는 voice library 음성이라 무료 계정에서 HTTP 402 `paid_plan_required`로 거절되므로, 그런 계정은 premade 음성 id를 여기에 넣는다. CLI `--voice` 플래그가 이 값보다 우선한다. 자세한 내용은 [`docs/voice.md`](docs/voice.md). |

## 3. 에셋 디렉터리

Remotion은 `remotion/public/` 아래의 정적 에셋을 서빙한다. 프로젝트 단위 에셋(`emoticon`, `img`, `company`, `font`)은 job마다가 아니라 한 번만 세팅되고, 없으면 `render._ensure_project_public_assets()`가 저장소 루트의 원본에서 심링크(실패 시 복사)로 복구한다.

| `public/` 이름 | 저장소 원본 | 상태 | 없으면 |
| --- | --- | --- | --- |
| `emoticon` | `emoticon/emoticon/` | **필수 / 이 저장소에 포함됨** (호랑이 PNG 60장) | 이모티콘 cue가 전부 404 → 시각 레이어가 비어 보인다 |
| `font` | — | 포함됨 (`remotion/public/font/`) | 자막 폰트가 시스템 기본으로 떨어진다 |
| `img` | `img/` (`img/header_logo.png`) | **선택 / 이 저장소에 없음** | `BrandHeader`가 로고 로드 실패를 감지해 텍스트 워드마크("데이브리핑")로 degrade한다 — 렌더는 실패하지 않는다 |
| `company` | `company/` | **선택 / 이 저장소에 없음** | 회사 로고 오버레이가 표시되지 않는다 (현재 cue를 만드는 코드도 없다 — 6절 참고) |

> `emoticon` 원본은 한 단계 더 깊은 `emoticon/emoticon/`이다(`remotion/public/emoticon -> ../../emoticon/emoticon`). PNG 60장이 들어 있는 그 디렉터리를 가리켜야 한다.

`img/`와 `company/`가 없어도 렌더는 정상 종료하지만, **운영 배포에는 두 디렉터리를 채워 넣는 것을 권한다.** 로고 없는 영상은 브랜드 포맷의 핵심 요소 하나가 빠진 상태다.

## 4. 실행 방법

기본(이모티콘 모드):

```bash
uv run md2short run docs/sample.md --visual-mode emoticon
```

성공하면 `jobs/<job_id>/`에 전체 산출물이 남고, 최종 MP4가 `output/ret.mp4`로 복사된다(`--output ""`로 복사를 끌 수 있다).

photo 모드(청크마다 `gpt-image-2` 이미지를 생성하므로 **`OPENAI_API_KEY`가 반드시 필요하고 비용이 훨씬 크다**):

```bash
uv run md2short run docs/sample.md --visual-mode photo
```

이미지 생성이 실패하면 job 전체가 emoticon 모드로 fallback하고 그 사실이 `jobs/<job_id>/logs/photo_fallback.json`에 기록된다.

자주 쓰는 다른 플래그: `--voice <id>`, `--speed 0.7~1.2`, `--chunk-size 1~24`, `--bg 0xFFFFFF`, `--model gpt-5-mini`, `--jobs-root jobs`.

job 상태 확인:

```bash
uv run md2short status <job_id>
```

오래된 job 정리 (먼저 `--dry-run`으로 대상만 확인할 것):

```bash
uv run md2short clean --older-than 24h --dry-run
uv run md2short clean --older-than 24h
```

`--older-than`은 `24h`, `7d`, `30m`, `3600s`, 또는 정수 초를 받는다.

## 5. 테스트

```bash
uv run pytest                 # 전체 Python 스위트
uv run pytest -m "not slow"   # 실제 npx remotion render 서브프로세스를 띄우는 테스트 제외
```

```bash
cd remotion && npx tsc --noEmit   # Remotion/TypeScript 타입 체크, 0 에러를 유지해야 한다
```

`remotion/public/jobs/sample/`은 테스트 fixture이며 `Root.tsx`의 `defaultProps`도 이 경로를 쓴다. **정리 작업 중에 지우면 안 된다.**

## 6. 알려진 제한사항

- **회사 로고 cue 생성(F-05)이 구현되어 있지 않다.** Remotion 렌더 쪽은 `companyLogoCues` prop과 `public/company/<fileName>` 오버레이를 지원하지만, Python 파이프라인에는 그 cue를 만들어 내는 모듈이 아직 없다 — `pipeline.run_job()`은 항상 빈 리스트를 넘긴다. 이는 **명시적으로 유보한 범위 공백이며 버그가 아니다.** 로고 오버레이가 필요하면 cue planner를 새로 구현해야 한다.
- **Scribe(ASR)가 긴 숫자 묶음을 간혹 잘못 끊는다.** 자막 텍스트에만 영향을 주는 표시 문제이며, 숫자 카드(`NumberCard`)는 원문 숫자 표기를 그대로 보여주므로 강조되는 수치 자체는 정확하다.
- **photo 모드의 moderation 재시도 체인이 실제 moderation 거절에 대해 검증되지 않았다.** 단위 테스트로만 덮여 있고, 실제 운영에서 `error.code` moderation 거절 경로가 돌아간 적은 없다.

## 7. 비용

`jobs/<job_id>/qc_report.json`의 `cost_estimate`가 job별 로컬 추정치를 담는다: ElevenLabs TTS(문자 수 기준), ElevenLabs STT(오디오 길이 기준), OpenAI 텍스트 모델(char/4 토큰 근사 × 고정 단가). **`gpt-image-2` 이미지 생성 비용은 `total_usd`에 합산하지 않는다** — photo 모드를 썼다면 실제 OpenAI 청구서로 따로 확인해야 한다.
