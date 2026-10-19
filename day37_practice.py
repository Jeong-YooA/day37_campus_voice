# day37_practice.py
# 캠퍼스 시설 안내 음성 도우미

import asyncio # 비동기 TTS 함수를 실행
import hashlib # 원본 음성의 SHA-256 해시값 계산
import json # 실행 기록을 JSONL 형식으로 저장
import os # DATABASE_URL 환경변수 조회
import tempfile # STT 입력을 위한 임시 WAV 파일 생성
import time # 처리 단계별 실행 시간 측정
import uuid # 요청별 고유 ID 생성
import wave # WAV 파일 헤더와 PCM 형식 검사
from datetime import datetime, timezone # UTC 시각 생성
from pathlib import Path # 파일 경로 처리
import streamlit as st # 웹 기반 사용자 인터페이스
import sys
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv() # .env 로딩

# 입력 파일 최대 크기를 10Mbytes로 제한
MAX_BYTES = 10 * 1024 * 1024

# 실행이력을 저장할 jsonl파일 경로
LOG_PATH = Path(__file__).with_name('voice_runs_campus.jsonl')

# 사용할 Whisper모델 크기
MODEL_SIZE = 'base'

# Edge TTS에서 사용하는 한국어 여성 음성 모델 ID
VOICE_ID = 'ko-KR-SunHiNeural'

FACILITIES = {
    "도서관": "도서관은 본관 옆 건물이며 평일 9시부터 18시까지 운영합니다.",
    "학생식당": "학생식당은 학생회관 1층에 있으며 11시부터 14시까지 운영합니다.",
    "취업지원센터": "취업지원센터는 학생회관 2층이며 취업 상담을 제공합니다.",
    "행정실": "행정실은 본관 1층이며 각종 증명서 발급을 안내합니다.",
}

# 질문에서 시설명을 찾아 해당 시설의 안내 문장을 반환하는 함수
def generate_response(question):

    # 질문의 공백을 제거
    question = "".join(question.split())

    for facility, guide in FACILITIES.items():
        # 시설명의 공백도 제거
        facility_name = "".join(facility.split())

        if facility_name in question:
            return guide

    facilities_text = ', '.join(FACILITIES.keys())

    return f'안내 가능한 시설은 {facilities_text}입니다. 시설명을 포함해서 다시 질문해 주세요'

# 데모용 음성 인식 결과를 반환하는 함수
def demo_stt():
    return "학생 식당 어디야?"



# 실제 WAV파일인지 확인하고 음성 파일의 기본 정보를 추출하는 함수
def validata_wav(audio):
    import io

    try:
        # 바이너리를 메모리 스트림으로 변환하여 WAV 헤더 읽기
        with wave.open(io.BytesIO(audio), 'rb') as wav:
            channels = wav.getnchannels() # 채널 수
            sample_rate = wav.getframerate()  # 샘플링레이트
            sample_width = wav.getsampwidth()  # 샘플당 바이트 수
            frames = wav.getnframes() # 전체 오디오 샘플 수
            compression = wav.getcomptype() # 압충 형식 (NONE = 비압축 PCM)

    except (wave.Error, EOFError, OSError) as e:
        raise ValueError('유효한 비압축 PCM WAV 파일을 입력해 주세요') from e

    if compression != 'NONE' or channels not in (1, 2) or sample_rate <= 0 or frames <= 0:
        raise ValueError('모노/스테레오 비압축 PCM WAV 파일이 필요합니다.')

    return {
        'channels': channels,
        'sample_rate': sample_rate,
        'sample_width': sample_width * 8,
        'duration_sec' : round(frames / sample_rate, 2)
    }
# faster-whisper 음성 인식 모델을 CPU에 로딩하는 함수
@st.cache_resource(show_spinner='Wisper 모델을 로딩하고 있습니다...')
def load_stt_model(model_size):
    return WhisperModel(model_size,device='cpu',compute_type='int8')


# WAV 데이터를 임시 하일로 저장한 후 실제 한국어 음성 인식 수행하는 함수
def transcribe_wav(audio,model_size):
    model = load_stt_model(model_size)

    # faster-whisper가 처리할 수 있도록 임시 WAV 파일 생성
    with tempfile.NamedTemporaryFile(suffix='wav', delete=False) as f:
        wav_path = Path(f.name) # 생성된 임시파일 경로
        f.write(audio)

    try:
        # 전사 후 segment들과 부가정보 획득
        segments, info = model.transcribe(
            str(wav_path),
            language = 'ko', # 언어를 한국어로 처리
            vad_filter=True # 음성 활동 탐지로 무음 구간 처리
        )
        # 구간별 텍스트를 공백으로 연결하여 하나의 문장으로 생성
        transcript = ' '.join(segments.text.strip() for segments in segments).strip()

        return transcript

    finally:
        # 인식 성공/실패 여부와 관계없이 임시 WAV파일 제거
        wav_path.unlink(missing_ok=True)

# 인식한 문장을 정리하고 간단한 규칙 기반 응답 문장을 생성하는 함수
def build_reply(transcript):
    # split/join으로 연속된 공백과 줄바꿈을 하나의 공백으로 통일
    clean = ' '.join(transcript.split())
    if not clean:
        return '음성 인식을 하지 못했습니다. 다시 말씀해 주세요.'
    return f'말씀하신 내용은 {clean}입니다.'

# Edge TTS로 한국어 음성을 합성하여 MP3 bytes로 반환하는 함수
async def generate_tts_mp3(text, voice):
    import edge_tts
    with tempfile.NamedTemporaryFile(suffix='mp3', delete=False) as f:
        mp3_path = Path(f.name)
    try:
        # 문장과 목소리 ID를 전달한 후 MP3로 저장
        await edge_tts.Communicate(text=text, voice=voice).save(str(mp3_path))
        return mp3_path.read_bytes()
    finally:
        mp3_path.unlink(missing_ok=True)


# 비동기 Edge TTS 함수를 동기코드에서 실행하는 함수
def synthesize_speech(text, voice_id):
    mp3_bytes = asyncio.run(generate_tts_mp3(text, voice_id))
    return mp3_bytes,'audio/mpeg' # MP3파일의 bytes와 MIME 반환

# 실행 정보를 JSONL 파일에 추가하고 선택적으로 PostgreSQL에 저장
def store_meta(meta):

    # JSONL은 JSON객체를 한 줄에 하나씩 기록하는 텍스트 파일
    with LOG_PATH.open('a', encoding='utf-8') as f:
        # json.dumps(딕셔너리) : 딕셔너리를 JSON객체 문자열로 변환
        # ensure_ascii=False : 한글로 저장 = ASCII로 변환해서 저장하지 않음
        f.write(json.dumps(meta, ensure_ascii=False) + '\n')

    # 환경변수의 DATABASE_URL 변수의 값
    database_url = os.getenv('DATABASE_URL','').strip()
    if not database_url:
        return

    try:
        import psycopg
        with psycopg.connect(database_url, connect_timeout=5) as conn:

            with conn.cursor() as cur:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS voice_runs(
                        request_id TEXT PRIMARY KEY, -- 요청 ID
                        created_at TIMESTAMP NOT NULL, --생성일시
                        status TEXT NOT NULL,--상태문자열
                        stt_ms DOUBLE PRECISION NOT NULL, -- STT 소요시간
                        reply_ms DOUBLE PRECISION NOT NULL, -- 응답 소요시간
                        tts_ms DOUBLE PRECISION NOT NULL, -- TTS 소요시간
                        total_ms DOUBLE PRECISION NOT NULL, -- 소요시간 합계
                        model_version TEXT NOT NULL, -- 모델 버전 문자열
                        audio_sha256 TEXT NOT NULL --오디오 해시값
                    )
                ''')
                cur.execute('''
                    INSERT INTO voice_runs
                    (request_id, created_at,status, stt_ms,reply_ms, tts_ms,total_ms,model_version,audio_sha256)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ''', (
                    meta['request_id'],meta['created_at'],meta['status'],meta['stt_ms']
                    ,meta['reply_ms'],meta['tts_ms'],meta['total_ms'],meta['model_version'],
                    meta['audio_sha256']
                ))

    except Exception as e:
        st.warning(f'DB 실행 이력 저장 실패: {type(e).__name__}: {e}')

# WAV 검증 > STT > 응담 생성 > TTS > 로그 저장 과정을 연결해서 실행하는 파이프라인 함수
def run_pipeline(audio,model_size,voice_id, stt_mode,tts_mode):

    # 요청을 구분할 고유 요청 ID
    request_id = str(uuid.uuid4())

    # 음성 원문을 기록하지 않고 해시값만 보관
    digest = hashlib.sha256(audio).hexdigest() # SHA256 16진수 해시값 생성

    # 단계별 실행시간 초기화 (ms)
    marks = {'stt_ms':0.0,'reply_ms':0.0,'tts_ms':0.0}

    # 전체 처리 시작 시간
    started = time.perf_counter()

    result = None # 성공한 경우 처리결과 저장
    error = None # 실패한 경우 예외처리

    try:
        # WAV 형식 검사
        validata_wav(audio)

        # STT단계 실행 후 소요시간 측정
        t0 = time.perf_counter()

        if stt_mode == '데모 STT':
            transcript = demo_stt()
        else:
            transcript = transcribe_wav(audio, model_size)
        marks['stt_ms'] = (time.perf_counter() - t0) * 1000

        # 규칙 기반 응답 생성 단계 실행 후 소요시간 측정
        t0 = time.perf_counter()
        reply = generate_response(transcript)
        marks['reply_ms'] = (time.perf_counter() - t0) * 1000

        # 실제 TTS 음성 합성 단계 실행 후 소요시간 측정
        t0 = time.perf_counter()

        if tts_mode == '비프음 데모':
            output, mime = demo_tts()
        else:
            output, mime = synthesize_speech(reply, voice_id)
        marks['tts_ms'] = (time.perf_counter() - t0) * 1000

        # UI에서 사용할 성공 결과를 튜플로 저장
        result = (transcript,reply,output,mime)

    except Exception as e:
        error = e

    # 전체 처리 시간에는 최초 모델 로딩 시간도 포함
    elapsed = (time.perf_counter() - started) * 1000

    stt_name = "demo-stt" if stt_mode == "데모 STT" else f"whisper-{model_size}"
    tts_name = "demo-beep" if tts_mode == "비프음 데모" else f"edge-tts/{voice_id}"

    meta = {
        'request_id': request_id,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'error' if error else 'success',
        **{key: round(value, 2) for key, value in marks.items()},
        'total_ms': round(elapsed, 2),
        'model_version': f'{stt_name}/{tts_name}',
        'audio_sha256': digest
    }

    store_meta(meta)

    return result, marks, meta, error

# 최근 실행 이력을 최신순으로 가져오는 함수
def load_history():
    if not LOG_PATH.exists():
        return []

    records = []

    with LOG_PATH.open('r',encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            records.append(record)

        return records[-10:][::-1]

# 데모용 비프음을 WAV 데이터로 반환하는 함수
def demo_tts():
    import io
    import math
    import struct

    sample_rate = 16000
    duration = 0.5
    frequency = 440

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        samples = []

        for i in range(int(sample_rate * duration)):
            value = int(
                5000 * math.sin(2 * math.pi * frequency * i / sample_rate)
            )
            samples.append(struct.pack('<h', value))
        wav.writeframes(b''.join(samples))

    return buffer.getvalue(), 'audio/wav'

def main():

# Streamlit 화면을 구성하는 함수

    st.title("캠퍼스 시설 안내 음성 도우미")

    st.write("시설명을 포함해서 질문해 주세요.")
    st.caption("안내 시설: 도서관, 학생식당, 취업지원센터, 행정실")

    question = st.text_input(
        "질문 입력",
        placeholder="예: 도서관 운영시간 알려줘"
    )

    if st.button("안내 받기"):
        if not question.strip():
            st.warning("질문을 입력해 주세요.")
            return

        answer = generate_response(question)

        st.subheader("안내 결과")
        st.write(answer)

# 녹음 기능 추가
    st.divider()
    st.subheader("음성으로 질문하기")

    stt_mode = st.selectbox(
        "음성 인식 방식",
        ["데모 STT", "실제 STT"]
    )


    tts_mode = st.selectbox(
        "음성 출력 방식",
        ["비프음 데모", "실제 TTS"]
    )
    audio = st.audio_input("시설에 대해 질문해 주세요.")

    if st.button("녹음 확인"):
        if audio is None:
            st.warning("먼저 음성을 녹음해 주세요.")
            return

        audio_bytes = audio.getvalue()

        if len(audio_bytes) > MAX_BYTES:
            st.error("음성 파일은 10MB 이하만 사용할 수 있습니다.")
            return

        st.success("녹음이 확인되었습니다.")
        st.audio(audio_bytes, format="audio/wav")

        st.write(f'음성 파일 크기: {len(audio_bytes):,}bytes')


        with st.spinner("음성을 인식하고 안내 음성을 만들고 있습니다..."):
            result, marks, meta, error = run_pipeline(
                audio_bytes,
                MODEL_SIZE,
                VOICE_ID,
                stt_mode,
                tts_mode
            )
        if error is not None:
            st.error(f'음성 처리 실패 : {type(error).__name__}: {error}')
            return

        transcript,reply,output,mime = result

        st.write('인식한 질문: ',transcript)
        st.write('시설 안내:',reply)
        st.audio(output, format=mime)

        st.subheader("단계별 처리시간")
        st.bar_chart({
            "STT" : marks['stt_ms'],
            "응답 생성" : marks['reply_ms'],
            "TTS" : marks['tts_ms']
        })

        st.caption("처리시간 단위: 밀리초(ms), 1,000ms = 1초")
        st.write(f"총 처리시간: {meta['total_ms']:,.2f} ms")

    # 여기부터 공백 4칸
    st.divider()
    st.subheader("최근 실행 이력")

    history = load_history()

    if history:
        st.dataframe(history)
    else:
        st.info("아직 저장된 실행 이력이 없습니다.")

if __name__ == '__main__':
    main()
























