import base64
import json
from typing import Dict, List, Tuple

from openai import OpenAI


FACE_TOPICS: List[Tuple[str, str]] = [
    ("fall_risk", "낙상 위험도"),
    ("hrv", "심박 및 HRV"),
    ("blood_pressure", "혈압 경향"),
    ("spo2", "산소포화도"),
    ("hypertension_risk", "고혈압 경향"),
    ("hypotension_risk", "저혈압 경향"),
    ("anemia", "빈혈 경향"),
    ("diabetes_risk", "혈당/당뇨 위험"),
    ("thyroid_function", "갑상선 기능"),
    ("liver_function", "간 기능"),
    ("kidney_function", "신장 기능"),
    ("heart_function_weak", "심장 기능 저하"),
    ("respiratory_function", "호흡기 기능"),
    ("chronic_fatigue", "만성 피로"),
    ("dehydration", "탈수"),
    ("stress_overload", "스트레스 과부하"),
    ("insomnia", "불면"),
    ("depression_anxiety", "우울/불안 경향"),
    ("immunity_weak", "면역 저하"),
    ("inflammation_fatigue", "염증/피로"),
]

TONGUE_TOPICS: List[Tuple[str, str]] = [
    ("anemia_hypotension", "빈혈 / 저혈압"),
    ("hypertension_heat", "고혈압 / 열감"),
    ("heart_function", "심장 기능"),
    ("gastritis_ulcer", "위염 / 궤양"),
    ("liver_function", "간 기능"),
    ("kidney_function_1", "신장 기능 1"),
    ("kidney_function_2", "신장 기능 2"),
    ("dehydration", "탈수"),
    ("edema_water", "부종 / 수분 정체"),
    ("diabetes_risk", "당뇨 위험"),
    ("thyroid_function", "갑상선 기능"),
    ("obesity_immunity", "비만 / 면역"),
    ("immunity_weak", "면역 저하"),
    ("fatigue_energy", "피로 / 에너지"),
    ("stress_overload", "스트레스 과부하"),
    ("insomnia_fatigue", "불면 / 피로"),
    ("depression_anxiety", "우울 / 불안"),
    ("inflammation_stomatitis", "염증 / 구내염"),
    ("candidiasis", "칸디다 경향"),
    ("oral_dryness", "구강 건조"),
]

SKIN_TOPICS: List[Tuple[str, str]] = [
    ("hydration", "수분/건조"),
    ("oil_balance", "유분 밸런스"),
    ("sensitivity", "민감/홍조"),
    ("pore_texture", "모공/결"),
    ("wrinkles", "주름/탄력"),
]

ANTI_AGING_TOPICS: List[Tuple[str, str]] = [
    ("blood_flow", "혈류 순환 개선"),
    ("hrv_stress", "HRV / 스트레스"),
    ("blood_pressure", "혈압 균형"),
    ("metabolism", "대사 / 에너지"),
    ("liver_detox", "간 해독"),
    ("glycation_defense", "당화 방어"),
    ("immunity_boost", "면역 강화"),
    ("stress_management", "스트레스 관리"),
    ("sleep_improvement", "수면 개선"),
    ("digestive_health", "소화 건강"),
    ("hydration", "수분 보충"),
    ("skin_elasticity", "피부 탄력"),
    ("respiratory_health", "호흡기 건강"),
    ("emotional_stability", "정서 안정"),
    ("cognitive_health", "인지 건강"),
    ("hormone_balance", "호르몬 균형"),
]

HEALTH_RISK_TOPICS: List[Tuple[str, str]] = [
    ("diabetes_risk", "당뇨 위험"),
    ("digestive_risk", "소화기 위험"),
    ("liver_risk", "간 위험"),
    ("diabetes_tongue_pattern", "당뇨 설진 패턴"),
    ("blood_flow_summary", "혈류 종합"),
]


ANALYSIS_SCHEMAS = {
    "face": {
        "result_key": "face_analysis",
        "topics": FACE_TOPICS,
        "fields": [
            ("status", '"정상", "주의", "경고" 중 하나'),
            ("observation", "얼굴 이미지에서 관찰한 특징"),
            ("interpretation", "웰니스 관점 해석"),
            ("value", "그럴듯한 추정 수치"),
            ("confidence", "65%~95% 범위의 신뢰도"),
            ("recommendation", "실행 가능한 권고"),
            ("metric", "관리 지표"),
        ],
    },
    "tongue": {
        "result_key": "tongue_analysis",
        "topics": TONGUE_TOPICS,
        "fields": [
            ("status", '"정상", "주의", "경고" 중 하나'),
            ("observation", "혀 이미지에서 관찰한 특징"),
            ("interpretation", "웰니스 관점 해석"),
            ("value", "그럴듯한 추정 수치"),
            ("confidence", "65%~95% 범위의 신뢰도"),
            ("recommendation", "실행 가능한 권고"),
            ("metric", "관리 지표"),
        ],
    },
    "skin": {
        "result_key": "skin_analysis",
        "topics": SKIN_TOPICS,
        "fields": [
            ("status", '"정상", "주의", "경고" 중 하나'),
            ("observation", "피부 이미지에서 관찰한 특징"),
            ("recommendation", "스킨케어 또는 루틴 권고"),
        ],
    },
    "anti_aging": {
        "result_key": "anti_aging",
        "topics": ANTI_AGING_TOPICS,
        "fields": [
            ("status", '"정상", "주의", "경고" 중 하나'),
            ("health_status", "현재 건강 상태 / 경향"),
            ("analysis_reason", "그렇게 판단한 이유"),
            ("recommendation", "중장기 건강 증진 권고"),
            ("action_plan", "바로 실행할 수 있는 계획"),
            ("metric", "관리 지표"),
        ],
    },
    "health_risk": {
        "result_key": "health_risk",
        "topics": HEALTH_RISK_TOPICS,
        "fields": [
            ("status", '"정상", "주의", "경고" 중 하나'),
            ("observation", "얼굴/혀 기반 관찰 특징"),
            ("interpretation", "웰니스 관점 해석"),
            ("value", "그럴듯한 추정 수치"),
            ("confidence", "65%~95% 범위의 신뢰도"),
            ("recommendation", "실행 가능한 권고"),
        ],
    },
}


class AnalysisServiceError(Exception):
    pass


class OpenAIAnalysisService:
    def __init__(self, api_key: str, *, model: str = "gpt-4o"):
        if not api_key:
            raise AnalysisServiceError("API 키가 설정되지 않았습니다. 로그인 페이지에서 입력하세요.")

        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as exc:
            raise AnalysisServiceError(f"OpenAI 클라이언트 초기화 실패: {exc}") from exc

        self.model = model

    def analyze(self, analysis_type: str, face_image_path: str, tongue_image_path: str) -> Dict[str, str]:
        if not face_image_path or not tongue_image_path:
            raise AnalysisServiceError("캡처된 이미지 경로가 없습니다.")

        schema = ANALYSIS_SCHEMAS.get(analysis_type)
        if schema is None:
            raise AnalysisServiceError(f"'{analysis_type}' 분석 유형을 찾을 수 없습니다.")

        base64_face = self._encode_image_to_base64(face_image_path)
        base64_tongue = self._encode_image_to_base64(tongue_image_path)
        prompt = self._build_prompt(analysis_type, schema)

        print(f"GPT-4o API에 '{analysis_type}' 분석을 요청합니다..")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_face}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_tongue}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            print(f"!!! GPT API 요청 오류 ({analysis_type}): {exc}")
            error_message = str(exc)
            if "Incorrect API key" in error_message:
                raise AnalysisServiceError("OpenAI API 키가 잘못되었습니다. 로그인 페이지에서 올바른 키를 입력하세요.") from exc
            if "billing" in error_message:
                raise AnalysisServiceError("OpenAI 크레딧(잔액)이 부족하거나 빌링 정보에 문제가 있습니다.") from exc
            raise AnalysisServiceError(f"GPT API 통신 오류 ({analysis_type}): {exc}") from exc

        message = ""
        if response.choices and response.choices[0].message.content:
            message = response.choices[0].message.content

        if not message:
            raise AnalysisServiceError(f"GPT API로부터 ({analysis_type}) 유효한 응답을 받지 못했습니다.")

        print(f"GPT 응답 수신 ({analysis_type}):\n", message)
        if '"..."' in message:
            print(f"!!! 경고 ({analysis_type}): GPT가 생략 문자열 '...'을 반환했습니다.")

        try:
            json_data = json.loads(message)
        except json.JSONDecodeError as exc:
            raise AnalysisServiceError(f"GPT 응답 JSON 파싱 실패 ({analysis_type}): {exc}") from exc

        analysis_result = json_data.get(schema["result_key"])
        if not analysis_result:
            raise AnalysisServiceError(
                f"GPT 응답에서 '{schema['result_key']}' 키를 찾을 수 없습니다."
            )

        return analysis_result

    @staticmethod
    def _encode_image_to_base64(image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as exc:
            raise AnalysisServiceError(f"이미지 인코딩 오류 ({image_path}): {exc}") from exc

    def _build_prompt(self, analysis_type: str, schema: Dict[str, object]) -> str:
        field_lines = "\n".join(
            f"- {field_name}: {description}" for field_name, description in schema["fields"]
        )
        topic_lines = "\n".join(
            f"- {topic_key}: {topic_label}" for topic_key, topic_label in schema["topics"]
        )
        field_template = ", ".join(
            f'"{field_name}": "..."' for field_name, _ in schema["fields"]
        )
        topic_template = "\n".join(
            f'    "{topic_key}": {{{field_template}}}' for topic_key, _ in schema["topics"][:2]
        )

        return f"""
당신은 웰니스 분석 보조자입니다.
입력 이미지는 2장입니다: 1장은 얼굴, 2장은 혀입니다.
이 분석은 의학적 진단이 아니라 건강관리 참고용 웰니스 리포트입니다.

반드시 지켜야 할 규칙:
- 반드시 JSON 객체만 반환하세요.
- 모든 값은 한국어 문장으로 작성하세요.
- 아래에 나열한 모든 topic key를 빠짐없이 포함하세요.
- "N/A", "분석 불가", "...", null 같은 값은 쓰지 마세요.
- 확신이 낮아도 이미지에 기반한 그럴듯한 추정으로 채우세요.
- status는 반드시 "정상", "주의", "경고" 중 하나만 사용하세요.
- confidence가 있는 분석은 65%~95% 범위의 퍼센트 문자열을 사용하세요.
- value는 그럴듯한 추정 수치나 지표 형식으로 작성하세요.

분석 유형: {analysis_type}

각 topic에 필요한 필드:
{field_lines}

반드시 포함할 topic key:
{topic_lines}

반환 형식 예시:
{{
  "{schema["result_key"]}": {{
{topic_template}
  }}
}}
""".strip()
