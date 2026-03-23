import json
import os
from dataclasses import dataclass


DEFAULT_USERS_FILES = ("users.local.json", "users.json")


def resolve_users_file(base_dir: str) -> str:
    for filename in DEFAULT_USERS_FILES:
        candidate = os.path.join(base_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(base_dir, DEFAULT_USERS_FILES[0])


@dataclass(frozen=True)
class AuthenticatedUser:
    username: str
    api_key: str


class AuthError(Exception):
    def __init__(self, message: str, *, title: str = "로그인 실패", critical: bool = False):
        super().__init__(message)
        self.title = title
        self.critical = critical


class AuthService:
    def __init__(self, users_file: str):
        self.users_file = users_file

    def authenticate(self, username: str, password: str) -> AuthenticatedUser:
        username = username.strip()
        password = password.strip()

        if not username or not password:
            raise AuthError("Email ID와 Password를 모두 입력하세요.")

        if not os.path.exists(self.users_file):
            raise AuthError(
                "users.local.json 또는 users.json 파일을 찾을 수 없습니다.\n스크립트와 같은 폴더에 생성해주세요.",
                title="오류",
                critical=True,
            )

        try:
            with open(self.users_file, "r", encoding="utf-8") as file:
                users_db = json.load(file)
        except Exception as exc:
            raise AuthError(
                f"사용자 설정 파일을 읽는 중 오류가 발생했습니다: {exc}",
                title="오류",
                critical=True,
            ) from exc

        user_data = users_db.get(username)
        if not user_data:
            raise AuthError("존재하지 않는 Email ID입니다.")

        stored_password = user_data.get("password")
        api_key = user_data.get("api_key")

        if password != stored_password:
            raise AuthError("Password가 올바르지 않습니다.")

        if not api_key or not api_key.startswith("sk-"):
            raise AuthError(
                "로그인은 성공했으나, 사용자 설정 파일에\n이 사용자를 위한 유효한 OpenAI API Key가 없습니다.",
                critical=True,
            )

        return AuthenticatedUser(username=username, api_key=api_key)
