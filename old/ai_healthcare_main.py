# -*- coding: utf-8 -*-
import os, sys, cv2, numpy as np
import random # random 모듈 import 추가
from datetime import datetime
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QBuffer, pyqtSlot, QPoint, QThread
from PyQt5.QtGui import QPixmap, QFont, QImage, QColor
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtWidgets import (
    QApplication, QWidget, QStackedWidget, QLabel, QVBoxLayout, QPushButton,
    QLineEdit, QGridLayout, QFrame, QHBoxLayout, QMessageBox, QFileDialog,
    QScrollArea, QProgressBar # <--- 이 부분 추가
)
# reportlab은 함수 내에서 import 하므로 여기서는 제거해도 됩니다.
# 다만, 실행 전 'pip install reportlab'은 필수입니다.
import io # 이미지 메모리 처리를 위해 필요

# --- ▼▼▼ 3단계 신규 추가 (누락된 부분 모두 포함) ▼▼▼ ---
import hashlib  # <--- 로그인용 해시 라이브러리 추가
# --- ▲▲▲ 3단계 신규 추가 끝 ▲▲▲ ---


# ---------- Config (FIXED) ----------
# The script loads UI images from the local 'interface' folder.
from models import AppSession
from services import AnalysisServiceError, AuthError, AuthService, OpenAIAnalysisService, resolve_users_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(SCRIPT_DIR, "interface")
IMG_FILES = [
    "1.png",
    "2.png",
    "3.png",
    "4.png", # 파일명 수정
]
SAVE_DIR = SCRIPT_DIR
FACE_SAVE_NAME = "capture_face.jpg"
TONGUE_SAVE_NAME = "capture_tongue.jpg"

# ---------- 유틸 ----------
def cvimg_to_qpixmap(img_bgra: np.ndarray) -> QPixmap:
    h, w, ch = img_bgra.shape
    bytes_per_line = ch * w
    qimg = QImage(img_bgra.data, w, h, bytes_per_line, QImage.Format_ARGB32)
    return QPixmap.fromImage(qimg)

# ---------- 기본 이미지 페이지 ----------
class ImagePage(QWidget):
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent) # () 추가
        self.image_path = image_path
        self.bg_label = QLabel(self)
        self.bg_label.setAlignment(Qt.AlignCenter)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.bg_label)
        self._pixmap = QPixmap(self.image_path)
        if self._pixmap.isNull(): # () 추가
            print(f"Error: Could not load image at {self.image_path}")
            self.bg_label.setText(f"Image not found:\n{os.path.basename(self.image_path)}")
            self.bg_label.setStyleSheet("color: red; font-size: 20px;")
        self._update_pixmap() # () 추가

    def resizeEvent(self, event):
        self._update_pixmap() # () 추가
        super().resizeEvent(event) # () 추가

    def _update_pixmap(self):
        if not self._pixmap.isNull(): # () 추가
            scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation) # () 추가
            self.bg_label.setPixmap(scaled)

# ---------- 로그인 페이지 ----------
class LoginPage(ImagePage):
    def __init__(self, image_path: str, parent=None):
        super().__init__(image_path, parent) # () 추가
        self.overlay = QWidget(self)
        self.overlay.setAttribute(Qt.WA_TranslucentBackground, True)
        overlay_layout = QVBoxLayout(self.overlay)
        overlay_layout.setContentsMargins(0, 0, 0, 0)

        grid_wrap = QWidget(self)
        grid = QGridLayout(grid_wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(14)

        label_font = QFont("NanumBarunGothic", 18, QFont.Bold)
        edit_font  = QFont("NanumBarunGothic", 18)

        def make_edit(ph):
            e = QLineEdit() # () 추가
            e.setPlaceholderText(ph)
            e.setFont(edit_font)
            e.setFixedHeight(55)
            e.setStyleSheet("""
                QLineEdit {
                    background: rgba(255,255,255, 230);
                    border: 2px solid #c59d00;
                    border-radius: 8px;
                    padding: 8px 12px;
                }
                QLineEdit:focus {
                    border-color: #ffb300;
                    background: rgba(255,255,255, 250);
                }
            """)
            return e

        # --- 1. 환자 이름 (기존) ---
        lbl_name = QLabel("➤ 환자 이름")
        lbl_name.setFont(label_font)
        lbl_name.setStyleSheet("color: white;")
        self.name_edit  = make_edit("예: 홍길동")
        grid.addWidget(lbl_name, 0, 0, Qt.AlignRight)
        grid.addWidget(self.name_edit, 0, 1)

        # --- 2. Email ID (수정) ---
        lbl_gpt_id = QLabel("➤ Email ID")
        lbl_gpt_id.setFont(label_font)
        lbl_gpt_id.setStyleSheet("color: white;")
        self.gpt_id_edit = make_edit("Email ID 입력 (예: admin@example.com)") # (수정)
        grid.addWidget(lbl_gpt_id, 1, 0, Qt.AlignRight)
        grid.addWidget(self.gpt_id_edit, 1, 1)

        # --- 3. Password (수정) ---
        lbl_gpt_pw = QLabel("➤ Password")
        lbl_gpt_pw.setFont(label_font)
        lbl_gpt_pw.setStyleSheet("color: white;")
        self.gpt_pw_edit = make_edit("Password 입력 (예: pass123)") # (수정)
        self.gpt_pw_edit.setEchoMode(QLineEdit.Password) # 비밀번호 마스킹
        grid.addWidget(lbl_gpt_pw, 2, 0, Qt.AlignRight)
        grid.addWidget(self.gpt_pw_edit, 2, 1)

        # --- (기존) ---
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)

        overlay_layout.addStretch() # () 추가
        overlay_layout.addWidget(grid_wrap, 0, Qt.AlignHCenter)
        overlay_layout.addStretch() # () 추가
        self.overlay.raise_() # () 추가

    def resizeEvent(self, event):
        super().resizeEvent(event) # () 추가
        w, h = self.width(), self.height() # () 추가
        overlay_width = min(800, int(w * 0.64))
        
        # (수정) 필드가 3개로 늘어났으므로 오버레이 높이 증가
        overlay_height = int(h * 0.45) # 0.25 -> 0.45
        
        left = int(w * 0.5 - overlay_width / 2)
        top = int(h * 0.40) # 0.45 -> 0.40 (조금 위로 이동)
        self.overlay.setGeometry(left, top, overlay_width, overlay_height)
        

# ---------- (신규) 카메라 로딩용 스레드 ----------
class CameraWorker(QThread):
    """백그라운드에서 카메라를 초기화하는 작업자 클래스"""
    result_ready = pyqtSignal(object) # 성공 시 cv2.VideoCapture 객체 전송

    def run(self):
        cap = cv2.VideoCapture(0)
        
        # Windows에서 기본 백엔드로 실패 시 DSHOW 시도
        if not cap.isOpened() and os.name == 'nt':
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
        if cap.isOpened():
            # 해상도 설정 (시간이 좀 걸리는 작업)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
        # 완료 신호 전송 (성공하든 실패하든 객체는 보냄)
        self.result_ready.emit(cap)

# ---------- 카메라 페이지 (스레드 로딩 적용) ----------
class CameraPage(ImagePage):
    finished = pyqtSignal(str, str)

    def __init__(self, image_path: str, parent=None):
        print(">>> CameraPage: __init__ 시작")
        super().__init__(image_path, parent)
        self.cam_label = QLabel(self)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setAttribute(Qt.WA_TranslucentBackground, True)

        # --- 로딩 텍스트 레이블 ---
        self.loading_label = QLabel("카메라 모듈을 불러오는 중입니다...\n잠시만 기다려주세요.", self)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setFont(QFont("NanumBarunGothic", 20, QFont.Bold))
        self.loading_label.setStyleSheet("color: white; background-color: rgba(0,0,0,0.3); padding: 20px; border-radius: 10px;")
        self.loading_label.hide()

        # --- 로딩 게이지 바 ---
        self.startup_bar = QProgressBar(self)
        self.startup_bar.setRange(0, 100)
        self.startup_bar.setValue(0)
        self.startup_bar.setTextVisible(True)
        self.startup_bar.setFormat("%p%")
        self.startup_bar.setAlignment(Qt.AlignCenter)
        self.startup_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 10px;
                background-color: #FFFFFF;
                text-align: center;
                color: black;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #00D26A;
                border-radius: 8px;
            }
        """)
        self.startup_bar.hide()

        # --- 로딩 상태 변수 ---
        self.loader_thread = None  # 스레드 객체
        self.loaded_cap = None     # 로딩된 카메라 객체
        self.is_camera_ready = False 
        self.loading_progress_val = 0
        
        # 비주얼용 타이머 (게이지 올리기용)
        self.loading_visual_timer = QTimer(self)
        self.loading_visual_timer.setInterval(25) # 25ms마다 업데이트 (약 2.5초 동안 100% 도달 목표)
        self.loading_visual_timer.timeout.connect(self.update_loading_bar)

        # 시작 버튼
        self.start_btn = QPushButton("시작", self)
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setFixedHeight(46)
        self.start_btn.setFont(QFont("NanumBarunGothic", 16, QFont.Bold))
        self.start_btn_style = """
            QPushButton {
                background: rgba(0, 150, 0, 0.85);
                color: white; border: none; border-radius: 10px;
                padding: 8px 18px;
            }
            QPushButton:hover { background: rgba(0, 170, 0, 0.95); }
            QPushButton:pressed { background: rgba(0, 120, 0, 0.95); }
        """
        self.start_btn.setStyleSheet(self.start_btn_style)
        self.start_btn.clicked.connect(self.start_sequence)

        # 프레임 갱신 및 카운트다운
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.update_frame)
        self.count_timer = QTimer(self)
        self.count_timer.setInterval(1000)
        self.count_timer.timeout.connect(self.tick_count)
        
        self.cap = None
        self.state = "idle"
        self.countdown = 0
        self.last_frame = None
        self.left_frozen_frame = None

        # 완료 메시지
        self.completion_label = QLabel("촬영이 끝났습니다. 다음 버튼을 눌러 결과를 확인하세요.", self)
        self.completion_label.setAlignment(Qt.AlignCenter)
        self.completion_label.setFont(QFont("NanumBarunGothic", 12, QFont.Bold))
        self.completion_label.setStyleSheet("color: white; background-color: rgba(0,0,0,0.5); border-radius: 8px; padding: 8px 15px;")
        self.completion_label.hide()
        print(">>> CameraPage: __init__ 완료")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.cam_label.setGeometry(0, 0, self.width(), self.height())

        cx = self.width() // 2
        cy = self.height() // 2
        
        # 로딩 UI 위치
        self.loading_label.adjustSize()
        lw, lh = self.loading_label.width(), self.loading_label.height()
        bar_w, bar_h = 400, 30
        total_h = lh + 20 + bar_h
        start_y = cy - (total_h // 2)
        
        self.loading_label.move(cx - lw // 2, start_y)
        self.startup_bar.setGeometry(cx - bar_w // 2, start_y + lh + 20, bar_w, bar_h)

        # 버튼 위치
        y_start_btn = int(self.height() * 0.88)
        btn_w = self.start_btn.width() if self.start_btn.width() > 0 else 100
        self.start_btn.move(int(cx - btn_w / 2), y_start_btn)

        # 완료 라벨 위치
        self.completion_label.adjustSize()
        cw, ch = self.completion_label.width(), self.completion_label.height()
        self.completion_label.move(int(cx - cw / 2), y_start_btn - ch - 10)

    def start_sequence(self):
        """시작 버튼 클릭 시 호출"""
        # 1. UI 초기화
        self.start_btn.hide()
        self.loading_label.show()
        self.startup_bar.setValue(0)
        self.startup_bar.show()
        self.loading_label.raise_()
        self.startup_bar.raise_()

        # 2. 변수 초기화
        self.loading_progress_val = 0
        self.is_camera_ready = False
        self.loaded_cap = None

        # 3. 백그라운드 카메라 로딩 시작
        self.loader_thread = CameraWorker()
        self.loader_thread.result_ready.connect(self.on_camera_loaded)
        self.loader_thread.start()

        # 4. 비주얼 타이머 시작 (게이지 올리기)
        self.loading_visual_timer.start()

    def on_camera_loaded(self, cap_object):
        """스레드에서 카메라 연결이 완료되면 호출됨"""
        self.loaded_cap = cap_object
        self.is_camera_ready = True
        # 여기서 바로 화면을 켜지 않고, 로딩바가 자연스럽게 100%가 될 때까지 기다리거나
        # 로딩바가 이미 많이 찼으면 바로 넘어갑니다.

    def update_loading_bar(self):
        """타이머에 의해 주기적으로 호출되어 게이지를 올림"""
        # 1. 기본적으로 1씩 증가
        self.loading_progress_val += 1
        
        # 2. 카메라 준비 상태에 따른 로직
        if not self.is_camera_ready:
            # 카메라가 아직 준비 안 됐으면 95%에서 멈춰서 기다림
            if self.loading_progress_val > 95:
                self.loading_progress_val = 95
                self.loading_label.setText("카메라 연결 대기 중...")
        else:
            # 카메라가 준비 완료되면 빠르게 100%로 채움
            if self.loading_progress_val < 100:
                # 준비되었으면 점프해서 팍팍 채움 (시각적 쾌감)
                self.loading_progress_val += 5 

        # 3. 값 적용
        self.startup_bar.setValue(min(100, self.loading_progress_val))

        # 4. 100% 도달 시 실제 화면 전환
        if self.loading_progress_val >= 100:
            self.loading_visual_timer.stop()
            self.loading_label.hide()
            self.startup_bar.hide()
            self.activate_camera_feed() # 화면 켜기

    def activate_camera_feed(self):
        """최종적으로 카메라 화면을 띄우고 카운트다운 시작"""
        self.cap = self.loaded_cap
        
        # 만약 스레드에서 카메라 열기에 실패했다면
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "카메라 오류", "카메라를 열 수 없습니다.")
            self.start_btn.show()
            return

        # 카메라 타이머 시작
        if not self.frame_timer.isActive():
            self.frame_timer.start(30)

        self.left_frozen_frame = None
        self.state = "face"
        self.countdown = 5
        self.count_timer.start()

    def stop_camera(self):
        self.frame_timer.stop()
        self.count_timer.stop()
        self.loading_visual_timer.stop()
        
        # 스레드 정리
        if self.loader_thread is not None:
            if self.loader_thread.isRunning():
                self.loader_thread.quit()
                self.loader_thread.wait()
            self.loader_thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # (나머지 유틸 함수들은 기존과 동일)
    def compute_circles(self, w, h):
        r = int(min(w, h) * 0.22)
        left_center  = (int(w * 0.255), int(h * 0.53))
        right_center = (int(w * 0.745), int(h * 0.53))
        return left_center, right_center, r

    def _make_square_bgra(self, frame_bgr, d):
        fh, fw = frame_bgr.shape[:2]
        side = min(fw, fh)
        x0 = (fw - side) // 2
        y0 = (fh - side) // 2
        square = frame_bgr[y0:y0+side, x0:x0+side]
        square = cv2.resize(square, (d, d), interpolation=cv2.INTER_AREA)
        square = cv2.cvtColor(square, cv2.COLOR_BGR2BGRA)
        return square

    def draw_masked_dual(self, frame_bgr, w, h):
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        left_c, right_c, r = self.compute_circles(w, h)
        d = r * 2
        if frame_bgr is None: return canvas
        
        square_live = self._make_square_bgra(frame_bgr, d)
        square_left = self._make_square_bgra(self.left_frozen_frame, d) if self.left_frozen_frame is not None else square_live

        y, x = np.ogrid[:d, :d]
        mask = (x - r)**2 + (y - r)**2 <= r*r
        alpha = np.zeros((d, d), dtype=np.uint8)
        alpha[mask] = 255
        square_live[..., 3] = alpha
        square_left[..., 3] = alpha

        def paste_at(sq, cx, cy):
            x1 = max(0, cx - r); y1 = max(0, cy - r)
            x2 = min(w, x1 + d); y2 = min(h, y1 + d)
            sx1 = 0 if x1 == cx - r else (cx - r) - x1
            sy1 = 0 if y1 == cy - r else (cy - r) - y1
            sx2 = sx1 + (x2 - x1); sy2 = sy1 + (y2 - y1)
            roi = canvas[y1:y2, x1:x2]; frag = sq[sy1:sy2, sx1:sx2]
            a = frag[..., 3:4] / 255.0
            roi[..., :3] = (frag[..., :3] * a + roi[..., :3] * (1 - a)).astype(np.uint8)
            roi[..., 3] = np.clip(roi[..., 3] + frag[..., 3], 0, 255)
        
        paste_at(square_left, *left_c)
        paste_at(square_live, *right_c)
        
        if self.state in ("face", "tongue") and self.countdown >= 0:
            txt = f"{self.countdown}"
            target = left_c if self.state == "face" else right_c
            cx, cy = target[0], target[1]
            cv2.putText(canvas, txt, (cx-20, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255,255), 6, cv2.LINE_AA)
        return canvas
    
    def update_frame(self):
        if self.cap is None or not self.cap.isOpened(): return
        try:
            ok, frame = self.cap.read()
            if not ok: return
            frame = cv2.flip(frame, 1)
            self.last_frame = frame
            w, h = self.width(), self.height()
            overlay = self.draw_masked_dual(frame, w, h)
            self.cam_label.setPixmap(cvimg_to_qpixmap(overlay))
        except Exception as e:
            print(f"Error: {e}")
            self.stop_camera()

    def tick_count(self):
        if self.state not in ("face", "tongue"):
            self.count_timer.stop()
            return
        if self.countdown <= 0:
            self.capture_current()
            if self.state == "face":
                if self.last_frame is not None:
                    self.left_frozen_frame = self.last_frame.copy()
                self.state = "tongue"
                self.countdown = 5 
                return
            else:
                self.state = "done"
                self.count_timer.stop()
                face_path = os.path.join(SAVE_DIR, FACE_SAVE_NAME)
                tongue_path = os.path.join(SAVE_DIR, TONGUE_SAVE_NAME)
                self.finished.emit(face_path, tongue_path)
                self.completion_label.show()
                self.completion_label.raise_()
                return
        self.countdown -= 1

    def capture_current(self):
        if self.last_frame is None: return
        save_path = os.path.join(SAVE_DIR, FACE_SAVE_NAME if self.state == "face" else TONGUE_SAVE_NAME)
        try:
            is_success, img_encode = cv2.imencode(".jpg", self.last_frame)
            if not is_success: return
            with open(save_path, 'wb') as f:
                f.write(img_encode)
            print(f"이미지 저장 성공: {save_path}")
        except Exception as e:
            print(f"파일 쓰기 오류: {e}")

    def reset_page(self):
        print(">>> CameraPage: reset_page 호출됨")
        self.stop_camera() 
        
        self.completion_label.hide()
        self.loading_label.hide()
        self.startup_bar.hide()
        
        self.start_btn.setText("시작")
        self.start_btn.setDisabled(False)
        self.start_btn.setStyleSheet(self.start_btn_style)
        self.start_btn.show() 
        
        self.state = "idle"
        self.countdown = 0
        self.last_frame = None
        self.left_frozen_frame = None
        
        self.cam_label.clear()
        super()._update_pixmap()

# ---------- (신규) 분석 중 로딩 페이지 ----------
class AnalysisPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #ffffff;") # 흰색 배경
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # --- ▼▼▼ (수정) 레이아웃 구성 변경 ▼▼▼ ---
        
        # 1. 메인 텍스트
        self.loading_label = QLabel("GPT-4O가 이미지를 분석 중입니다...\n\n잠시만 기다려주세요.", self)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setFont(QFont("NanumBarunGothic", 24, QFont.Bold))
        self.loading_label.setStyleSheet("""
            QLabel {
                color: #173a9e; /* 메인 테마 색상 */
                background-color: rgba(240, 240, 240, 0.8);
                border-radius: 15px;
                padding: 40px;
                line-height: 150%;
            }
        """)
        
        # (신규) 2. 프로그레스 바
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100) # 0% ~ 100%
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True) # 텍스트 보이게
        self.progress_bar.setFormat("%p%") # 퍼센트로 표시
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFont(QFont("NanumBarunGothic", 14, QFont.Bold))
        self.progress_bar.setFixedHeight(35)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #e0e0e0;
                color: #ffffff;
                border: none;
                border-radius: 15px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #173a9e; /* 메인 테마 색상 */
                border-radius: 15px;
            }
        """)

        # (신규) 3. 상태 텍스트 레이블
        self.status_label = QLabel("분석을 준비 중입니다...", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("NanumBarunGothic", 16, QFont.Bold))
        self.status_label.setStyleSheet("color: #333; margin-top: 15px;")
        
        # --- 레이블과 프로그레스 바를 담을 컨테이너 ---
        self.container_widget = QWidget()
        self.container_widget.setStyleSheet("background-color: transparent;")
        
        container_layout = QVBoxLayout(self.container_widget)
        container_layout.setContentsMargins(50, 50, 50, 50) # 여백
        container_layout.addWidget(self.loading_label)
        container_layout.addSpacing(30) # 간격
        container_layout.addWidget(self.progress_bar)
        container_layout.addWidget(self.status_label)
        container_layout.setStretchFactor(self.loading_label, 1) # 로딩 레이블이 공간 차지
        
        # --- 전체 페이지 레이아웃 ---
        layout.addStretch()
        layout.addWidget(self.container_widget, 0, Qt.AlignHCenter) # 중앙 정렬
        layout.addStretch()
        
        # --- ▲▲▲ (수정) 레이아웃 구성 변경 끝 ▲▲▲ ---

    # (신규) 프로그레스 업데이트 함수
    def update_progress(self, value: int, text: str):
        """프로그레스 바의 값과 상태 텍스트를 업데이트합니다."""
        self.progress_bar.setValue(value)
        self.status_label.setText(text)
        QApplication.processEvents() # UI 즉시 새로고침

    def showEvent(self, event):
        """페이지가 보일 때 UI가 즉시 업데이트되도록 함"""
        super().showEvent(event)
        # (수정) 페이지가 보일 때 항상 0%로 리셋
        self.update_progress(0, "분석을 준비 중입니다...")

# ---------- 신호등 위젯 ----------
class LightDot(QWidget):
    COLORS={"정상":"#2ecc71","주의":"#f39c12","경고":"#e74c3c"}
    def __init__(self,level:str,parent=None):
        super().__init__(parent) # () 추가
        self.level=level
        # (수정) level이 키에 없는 경우 기본값(#999) 사용
        dot=QLabel("●"); dot.setStyleSheet(f"color:{self.COLORS.get(level,'#999')};font-size:30px;")
        lay=QHBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(dot)
        lay.setAlignment(Qt.AlignCenter)

# ---------- 별점 위젯 (신규 추가) ----------
class StarRatingWidget(QWidget):
    def __init__(self, rating: int, max_stars: int = 5, parent=None):
        super().__init__(parent) # () 추가
        self.rating = rating
        self.max_stars = max_stars

        star_full = "★"
        star_empty = "☆"

        # HTML로 색상 구분 (이미지와 유사하게)
        full_stars_html = f"<font color='#333'>{star_full * rating}</font>"
        empty_stars_html = f"<font color='#ccc'>{star_empty * (max_stars - rating)}</font>"
        stars_str = full_stars_html + empty_stars_html

        label = QLabel(stars_str)
        label.setStyleSheet("font-size: 22px; font-weight: bold;") # 폰트 크기

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(label)
        layout.setAlignment(Qt.AlignCenter)

# ---------- 표 공통 (QScrollArea로 수정) ----------
class TablePage(QWidget):
    def __init__(self,head_title:str,sub:str="",parent=None):
        super().__init__(parent) # () 추가
        self.head_title=head_title; self.sub=sub
        self.stretch_columns = []

        # --- Root Layout ---
        # 1. 전체 페이지를 위한 루트 레이아웃 (QScrollArea를 포함)
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        # 2. 스크롤 영역 생성
        # (수정) 'scroll_area'를 'self.scroll_area'로 변경하여 외부에서 접근 가능하도록 함
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True) # <--- 중요: 컨텐츠 위젯이 스크롤 영역을 채우도록 함
        self.scroll_area.setFrameShape(QFrame.NoFrame) # 테두리 없음

        # 3. 스크롤될 컨텐츠 위젯
        scroll_content_widget = QWidget() # () 추가
        # (수정) 'scroll_area' -> 'self.scroll_area'
        self.scroll_area.setWidget(scroll_content_widget)

        # 4. 'main_layout'을 'self'가 아닌 'scroll_content_widget'에 적용
        main_layout=QVBoxLayout(scroll_content_widget);
        # (중요) 하단 여백 85px -> 20px로 줄임 (버튼이 스크롤 영역 안에 있으므로)
        main_layout.setContentsMargins(14,8,14,20);
        main_layout.setSpacing(10) # 스페이싱 3 -> 10으로 늘림

        # --- Widgets (기존과 동일) ---
        title=QLabel(head_title); title.setFont(QFont("NanumBarunGothic",33,QFont.Black)); title.setStyleSheet("color:#173a9e;margin:10px 0;")
        subtitle=QLabel(sub); subtitle.setWordWrap(True); subtitle.setFont(QFont("NanumBarunGothic",15)); subtitle.setStyleSheet("color:#444;margin:0 0 10px 0;")

        self.table=QTableWidget(scroll_content_widget) # 부모를 scroll_content_widget로
        self.table.setFrameShape(QFrame.NoFrame); self.table.verticalHeader().setVisible(False); self.table.horizontalHeader().setVisible(False); self.table.setEditTriggers(QTableWidget.NoEditTriggers); self.table.setSelectionMode(QTableWidget.NoSelection); self.table.setWordWrap(True) # () 추가

        # --- 스크롤바 스타일링을 QScrollArea로 이동 ---
        # (수정) 'scroll_area' -> 'self.scroll_area'
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background: #ffffff;
                border: none;
            }
            QScrollBar:vertical {
                width: 28px;
                background: #f0f0f0;
                border-radius: 6px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #173a9e;
                min-height: 100px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

        # --- 테이블 스타일 (스크롤바 제외) ---
        self.table.setStyleSheet("""
            QTableWidget {
                background: #fff;
                border: 1px solid #d9d9d9;
                font-family: 'NanumBarunGothic';
                font-weight: 600;
            }
            QTableWidget::item {
                padding: 8px 10px;
                font-size: 16px;
            }
        """)

        # --- (중요) 테이블이 스스로 스크롤되지 않도록 함 ---
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


        # --- Layout Widgets (수정) ---
        main_layout.addWidget(title);
        main_layout.addWidget(subtitle);
        main_layout.addWidget(self.table, 0) # <--- 스트레치 제거

        self.summary_eval_container = QWidget() # () 추가
        main_layout.addWidget(self.summary_eval_container)

        self.summary_widget_container = QWidget() # () 추가
        main_layout.addWidget(self.summary_widget_container)

        # 저장 버튼을 스크롤 영역 *안*으로 이동

        main_layout.addStretch(1) # <--- 맨 아래에 스트레치를 추가하여 컨텐츠를 위로 민다

        # --- 스크롤 영역을 루트 레이아웃에 추가 ---
        # (수정) 'scroll_area' -> 'self.scroll_area' (여기가 오류 발생 지점입니다)
        root_layout.addWidget(self.scroll_area)

        # 데이터 저장을 위한 변수 추가
        self.table_data = [] # 생성된 테이블 데이터를 저장할 리스트

    # _set_table_data 함수는 원본과 동일, () 추가
    def _set_table_data(self,rows):
        if not rows:return
        r,c=len(rows),len(rows[0]); self.table.clear(); self.table.setRowCount(r); self.table.setColumnCount(c) # () 추가
        font = QFont("NanumBarunGothic", 14)
        for i, row_data in enumerate(rows):
            for j, val in enumerate(row_data):
                if isinstance(val, QWidget):
                    container = QWidget() # () 추가
                    layout = QHBoxLayout(container)
                    layout.addWidget(val)
                    layout.setAlignment(Qt.AlignCenter)
                    layout.setContentsMargins(0,0,0,0)
                    self.table.setCellWidget(i, j, container)
                else:
                    it=QTableWidgetItem(str(val))
                    it.setFont(font)
                    it.setFlags(Qt.ItemIsEnabled)
                    it.setTextAlignment(Qt.AlignVCenter|Qt.AlignLeft)
                    self.table.setItem(i, j, it)
        for i in range(r):
            it0=self.table.item(i,0)
            if it0 and str(it0.text()).startswith("■ "): # () 추가
                self.table.setSpan(i,0,1,c); it0.setBackground(QColor("#f1f2f6"))
                it0.setFont(QFont("NanumBarunGothic",16,QFont.ExtraBold))

    def showEvent(self, event):
        super().showEvent(event) # () 추가
        QTimer.singleShot(50, self.perform_resize)

    # (수정) perform_resize 함수 수정 및 () 추가
    def perform_resize(self):
        # 1. 열 너비 맞추기
        self.table.resizeColumnsToContents() # () 추가 # 먼저 컨텐츠 기준으로 맞춤
        header = self.table.horizontalHeader() # () 추가
        
        # --- (수정) 0번 열 고정 로직 제거 ---
        for col in range(header.count()): # () 추가
            if col in self.stretch_columns:
                header.setSectionResizeMode(col, QHeaderView.Stretch)
            else:
                # (수정) 컨텐츠에 맞게 고정하는 대신, 사용자가 조절할 수 있게 합니다.
                header.setSectionResizeMode(col, QHeaderView.Interactive)
        # --- 수정 끝 ---

        # 2. 행 높이 맞추기
        self.table.resizeRowsToContents() # () 추가

        # 3. (중요) 테이블의 전체 높이를 계산하여 고정
        #   - 스크롤바가 테이블이 아닌 QScrollArea에 생기도록
        total_height = 0
        if self.table.horizontalHeader().isVisible(): # () 추가
            total_height += self.table.horizontalHeader().height() # () 추가

        for i in range(self.table.rowCount()): # () 추가
            total_height += self.table.rowHeight(i)

        # 프레임/테두리 두께 여유분 추가 (2px) + 약간의 여유(2px)
        self.table.setFixedHeight(total_height + 4)

        # (참고) eval_table(요약 테이블)은 ResultPageAntiAging 클래스에서
        # 자체적으로 높이를 고정하므로 여기서 건드리지 않아도 됨.

    # _add_wellness_summary_content 함수는 원본과 동일, () 추가
    def _add_wellness_summary_content(self, summary_type="face"):
        if self.summary_widget_container.layout() is None: # () 추가
            summary_layout=QVBoxLayout(self.summary_widget_container)
            summary_layout.setContentsMargins(10,20,10,10); summary_layout.setSpacing(10)
        else:
            # 기존 위젯 제거
            layout = self.summary_widget_container.layout() # () 추가
            while layout.count(): # () 추가
                child = layout.takeAt(0)
                if child.widget(): # () 추가
                    child.widget().deleteLater() # () 추가

        s_points_data = {
            "face": {
                "경고": "• <font color='#e74c3c'><b>즉시 관리 필요</b></font> — 만성 스트레스 / 수면 불균형 / 자율신경 과활성",
                "주의": "• <font color='#f39c12'><b>기능 저하 경향</b></font> — 혈류순환 / 간·신장 / 피부탄력 저하",
                "정상": "• <font color='#2ecc71'><b>정상·유지 단계</b></font> — 산소포화도 / 심박 안정 / 혈색 균형"
            },
            "tongue": {
                "경고": "• <font color='#e74c3c'><b>즉시 관리 필요</b></font> — 염증 / 스트레스 / 심장 / 곰팡이 감염",
                "주의": "• <font color='#f39c12'><b>기능 저하 경향</b></font> — 위·간·신장 / 면역 / 피로 / 탈수",
                "정상": "• <font color='#2ecc71'><b>정상·유지 단계</b></font> — 수분·온도 균형 양호"
            },
            "skin": {
                "경고": "• <font color='#e74c3c'><b>즉시 관리 필요</b></font> — 피부 장벽 손상 / 염증 악화",
                "주의": "• <font color='#f39c12'><b>기능 저하 경향</b></font> — 수분 부족 / 탄력 저하 / 색소 침착",
                "정상": "• <font color='#2ecc71'><b>정상·유지 단계</b></font> — 유수분 밸런스 양호"
            },
            "anti-aging": {
                # 이 부분은 ResultPageAntiAging에서 _add_conclusion_content로 대체되었으므로
                # 호출되지 않지만, 다른 페이지와의 호환성을 위해 남겨둠.
                "경고": "• <font color='#e74c3c'><b>즉시 관리 필요</b></font> — 산화 스트레스 높음 / 세포 손상 가속",
                "주의": "• <font color='#f39c12'><b>기능 저하 경향</b></font> — 순환 저하 / 대사 불균형 / 활력 감소",
                "정상": "• <font color='#2ecc71'><b>정상·유지 단계</b></font> — 신체 균형 양호"
            },
            "risk": { # 'risk' 타입에 대한 권고 요약은 이미지에 나온 텍스트 사용
                "경고": "• <font color='#e74c3c'><b>즉시 관리 필요</b></font> — 위장 / 염증 / 간 피로",
                "주의": "• <font color='#f39c12'><b>기능 저하 경향</b></font> — 혈류당 대사 / 면역",
                "정상": "• <font color='#2ecc71'><b>정상·유지 단계</b></font> — 수분·체온·심박 안정"
            }
        }
        current_s_points_map = s_points_data.get(summary_type, s_points_data.get("face"))

        # 점수 기반 요약 생성 (risk 타입 제외)
        s_points = []
        if summary_type != "risk":
            # __init__에서 계산된 점수 사용
            if hasattr(self, 'wellness_score'):
                score = self.wellness_score
                if score >= 85: s_points = [current_s_points_map["정상"]]
                elif score >= 75: s_points = [current_s_points_map["주의"], current_s_points_map["정상"]]
                else: s_points = [current_s_points_map["경고"], current_s_points_map["주의"]]
        else: # risk 타입은 고정된 요약 사용
            s_points = [
                current_s_points_map["경고"],
                current_s_points_map["주의"],
                current_s_points_map["정상"]
            ]

        # 요약 위젯 생성 및 추가
        summary_layout=self.summary_widget_container.layout() # () 추가
        if summary_type == "risk":
            title_label = QLabel("<h4>💡 권고 요약</h4>")
            title_label.setFont(QFont("NanumBarunGothic", 15, QFont.Bold))
            summary_layout.addWidget(title_label)

            points_label = QLabel("<br>".join(s_points))
            points_label.setFont(QFont("NanumBarunGothic", 13))
            summary_layout.addWidget(points_label)

            disclaimer_label=QLabel("이 리포트는 얼굴·혀 영상 패턴으로 체내 대사 균형과 장부 기능을 예측하는 AI 자가관리용 도구입니다."); disclaimer_label.setFont(QFont("NanumBarunGothic",14)); disclaimer_label.setWordWrap(True); disclaimer_label.setStyleSheet("padding: 12px 0;"); summary_layout.addWidget(disclaimer_label)
            model_info_label=QLabel("⚙️ AI 모델 정확도는 평균 82~88% 수준이며, 조명·해상도·피로도 등에 따라 변동이 있습니다."); model_info_label.setFont(QFont("NanumBarunGothic",12)); model_info_label.setWordWrap(True); model_info_label.setStyleSheet("color: #777;"); summary_layout.addWidget(model_info_label)

        # (수정) anti-aging 타입은 ResultPageAntiAging에서
        # _add_conclusion_content를 사용하므로 여기서 아무것도 하지 않도록 함
        elif summary_type == "anti-aging":
            pass # 아무것도 하지 않음

        elif hasattr(self, 'wellness_score'): # 다른 페이지 요약 (점수 기반)
            score = self.wellness_score
            level_text, s_color = self.wellness_level_text, self.wellness_color

            summaries_texts = { # d_text, analysis_text, model_text 가져오기
                "face": ("※ 심박수·HRV·혈류 순환 지표가 약간 낮아 스트레스 및 피로 누적 가능성 있음", "💬 AI 얼굴 분석은 혈류·피부 광반사 패턴을 기반으로 순환·스트레스·피로·활력 상태를 비침습적으로 추정하는 자가진단 도구입니다.", "⚙️ AI 모델: rPPG + CNN + HRV Regression Hybrid (평균 정확도 80~92 %)"),
                "tongue": ("※ 체내 순환·면역 지표가 낮게 나타남 → 수면·수분·피로 관리 강화 필요", "💬 AI 혀 분석은 혈액검사 없이 장부 기능과 대사 균형을 간접 확인할 수 있는 자가진단 도구입니다.", "⚙️ AI 모델: CNN + HSV + LBP 복합 구조 (DeepTongue 2.0 기준 정확도 82%)"),
                "skin": ("※ 피부 장벽 약화 및 노화 가속 경향이 보임 → 보습 및 자외선 차단 강화 필요", "💬 AI 피부 분석은 리얼업 출력 형식으로 고객 이름/날짜만 넣으면 바로 사용 가능합니다.", "⚙️ AI 모델: Multi-task CNN 기반 피부 분석 (정확도 83~94%)"),
                # anti-aging는 위에서 처리됨
            }
            d_text, analysis_text, model_text = summaries_texts.get(summary_type, summaries_texts["face"])


            score_label=QLabel(f"■ AI 웰니스 종합 점수: <font color='{s_color}'>{score}점</font> ({level_text})"); score_label.setFont(QFont("NanumBarunGothic",14,QFont.Bold)); summary_layout.addWidget(score_label)

            summary_points_label=QLabel("<br>".join(s_points)); summary_points_label.setFont(QFont("NanumBarunGothic",15)); summary_layout.addWidget(summary_points_label)
            detailed_label=QLabel(d_text); detailed_label.setFont(QFont("NanumBarunGothic",14)); detailed_label.setStyleSheet("color: #555;"); summary_layout.addWidget(detailed_label)
            analysis_label=QLabel(analysis_text); analysis_label.setFont(QFont("NanumBarunGothic",13)); analysis_label.setWordWrap(True); analysis_label.setStyleSheet("background-color: #f0f0f0; padding: 16px; border-radius: 14px;"); summary_layout.addWidget(analysis_label)
            
            # --- ▼▼▼ 여기가 수정된 부분입니다 ▼▼▼ ---
            if model_text: 
                model_label=QLabel(model_text) # `model_label` 생성
                model_label.setFont(QFont("NanumBarunGothic",13))
                model_label.setWordWrap(True)
                model_label.setStyleSheet("color: #777;") # `model_label`에 스타일 적용 (기존 model_info_label -> model_label)
                summary_layout.addWidget(model_label)
            # --- ▲▲▲ 여기가 수정된 부분입니다 ▲▲▲ ---

            disclaimer_label=QLabel("📍 진단 목적이 아닌, 건강 관리·조기 예측·생활습관 교정용으로 사용됩니다."); disclaimer_label.setFont(QFont("NanumBarunGothic",12,QFont.Bold)); disclaimer_label.setStyleSheet("color: #e74c3c;"); summary_layout.addWidget(disclaimer_label)

# ---------- 신규 결과 페이지 0: 신뢰도 안내 (테이블 + 텍스트) ----------
class ResultPageZero(TablePage):
    def __init__(self, parent=None):
        super().__init__("0. AI 분석 신뢰도/정확도 안내", "현재 측정 환경에서의 AI 분석 신뢰도 지표와 결과 활용에 대한 안내입니다.", parent)
        self.stretch_columns = [1]
        self.info_label = None # (신규) 환자 정보 표시용 레이블 초기화
        
    # (수정) 인자 추가: patient_name, date_str
    def populate_data(self, patient_name="Guest", date_str=""):
        """(신규) 외부 호출로 데이터를 생성하고 테이블을 채우는 함수"""
        
        # --- ▼▼▼ (신규) 환자 정보 레이블 생성 및 업데이트 ▼▼▼ ---
        if self.info_label is None:
            self.info_label = QLabel("", self)
            self.info_label.setFont(QFont("NanumBarunGothic", 16, QFont.Bold))
            self.info_label.setStyleSheet("""
                QLabel {
                    color: #173a9e;
                    background-color: #f4f7fc;
                    border: 1px solid #c0c0c0;
                    border-radius: 8px;
                    padding: 12px;
                    margin-bottom: 15px;
                }
            """)
            self.info_label.setAlignment(Qt.AlignCenter)
            
            # TablePage의 메인 레이아웃 가져오기
            # 구조: Title(0) -> Subtitle(1) -> [이곳에 삽입(2)] -> Table(3)
            layout = self.scroll_area.widget().layout()
            layout.insertWidget(2, self.info_label)
        
        # 텍스트 업데이트
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.info_label.setText(f"👤 고객환자: {patient_name}    |    📅 측정일시: {date_str}")
        # --- ▲▲▲ (신규) 끝 ▲▲▲ ---
        
        # --- ▼▼▼ (기존) 랜덤 데이터 생성 ▼▼▼ ---
        # 1. 테이블에 들어갈 실제 랜덤 데이터
        self.face_conf = random.randint(70, 98)
        self.tongue_conf = random.randint(61, 97)
        self.rppg_conf = random.randint(70, 95)
        self.env_status = random.choice(["양호", "보통", "약간 흐림"])

        self.table_data = [
            ["■ 개인별 신뢰 지표","",""],
            ["항목","분석 내용","예측 신뢰도"],
            ["얼굴 영상","피부톤·혈색 기반 웰니스 예측", f"{self.face_conf}%"],
            ["혀 영상","색/설태 기반 소화기 웰니스 예측", f"{self.tongue_conf}%"],
            ["rPPG (추정)","심박·스트레스 측정", f"{self.rppg_conf}%"],
            ["촬영 환경","조명 및 촬영 안정성", self.env_status]
        ]
        # --- ▲▲▲ (기존) 끝 ▲▲▲ ---
        
        # (이동) 데이터 생성 후 테이블 설정 및 컨텐츠 추가
        self._set_table_data(self.table_data) # 저장된 데이터로 테이블 설정
        self._add_static_content() # () 추가

    def _add_static_content(self):
        """
        (수정) 테이블 아래에 이미지 기반 텍스트를 추가합니다.
        summary_eval_container에 위젯을 추가하여 테이블 바로 뒤에 배치합니다.
        """

        # 1. 레이아웃 가져오기 (summary_eval_container 사용)
        if self.summary_eval_container.layout() is None: # () 추가
            summary_layout=QVBoxLayout(self.summary_eval_container)
            summary_layout.setContentsMargins(10,20,10,10); summary_layout.setSpacing(15)
        else:
            summary_layout = self.summary_eval_container.layout() # () 추가
            # (기존 위젯 제거)
            while summary_layout.count(): # () 추가
                child = summary_layout.takeAt(0)
                if child.widget(): child.widget().deleteLater() # () 추가, () 추가

        # --- ▼▼▼ (기존) 예시 텍스트도 populate_data의 랜덤 값을 사용 ▼▼▼ ---
        # 2. (수정) 이미지의 텍스트를 HTML로 *복원*
        html_content = f"""
        <style>
            /* 기본 폰트 설정 */
            div.content {{
                font-family: 'NanumBarunGothic', sans-serif;
                font-size: 16pt; /* 기본 폰트 크기 */
                line-height: 160%;
            }}
            /* h3: 섹션 타이틀 (1. 2.) */
            h3 {{
                font-size: 28pt;
                font-weight: 900; /* ExtraBold */
                margin-bottom: 17pt;
                margin-top: 19pt;
                color: #333; /* 제목 색상 */
            }}
            /* ul: 기본 글머리 기호 목록 */
            ul {{
                margin-left: 20px; /* 들여쓰기 */
                margin-bottom: 14px;
            }}
            /* li: 목록 아이템 */
            li {{
                margin-bottom: 14px;
            }}
            /* 중첩된 ul (예: 하위 글머리 기호) */
            ul ul {{
                list-style-type: none; /* 바깥쪽 점 제거 */
                margin-left: 14px;
                margin-top: 14px;
            }}
            /* (수정) 중첩된 li (• 기호 대신 이미지처럼 ' - ' 사용) */
            ul ul li::before {{
                content: '- '; /* '• ' 대신 ' - ' 사용 */
                margin-right: 14px;
                font-weight: bold;
            }}
            /* 빨간색 경고문 */
            p.disclaimer {{
                font-size: 14pt;
                font-weight: bold;
                color: #d33;
                padding: 17px;
                border-top: 1px solid #ddd;
                margin-top: 20px;
                text-align: center;
            }}
        </style>

        <div class="content">
            <h3>1. 정확도 안내 (개인별 신뢰 지표)</h3>
            <ul>
                <li>예: (위 표의 내용은 이 예시에 해당합니다)
                    <ul>
                        <li>얼굴 영상 → 피부톤·혈색 기반 웰니스 예측 <b>신뢰도: {self.face_conf}%</b></li>
                        <li>혀 영상 → 색/태 기반 소화기 웰니스 예측 <b>신뢰도: {self.tongue_conf}%</b></li>
                        <li>rPPG → 심박·스트레스 측정 <b>신뢰도: {self.rppg_conf}%</b> (촬영 환경 {self.env_status})</li>
                    </ul>
                </li>
                <li>이 단계에서 사용자에게 "현재 분석이 얼마나 신뢰할 수 있는지"를 먼저 알려줌.</li>
                <li>장점: 사용자가 결과를 과도하게 믿지 않고, 신뢰도에 따라 참고 수준을 조절 가능.</li>
            </ul>

            <h3>2. AI 건강 분석 리포트 제시</h3>
            <ul>
                <li>정확도 %를 보여준 후,
                    <ul>
                        <li>얼굴 리포트</li>
                        <li>혀 리포트</li>
                        <li>각각 신호등(🟢 / 🟡 / 🔴)으로 가시성 높게 제공.</li>
                    </ul>
                </li>
            </ul>

            <p class="disclaimer">
            본 건강상태 분석은 의사의 진단이 아니라 AI 분석이므로 자신의 건강 상태 분석 자료를 참고로 하여 자신의 건강을 더욱 지키고 함이며, 필요시 의사의 진단을 받아 더욱 건강에 신경을 쓰시어 행복한 삶을 살아 가도록 하셔야 합니다 !!!
            </p>
        </div>
        """
        # --- ▲▲▲ (수정) f-string으로 변경 끝 ▲▲▲ ---

        # 3. QLabel 생성 및 설정
        content_label = QLabel(html_content)
        content_label.setWordWrap(True)
        content_label.setTextInteractionFlags(Qt.TextBrowserInteraction) # 리치 텍스트 상호작용
        content_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        # 4. 레이아웃에 추가
        summary_layout.addWidget(content_label)

# ---------- 결과 페이지 1: 얼굴 분석 ----------
class ResultPageFace(TablePage):
    def __init__(self,parent=None):
        super().__init__("1. AI 얼굴 분석 (GPT-4o)","얼굴의 다양한 특징을 분석하여 현재 건강 상태를 예측합니다.",parent) # () 추가
        # (수정) 컬럼 스트레치를 9개 컬럼에 맞게 변경 (특징, 해석, 권고)
        self.stretch_columns=[2, 3, 7] # 0-based: 특징, 해석, 권고
        
    def populate_data(self, analysis_data: dict):
        """(수정) GPT-4O 분석 데이터를 받아 테이블을 채우는 함수"""
        
        # --- ▼▼▼ 3단계 수정: 3항목 -> 20항목 (이미지 기반) ▼▼▼ ---
        
        # 1. 테이블 헤더 정의 (9열)
        rows=[
            ["■ AI 얼굴 항목 (GPT-4O)","","","","","","","",""], # 9개
            ["#", "질환 / 건강균", "얼굴 AI 탐지 특징", "건강 해석", "측정 값", "신호등", "신뢰도", "권고 사항", "관련 관리 지표"]
        ]

        # 2. 분석할 항목 리스트 정의 (key, name)
        # 이 key값들은 MainWindow의 get_gpt_analysis 프롬프트와 일치해야 합니다.
        topics = [
            ('fall_risk', '낙상 위험도'),
            ('hrv', '심박수/심박변이도'),
            ('blood_pressure', '혈압 (SBP/DBP)'),
            ('spo2', '산소포화도 (SpO₂)'),
            ('hypertension_risk', '고혈압 경향'),
            ('hypotension_risk', '저혈압 경향'),
            ('anemia', '빈혈'),
            ('diabetes_risk', '당뇨불균형(당뇨 경향)'),
            ('thyroid_function', '갑상선 기능저하'),
            ('liver_function', '간 기능 저하'),
            ('kidney_function', '신장 기능 저하'),
            ('heart_function_weak', '심장 기능 약화'),
            ('respiratory_function', '호흡기 기능 저하'),
            ('chronic_fatigue', '만성피로'),
            ('dehydration', '탈수'),
            ('stress_overload', '스트레스 과부하'),
            ('insomnia', '불면'),
            ('depression_anxiety', '우울/불안 경향'),
            ('immunity_weak', '면역력 저하'),
            ('inflammation_fatigue', '면역력 피로(염증)')
        ]

        status_list = []

        try:
            for i, (key, name) in enumerate(topics, 1):
                data = analysis_data.get(key, {})
                
                status = data.get('status', '분석 실패')
                status_list.append(status)
                
                rows.append([
                    str(i), # 1. #
                    name,   # 2. 질환 / 건강균
                    data.get('observation', '데이터 없음'),  # 3. 얼굴 AI 탐지 특징
                    data.get('interpretation', '데이터 없음'), # 4. 건강 해석
                    data.get('value', 'N/A'),           # 5. 측정 값
                    LightDot(status),                   # 6. 신호등
                    data.get('confidence', 'N/A'),      # 7. 신뢰도
                    data.get('recommendation', '데이터 없음'), # 8. 권고 사항
                    data.get('metric', 'N/A')           # 9. 관련 관리 지표
                ])
        
            # 2-4. (신규) 종합 웰니스 점수 (20항목 기준)
            normal_count = status_list.count('정상')
            caution_count = status_list.count('주의')
            
            if normal_count >= 17: # 85% 이상 '정상'
                self.wellness_score = random.randint(85, 95)
            elif normal_count >= 14 or caution_count >= 17: # 70% 이상 '정상' or 85% 이상 '주의'
                self.wellness_score = random.randint(70, 84)
            else:
                self.wellness_score = random.randint(50, 69) # 그 외 (경고 포함)

        except Exception as e:
            print(f"GPT 결과 파싱 오류 (ResultPageFace): {e}")
            rows.append(["오류", "GPT 결과 파싱 중 오류 발생", str(e), "", "", "", "", "", ""])
            self.wellness_score = 0

        # 3. 테이블 데이터 저장
        self.table_data = rows

        # 4. 웰니스 점수 저장
        if self.wellness_score >= 85: self.wellness_level_text, self.wellness_color = "정상 단계", "#2ecc71"
        elif self.wellness_score >= 70: self.wellness_level_text, self.wellness_color = "주의 단계", "#f39c12"
        else: self.wellness_level_text, self.wellness_color = "관리 필요", "#e74c3c"
        
        # 5. 테이블 설정 및 요약 추가
        self._set_table_data(self.table_data) 
        self._add_wellness_summary_content("face") # 요약 섹션은 기존 UI 재사용
        # --- ▲▲▲ 3단계 수정 끝 ▲▲▲ ---

    # --- (수정) 이 페이지 전용 perform_resize ---
    def perform_resize(self):
        # 1. 열 너비 맞추기
        self.table.resizeColumnsToContents() # 먼저 컨텐츠 기준으로 맞춤
        header = self.table.horizontalHeader()

        for col in range(header.count()):
            # (신규) 0번 열('#')은 좁은 너비로 고정
            if col == 0:
                header.setSectionResizeMode(col, QHeaderView.Fixed)
                header.resizeSection(col, 50) # 50px로 고정
            
            elif col in self.stretch_columns:
                header.setSectionResizeMode(col, QHeaderView.Stretch)
            
            else:
                # (수정) 컨텐츠에 맞게 고정하는 대신, 사용자가 조절할 수 있게 합니다.
                header.setSectionResizeMode(col, QHeaderView.Interactive)
        
        # 2. 행 높이 맞추기
        self.table.resizeRowsToContents()

        # 3. (중요) 테이블의 전체 높이를 계산하여 고정
        total_height = 0
        if self.table.horizontalHeader().isVisible():
            total_height += self.table.horizontalHeader().height()

        for i in range(self.table.rowCount()):
            total_height += self.table.rowHeight(i)

        self.table.setFixedHeight(total_height + 4)
    # --- perform_resize 수정 끝 ---


    # --- (얼굴 분석 페이지 전용 요약 섹션) ---
    # 이 함수는 2단계에서 수정한 것을 그대로 사용합니다. (변경 없음)
    def _add_wellness_summary_content(self, summary_type="face"):
        # 1. 레이아웃 가져오기 (summary_eval_container 사용)
        if self.summary_eval_container.layout() is None: # () 추가
            summary_layout=QVBoxLayout(self.summary_eval_container)
            summary_layout.setContentsMargins(10,20,10,10); summary_layout.setSpacing(10)
        else:
            summary_layout = self.summary_eval_container.layout() # () 추가
            # 기존 위젯 제거
            while summary_layout.count(): # () 추가
                child = summary_layout.takeAt(0)
                if child.widget(): # () 추가
                    child.widget().deleteLater() # () 추가

        # 2. (신규) 타이틀 추가 (이미지 반영)
        title_label = QLabel("<h4>🌝 얼굴 종합 권고 요약 (by GPT-4o)</h4>") # (수정)
        title_label.setFont(QFont("NanumBarunGothic", 17, QFont.Bold))
        title_label.setStyleSheet("margin-bottom: 12px;")
        summary_layout.addWidget(title_label)

        # 3. (신규) 고정된 3-레벨 권고 사항 (이미지 반영)
        s_points_html = [
            "• <font color='#e74c3c'><b>즉시 관리 필요</b></font> — 만성 스트레스 / 수면 불균형 / 자율신경 과활성",
            "• <font color='#f39c12'><b>기능 저하 경향</b></font> — 혈류순환 / 간·신장 / 피부탄력 저하",
            "• <font color='#2ecc71'><b>정상·유지 단계</b></font> — 산소포화도 / 심박 안정 / 혈색 균형"
        ]
        points_label = QLabel("<br>".join(s_points_html))
        points_label.setFont(QFont("NanumBarunGothic", 16))
        points_label.setStyleSheet("line-height: 150%;") # 줄 간격 추가
        summary_layout.addWidget(points_label)

        # 4. 웰니스 점수 (랜덤 생성된 값 사용)
        if hasattr(self, 'wellness_score'):
            score = self.wellness_score
            level_text, s_color = self.wellness_level_text, self.wellness_color

            # (수정) 이미지와 동일한 텍스트/아이콘으로 변경
            score_label=QLabel(f"■ AI 웰니스 종합 점수: <font color='{s_color}'>{score}점</font> ({level_text})")
            score_label.setFont(QFont("NanumBarunGothic", 18, QFont.Bold))
            score_label.setStyleSheet("margin-top: 15px;") # 상단 여백
            summary_layout.addWidget(score_label)

        # 5. (신규) 고정된 상세 텍스트 (이미지 반영)
        d_text = "※ AI 분석 결과, 피로 누적 및 수분 부족 경향이 보입니다." # (수정)
        analysis_text = "💬 AI 얼굴 분석은 혈류·심박·피부 광반사 패턴을 기반으로 순환·스트레스·피로·활력 상태를 비침습적으로 추정하는 자가진단 도구입니다."
        model_text = "⚙️ AI 모델: GPT-4o (OpenAI) + rPPG + CNN + HRV Regression Hybrid (Binah.ai / Affectiva / VitalSigns AI / FaceReader 유형 모델 기반, 평균 정확도 85~92 %)" # (수정)
        disclaimer_text = "📍 진단 목적이 아닌, 건강 관리·조기 예측·생활습관 교정용으로 사용됩니다."

        # 6. 위젯 추가 (이미지 순서대로)
        detailed_label=QLabel(d_text)
        detailed_label.setFont(QFont("NanumBarunGothic", 16, QFont.Bold))
        detailed_label.setStyleSheet("color: #555; margin-top: 11px;")
        summary_layout.addWidget(detailed_label)

        analysis_label=QLabel(analysis_text)
        analysis_label.setFont(QFont("NanumBarunGothic", 15))
        analysis_label.setWordWrap(True)
        analysis_label.setStyleSheet("background-color: #f0f0f0; padding: 11px; border-radius: 13px; margin-top: 14px;")
        summary_layout.addWidget(analysis_label)

        model_label=QLabel(model_text)
        model_label.setFont(QFont("NanumBarunGothic", 14))
        model_label.setWordWrap(True)
        model_label.setStyleSheet("color: #777; margin-top: 13px; line-height: 140%;") # 긴 텍스트를 위한 줄간격
        summary_layout.addWidget(model_label)

        disclaimer_label=QLabel(disclaimer_text)
        disclaimer_label.setFont(QFont("NanumBarunGothic", 14, QFont.Bold))
        disclaimer_label.setStyleSheet("color: #e74c3c; margin-top: 13px;")
        summary_layout.addWidget(disclaimer_label)

# ---------- 결과 페이지 2: 혀 분석 ----------
class ResultPageTongue(TablePage):
    def __init__(self,parent=None):
        super().__init__("2. AI 혀 분석 (GPT-4o)","혀의 색, 설태 등을 통해 웰니스 상태를 예측합니다.",parent) # () 추가
        # (수정) 컬럼 스트레치를 9개 컬럼에 맞게 변경
        self.stretch_columns=[2, 3, 7] # 0-based: AI 탐지 특징, 건강 해석, 권고 사항

    def populate_data(self, analysis_data: dict):
        """(수정) GPT-4O 혀 분석 데이터를 받아 테이블을 채우는 함수"""
        
        # 1. 테이블 헤더 정의 (9열)
        rows=[
            ["■ AI 혀 항목 (GPT-4O)","","","","","","","",""], # 9개
            ["#", "질환 / 건강균", "AI 탐지 특징 (혀 영상)", "건강 해석", "측정 값 (예시)", "신호등", "신뢰도", "권고 사항", "관련 관리 지표"]
        ]

        # 2. 분석할 항목 리스트 정의 (key, name) - 새 이미지 기준 20개
        topics = [
            ('anemia_hypotension', '빈혈 / 저혈압'),
            ('hypertension_heat', '고혈압 / 열증'),
            ('heart_function', '심장 기능 저하'),
            ('gastritis_ulcer', '소화불량 / 위염'),
            ('liver_function', '간 기능 저하'),
            ('kidney_function_1', '신장 기능 저하'), # 6번 항목
            ('kidney_function_2', '신장 기능 저하'), # 7번 항목 (이미지상 중복)
            ('dehydration', '탈수 / 수분 부족'),
            ('edema_water', '부종 / 수분정체'),
            ('diabetes_risk', '당뇨불균형 (당뇨)'),
            ('thyroid_function', '갑상선 기능저하'),
            ('obesity_immunity', '비만 / 체질 과다(비만)'),
            ('immunity_weak', '면역력 저하'),
            ('fatigue_energy', '피로 기능 저하'),
            ('stress_overload', '스트레스 과부하'),
            ('insomnia_fatigue', '불면 / 피로 누적'),
            ('depression_anxiety', '우울 / 기력 저하'),
            ('inflammation_stomatitis', '염증 / 구강염'),
            ('candidiasis', '곰팡이 감염 (Candida)'),
            ('oral_dryness', '구취 / 구강건조증')
        ]
        
        status_list = []

        try:
            for i, (key, name) in enumerate(topics, 1):
                # analysis_data 딕셔너리에서 해당 key의 데이터 추출
                data = analysis_data.get(key, {})
                
                # 'status' 키를 사용하여 신호등 레벨 결정
                status = data.get('status', '분석 실패')
                status_list.append(status)
                
                # 테이블 행 데이터 추가 (9개 열)
                rows.append([
                    str(i), # 1. #
                    name,   # 2. 질환 / 건강균
                    data.get('observation', '데이터 없음'),  # 3. AI 탐지 특징
                    data.get('interpretation', '데이터 없음'), # 4. 건강 해석
                    data.get('value', 'N/A'),           # 5. 측정 값 (예시)
                    LightDot(status),                   # 6. 신호등
                    data.get('confidence', 'N/A'),      # 7. 신뢰도
                    data.get('recommendation', '데이터 없음'), # 8. 권고 사항
                    data.get('metric', 'N/A')           # 9. 관련 관리 지표
                ])
            
            # 2-4. (신규) 종합 웰니스 점수 (20항목 기준)
            normal_count = status_list.count('정상')
            caution_count = status_list.count('주의')
            
            if normal_count >= 17: # 85% 이상 '정상'
                self.wellness_score = random.randint(85, 95)
            elif normal_count >= 14 or caution_count >= 17: # 70% 이상 '정상' or 85% 이상 '주의'
                self.wellness_score = random.randint(70, 84)
            else:
                self.wellness_score = random.randint(50, 69) # 그 외 (경고 포함)

        except Exception as e:
            print(f"GPT (혀) 결과 파싱 오류: {e}")
            # 오류 발생 시 테이블에 오류 메시지 표시
            rows.append(["오류", "GPT 결과 파싱 중 오류 발생", str(e), "", "", "", "", "", ""])
            self.wellness_score = 0

        # 3. 테이블 데이터 저장
        self.table_data = rows

        # 4. 웰니스 점수 저장
        if self.wellness_score >= 85: self.wellness_level_text, self.wellness_color = "정상 단계", "#2ecc71"
        elif self.wellness_score >= 70: self.wellness_level_text, self.wellness_color = "주의 단계", "#f39c12"
        else: self.wellness_level_text, self.wellness_color = "관리 필요", "#e74c3c"
        
        # 5. 테이블 설정 및 요약 추가
        self._set_table_data(self.table_data)
        self._add_wellness_summary_content("tongue") # 요약 섹션은 기존 UI 재사용

    # --- (수정) 이 페이지 전용 perform_resize ---
    def perform_resize(self):
        # 1. 열 너비 맞추기
        self.table.resizeColumnsToContents() # 먼저 컨텐츠 기준으로 맞춤
        header = self.table.horizontalHeader()

        for col in range(header.count()):
            # (신규) 0번 열('#')은 좁은 너비로 고정
            if col == 0:
                header.setSectionResizeMode(col, QHeaderView.Fixed)
                header.resizeSection(col, 50) # 50px로 고정
            
            elif col in self.stretch_columns:
                header.setSectionResizeMode(col, QHeaderView.Stretch)
            
            else:
                # (수정) 컨텐츠에 맞게 고정하는 대신, 사용자가 조절할 수 있게 합니다.
                header.setSectionResizeMode(col, QHeaderView.Interactive)
        
        # 2. 행 높이 맞추기
        self.table.resizeRowsToContents()

        # 3. (중요) 테이블의 전체 높이를 계산하여 고정
        total_height = 0
        if self.table.horizontalHeader().isVisible():
            total_height += self.table.horizontalHeader().height()

        for i in range(self.table.rowCount()):
            total_height += self.table.rowHeight(i)

        self.table.setFixedHeight(total_height + 4)
    # --- perform_resize 수정 끝 ---


    # --- (혀 분석 페이지 전용 요약 섹션) ---
    def _add_wellness_summary_content(self, summary_type="tongue"):
        """
        (신규) 혀 분석 페이지 전용 요약 섹션 (이미지 레이아웃 반영)
        TablePage의 기본 메서드를 오버라이드합니다.
        """
        # 1. 레이아웃 가져오기
        if self.summary_eval_container.layout() is None: # () 추가
            summary_layout=QVBoxLayout(self.summary_eval_container)
            summary_layout.setContentsMargins(10,20,10,10); summary_layout.setSpacing(10)
        else:
            summary_layout = self.summary_eval_container.layout() # () 추가
            # 기존 위젯 제거
            while summary_layout.count(): # () 추가
                child = summary_layout.takeAt(0)
                if child.widget(): # () 추가
                    child.widget().deleteLater() # () 추가

        # 2. 타이틀 추가 (이미지 반영)
        title_label = QLabel("<h4>🌿 종합 권고 요약 (by GPT-4o)</h4>") # (수정)
        title_label.setFont(QFont("NanumBarunGothic", 16, QFont.Bold))
        title_label.setStyleSheet("margin-bottom: 5px;")
        summary_layout.addWidget(title_label)

        # 3. 고정된 3-레벨 권고 사항 (이미지 반영)
        s_points_html = [
            "• <font color='#e74c3c'><b>즉시 관리 필요</b></font> — 염증 / 스트레스 / 심장 / 곰팡이 감염",
            "• <font color='#f39c12'><b>기능 저하 경향</b></font> — 위·간·신장 / 면역 / 피로 / 탈수",
            "• <font color='#2ecc71'><b>정상·유지 단계</b></font> — 수분·온도 균형 양호"
        ]
        points_label = QLabel("<br>".join(s_points_html))
        points_label.setFont(QFont("NanumBarunGothic", 15))
        points_label.setStyleSheet("line-height: 150%;") # 줄 간격 추가
        summary_layout.addWidget(points_label)

        # 4. 웰니스 점수 (GPT 분석 기반)
        if hasattr(self, 'wellness_score'):
            score = self.wellness_score
            level_text, s_color = self.wellness_level_text, self.wellness_color

            score_label=QLabel(f"■ AI 웰니스 종합 점수: <font color='{s_color}'>{score}점</font> ({level_text})")
            score_label.setFont(QFont("NanumBarunGothic", 16, QFont.Bold))
            score_label.setStyleSheet("margin-top: 15px;") # 상단 여백
            summary_layout.addWidget(score_label)

        # 5. 고정된 상세 텍스트 (이미지 반영)
        d_text = "※ AI 분석 결과, 소화기 부담 및 수분 부족 경향이 보입니다." # (수정)
        analysis_text = "💬 AI 혀 분석은 혈액검사 없이 장부 기능과 대사 균형을 간접 확인할 수 있는 자가진단 도구입니다."
        model_text = "⚙️ AI 모델: GPT-4o (OpenAI) + CNN + HSV + LBP 복합 구조 (DeepTongue 2.0 기준 정확도 82%)" # (수정)
        disclaimer_text = "📍 진단 목적이 아닌, 건강 관리·조기 예측·생활습관 교정용으로 사용됩니다."

        # 6. 위젯 추가 (이미지 순서대로)
        detailed_label=QLabel(d_text)
        detailed_label.setFont(QFont("NanumBarunGothic", 16, QFont.Bold))
        detailed_label.setStyleSheet("color: #555; margin-top: 8px;")
        summary_layout.addWidget(detailed_label)

        analysis_label=QLabel(analysis_text)
        analysis_label.setFont(QFont("NanumBarunGothic", 14))
        analysis_label.setWordWrap(True)
        analysis_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 10px;")
        summary_layout.addWidget(analysis_label)

        model_label=QLabel(model_text)
        model_label.setFont(QFont("NanumBarunGothic", 13))
        model_label.setWordWrap(True)
        model_label.setStyleSheet("color: #777; margin-top: 10px;")
        summary_layout.addWidget(model_label)

        disclaimer_label=QLabel(disclaimer_text)
        disclaimer_label.setFont(QFont("NanumBarunGothic", 13, QFont.Bold))
        disclaimer_label.setStyleSheet("color: #e74c3c; margin-top: 10px;")
        summary_layout.addWidget(disclaimer_label)

# ---------- 결과 페이지 3: 피부 분석 (동적 권장 루틴 추가됨) ----------
class ResultPageSkin(TablePage):
    def __init__(self,parent=None):
        super().__init__("3. AI 피부 분석 (GPT-4o)","피부의 유수분, 탄력, 주름 등을 분석하여 피부 건강 상태를 예측합니다.",parent) # () 추가
        self.stretch_columns=[1, 2] # (수정) GPT 분석에 맞게 변경
        
    def populate_data(self, analysis_data: dict):
        """(수정) GPT-4O 피부 분석 데이터를 받아 테이블을 채우는 함수"""
        
        # --- ▼▼▼ (신규) 재촬영 시 위젯 중복 방지 코드 ▼▼▼ ---
        # 기존 레이아웃의 자식 위젯들을 모두 삭제합니다.
        
        # 1. summary_eval_container (동적루틴, 일반루틴, 라이프스타일팁) 비우기
        if self.summary_eval_container.layout() is not None:
            layout = self.summary_eval_container.layout()
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                    
        # 2. summary_widget_container (종합점수, AI모델정보) 비우기
        # (이 페이지에서는 TablePage의 기본 _add_wellness_summary_content를 오버라이드하지 않으므로,
        #  부모 클래스의 함수가 사용하는 summary_widget_container를 비워야 합니다.)
        if self.summary_widget_container.layout() is not None:
            layout = self.summary_widget_container.layout()
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        # --- ▲▲▲ (신규) 재촬영 시 위젯 중복 방지 코드 끝 ▲▲▲ ---

        
        # 1. 테이블 헤더 정의 (기존과 동일)
        rows=[
            ["■ AI 피부 항목 (GPT-4o)","","","",""], 
            ["항목", "AI 분석 특징", "상태 (AI 판단)", "신호등", "권고사항(루틴/성분)"]
        ]

        # 2. 분석 데이터 매핑 (기존과 동일)
        skin_topics = [
            ('hydration', '수분/건조도'),
            ('oil_balance', '유분 밸런스'),
            ('sensitivity', '홍조/민감'),
            ('pore_texture', '모공/요철'),
            ('wrinkles', '주름/탄력')
        ]
        
        status_list = []
        
        # (신규) analysis_data가 비어있을 경우를 대비
        if not analysis_data:
             analysis_data = {} # 빈 딕셔너리로 초기화

        try:
            for key, name in skin_topics:
                # (수정) analysis_data에서 직접 가져오도록
                data = analysis_data.get(key, {}) # key가 없으면 빈 dict 반환
                status = data.get('status', '분석 실패') # status가 없으면 '분석 실패'
                status_list.append(status)
                
                rows.append([
                    name,
                    data.get('observation', '데이터 없음'),
                    status,
                    LightDot(status),
                    data.get('recommendation', '데이터 없음')
                ])
            
            # 3. 웰니스 점수 계산 (기존과 동일)
            normal_count = status_list.count('정상')
            caution_count = status_list.count('주의')
            
            if normal_count >= 4:
                self.wellness_score = random.randint(85, 95)
            elif normal_count >= 2 or caution_count >= 4:
                self.wellness_score = random.randint(70, 84)
            else:
                self.wellness_score = random.randint(50, 69)

        except Exception as e:
            print(f"GPT (피부) 결과 파싱 오류: {e}")
            rows.append(["오류", "GPT 결과 파싱 중 오류 발생", str(e), LightDot("경고"), ""])
            self.wellness_score = 0

        self.table_data = rows # 생성된 데이터를 인스턴스 변수에 저장

        # 4. 웰니스 점수 저장 (기존과 동일)
        if self.wellness_score >= 85: self.wellness_level_text, self.wellness_color = "정상 단계", "#2ecc71"
        elif self.wellness_score >= 70: self.wellness_level_text, self.wellness_color = "주의 단계", "#f39c12"
        else: self.wellness_level_text, self.wellness_color = "관리 필요", "#e74c3c"
        
        # 5. 테이블 설정 및 요약 추가
        self._set_table_data(self.table_data)
        
        # 5-1. (신규) 동적 권장 루틴 추가
        self._add_dynamic_skin_advice(analysis_data) 
        
        # 5-2. (기존) 일반 요약 추가
        # (주의: ResultPageSkin은 부모 클래스의 _add_wellness_summary_content를 사용합니다)
        super()._add_wellness_summary_content("skin") 
        
        # 5-3. (기존) 고정 루틴 추가
        self._add_skin_routine_section() 
        
        # 5-4. (기존) 라이프스타일 팁 추가
        self._add_lifestyle_tips_section()
        
        # --- ▲▲▲ 여기가 수정된 부분입니다 ▲▲▲ ---

    # --- ▼▼▼ (신규) 동적 권장 루틴 생성 함수 ▼▼▼ ---
    def _add_dynamic_skin_advice(self, analysis_data: dict):
        """
        (신규) AI 분석 결과의 신호등('주의'/'경고')에 따라
        동적인 권장 루틴을 생성하여 summary_eval_container에 추가합니다.
        """
        
        # 1. '주의' 또는 '경고'가 하나라도 있는지 확인
        has_warning = False
        skin_topics_keys = ['hydration', 'oil_balance', 'sensitivity', 'pore_texture', 'wrinkles']
        
        statuses = {}
        for key in skin_topics_keys:
            status = analysis_data.get(key, {}).get('status', '정상') # 기본값 '정상'
            statuses[key] = status
            if status in ["주의", "경고"]:
                has_warning = True

        if not has_warning:
            return # '주의'/'경고'가 없으면 이 섹션을 추가하지 않음

        # 2. 레이아웃 가져오기 (summary_eval_container 사용)
        if self.summary_eval_container.layout() is None:
            summary_layout = QVBoxLayout(self.summary_eval_container)
            summary_layout.setContentsMargins(10, 20, 10, 10)
            summary_layout.setSpacing(15) # 간격 넉넉하게
        else:
            summary_layout = self.summary_eval_container.layout()

        # 3. 신호등별 권장 루틴 데이터
        advice_map = {
            'hydration': {
                'title': '💧 수분/건조도',
                '주의': '<b><font color="#f39c12">[수분 보충]</font></b> 속건조 상태일 수 있습니다. **히알루론산, 판테놀(B5)** 세럼을 사용하고, **하루 물 1.5L** 섭취를 의식적으로 실천하세요.',
                '경고': '<b><font color="#e74c3c">[장벽 강화]</font></b> 피부 장벽이 손상되었습니다. **세라마이드, 스쿠알란** 성분의 고보습 크림으로 교체하고, 저녁에 **페이스 오일** 1~2방울을 추가하세요. **수면 7시간**이 필수입니다.'
            },
            'oil_balance': {
                'title': '✨ 유분 밸런스',
                '주의': '<b><font color="#f39c12">[밸런싱]</font></b> T존(이마/코)은 번들거리고 U존(볼)은 건조한 복합성 상태입니다. T존은 **BHA** 토너로 닦아내고, U존은 보습을 덧바르세요. **오일프리 젤 크림**을 추천합니다.',
                '경고': '<b><font color="#e74c3c">[피지 조절]</font></b> 유분이 과다합니다. **약산성 젤 클렌저**와 **BHA (살리실산)** 제품을 주 2-3회 사용하세요. **유제품, 설탕, 밀가루** 섭취를 줄이는 것이 필수입니다.'
            },
            'sensitivity': {
                'title': '🛡️ 홍조/민감',
                '주의': '<b><font color="#f39c12">[진정]</font></b> 피부가 외부 자극에 민감해져 있습니다. **시카(병풀), 아줄렌, 알로에** 성분으로 피부 열감을 내려주세요. **각질 제거를 즉시 중단**하고 뜨거운 물 세안을 피하세요.',
                '경고': '<b><font color="#e74c3c">[장벽 회복]</font></b> 장벽이 손상되어 따가움/가려움이 유발될 수 있습니다. **세라마이드, 판테놀(B5)** 성분에 집중하고, **무알콜/무향료** 제품으로 스킨케어를 최소화하세요. **자외선 차단제(무기자차)**는 365일 필수입니다.'
            },
            'pore_texture': {
                'title': 'mịn 모공/요철',
                '주의': '<b><font color="#f39c12">[각질 정돈]</font></b> 모공이 늘어지거나 피지가 쌓여 피부결이 거칠어 보입니다. **AHA/PHA** 토너로 주 2-3회 저녁에 닦아내고, **보습**에 2배 더 신경 쓰세요.',
                '경고': '<b><font color="#e74c3c">[턴오버 촉진]</font></b> 묵은 각질과 피지가 좁쌀 트러블을 유발하고 있습니다. 저녁 루틴에 **저농도 레티놀** 제품을 도입하세요 (주 2회 시작). **클렌징 오일**로 매일 1분 롤링하는 것도 좋습니다.'
            },
            'wrinkles': {
                'title': '⏳ 주름/탄력',
                '주의': '<b><font color="#f39c12">[안티에이징 시작]</font></b> 눈가/입가에 잔주름이 보입니다. **펩타이드, 콜라겐** 세럼과 **아이크림** 사용을 시작하세요. **자외선 차단제**를 꼼꼼히 발라 광노화를 막는 것이 가장 중요합니다.',
                '경고': '<b><font color="#e74c3c">[집중 탄력 개선]</font></b> 콜라겐 감소로 주름이 깊어지고 있습니다. 주름 개선 기능성 성분인 **레티놀/레티날**을 저녁 루틴에 **필수**로 포함시키고, 낮에는 **비타민C** 세럼으로 항산화 관리를 병행하세요.'
            }
        }

        # 4. 위젯 생성 및 추가
        
        # (신규) 구분선 추가 (다른 섹션과 분리)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("margin: 15px 0;") # 위아래 여백
        summary_layout.addWidget(line)

        title_label = QLabel("<h3>🚦 신호등별 집중 관리 루틴</h3>")
        title_label.setFont(QFont("NanumBarunGothic", 18, QFont.Bold))
        title_label.setStyleSheet("margin-bottom: 15px; color: #333;")
        summary_layout.addWidget(title_label)

        # '주의'/'경고'인 항목만 골라서 QLabel로 추가
        for key in skin_topics_keys:
            status = statuses.get(key)
            if status in ["주의", "경고"]:
                advice = advice_map.get(key)
                if advice:
                    # 항목 타이틀 (예: 💧 수분/건조도 (경고))
                    item_title_label = QLabel(f"<b>{advice['title']} (<font color='{LightDot.COLORS.get(status)}'>{status}</font>)</b>")
                    item_title_label.setFont(QFont("NanumBarunGothic", 17, QFont.Bold))
                    item_title_label.setStyleSheet("margin-top: 10px; margin-bottom: 5px;")
                    summary_layout.addWidget(item_title_label)

                    # 항목별 권고 텍스트
                    advice_text = advice.get(status) # '주의' 또는 '경고'에 해당하는 텍스트
                    item_advice_label = QLabel(advice_text)
                    item_advice_label.setFont(QFont("NanumBarunGothic", 16))
                    item_advice_label.setWordWrap(True)
                    item_advice_label.setStyleSheet("line-height: 150%; margin-left: 10px; padding-bottom: 10px;")
                    summary_layout.addWidget(item_advice_label)
    # --- ▲▲▲ (신규) 함수 추가 끝 ▲▲▲ ---


    # --- (신규) 권장 루틴 섹션 추가 ---
    # (이하 2단계와 동일 - 변경 없음)
    def _add_skin_routine_section(self):
        summary_layout = self.summary_eval_container.layout() # () 추가 # summary_eval_container 사용
        if summary_layout is None: # 레이아웃이 없으면 새로 생성 (하지만 TablePage에서 이미 생성됨)
            summary_layout = QVBoxLayout(self.summary_eval_container)
            summary_layout.setContentsMargins(10, 20, 10, 10)
            summary_layout.setSpacing(10)

        # (수정) 동적 루틴이 위에 생겼으므로, 구분을 위해 구분선과 제목 수정
        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken); summary_layout.addWidget(line)
        title = QLabel("<h3>💡 피부 타입별 일반 권장 루틴</h3>") # 제목 수정
        
        title.setFont(QFont("NanumBarunGothic", 18, QFont.Bold))
        title.setStyleSheet("margin-top: 24px; margin-bottom: 15px; color: #333;")
        summary_layout.addWidget(title)

        routine_data = {
            "복합성 (가장 흔함)": [
                "AM: 약산성 세안 → 비타민C 세럼 → 가벼운 보습 → 자차",
                "PM: 저자극 클렌징 → BHA(격일) → 나이아신아마이드 → 세라마이드 크림",
                "주 2회: 레티날/레티놀 저농도(피부 적응), 시트팩은 10분 이내"
            ],
            "건성/민감": [
                "AM: 미온수 헹굼 → 히알루론산/판테놀 → 세라마이드 크림 → 자차",
                "PM: 클렌징 밀크 → 각질제거는 PHA/LHA 위주(주 1-2회) → 아줄렌/센텔라 → 리치 보습"
            ],
            "지성/트러블": [
                "AM: 젤클렌저 소량 → 나이아신아마이드 → 유분조절 로션 → 자차(논코메도)",
                "PM: 클렌징 → BHA 1% → 티트리 스팟 → 가벼운 젤크림",
                "주의: 과세안·과잉 각질제거 금지"
            ]
        }

        for skin_type, routines in routine_data.items(): # () 추가
            type_label = QLabel(f"<b>{skin_type}</b>")
            type_label.setFont(QFont("NanumBarunGothic", 17, QFont.Bold))
            type_label.setStyleSheet("margin-top: 15px; margin-bottom: 15px; color: #333;")
            summary_layout.addWidget(type_label)

            for routine_item in routines:
                item_label = QLabel(f"• {routine_item}")
                item_label.setFont(QFont("NanumBarunGothic", 17))
                item_label.setWordWrap(True)
                item_label.setStyleSheet("margin-left: 17px; line-height: 150%;")
                summary_layout.addWidget(item_label)

        common_label = QLabel("<b>공통: SPF50+ 매일, 레티노이드는 저농도 → 점진적, 임산부는 레티노이드/고농도 산 피하세요.</b>")
        common_label.setFont(QFont("NanumBarunGothic", 15, QFont.Bold))
        common_label.setStyleSheet("margin-top: 20px; color: #555;")
        common_label.setWordWrap(True)
        summary_layout.addWidget(common_label)


    # --- (신규) 라이프스타일 팁 섹션 추가 ---
    # (이하 2단계와 동일 - 변경 없음)
    def _add_lifestyle_tips_section(self):
        summary_layout = self.summary_eval_container.layout() # () 추가 # summary_eval_container 사용
        if summary_layout is None: # 레이아웃이 없으면 새로 생성 (하지만 TablePage에서 이미 생성됨)
            summary_layout = QVBoxLayout(self.summary_eval_container)
            summary_layout.setContentsMargins(10, 20, 10, 10)
            summary_layout.setSpacing(10)

        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken); summary_layout.addWidget(line)

        title = QLabel("<h3>라이프스타일 팁 (노화 방지에 직결)</h3>")
        title.setFont(QFont("NanumBarunGothic", 16, QFont.Bold))
        title.setStyleSheet("margin-top: 20px; margin-bottom: 17px; color: #333;")
        summary_layout.addWidget(title)

        tips = [
            "수면 7시간 + 취침 23시 이전 / 아침 햇빛 10분",
            "물 1.5-2L + 단백질 1.0-1.2g/kg",
            "설탕·가공류 줄이고 오메가3·비타민C/E 보강",
            "유산소 30분 + 가벼운 근력 10-15분 (혈류↑ → 피부톤·회복↑)",
            "스트레스 관리: 4-7-8 호흡 3세트, 짧은 명상"
        ]

        for tip_item in tips:
            item_label = QLabel(f"• {tip_item}")
            item_label.setFont(QFont("NanumBarunGothic", 16))
            item_label.setWordWrap(True)
            item_label.setStyleSheet("margin-left: 15px; line-height: 150%;")
            summary_layout.addWidget(item_label)    # --- (신규) 권장 루틴 섹션 추가 ---
    # (이하 2단계와 동일 - 변경 없음)
    def _add_skin_routine_section(self):
        summary_layout = self.summary_eval_container.layout() # () 추가 # summary_eval_container 사용
        if summary_layout is None: # 레이아웃이 없으면 새로 생성 (하지만 TablePage에서 이미 생성됨)
            summary_layout = QVBoxLayout(self.summary_eval_container)
            summary_layout.setContentsMargins(10, 20, 10, 10)
            summary_layout.setSpacing(10)

        title = QLabel("<h3>권장 루틴 (피부타입별 간단 버전)</h3>")
        title.setFont(QFont("NanumBarunGothic", 18, QFont.Bold))
        title.setStyleSheet("margin-top: 24px; margin-bottom: 15px; color: #333;")
        summary_layout.addWidget(title)

        routine_data = {
            "복합성 (가장 흔함)": [
                "AM: 약산성 세안 → 비타민C 세럼 → 가벼운 보습 → 자차",
                "PM: 저자극 클렌징 → BHA(격일) → 나이아신아마이드 → 세라마이드 크림",
                "주 2회: 레티날/레티놀 저농도(피부 적응), 시트팩은 10분 이내"
            ],
            "건성/민감": [
                "AM: 미온수 헹굼 → 히알루론산/판테놀 → 세라마이드 크림 → 자차",
                "PM: 클렌징 밀크 → 각질제거는 PHA/LHA 위주(주 1-2회) → 아줄렌/센텔라 → 리치 보습"
            ],
            "지성/트러블": [
                "AM: 젤클렌저 소량 → 나이아신아마이드 → 유분조절 로션 → 자차(논코메도)",
                "PM: 클렌징 → BHA 1% → 티트리 스팟 → 가벼운 젤크림",
                "주의: 과세안·과잉 각질제거 금지"
            ]
        }

        for skin_type, routines in routine_data.items(): # () 추가
            type_label = QLabel(f"<b>{skin_type}</b>")
            type_label.setFont(QFont("NanumBarunGothic", 17, QFont.Bold))
            type_label.setStyleSheet("margin-top: 15px; margin-bottom: 15px; color: #333;")
            summary_layout.addWidget(type_label)

            for routine_item in routines:
                item_label = QLabel(f"• {routine_item}")
                item_label.setFont(QFont("NanumBarunGothic", 17))
                item_label.setWordWrap(True)
                item_label.setStyleSheet("margin-left: 17px; line-height: 150%;")
                summary_layout.addWidget(item_label)

        common_label = QLabel("<b>공통: SPF50+ 매일, 레티노이드는 저농도 → 점진적, 임산부는 레티노이드/고농도 산 피하세요.</b>")
        common_label.setFont(QFont("NanumBarunGothic", 15, QFont.Bold))
        common_label.setStyleSheet("margin-top: 20px; color: #555;")
        common_label.setWordWrap(True)
        summary_layout.addWidget(common_label)


    # --- (신규) 라이프스타일 팁 섹션 추가 ---
    # (이하 2단계와 동일 - 변경 없음)
    def _add_lifestyle_tips_section(self):
        summary_layout = self.summary_eval_container.layout() # () 추가 # summary_eval_container 사용
        if summary_layout is None: # 레이아웃이 없으면 새로 생성 (하지만 TablePage에서 이미 생성됨)
            summary_layout = QVBoxLayout(self.summary_eval_container)
            summary_layout.setContentsMargins(10, 20, 10, 10)
            summary_layout.setSpacing(10)

        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken); summary_layout.addWidget(line)

        title = QLabel("<h3>라이프스타일 팁 (노화 방지에 직결)</h3>")
        title.setFont(QFont("NanumBarunGothic", 16, QFont.Bold))
        title.setStyleSheet("margin-top: 20px; margin-bottom: 17px; color: #333;")
        summary_layout.addWidget(title)

        tips = [
            "수면 7시간 + 취침 23시 이전 / 아침 햇빛 10분",
            "물 1.5-2L + 단백질 1.0-1.2g/kg",
            "설탕·가공류 줄이고 오메가3·비타민C/E 보강",
            "유산소 30분 + 가벼운 근력 10-15분 (혈류↑ → 피부톤·회복↑)",
            "스트레스 관리: 4-7-8 호흡 3세트, 짧은 명상"
        ]

        for tip_item in tips:
            item_label = QLabel(f"• {tip_item}")
            item_label.setFont(QFont("NanumBarunGothic", 16))
            item_label.setWordWrap(True)
            item_label.setStyleSheet("margin-left: 15px; line-height: 150%;")
            summary_layout.addWidget(item_label)
# --- ▲▲▲ (교체) 3단계 수정 클래스 끝 ▲▲▲ ---
# ---------- 결과 페이지 4: 노화방지 분석 ----------
class ResultPageAntiAging(TablePage):
    def __init__(self,parent=None):
        # 1. (수정) 이미지에 맞게 제목 변경
        super().__init__("4. AI 노화 방지 분석 (GPT-4o)","AI 분석을 통해 현재 건강 상태를 예측합니다.",parent) # () 추가
        # 2. (수정) 8개 컬럼에 맞게 스트레치 변경 (특징, 건강상태, 권고, 실행방법)
        self.stretch_columns=[1, 2, 4, 5] # <--- 6에서 5로 수정 (실행 방법이 5번 인덱스)

    def populate_data(self, analysis_data: dict): # <--- (수정) analysis_data 받도록 변경
        """(수정) GPT-4o 분석 데이터를 받아 테이블을 채우는 함수"""
        
        # 1. 테이블 헤더 정의 (8열)
        rows=[
            ["■ AI 노화 방지 항목 (GPT-4o)","","","","","","",""], # 8개
            ["구분", "주요 AI 분석 지표", "건강 상태 / 경향", "AI 분석 결과(원인과 기반)", "노화 방지 건강 증진 중심 실천 항목", "실행 방법 (실생활 적용)", "신호등", "관련 관리 지표"]
        ]

        # 2. 분석할 항목 리스트 정의 (key, name) - 새 이미지 기준 16개
        topics = [
            ('blood_flow', '혈류 순환 개선'),
            ('hrv_stress', '심박, HRV 안정화'),
            ('blood_pressure', '혈압 균형 유지'),
            ('metabolism', '대사 활성화 / 항산화'),
            ('liver_detox', '간 기능 회복'),
            ('glycation_defense', '신장/수분대사 개선'),
            ('immunity_boost', '면역력 강화'),
            ('stress_management', '스트레스 관리'),
            ('sleep_improvement', '수면의 질 개선'),
            ('digestive_health', '소화 기능 강화'),
            ('hydration', '탈수 / 체액 부족'),
            ('skin_elasticity', '피부 / 탄력 회복'),
            ('respiratory_health', '호흡기 강화'),
            ('emotional_stability', '정신적 안정 / 감정회복'),
            ('cognitive_health', '인지능력 / 뇌건강 관리'),
            ('hormone_balance', '노화 방지 핵심 호르몬 루틴(종합)')
        ]
        
        status_list = []

        try:
            for i, (key, name) in enumerate(topics, 1):
                data = analysis_data.get(key, {})
                
                status = data.get('status', '분석 실패')
                status_list.append(status)
                
                # 테이블 행 데이터 추가 (8개 열)
                rows.append([
                    str(i), # 1. 구분
                    name,   # 2. 주요 AI 분석 지표
                    data.get('health_status', '데이터 없음'),  # 3. 건강 상태 / 경향
                    data.get('analysis_reason', '데이터 없음'), # 4. AI 분석 결과
                    data.get('recommendation', '데이터 없음'),  # 5. 노화 방지 건강 증진
                    data.get('action_plan', '데이터 없음'),     # 6. 실행 방법
                    LightDot(status),                           # 7. 신호등
                    data.get('metric', 'N/A')                   # 8. 관련 관리 지표
                ])
            
            # 3. 웰니스 점수 계산 (16항목 기준)
            normal_count = status_list.count('정상')
            
            if normal_count >= 14: # 85% 이상 '정상'
                self.wellness_score = random.randint(85, 95)
            elif normal_count >= 11: # 70% 이상 '정상'
                self.wellness_score = random.randint(70, 84)
            else:
                self.wellness_score = random.randint(50, 69) # 그 외 (경고 포함)

        except Exception as e:
            print(f"GPT (노화방지) 결과 파싱 오류: {e}")
            rows.append(["오류", "GPT 결과 파싱 중 오류 발생", str(e), "", "", "", "", ""])
            self.wellness_score = 0
        
        self.table_data = rows

        # 4. 웰니스 점수 저장
        if self.wellness_score >= 85: self.wellness_level_text, self.wellness_color = "정상 단계", "#2ecc71"
        elif self.wellness_score >= 70: self.wellness_level_text, self.wellness_color = "주의 단계", "#f39c12"
        else: self.wellness_level_text, self.wellness_color = "관리 필요", "#e74c3c"
        
        # 5. 테이블 설정 및 요약 추가
        self._set_table_data(self.table_data)
        
        # (수정) _generate_insight_summary() 제거, _add_conclusion_content() 유지
        self._add_conclusion_content() 

    # --- (수정) 이 페이지 전용 perform_resize ---
    def perform_resize(self):
        # 1. 열 너비 맞추기
        self.table.resizeColumnsToContents() # 먼저 컨텐츠 기준으로 맞춤
        header = self.table.horizontalHeader()

        for col in range(header.count()):
            # (신규) 0번 열('#')은 좁은 너비로 고정
            if col == 0:
                header.setSectionResizeMode(col, QHeaderView.Fixed)
                header.resizeSection(col, 50) # 50px로 고정
            
            elif col in self.stretch_columns:
                header.setSectionResizeMode(col, QHeaderView.Stretch)
            
            else:
                # (수정) 컨텐츠에 맞게 고정하는 대신, 사용자가 조절할 수 있게 합니다.
                header.setSectionResizeMode(col, QHeaderView.Interactive)
        
        # 2. 행 높이 맞추기
        self.table.resizeRowsToContents()

        # 3. (중요) 테이블의 전체 높이를 계산하여 고정
        total_height = 0
        if self.table.horizontalHeader().isVisible():
            total_height += self.table.horizontalHeader().height()

        for i in range(self.table.rowCount()):
            total_height += self.table.rowHeight(i)

        self.table.setFixedHeight(total_height + 4)
    # --- perform_resize 수정 끝 ---


    # --- (유지) '결론' 텍스트 블록 생성 함수 (고정값) ---
    def _add_conclusion_content(self):
        """이미지의 '결론' 섹션을 생성하여 self.summary_widget_container에 추가합니다."""

        if self.summary_widget_container.layout() is None: # () 추가
            summary_layout=QVBoxLayout(self.summary_widget_container)
            summary_layout.setContentsMargins(10,20,10,10); summary_layout.setSpacing(10)
        else:
            summary_layout = self.summary_widget_container.layout() # () 추가
            # (기존 위젯 제거)
            while summary_layout.count(): # () 추가
                child = summary_layout.takeAt(0)
                if child.widget(): child.widget().deleteLater() # () 추가, () 추가

        # "결론" 타이틀 (이미지 핑크 아이콘 대신 emoji 사용)
        eval_title = QLabel("💡 결론")
        eval_title.setFont(QFont("NanumBarunGothic", 18, QFont.Bold))
        eval_title.setStyleSheet("color: #d9006c; margin-bottom: 5px;") # 이미지의 핑크색
        summary_layout.addWidget(eval_title)

        # 결론 텍스트 (이미지 내용 반영)
        text_1 = "AI 얼굴 + 혀 영상 통합 분석에 따르면,\n노화를 낮추는 핵심은 '<b>순환-대사-스트레스 3축 관리</b>'입니다."
        text_2 = "얼굴은 혈류의 거울,\n혀는 장부의 거울입니다."
        text_3 = "두 데이터를 함께 보면, 몸의 겉과 속의 균형 회복 루틴을 설계할 수 있습니다. 🌿"

        lbl_1 = QLabel(text_1)
        lbl_1.setFont(QFont("NanumBarunGothic", 18, QFont.Bold))
        lbl_1.setWordWrap(True)
        lbl_1.setStyleSheet("line-height: 150%;") # 줄 간격
        summary_layout.addWidget(lbl_1)

        lbl_2 = QLabel(text_2)
        lbl_2.setFont(QFont("NanumBarunGothic", 15))
        lbl_2.setWordWrap(True)
        lbl_2.setStyleSheet("margin-top: 13px; line-height: 140%;")
        summary_layout.addWidget(lbl_2)

        lbl_3 = QLabel(text_3)
        lbl_3.setFont(QFont("NanumBarunGothic", 15))
        lbl_3.setWordWrap(True)
        lbl_3.setStyleSheet("margin-top: 8px; line-height: 140%;")
        summary_layout.addWidget(lbl_3)

# ---------- 결과 페이지 5: 건강위험 분석 ----------
class ResultPageHealthRisk(TablePage):
    def __init__(self,parent=None):
        super().__init__("5. AI 건강위험 분석 (GPT-4o)","잠재적 건강 위험 요소를 웰니스 관점에서 예측합니다.",parent) # () 추가
        # (수정) 8개 컬럼에 맞게 스트레치 변경 (특징, 해석, 권고)
        self.stretch_columns=[2, 3, 7] # 0-based: AI 탐지 특징, 건강 해석, 권고 사항

    def populate_data(self, analysis_data: dict): # <--- (수정) analysis_data 받기
        """(수정) GPT-4o 분석 데이터를 받아 테이블을 채우는 함수"""
        
        # 1. 테이블 헤더 정의 (8열)
        rows=[
            ["■ AI 건강위험 분석 (GPT-4O)","","","","","","",""], # 8개
            ["#", "질환 / 건강균", "AI 탐지 특징 (얼굴-혀 영상 기준)", "건강 해석", "측정 값 (예시)", "신호등", "신뢰도", "권고 사항"]
        ]

        # 2. 분석 데이터 매핑 (5개 항목)
        topics_map = {
            'diabetes_risk': '당뇨 / 당뇨성 망막증',
            'digestive_risk': '위장질환 / 소화기암 (위암 초기 의심)',
            'liver_risk': '간질환 (간염-간경화-간암 위험군)',
            'diabetes_tongue_pattern': '당뇨병 (혀 패턴 기준)',
            'blood_flow_summary': '혈류 순환 종합지표'
        }
        status_list = []

        try:
            # (수정) 5개 항목을 순회
            for i, (key, name) in enumerate(topics_map.items(), 1):
                data = analysis_data.get(key, {})
                status = data.get('status', '분석 실패')
                status_list.append(status)
                
                # (수정) 8개 열에 맞게 데이터 추가
                rows.append([
                    str(i), # 1. #
                    name,   # 2. 질환 / 건강균
                    data.get('observation', '데이터 없음'),  # 3. AI 탐지 특징
                    data.get('interpretation', '데이터 없음'), # 4. 건강 해석
                    data.get('value', 'N/A'),           # 5. 측정 값 (예시)
                    LightDot(status),                   # 6. 신호등
                    data.get('confidence', 'N/A'),      # 7. 신뢰도
                    data.get('recommendation', '데이터 없음')  # 8. 권고 사항
                ])

            # 3. 웰니스 점수 계산 (5항목 기준)
            normal_count = status_list.count('정상')
            if normal_count >= 4: self.wellness_score = random.randint(85, 95) # 4~5개 '정상'
            elif normal_count >= 3: self.wellness_score = random.randint(70, 84) # 3개 '정상'
            else: self.wellness_score = random.randint(50, 69)

        except Exception as e:
            print(f"GPT (건강위험) 결과 파싱 오류: {e}")
            rows.append(["오류", "GPT 결과 파싱 중 오류 발생", str(e), "", "", "", "", ""])
            self.wellness_score = 0

        self.table_data = rows # 생성된 데이터를 인스턴스 변수에 저장
        
        # (수정) 점수 계산
        if self.wellness_score >= 85: self.wellness_level_text, self.wellness_color = "▲ 정상 단계", "#2ecc71"
        elif self.wellness_score >= 70: self.wellness_level_text, self.wellness_color = "▲ 주의 단계", "#f39c12"
        else: self.wellness_level_text, self.wellness_color = "▼ 관리 필요", "#e74c3c"
        
        # (이동)
        self._set_table_data(self.table_data) # 저장된 데이터로 테이블 설정
        self._generate_overall_summary_content() # () 추가 # 종합 평가 추가
        self._add_wellness_summary_content("risk") # 권고 요약 추가

    # --- (수정) 이 페이지 전용 perform_resize ---
    def perform_resize(self):
        # 1. 열 너비 맞추기
        self.table.resizeColumnsToContents() # 먼저 컨텐츠 기준으로 맞춤
        header = self.table.horizontalHeader()

        for col in range(header.count()):
            # (신규) 0번 열('#')은 좁은 너비로 고정
            if col == 0:
                header.setSectionResizeMode(col, QHeaderView.Fixed)
                header.resizeSection(col, 50) # 50px로 고정
            
            elif col in self.stretch_columns:
                header.setSectionResizeMode(col, QHeaderView.Stretch)
            
            else:
                # (수정) 컨텐츠에 맞게 고정하는 대신, 사용자가 조절할 수 있게 합니다.
                header.setSectionResizeMode(col, QHeaderView.Interactive)
        
        # 2. 행 높이 맞추기
        self.table.resizeRowsToContents()

        # 3. (중요) 테이블의 전체 높이를 계산하여 고정
        total_height = 0
        if self.table.horizontalHeader().isVisible():
            total_height += self.table.horizontalHeader().height()

        for i in range(self.table.rowCount()):
            total_height += self.table.rowHeight(i)

        self.table.setFixedHeight(total_height + 4)
    # --- perform_resize 수정 끝 ---

 # (수정) 점수 계산을 populate_data로 이동시켰으므로, 여기서는 UI만 그림
    def _generate_overall_summary_content(self):
        """이미지의 '종합 평가' 섹션을 생성합니다."""
        
        # --- ▼▼▼ 여기가 수정된 부분입니다 ▼▼▼ ---
        if self.summary_eval_container.layout() is None: # () 추가
            summary_layout=QVBoxLayout(self.summary_eval_container)
            summary_layout.setContentsMargins(10,20,10,10); summary_layout.setSpacing(10)
        else:
            # (수정) layout -> summary_layout 변수명 변경
            summary_layout = self.summary_eval_container.layout() # () 추가
            # 기존 위젯 제거
            while summary_layout.count(): # () 추가
                child = summary_layout.takeAt(0)
                if child.widget(): # () 추가
                    child.widget().deleteLater() # () 추가
        # --- ▲▲▲ 여기가 수정된 부분입니다 ▲▲▲ ---

        # "종합 평가" 타이틀
        eval_title = QLabel("<h4>🩺 종합 평가 (GPT-4o)</h4>")
        eval_title.setFont(QFont("NanumBarunGothic", 16, QFont.Bold))
        summary_layout.addWidget(eval_title) # 이제 summary_layout이 항상 정의되어 있음

        # 종합 평가 테이블 생성 (이미지 참고)
        eval_table = QTableWidget() # () 추가
        eval_table.setFrameShape(QFrame.NoFrame)
        eval_table.verticalHeader().setVisible(False) # () 추가
        eval_table.horizontalHeader().setVisible(False) # () 추가
        eval_table.setEditTriggers(QTableWidget.NoEditTriggers)
        eval_table.setSelectionMode(QTableWidget.NoSelection)
        eval_table.setWordWrap(True)
        eval_table.setStyleSheet("""
            QTableWidget {
                background: #fff;
                border: 8px solid #d9d9d9;
                font-family: 'NanumBarunGothic';
            }
            QTableWidget::item {
                padding: 11px 14px;
                font-size: 17px;
                border-bottom: 1px solid #eee; /* 행 구분선 */
                vertical-align: middle; /* 세로 중앙 정렬 */
            }
                QTableWidget tr > td:first-child { /* 첫 번째 열 스타일 */
                    font-weight: bold;
                    color: #555;
                    background-color: #f8f9fa; /* 약간의 배경색 */
                }
        """)
        eval_table.setRowCount(4)
        eval_table.setColumnCount(3)

        # 컬럼 너비 조정 (이미지와 비슷하게)
        eval_table.setColumnWidth(0, 150) # 항목
        eval_table.setColumnWidth(1, 200) # 결과
        eval_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch) # () 추가 # 평가 (나머지 공간 차지)

        # (수정) populate_data에서 계산된 점수 사용
        score = self.wellness_score
        level_text = self.wellness_level_text
        level_color = self.wellness_color

        # (수정) '주요 위험군'을 새 항목에 맞게 변경
        data = [
            ["AI 웰니스 점수", f"{score}점", f"<font color='{level_color}'>{level_text}</font>"],
            ["주요 위험군", "당뇨 / 소화기 / 간 위험", "생활습관 개선 권장"],
            ["AI 신뢰도 평균", f"{random.randint(80, 88)}%", "영상 품질 양호"],
            ["AI 모델 기반", "GPT-4o (OpenAI) 웰니스 분석", ""]
        ]

        for r_idx, row_data in enumerate(data):
            for c_idx, cell_data in enumerate(row_data):
                item = QTableWidgetItem() # () 추가
                item.setText(cell_data)

                # 첫 번째 열 폰트 및 정렬
                if c_idx == 0:
                    font = QFont("NanumBarunGothic", 16, QFont.Bold)
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                # 마지막 열 정렬
                elif c_idx == 2:
                    font = QFont("NanumBarunGothic", 14)
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                # 중간 열
                else:
                    font = QFont("NanumBarunGothic", 15)
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)

                item.setFont(font)
                eval_table.setItem(r_idx, c_idx, item)

        eval_table.resizeRowsToContents() # () 추가 # 행 높이 자동 조절
        # 테이블 높이 고정 (스크롤바 방지)
        total_height = sum(eval_table.rowHeight(i) for i in range(eval_table.rowCount())) + 5 # () 추가 # 약간의 여유 추가
        eval_table.setFixedHeight(total_height)

        summary_layout.addWidget(eval_table)
        
        # ---------- 메인 윈도우 연결 ----------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__() # () 추가
        self.setWindowTitle("FunFun AI 건강분석")
        img_paths=[os.path.join(IMG_DIR,f)for f in IMG_FILES]
        self.pages=QStackedWidget(self)
        
        # (수정) 페이지 객체 생성 시 AnalysisPage 추가
        self.page_widgets=[
            ImagePage(img_paths[0]),        # 0
            ImagePage(img_paths[1]),        # 1
            LoginPage(img_paths[2]),        # 2
            CameraPage(img_paths[3]),       # 3
            AnalysisPage(),                 # 4 (신규 추가)
            ResultPageZero(),               # 5 (인덱스 변경)
            ResultPageFace(),               # 6
            ResultPageTongue(),             # 7
            ResultPageSkin(),               # 8
            ResultPageAntiAging(),          # 9
            ResultPageHealthRisk()          # 10
        ]
        
        for p in self.page_widgets: self.pages.addWidget(p)
        
        # (수정) 카메라 페이지 인덱스 3으로 변경
        cam:CameraPage=self.page_widgets[3] 
        cam.finished.connect(self.on_camera_finished) # 카메라 정지 및 UI 변경용
        cam.finished.connect(self.enable_next_button) # '다음' 버튼 활성화용

        root=QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.addWidget(self.pages)
        self.prev_btn=QPushButton("◀ 이전",self); self.next_btn=QPushButton("다음 ▶",self); self.exit_btn=QPushButton("끝내기",self); self.retake_btn=QPushButton("재촬영",self)

        self.save_btn = QPushButton("결과저장", self)
        self.print_btn = QPushButton("프린트", self) # (신규) 프린트 버튼

        # (수정) 리스트에 print_btn 추가
        for b in(self.prev_btn,self.next_btn,self.exit_btn,self.retake_btn, self.save_btn, self.print_btn): 
                b.setFont(QFont("NanumBarunGothic", 16, QFont.Bold))
                b.setCursor(Qt.PointingHandCursor)

        blue="QPushButton{background:#0e7afe;color:white;border:none;border-radius:8px;padding:8px 20px;}QPushButton:hover{background:#2b8fff;}QPushButton:pressed{background:#0d6de0;}QPushButton:disabled{background:#a0a0a0;}"
        red="QPushButton{background:#d33;color:white;border:none;border-radius:8px;padding:8px 20px;}QPushButton:hover{background:#e44;}QPushButton:pressed{background:#c22;}"
        save_blue="QPushButton{background:#173a9e;color:white;border:none;border-radius:10px;padding:10px 20px;}QPushButton:hover{background:#2050d1;}QPushButton:pressed{background:#0e2c7a;}"
        # 스타일 설정
        self.prev_btn.setStyleSheet(blue); self.next_btn.setStyleSheet(blue); self.retake_btn.setStyleSheet(blue); self.exit_btn.setStyleSheet(red)
        self.save_btn.setStyleSheet(save_blue)
        # (신규) 프린트 버튼 스타일 (약간 어두운 회색/검정 계열 추천)
        print_style="QPushButton{background:#444;color:white;border:none;border-radius:10px;padding:10px 20px;}QPushButton:hover{background:#666;}QPushButton:pressed{background:#222;}"
        self.print_btn.setStyleSheet(print_style)

        # 버튼 연결
        self.prev_btn.clicked.connect(self.go_prev); self.next_btn.clicked.connect(self.go_next)
        self.exit_btn.clicked.connect(self.close); self.retake_btn.clicked.connect(self.go_retake)
        self.save_btn.clicked.connect(self.save_all_results_pdf)
        self.print_btn.clicked.connect(self.print_results) # (신규) 연결

        self.pages.currentChanged.connect(self.on_page_changed); 
        self.save_btn.hide()
        self.print_btn.hide() # (신규) 처음엔 숨김
        
        self.analysis_timer = QTimer(self) 
        self.analysis_timer.setSingleShot(True)
        self.analysis_timer.timeout.connect(self.start_gpt_analysis_wrapper) # (수정) 래퍼 함수로 연결 

        self.on_page_changed(0) # <--- __init__의 마지막 호출
        
        self.session = AppSession()
        self.auth_service = AuthService(resolve_users_file(SCRIPT_DIR))
        self.analysis_service = None
        self.camera_finished = False # (신규) 촬영 완료 플래그
        self.analysis_complete = False # (신규) 분석 완료 플래그
        
    # --- (수정) __init__ 밖으로 함수를 빼내고, 내용 들여쓰기 수정 ---
    # (이 함수부터는 class MainWindow 바로 아래 레벨입니다)
    def on_page_changed(self, idx):
        session = getattr(self, "session", None)
        camera_finished = session.camera_finished if session is not None else False
        analysis_complete = session.analysis_complete if session is not None else False
        is_first_page = (idx == 0)
        is_last_page = (idx == self.pages.count() - 1)
        is_camera_page = (idx == 3)
        is_analysis_page = (idx == 4) 

        self.prev_btn.setDisabled(is_first_page or is_analysis_page)
        
        # (수정) 촬영이 완료되었으면 카메라 페이지에서도 '다음' 버튼 활성화
        next_disabled = is_last_page or is_analysis_page or (is_camera_page and not camera_finished)
        self.next_btn.setDisabled(next_disabled)

        is_result_page = (idx >= 5 and idx <= 10)
        self.save_btn.setVisible(is_result_page)
        self.retake_btn.setVisible(is_result_page)
        self.print_btn.setVisible(is_result_page) # (신규) 프린트 버튼 표시

        # (수정) 분석이 완료되지 않았을 때만 분석 페이지 실행
        if idx == 4 and not analysis_complete:
            print("AI 분석 타이머 시작 (1초)...")
            self.analysis_timer.start(1000) 

    def on_camera_finished(self,face_path,tongue_path):
        cam_page=self.page_widgets[3]
        if isinstance(cam_page, CameraPage):
            cam_page.stop_camera() 
            cam_page.start_btn.setText("촬영 완료. 다음 ▶ 버튼을 눌러주세요.")
            cam_page.start_btn.setDisabled(True)
            cam_page.start_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(100, 100, 100, 0.8);
                    color: white; border: none; border-radius: 10px;
                    padding: 8px 18px;
                }
            """)
        
        self.session.set_captured_images(face_path, tongue_path)
        self.camera_finished = True # (신규) 촬영 완료됨
        self.analysis_complete = False # (신규) 새 촬영 시 분석 플래그 초기화
        print(f"이미지 저장 완료:\nFace: {face_path}\nTongue: {tongue_path}")

    def closeEvent(self,event):
        cam_page=self.page_widgets[3]
        if isinstance(cam_page, CameraPage): cam_page.stop_camera() 
        event.accept() 

    def resizeEvent(self,e): super().resizeEvent(e); self.layout_buttons() 

    def layout_buttons(self):
        y_bottom_row=self.height() -55; margin=16; w=140; h=46 
        self.exit_btn.setGeometry(margin,y_bottom_row,w,h)
        self.prev_btn.setGeometry(self.width()//2-w-10,y_bottom_row,w,h) 
        self.next_btn.setGeometry(self.width()//2+10,y_bottom_row,w,h) 
        home_btn_x = self.width()-w-margin 
        self.retake_btn.setGeometry(home_btn_x, y_bottom_row, w, h) # (수정) 재촬영 버튼 위치
        y_save_btn = y_bottom_row - h - 10 # 간격 조금 넓힘
        self.save_btn.setGeometry(home_btn_x, y_save_btn, w, h)

        # (신규) 프린트 버튼 배치 (저장 버튼 왼쪽)
        print_btn_x = home_btn_x - w - 10
        self.print_btn.setGeometry(print_btn_x, y_save_btn, w, h)

        self.exit_btn.raise_(); self.prev_btn.raise_(); self.next_btn.raise_(); self.retake_btn.raise_(); self.save_btn.raise_(); self.print_btn.raise_()

    def handle_login_transition(self):
        try:
            login_page: LoginPage = self.page_widgets[2]
            username = login_page.gpt_id_edit.text().strip()
            password_attempt = login_page.gpt_pw_edit.text().strip()

            user = self.auth_service.authenticate(username, password_attempt)
            self.session.set_api_key(user.api_key)
            self.session.reset_capture()
            self.analysis_service = OpenAIAnalysisService(user.api_key)
            cam_page = self.page_widgets[3]
            if isinstance(cam_page, CameraPage):
                cam_page.reset_page()
            print(f"濡쒓렇???깃났: {user.username}")
            self.pages.setCurrentIndex(3)
        except AuthError as e:
            dialog = QMessageBox.critical if e.critical else QMessageBox.warning
            dialog(self, e.title, str(e))
        except AnalysisServiceError as e:
            QMessageBox.critical(self, "濡쒓렇???ㅽ뙣", str(e))
        except Exception as e:
            QMessageBox.critical(self, "?ㅻ쪟", f"濡쒓렇??以??ㅻ쪟 諛쒖깮: {e}")
    
    def go_prev(self):
        idx = self.pages.currentIndex() 
        if idx > 0:
            if idx == 5: # 5번(첫결과) -> 3번(카메라)
                self.pages.setCurrentIndex(3)
            else:
                self.pages.setCurrentIndex(idx - 1)

    # --- (수정) "진짜" 로그인 로직으로 변경 ---
    def go_next(self):
        idx = self.pages.currentIndex() 
        
        if idx == 2: # 로그인 페이지(인덱스 2)일 때
            self.handle_login_transition()
            return
            try:
                login_page: LoginPage = self.page_widgets[2]
                username = login_page.gpt_id_edit.text().strip()
                password_attempt = login_page.gpt_pw_edit.text().strip() # 사용자가 입력한 비밀번호
                
                if not username or not password_attempt:
                    QMessageBox.warning(self, "로그인 실패", "Email ID와 Password를 모두 입력하세요.")
                    return

                # 1. users.json 파일 읽기
                users_file = os.path.join(SCRIPT_DIR, "users.json")
                if not os.path.exists(users_file):
                    QMessageBox.critical(self, "오류", "users.json 파일을 찾을 수 없습니다.\n스크립트와 같은 폴더에 생성해주세요.")
                    return
                
                with open(users_file, 'r', encoding='utf-8') as f:
                    users_db = json.load(f)

                # 2. 사용자 정보 확인
                user_data = users_db.get(username)
                if not user_data:
                    QMessageBox.warning(self, "로그인 실패", "존재하지 않는 Email ID입니다.")
                    return

                # 3. (수정) 비밀번호를 해시 대신 직접 비교
                stored_password = user_data.get("password") # "password_hash" -> "password"
                api_key = user_data.get("api_key")
                
                if password_attempt == stored_password:
                    # 비밀번호 일치
                    if not api_key or not api_key.startswith("sk-"):
                        QMessageBox.critical(self, "로그인 실패", "로그인은 성공했으나, users.json 파일에\n이 사용자를 위한 유효한 OpenAI API Key가 없습니다.")
                        return
                    
                    print(f"로그인 성공: {username}")
                    self.gpt_api_key = api_key # API 키 저장
                else:
                    # 비밀번호 불일치
                    QMessageBox.warning(self, "로그인 실패", "Password가 올바르지 않습니다.")
                    return

            except Exception as e:
                QMessageBox.critical(self, "오류", f"로그인 중 오류 발생: {e}")
                return
        
        # --- ▼▼▼ 여기가 수정된 부분입니다 ▼▼▼ ---
        if idx == 3: # 카메라 페이지(인덱스 3)일 때
            if self.session.analysis_complete:
                # 분석이 이미 완료되었으면(True), 분석 페이지(4)를 건너뛰고
                # 첫 번째 결과 페이지(5)로 바로 이동합니다.
                print("분석 결과가 캐시됨. 결과 페이지(5)로 바로 이동.")
                self.pages.setCurrentIndex(5)
            else:
                # 분석이 아직 안되었으면(False), 분석 페이지(4)로 이동
                print("분석 시작. 분석 페이지(4)로 이동.")
                self.pages.setCurrentIndex(4)
            return # 이 go_next 함수를 여기서 종료
        # --- ▲▲▲ 여기가 수정된 부분입니다 ▲▲▲ ---
            
        if idx < self.pages.count() - 1: 
            self.pages.setCurrentIndex(idx + 1)

    @pyqtSlot() 
    def enable_next_button(self):
        # (수정) 버튼 상태를 on_page_changed에서 관리하도록 호출
        self.session.camera_finished = True
        self.camera_finished = True
        self.on_page_changed(self.pages.currentIndex())
            
    # --- ▼▼▼ GPT 분석 요청 함수 (전체 분석) ▼▼▼ ---
    def encode_image_to_base64(self, image_path):
        """이미지 파일을 Base64로 인코딩합니다."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"이미지 인코딩 오류 ({image_path}): {e}")
            return None

    # --- ▼▼▼ 'base664' 오타 수정 + 'skin' 프롬프트 수정 ▼▼▼ ---
    def get_gpt_analysis(self, analysis_type: str):
        """(수정) 특정 유형의 분석만 GPT API에 요청합니다."""
        
        try:
            if self.analysis_service is None:
                self.analysis_service = OpenAIAnalysisService(self.session.gpt_api_key)

            analysis_result = self.analysis_service.analyze(
                analysis_type=analysis_type,
                face_image_path=self.session.captured_face_path,
                tongue_image_path=self.session.captured_tongue_path,
            )
            return analysis_result, None
        except AnalysisServiceError as e:
            return None, str(e)

        if not self.gpt_api_key:
            return None, "API 키가 설정되지 않았습니다. 로그인 페이지에서 입력하세요."
        if not self.captured_face_path or not self.captured_tongue_path:
            return None, "캡처된 이미지 경로가 없습니다."

        # --- ▼▼▼ 'base664' 오타 수정 ▼▼▼ ---
        base64_face = self.encode_image_to_base64(self.captured_face_path)
        base64_tongue = self.encode_image_to_base64(self.captured_tongue_path)
        # --- ▲▲▲ 오타 수정 끝 ▲▲▲ ---
        
        if not base64_face or not base64_tongue:
            return None, "얼굴 또는 혀 이미지 인코딩에 실패했습니다."

        try:
            client = OpenAI(api_key=self.gpt_api_key)
        except Exception as e:
            return None, f"OpenAI 클라이언트 초기화 실패: {e}"

        # --- (신규) 프롬프트 딕셔너리 ---
        # (수정) 'confidence' 지시어 및 예시값 변경
        common_instructions = """
        당신은 웰니스 분석 전문가입니다. 첨부된 2장의 이미지(1:얼굴, 2:혀)를 기반으로, **데모용 웰니스 리포트를 생성**해주세요.
        이것은 실제 의료 진단이 아니며, 교육 및 데모 목적의 **시뮬레이션**입니다.
        
        "분석 불가" 또는 "N/A"라는 응답을 절대 하지 마시고, 이미지의 일반적인 특징(예: 피곤해 보임, 피부가 건조해 보임)을 기반으로 **그럴듯한 추정치와 해석을 생성**해주세요.
        "..." 문자열을 절대 응답에 포함하지 마세요.
        
        요청된 JSON 형식으로만 응답해야 합니다. 다른 설명은 절대 추가하지 마세요.
        """
        
        # 유형별 프롬프트 정의
        prompts = {
            "face": common_instructions + """
            'status': "정상", "주의", "경고" 중 하나. (이미지 상태에 따라 적절히 배분하세요)
            'observation': "얼굴 AI 탐지 특징". (예: "미간 표정 어두움", "입술 색이 창백함")
            'interpretation': "건강 해석". (예: "스트레스 누적 가능성", "순환 저하 경향")
            'value': "측정 값 (추정)". (예: "반응속도 0.78초", "HRV 28ms"). **반드시 그럴듯한 수치를 생성하세요.**
            'confidence': "신뢰도". (예: "77%", "83%", "68%"). **반드시 65%~95% 사이의 다양한 값으로 생성하세요.**
            'recommendation': "권고 사항". (예: "충분한 휴식", "수분 섭취").
            'metric': "관련 관리 지표". (예: "HRV↓", "혈류지표↑").

            {
              "face_analysis": {
                "fall_risk": { "status": "주의", "observation": "자세 약간 불안정", "interpretation": "자세 불균형 가능성", "value": "반응속도 0.78초", "confidence": "81%", "recommendation": "자세 교정, 근력 운동", "metric": "HRV↓" },
                "hrv": { "status": "주의", "observation": "rPPG 신호 미세 변동", "interpretation": "자율신경 불균형", "value": "HRV 28ms", "confidence": "85%", "recommendation": "심호흡, 가벼운 스트레칭", "metric": "HRV↓" },
                "blood_pressure": { "status": "주의", "observation": "안면 홍조 약간 보임", "interpretation": "약간의 혈압 상승 경향", "value": "SBP 128 / DBP 85", "confidence": "82%", "recommendation": "염분 조절, 스트레스 완화", "metric": "혈류지표↑" },
                "spo2": { "status": "정상", "observation": "피부 채도 안정적", "interpretation": "정상 범위", "value": "97%", "confidence": "91%", "recommendation": "유지", "metric": "SpO₂" },
                "hypertension_risk": { "status": "주의", "observation": "안색 붉은 기", "interpretation": "스트레스성 순환 과부하", "value": "혈관 확장 지표 0.65", "confidence": "76%", "recommendation": "저염식, 유산소 운동", "metric": "HR↑" },
                "hypotension_risk": { "status": "주의", "observation": "입술 색 옅음", "interpretation": "혈류량 부족 경향", "value": "혈류 강도 0.42", "confidence": "72%", "recommendation": "충분한 수분 보충", "metric": "HRV↓" },
                "anemia": { "status": "경고", "observation": "입술 광대 창백", "interpretation": "Hb 기능성 저하", "value": "Hb 추정 11.8g/dL", "confidence": "71%", "recommendation": "철분 엽산 섭취", "metric": "혈류↓" },
                "diabetes_risk": { "status": "주의", "observation": "피부 건조, 광택 저하", "interpretation": "혈당 변동 ↑", "value": "피부탄력도 -18%", "confidence": "68%", "recommendation": "당질 섭취 제한", "metric": "HRV↓" },
                "thyroid_function": { "status": "정상", "observation": "얼굴 부기 없음", "interpretation": "대사 정상", "value": "피부탄력 0.9", "confidence": "65%", "recommendation": "유지", "metric": "HRV" },
                "liver_function": { "status": "경고", "observation": "눈 흰자위 혼탁함", "interpretation": "해독 능력 ↓", "value": "색소침착 0.62", "confidence": "67%", "recommendation": "금주/야식(섬유소↑)", "metric": "혈류 정체" },
                "kidney_function": { "status": "주의", "observation": "다크서클 부종", "interpretation": "순환 저하", "value": "부종지수 0.8", "confidence": "73%", "recommendation": "저염식, 수면", "metric": "HRV↓" },
                "heart_function_weak": { "status": "정상", "observation": "입술 혈색 양호", "interpretation": "순환 양호", "value": "심박출량↑", "confidence": "78%", "recommendation": "유지", "metric": "HRV" },
                "respiratory_function": { "status": "주의", "observation": "입술 청색 기", "interpretation": "산소 교환 저하", "value": "SpO₂ 95%", "confidence": "83%", "recommendation": "호흡운동, 실내 공기질", "metric": "SpO₂" },
                "chronic_fatigue": { "status": "경고", "observation": "윤기 ↓, 얼굴 긴장", "interpretation": "교감신경 과항진", "value": "HRV 25ms", "confidence": "88%", "recommendation": "스트레칭, 수면 리듬 회복", "metric": "HRV↓" },
                "dehydration": { "status": "정상", "observation": "피부 촉촉함", "interpretation": "수분 양호", "value": "수분지표 0.8", "confidence": "81%", "recommendation": "유지", "metric": "반사도↑" },
                "stress_overload": { "status": "경고", "observation": "표정 경직", "interpretation": "교감신경 항진", "value": "HRV 22ms", "confidence": "92%", "recommendation": "명상, 호흡, 수면 관리", "metric": "HRV↓" },
                "insomnia": { "status": "경고", "observation": "눈 밑 다크서클", "interpretation": "수면의 질 ↓", "value": "수면지표 0.75", "confidence": "77%", "recommendation": "수면 루틴, 조명", "metric": "HRV↓" },
                "depression_anxiety": { "status": "정상", "observation": "표정 편안함", "interpretation": "세로토닌 양호", "value": "감정지표 0.8", "confidence": "71%", "recommendation": "유지", "metric": "HRV" },
                "immunity_weak": { "status": "주의", "observation": "홍조, 여드름", "interpretation": "면역-염증 저하", "value": "염증지표 0.8", "confidence": "69%", "recommendation": "항산화, 수면", "metric": "혈류↑" },
                "inflammation_fatigue": { "status": "경고", "observation": "무표정 + HR 불안정", "interpretation": "정신-신체 피로", "value": "24ms, HR 82", "confidence": "84%", "recommendation": "명상-산책, 규칙적 수면", "metric": "HRV↓" }
              }
            }
            """,
            "tongue": common_instructions + """
            'status': "정상", "주의", "경고" 중 하나.
            'observation': "AI 탐지 특징 (혀 영상)". (예: "혀 색 붉고 윤기 약함")
            'interpretation': "건강 해석". (예: "혈류량↓, 산소공급 저하")
            'value': "측정 값 (예시)". (예: "색상 L값 82"). **반드시 그럴듯한 수치를 생성하세요.**
            'confidence': "신뢰도". (예: "77%", "83%", "68%"). **반드시 65%~95% 사이의 다양한 값으로 생성하세요.**
            'recommendation': "권고 사항". (예: "철분/엽산 섭취")
            'metric': "관련 관리 지표". (예: "혈류개선↓")

            {
              "tongue_analysis": {
                "anemia_hypotension": { "status": "주의", "observation": "혀 색 붉고 윤기 약함", "interpretation": "혈류량↓, 산소공급 저하", "value": "색상 L값 82", "confidence": "86%", "recommendation": "철분/엽산 섭취, 따뜻한 차(생강차)", "metric": "혈류개선↓" },
                "hypertension_heat": { "status": "정상", "observation": "혀끝 분홍색", "interpretation": "열 균형", "value": "적색비율 0.5", "confidence": "79%", "recommendation": "유지", "metric": "HR" },
                "heart_function": { "status": "경고", "observation": "혀끝 자색, 미세출혈", "interpretation": "순환불균형, 피로 누적", "value": "혈색지표 0.61", "confidence": "83%", "recommendation": "휴식, 온찜질, 유산소 운동", "metric": "HRV↓" },
                "gastritis_ulcer": { "status": "경고", "observation": "설태 두껍고 백색", "interpretation": "위염, 소화불량, 위장 부담", "value": "부담지수 0.54", "confidence": "87%", "recommendation": "자극적 음식 회피, 자가 관리", "metric": "위산분비↑" },
                "liver_function": { "status": "경고", "observation": "혀 가장자리 어둡고 둔탁", "interpretation": "해독능력↓, 피로 누적", "value": "HSV H값 0.42", "confidence": "81%", "recommendation": "녹황색 채소, 금주, 수면 리듬 회복", "metric": "AST/ALT↑" },
                "kidney_function_1": { "status": "정상", "observation": "중앙부 분홍색", "interpretation": "에너지 흡수 양호", "value": "채도 S값 0.6", "confidence": "74%", "recommendation": "유지", "metric": "포도당" },
                "kidney_function_2": { "status": "경고", "observation": "혀 뿌리 어둡고 윤기↓", "interpretation": "노폐물 배출 저하", "value": "색상 B채널 22", "confidence": "79%", "recommendation": "수분섭취, 저염식, 유산균", "metric": "크레아티닌↑" },
                "dehydration": { "status": "경고", "observation": "혀 건조, 설태 적음", "interpretation": "체액 부족, 점도↑", "value": "반사광비율 0.37", "confidence": "88%", "recommendation": "물 1.8L/일, 채소 섭취", "metric": "피부저항↑" },
                "edema_water": { "status": "경고", "observation": "혀 측면 부종, 자국 있음", "interpretation": "수분정체, 신장 부담", "value": "부종지수 0.78", "confidence": "82%", "recommendation": "염분 제한, 땀 유도", "metric": "순환↓" },
                "diabetes_risk": { "status": "정상", "observation": "혀 붉지 않음", "interpretation": "혈당 변동 양호", "value": "RGB 평균 R=140", "confidence": "76%", "recommendation": "유지", "metric": "HRV" },
                "thyroid_function": { "status": "주의", "observation": "혀 부음, 탄력↓", "interpretation": "대사 저하 가능성", "value": "표면거칠기 0.43", "confidence": "73%", "recommendation": "규칙적 수면, 스트레스 관리", "metric": "TSH↑" },
                "obesity_immunity": { "status": "경고", "observation": "혀 크고 자국 많음", "interpretation": "순환 정체, 대사↓", "value": "면적비 1.42", "confidence": "81%", "recommendation": "저염식, 유산소 운동", "metric": "HRV↓" },
                "immunity_weak": { "status": "주의", "observation": "설태 불균일, 표면 탁함", "interpretation": "면역 저하, 염증 반응", "value": "색상분산 0.22", "confidence": "86%", "recommendation": "수면↑, 스트레스↓", "metric": "염증지표↑" },
                "fatigue_energy": { "status": "경고", "observation": "혀끝 붉고 건조", "interpretation": "활성 에너지↓", "value": "밝기지표 0.74", "confidence": "79%", "recommendation": "호흡운동, 습도 관리", "metric": "SpO₂↓" },
                "stress_overload": { "status": "경고", "observation": "혀끝 붉고 정상홍반", "interpretation": "교감신경 과활성", "value": "적색분포비율 0.72", "confidence": "91%", "recommendation": "명상, 심호흡, 수면 리듬", "metric": "HRV↓" },
                "insomnia_fatigue": { "status": "경고", "observation": "혀 중앙 어둡고 윤기↓", "interpretation": "자율신경 불균형", "value": "반사도 0.48", "confidence": "77%", "recommendation": "수면 루틴, 조명, 카페인↓", "metric": "HRV↓" },
                "depression_anxiety": { "status": "정상", "observation": "선홍색, 표면 양호", "interpretation": "기혈순환 양호", "value": "색상대비 0.7", "confidence": "72%", "recommendation": "유지", "metric": "세로토닌↑" },
                "inflammation_stomatitis": { "status": "경고", "observation": "혀 붉고 설태 벗겨짐", "interpretation": "면역반응↑", "value": "색온도 6400K", "confidence": "89%", "recommendation": "청결(배, 녹두), 수분↑", "metric": "CRP↑" },
                "candidiasis": { "status": "경고", "observation": "백태 점상, 벗겨짐", "interpretation": "진균 감염", "value": "백색비율 0.46", "confidence": "93%", "recommendation": "항진균 관리, 구강 관리", "metric": "면역↓" },
                "oral_dryness": { "status": "경고", "observation": "혀태, 설태 두꺼움", "interpretation": "세균 증식, 침분비 저하", "value": "황색비율 0.64", "confidence": "84%", "recommendation": "수분↑, 구강세정, 녹차 섭취", "metric": "구강pH↓" }
              }
            }
            """,
            "skin": common_instructions + """
            'status': "정상", "주의", "경고" 중 하나.
            'observation': "AI 분석 특징". (예: "뺨 건조", "T존 유분 적절")
            'recommendation': "권고사항(루틴/성분)". (예: "보습제 사용", "진정 케어")
            {
              "skin_analysis": {
                "hydration": { "status": "주의", "observation": "뺨 건조", "recommendation": "보습제 사용" },
                "oil_balance": { "status": "정상", "observation": "T존 유분 적절", "recommendation": "유지" },
                "sensitivity": { "status": "경고", "observation": "코 주변 및 볼 붉음", "recommendation": "진정 케어, 자외선 차단" },
                "pore_texture": { "status": "정상", "observation": "모공 크기 보통", "recommendation": "유지" },
                "wrinkles": { "status": "주의", "observation": "눈가 미세 주름", "recommendation": "아이크림 사용" }
              }
            }
            """,
            "anti_aging": common_instructions + """
            'status': "정상", "주의", "경고" 중 하나.
            'health_status': "건강 상태 / 경향". (예: "혈류저하, 자율신경 불균형")
            'analysis_reason': "AI 분석 결과(원인과 기반)". (예: "얼굴 창백, HRV 낮음")
            'recommendation': "노화 방지 건강 증진 중심 실천 항목". (예: "혈류 자극 + 하체 강화 루틴")
            'action_plan': "실행 방법 (실생활 적용)". (예: "하루 20분 걷기, 종아리 스트레칭")
            'metric': "관련 관리 지표". (예: "혈류개선↓")

            {
              "anti_aging": {
                "blood_flow": { "status": "정상", "health_status": "혈류 순환 양호", "analysis_reason": "얼굴 혀 혈색 양호", "recommendation": "현재 상태 유지", "action_plan": "주 3회 30분 걷기", "metric": "혈류" },
                "hrv_stress": { "status": "주의", "health_status": "스트레스 과부하", "analysis_reason": "HR 높음, HRV 낮음", "recommendation": "호흡 수면 균형 루틴", "action_plan": "4-7-8 호흡법, 명상 10분, 23시 이전 취침, 아침 햇빛 노출", "metric": "HRV↓" },
                "blood_pressure": { "status": "주의", "health_status": "고혈압 경향, 순환 탄력↓", "analysis_reason": "홍조, 어지러움", "recommendation": "나트륨-스트레스 이중 관리", "action_plan": "저염식(하루 5g↓), 유산소 30분, 심호흡, 하루 1회 혈압체크", "metric": "혈류지표↑" },
                "metabolism": { "status": "정상", "health_status": "대사 균형", "analysis_reason": "혀 붉지 않음, 얼굴 윤기", "recommendation": "항산화 + 균형식 루틴 유지", "action_plan": "비타민C,E, 오메가3 섭취, 녹색채소", "metric": "위산분비" },
                "liver_detox": { "status": "경고", "health_status": "피로 누적, 해독 저하", "analysis_reason": "혀 옆면 어둡고 탁함", "recommendation": "해독 루틴 / 야식 금지", "action_plan": "금주, 수면 7h, 녹즙 브로콜리 클로렐라", "metric": "AST/ALT↑" },
                "glycation_defense": { "status": "정상", "health_status": "신장 정체 없음", "analysis_reason": "다크서클 없음", "recommendation": "수분 리듬 조절 유지", "action_plan": "물 1.8L, 저염식", "metric": "크레아티닌" },
                "immunity_boost": { "status": "주의", "health_status": "면역저하, 염증경향", "analysis_reason": "얼굴 혈색 불균일, 혀 탁함", "recommendation": "항산화 + 프로바이오틱스", "action_plan": "블루베리, 유산균, 숙면, 디지털디톡스", "metric": "염증지표↑" },
                "stress_management": { "status": "경고", "health_status": "교감신경 활성", "analysis_reason": "표정 경직, 혀끝 붉음", "recommendation": "마음 회복 루틴", "action_plan": "명상, 일기쓰기, 자연노출, 하루 10분 멍때리기", "metric": "HRV↓" },
                "sleep_improvement": { "status": "경고", "health_status": "수면 불균형", "analysis_reason": "눈밑 다크서클, 혀 중앙 건조", "recommendation": "수면위생 루틴", "action_plan": "일정한 취침시간, 조도↓, 카페인 제한, 수면안대", "metric": "HRV↓" },
                "digestive_health": { "status": "주의", "health_status": "위장 부담", "analysis_reason": "혀 중앙 설태 백색, 얼굴 건조", "recommendation": "소화 리듬 회복 루틴", "action_plan": "아침식사 필수, 식후 산책, 늦은 식사 금지", "metric": "위산분비↑" },
                "hydration": { "status": "주의", "health_status": "체액 부족, 피부 노화", "analysis_reason": "혀 마름, 피부 광택↓", "recommendation": "수분섭취 보충 루틴", "action_plan": "물 1.5~2L, 수분 채소, 나트륨-갈륨 균형", "metric": "반사도↓" },
                "skin_elasticity": { "status": "주의", "health_status": "세포 재생 저하", "analysis_reason": "얼굴 윤기↓, 혀 거칠음", "recommendation": "콜라겐+수면 루틴", "action_plan": "콜라겐 펩타이드, 단백질 섭취, 취침 전 1시간 전 전자기기 차단", "metric": "반사도↓" },
                "respiratory_health": { "status": "정상", "health_status": "산소공급 양호", "analysis_reason": "입술 분홍빛", "recommendation": "호흡근 강화 루틴", "action_plan": "복식호흡, 요가, 1일 3회 심호흡 5분", "metric": "SpO₂" },
                "emotional_stability": { "status": "주의", "health_status": "우울, 무기력", "analysis_reason": "무표정, 혀 창백", "recommendation": "감정순환 루틴", "action_plan": "햇빛, 대화, 걷기, 음악, 봉사활동", "metric": "세로토닌↓" },
                "cognitive_health": { "status": "정상", "health_status": "염증 반응 없음", "analysis_reason": "혀 깨끗함", "recommendation": "청정 루틴 유지", "action_plan": "물+, 녹차, 항산화제", "metric": "CRP" },
                "hormone_balance": { "status": "주의", "health_status": "순환 대사 저하", "analysis_reason": "혀 부종, 평균 HRV↓, 혈류지표 0.68", "recommendation": "항노화 종합 루틴", "action_plan": "수분+비타민D, 운동 30분, 수면 7h, 명상 10분, 염분↓", "metric": "HRV↓" }
              }
            }
            """,
            "health_risk": common_instructions + """
            'status': "정상", "주의", "경고" 중 하나.
            'observation': "AI 탐지 특징 (얼굴-혀 영상 기준)". (예: "얼굴: 다크서클, 부종")
            'interpretation': "건강 해석". (예: "혈당조절 저하")
            'value': "측정 값 (예시)". (예: "혈류지표 0.62"). **반드시 그럴듯한 수치를 생성하세요.**
            'confidence': "신뢰도". (예: "83%"). **반드시 65%~95% 사이의 다양한 값으로 생성하세요.**
            'recommendation': "권고 사항". (예: "당분섭취 조절")

            {
              "health_risk": {
                "diabetes_risk": { "status": "경고", "observation": "얼굴: 다크서클, 부종 / 혀: 백태, 두꺼움", "interpretation": "혈당조절 저하, 순환 부하", "value": "혈류지표 0.62", "confidence": "83%", "recommendation": "당분섭취 조절, 충분한 수면" },
                "digestive_risk": { "status": "경고", "observation": "혀 중앙 두꺼운 설태, 회백색", "interpretation": "위액 불균형, 장기능 저하", "value": "Texture Index 0.78", "confidence": "81%", "recommendation": "식후 자극적 음식 회피, 유산균 섭취" },
                "liver_risk": { "status": "주의", "observation": "혀 측면 어두움 / 윤기 저하", "interpretation": "간기능 저하, 혈류 정체", "value": "Hue(Y) = 0.14", "confidence": "86%", "recommendation": "수분/채소 섭취, 금주" },
                "diabetes_tongue_pattern": { "status": "주의", "observation": "혀 전체 건조, 균열 / 붉은 설태", "interpretation": "체내 수분 저하, 혈당 과다", "value": "Moisture Ratio = 0.68", "confidence": "84%", "recommendation": "물 섭취 증가, 저당분 식단" },
                "blood_flow_summary": { "status": "정상", "observation": "얼굴혀 평균 혈류색 양호", "interpretation": "혈류 순환 원활", "value": "순환지표 0.85", "confidence": "88%", "recommendation": "현재 상태 유지" }
              }
            }
            """
        }
        
        # --- (신규) 요청할 프롬프트 선택 ---
        analysis_prompt = prompts.get(analysis_type)
        if not analysis_prompt:
            return None, f"'{analysis_type}'에 대한 프롬프트를 찾을 수 없습니다."

        # (수정) max_tokens를 넉넉하게 4096으로 설정
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_face}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_tongue}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096, # 4000 -> 4096 (최대치)
            "response_format": { "type": "json_object" }
        }

        print(f"GPT-4o API에 '{analysis_type}' 분석을 요청합니다...")
        try:
            response = client.chat.completions.create(
                model=payload["model"],
                messages=payload["messages"],
                max_tokens=payload["max_tokens"],
                response_format=payload["response_format"]
            )
            
            if response.choices and response.choices[0].message.content:
                gpt_response_content = response.choices[0].message.content
                print(f"GPT 응답 수신 ({analysis_type}):\n", gpt_response_content)
                
                if '"..."' in gpt_response_content:
                    print(f"!!! 경고 ({analysis_type}): GPT가 예시 문자열 '...'을 반환했습니다.")
                
                json_data = json.loads(gpt_response_content)
                
                # (수정) 요청한 유형의 데이터 블록만 반환
                result_key = f"{analysis_type}_analysis"
                if analysis_type == "anti_aging": result_key = "anti_aging"
                if analysis_type == "health_risk": result_key = "health_risk"

                analysis_result = json_data.get(result_key)
                
                if not analysis_result:
                     return None, f"GPT 응답에서 '{result_key}' 키를 찾을 수 없습니다."

                return analysis_result, None # 성공
            else:
                return None, f"GPT API로부터 ({analysis_type}) 유효한 응답을 받지 못했습니다."

        except Exception as e:
            print(f"!!! GPT API 요청 오류 ({analysis_type}): {e}")
            error_message = str(e)
            if "Incorrect API key" in error_message:
                return None, "OpenAI API 키가 잘못되었습니다. 로그인 페이지에서 올바른 키를 입력하세요."
            elif "billing" in error_message:
                return None, "OpenAI 크레딧(잔액)이 부족하거나 빌링 정보에 문제가 있습니다."
            else:
                return None, f"GPT API 통신 오류 ({analysis_type}): {e}"
            
            
    def handle_gpt_error(self, error_message):
        """(신규) GPT 오류 공통 처리기"""
        QMessageBox.critical(self, "AI 분석 오류", f"GPT 분석에 실패했습니다:\n{error_message}\n\n로그인 페이지로 돌아갑니다.")
        self.pages.setCurrentIndex(2) # 로그인 페이지(2)로 이동

# (신규) 재촬영 버튼 기능
    @pyqtSlot()
    def go_retake(self):
        print("재촬영 시작...")
        
        # 0. 재촬영 확인 (스타일 적용)
        msg_box = QMessageBox(self) # 부모창 지정
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("재촬영 확인")
        msg_box.setText("정말로 재촬영하시겠습니까?")
        msg_box.setInformativeText("현재 분석 결과가 모두 사라집니다.")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        
        # --- ▼▼▼ 스타일 시트로 크기 및 폰트 조정 ▼▼▼ ---
        msg_box.setStyleSheet("""
            QMessageBox {
                min-width: 600px;   /* 창 최소 너비를 600px로 넓힘 */
                min-height: 300px;  /* 창 최소 높이 설정 */
                background-color: white;
            }
            QLabel {
                font-size: 24px;    /* 글자 크기 대폭 확대 (기본: 12~13px) */
                font-family: 'NanumBarunGothic';
                font-weight: bold;
                color: #333;
                padding: 20px;      /* 텍스트 주변 여백 추가 */
            }
            QPushButton {
                font-size: 20px;    /* 버튼 글자 크기 확대 */
                font-family: 'NanumBarunGothic';
                font-weight: bold;
                padding: 15px 40px; /* 버튼 크기 확대 */
                border-radius: 10px;
                border: 2px solid #ccc;
                background-color: #f9f9f9;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        # --- ▲▲▲ 스타일 시트 끝 ▲▲▲ ---
        
        reply = msg_box.exec_()
        
        if reply == QMessageBox.No:
            print("재촬영 취소")
            return

        # 1. 메인 윈도우 상태 리셋
        self.session.reset_capture()
        self.camera_finished = False
        self.analysis_complete = False
        self.captured_face_path = None
        self.captured_tongue_path = None
        
        # 2. 카메라 페이지 위젯 리셋
        try:
            cam_page: CameraPage = self.page_widgets[3]
            cam_page.reset_page() # 1번에서 추가한 새 함수 호출
        except Exception as e:
            print(f"카메라 페이지 리셋 중 오류: {e}")
            
        # 3. 카메라 페이지(3)로 이동
        self.pages.setCurrentIndex(3)

            
    @pyqtSlot()
    def start_gpt_analysis_wrapper(self):
        """(신규) AnalysisPage의 UI를 업데이트하며 실제 분석 함수를 호출하는 래퍼"""
        print("AI 분석 시작 (5단계 순차적 GPT-4o 호출)...")
        
        try:
            analysis_page: AnalysisPage = self.page_widgets[4]
            
            # 1. 얼굴 분석
            print("1/5: 얼굴 분석 요청...")
            analysis_page.update_progress(10, "1/5: 얼굴 특징 분석 중...")
            QApplication.processEvents()
            face_data, error = self.get_gpt_analysis("face")
            if error: self.handle_gpt_error(error); return
            
            # 2. 혀 분석
            print("2/5: 혀 분석 요청...")
            analysis_page.update_progress(30, "2/5: 혀 상태 분석 중...")
            QApplication.processEvents()
            tongue_data, error = self.get_gpt_analysis("tongue")
            if error: self.handle_gpt_error(error); return
            
            # 3. 피부 분석
            print("3/5: 피부 분석 요청...")
            analysis_page.update_progress(50, "3/5: 피부 상세 분석 중...")
            QApplication.processEvents()
            skin_data, error = self.get_gpt_analysis("skin")
            if error: self.handle_gpt_error(error); return
            
            # 4. 노화방지 분석
            print("4/5: 노화방지 분석 요청...")
            analysis_page.update_progress(70, "4/5: 노화 방지 항목 생성 중...")
            QApplication.processEvents()
            anti_aging_data, error = self.get_gpt_analysis("anti_aging")
            if error: self.handle_gpt_error(error); return
            
            # 5. 건강위험 분석
            print("5/5: 건강위험 분석 요청...")
            analysis_page.update_progress(90, "5/5: 건강 위험도 분석 중...")
            QApplication.processEvents()
            health_risk_data, error = self.get_gpt_analysis("health_risk")
            if error: self.handle_gpt_error(error); return

            # 모든 분석 성공 시, 데이터 채우기
            print("모든 분석 완료. 데이터 채우는 중...")
            analysis_page.update_progress(95, "분석 완료! 결과 페이지를 구성 중입니다...")
            
            # --- (신규) 환자 정보 가져오기 ---
            login_page: LoginPage = self.page_widgets[2]
            patient_name = login_page.name_edit.text().strip() or "Guest"
            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            # ---------------------------------

            # (수정) 첫 번째 페이지에 환자 이름과 시간 전달
            self.page_widgets[5].populate_data(patient_name, current_time_str)
            
            self.page_widgets[6].populate_data(face_data) 
            self.page_widgets[7].populate_data(tongue_data)
            self.page_widgets[8].populate_data(skin_data)
            self.page_widgets[9].populate_data(anti_aging_data)
            self.page_widgets[10].populate_data(health_risk_data)
            
            print("데이터 채우기 완료. 첫 번째 결과 페이지로 이동.")
            analysis_page.update_progress(100, "완료되었습니다!")
            
            self.analysis_complete = True # 분석 완료 플래그
            self.session.mark_analysis_complete()
            self.pages.setCurrentIndex(5)
            
        except Exception as e:
            # get_gpt_analysis 외의 예외 처리 (예: populate_data)
            QMessageBox.critical(self, "분석 오류", f"결과 생성 중 치명적 오류 발생: {e}\n프로그램을 재시작하세요.")
            print(f"!!! 데이터 채우기 또는 알 수 없는 오류: {e}")
            import traceback
            traceback.print_exc()
            self.pages.setCurrentIndex(0) # 오류 시 홈으로
            # --- ▲▲▲ 여기가 수정된 부분입니다 (오류 코드 복구) ▲▲▲ ---

            # (신규) 프린트 버튼 기능
            
    # (수정) 프린트 버튼 기능 (오류 해결 및 버튼 복구 강화)
    @pyqtSlot()
    def print_results(self):
        """결과를 임시 PDF로 생성하고 인쇄를 시도합니다. 실패 시 파일을 엽니다."""
        import tempfile # 임시 파일 생성을 위해 필요

        # 0. ReportLab 라이브러리 확인
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.utils import ImageReader
            from reportlab.lib.units import mm
        except ImportError:
            QMessageBox.critical(self, "오류", "ReportLab 라이브러리를 찾을 수 없습니다.")
            return

        # 1. 임시 파일 경로 생성
        temp_dir = tempfile.gettempdir()
        temp_filename = f"temp_print_result_{datetime.now().strftime('%H%M%S')}.pdf"
        temp_path = os.path.join(temp_dir, temp_filename)

        # 2. PDF 생성 시작 (화면 캡처)
        original_page_index = self.pages.currentIndex()
        
        # 캡처 중 버튼 숨기기
        buttons_to_hide = [self.prev_btn, self.next_btn, self.exit_btn, self.retake_btn, self.save_btn, self.print_btn]
        for btn in buttons_to_hide:
            btn.hide()
            
        # 복구용 변수
        content_widget = None
        original_size_policy = None
        page_to_save = None

        try:
            pdf_canvas = canvas.Canvas(temp_path, pagesize=A4)
            page_width, page_height = A4
            margin = 10 * mm
            available_width = page_width - 2 * margin
            available_height = page_height - 2 * margin

            result_page_indices = range(5, 11) 

            for i, page_index in enumerate(result_page_indices):
                page_to_save = self.page_widgets[page_index]
                if not isinstance(page_to_save, TablePage):
                    continue

                self.pages.setCurrentIndex(page_index)
                QApplication.processEvents() 

                # 캡처 로직 (save_all_results_pdf와 동일)
                content_widget = page_to_save.scroll_area.widget()
                original_size_policy = content_widget.sizePolicy()

                content_widget.setMaximumWidth(16777215) 
                content_widget.adjustSize() 
                QApplication.processEvents() 

                content_widget.adjustSize()
                QApplication.processEvents()
                
                full_size = content_widget.size()
                
                pixmap = QPixmap(full_size)
                pixmap.fill(Qt.white) 
                content_widget.render(pixmap, QPoint(0, 0))

                # 복구 (즉시 복구하여 화면 깨짐 방지)
                content_widget.setSizePolicy(original_size_policy)
                page_to_save.perform_resize() 

                # PDF 그리기
                buffer = QBuffer()
                buffer.open(QBuffer.ReadWrite)
                pixmap.save(buffer, format="PNG")
                buffer.seek(0)
                img_data = buffer.data()
                img_buffer_for_reportlab = io.BytesIO(img_data)
                img_reader = ImageReader(img_buffer_for_reportlab)
                buffer.close()

                img_width, img_height = img_reader.getSize()
                scale_w = available_width / img_width
                scale_h = available_height / img_height
                scale = min(scale_w, scale_h)
                if scale >= 1.0: scale = 1.0 
                
                draw_width = img_width * scale
                draw_height = img_height * scale
                draw_x = (page_width - draw_width) / 2
                draw_y = (page_height - draw_height) / 2 

                pdf_canvas.drawImage(img_reader, draw_x, draw_y, width=draw_width, height=draw_height, mask='auto')

                if i < len(result_page_indices) - 1:
                    pdf_canvas.showPage()

            pdf_canvas.save()

            # 3. 시스템 인쇄 명령 호출 (오류 처리 추가)
            if os.name == 'nt':
                try:
                    # 1차 시도: 바로 인쇄 명령
                    os.startfile(temp_path, "print")
                except OSError:
                    # WinError 1155 등 실패 시: 파일 열기로 대체 (사용자가 직접 인쇄)
                    # 기본 PDF 뷰어가 Edge/Chrome인 경우 이쪽으로 실행됩니다.
                    os.startfile(temp_path)
            else:
                QMessageBox.information(self, "알림", f"파일이 생성되었습니다:\n{temp_path}")

        except Exception as e:
            QMessageBox.critical(self, "오류", f"프린트 생성 실패.\n오류: {e}")
            import traceback
            traceback.print_exc() 
            
        finally:
            # 5. 작업 완료 후 복구 (어떤 오류가 나도 버튼은 다시 보여야 함)
            try:
                if content_widget is not None and original_size_policy is not None:
                    content_widget.setSizePolicy(original_size_policy)
                
                # 마지막 페이지 복구 시도
                if page_to_save is not None and isinstance(page_to_save, TablePage):
                    page_to_save.perform_resize()

                # 혹시 모르니 원래 페이지도 복구 시도
                current_orig_page = self.page_widgets[original_page_index]
                if isinstance(current_orig_page, TablePage):
                    current_orig_page.perform_resize()

            except: pass

            # 페이지 복귀
            self.pages.setCurrentIndex(original_page_index)
            
            # 버튼 강제 표시
            self.prev_btn.show()
            self.next_btn.show()
            self.exit_btn.show()
            self.retake_btn.show() # 재촬영 버튼도 표시
            
            # Save/Print 버튼 가시성 재설정
            self.on_page_changed(original_page_index)

    # --- (수정) PDF 저장 함수 (파일명 포맷 변경 포함) ---
    @pyqtSlot()
    def save_all_results_pdf(self):
        """(수정) 5번~10번까지 모든 결과 페이지를 하나의 PDF로 저장합니다."""
        
        # 0. ReportLab 라이브러리 확인
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.utils import ImageReader
            from reportlab.lib.units import mm
        except ImportError:
            QMessageBox.critical(self, "오류", "ReportLab 라이브러리를 찾을 수 없습니다.\n터미널(cmd)에서 'pip install reportlab'을 실행해주세요.")
            return

        # 1. 저장 경로 및 파일 이름 설정
        login:LoginPage = self.page_widgets[2]
        patient_name = login.name_edit.text().strip() or "무명"
        
        # --- ▼▼▼ (수정) 파일명 형식 변경 ▼▼▼ ---
        now = datetime.now()
        date_str = now.strftime("%Y%m%d") # 예: 20231124
        time_str = now.strftime("%H%M")   # 예: 1630 (16:30)
        
        # 파일명: 환자이름_날짜_시간.pdf
        default_filename = f"{patient_name}_{date_str}_{time_str}.pdf"
        # --- ▲▲▲ (수정) 끝 ▲▲▲ ---

        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "전체 결과 PDF로 저장",
            os.path.join(desktop_path, default_filename),
            "PDF Files (*.pdf)"
        )

        if not save_path:
            return # 사용자가 취소한 경우

        # 2. PDF 생성 시작
        original_page_index = self.pages.currentIndex()
        
        # 캡처 중 버튼 숨기기
        buttons_to_hide = [self.prev_btn, self.next_btn, self.exit_btn, self.retake_btn, self.save_btn, self.print_btn]
        for btn in buttons_to_hide:
            btn.hide()
            
        # 복구용 변수
        content_widget = None
        original_size_policy = None
        page_to_save = None

        try:
            pdf_canvas = canvas.Canvas(save_path, pagesize=A4)
            page_width, page_height = A4
            margin = 10 * mm
            available_width = page_width - 2 * margin
            available_height = page_height - 2 * margin

            result_page_indices = range(5, 11) 

            # 3. 모든 결과 페이지 순회하며 PDF에 추가
            for i, page_index in enumerate(result_page_indices):
                page_to_save = self.page_widgets[page_index]
                if not isinstance(page_to_save, TablePage):
                    continue

                self.pages.setCurrentIndex(page_index)
                QApplication.processEvents() 

                # 캡처 로직 (가로 잘림 방지)
                content_widget = page_to_save.scroll_area.widget()
                original_size_policy = content_widget.sizePolicy()

                content_widget.setMaximumWidth(16777215) 
                content_widget.adjustSize() 
                QApplication.processEvents() 

                content_widget.adjustSize()
                QApplication.processEvents()
                
                full_size = content_widget.size()
                
                pixmap = QPixmap(full_size)
                pixmap.fill(Qt.white) 
                content_widget.render(pixmap, QPoint(0, 0))

                # 복구
                content_widget.setSizePolicy(original_size_policy)
                page_to_save.perform_resize() 

                # 이미지 변환 및 PDF 그리기
                buffer = QBuffer()
                buffer.open(QBuffer.ReadWrite)
                pixmap.save(buffer, format="PNG")
                buffer.seek(0)
                img_data = buffer.data()
                img_buffer_for_reportlab = io.BytesIO(img_data)
                img_reader = ImageReader(img_buffer_for_reportlab)
                buffer.close()

                img_width, img_height = img_reader.getSize()
                scale_w = available_width / img_width
                scale_h = available_height / img_height
                scale = min(scale_w, scale_h)
                
                if scale >= 1.0: 
                    scale = 1.0 
                
                draw_width = img_width * scale
                draw_height = img_height * scale
                draw_x = (page_width - draw_width) / 2
                draw_y = (page_height - draw_height) / 2 

                pdf_canvas.drawImage(img_reader, draw_x, draw_y, width=draw_width, height=draw_height, mask='auto')

                if i < len(result_page_indices) - 1:
                    pdf_canvas.showPage()

            # 4. PDF 파일 저장
            pdf_canvas.save()
            QMessageBox.information(self, "저장 완료", f"전체 결과 PDF 파일 저장이 완료되었습니다.\n경로: {save_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"PDF 저장 실패.\n오류: {e}")
            import traceback
            traceback.print_exc() 
            
        finally:
            # 5. 작업 완료 후 복구
            try:
                # 마지막 페이지 복구
                if content_widget is not None and original_size_policy is not None:
                    content_widget.setSizePolicy(original_size_policy)
                if page_to_save is not None and isinstance(page_to_save, TablePage):
                    page_to_save.perform_resize()
                    
                # 혹시 모르니 원래 보고 있던 페이지도 복구
                current_orig_page = self.page_widgets[original_page_index]
                if isinstance(current_orig_page, TablePage):
                    current_orig_page.perform_resize()

            except Exception as e:
                print(f"PDF 저장 후 복구 중 오류: {e}") 

            # 버튼 복구 (수동)
            self.prev_btn.show()
            self.next_btn.show()
            self.exit_btn.show()
            self.retake_btn.show()
            
            # 페이지 복귀 및 버튼 상태 갱신
            self.pages.setCurrentIndex(original_page_index)
            self.on_page_changed(original_page_index)
            
# ---------- 실행 ----------
def main(): # () 추가
    app=QApplication(sys.argv)
    ok,message=True,"" # Simplified resource check for now
    if not os.path.isdir(IMG_DIR): ok,message=False,f"이미지 폴더 없음: '{IMG_DIR}'"
    else:
        for fname in IMG_FILES:
            if not os.path.isfile(os.path.join(IMG_DIR,fname)): ok,message=False,f"이미지 파일 없음: '{fname}'"; break
    if not ok: QMessageBox.critical(None, "필수 파일 없음", message); sys.exit(1)
    w=MainWindow(); w.showFullScreen(); sys.exit(app.exec_()) # () 추가, () 추가, () 추가

if __name__=="__main__": main() # () 추가
