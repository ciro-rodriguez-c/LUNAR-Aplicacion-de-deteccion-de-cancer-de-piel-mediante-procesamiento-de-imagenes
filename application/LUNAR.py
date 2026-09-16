import sys
import cv2
import os
import time
import numpy as np
import qrcode
from PIL import Image
from picamera2 import Picamera2
from libcamera import controls
from gpiozero import Button, LED
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QStackedWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PyQt5.QtGui import QImage, QPixmap, QFont
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image

#credenciales firebase

clave_secreta_incrustada = {
  ":D"
}

class ActualizadorFirebase:
    def __init__(self, clave_servicio, bucket_url):
        print("🔧 [Firebase] Creando instancia de ActualizadorFirebase...")
        try:
            if not firebase_admin._apps:
                ...
                firebase_admin.initialize_app(cred, {'storageBucket': bucket_url})
            self.db = firestore.client()
            self.bucket = storage.bucket()
            print("[Firebase] Conexión con Firebase lista.")
        except Exception as e:
            print(f"[Firebase] Error fatal durante la inicialización: {e}")
            raise
    def actualizar_datos(self, coleccion, documento, resultado_diagnostico, nombre_imagen_storage, ruta_imagen_local, ruta_map,  nombre_map, ruta_cam, nombre_cam):
        print(f"[Firebase] Actualizando datos con resultado: '{resultado_diagnostico}'...")
        try:
            doc_ref = self.db.collection(coleccion).document(documento)
            nuevos_datos = {'mensaje': resultado_diagnostico, 'imagen': nombre_imagen_storage, 'imagen_map': nombre_map, 'imagen_cam':nombre_cam}
            doc_ref.set(nuevos_datos)
            print(f"[Firebase] Firestore: Documento '{documento}' actualizado.")
        except Exception as e:
            print(f"[Firebase] Firestore: Error al actualizar el documento: {e}")
            return
        try:
            self.bucket.blob(nombre_imagen_storage).upload_from_filename(ruta_imagen_local)
            print(f"✅ [Firebase] Storage: Imagen '{nombre_imagen_storage}' subida.")

            self.bucket.blob(nombre_map).upload_from_filename(ruta_map)
            print(f"✅ [Firebase] Storage: Imagen '{nombre_map}' subida.")

            self.bucket.blob(nombre_cam).upload_from_filename(ruta_cam)
            print(f"✅ [Firebase] Storage: Imagen '{nombre_cam}' subida.")

        except Exception as e:
            print(f"[Firebase] Storage: Error al subir la imagen: {e}")

class SistemaDeHardwareRPi:
    def __init__(self):
        print("[Hardware] Inicializando hardware de Raspberry Pi...")
        
        self.leds = self._configurar_leds()
        self.boton = Button(25, pull_up=True) # PIN_BOTON
        
        self.carpeta = "/home/czair/LUNAR/"
        self.modelo_path = "/home/czair/LUNAR/modelo.pt"
        if not os.path.exists(self.carpeta):
            os.makedirs(self.carpeta)
            
        self.desactivar_leds()

    def _configurar_leds(self):
        leds = LED(24)
        print("[Hardware] LEDs configurados.")
        return leds

    def activar_leds(self):
        print("[Hardware] Encendiendo LEDs de estado ACTIVO.")
        self.leds.on()

    def desactivar_leds(self):
        print("[Hardware] Encendiendo LEDs de estado EN ESPERA.")
        self.leds.off()

    def guardar_foto(self, frame):
        nombre_archivo = os.path.join(self.carpeta, f"foto.jpg")
        if os.path.exists(nombre_archivo):
            os.remove(nombre_archivo)  # Elimina el archivo si ya existe
        cv2.imwrite(nombre_archivo, frame)
        print(f"[Hardware] Foto guardada en: {nombre_archivo}")
        return nombre_archivo

    def procesar_imagen(self, imagen_path):
        from ultralytics import YOLO
        nombreImagenBbox = os.path.join(self.carpeta, f"Bbox.jpg")
        nombreCamImagen = os.path.join(self.carpeta, f'grad_CAM.jpg')
        if os.path.exists(nombreImagenBbox):
            os.remove(nombreImagenBbox)  # Elimina el archivo si ya existe
        if os.path.exists(nombreCamImagen):
            os.remove(nombreCamImagen)  # Elimina el archivo si ya existe
        print(f"[IA] Analizando imagen: {imagen_path}...")
        try:
            #Carga del modelo
            modelo = YOLO(self.modelo_path)
            modelo.cpu()
            #Carga de la imagen a utilizar
            imagen = cv2.imread(imagen_path)
            img = imagen.copy()
            img = cv2.resize(img, (640, 640))
            rgb_img = img.copy()
            img = np.float32(img) / 255
            
            #Inferencia del modelo
            resultado = modelo(imagen)      #resultado = modelo(img) o resultado = modelo(rgb_img)
            #Imagen bouding box
            imagenBbox = resultado[0].plot()
            nombreImagenBbox = os.path.join(self.carpeta, f"Bbox.jpg")
            cv2.imwrite(nombreImagenBbox, imagenBbox)
            #Grad CAM
            #Configuracion
            target_layers = [modelo.model.model[-2]]
            cam = EigenCAM(modelo, target_layers, task='od')
            grayscale_cam = cam(rgb_img)[0, :, :]
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            cam_image = 255 - cam_image     #Inversión de colores (Quizas sde deba cambiar por la solución de arriba XD)
            nombreCamImagen = os.path.join(self.carpeta, f'grad_CAM.jpg')
            cv2.imwrite(nombreCamImagen, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
            #Logica del etiquetado
            if len(resultado[0].boxes.cls) > 0:
                etiqueta = resultado[0].names[resultado[0].boxes.cls[0].item()]
            else:
                etiqueta = "Sin deteccion"
            del modelo
            print(f"Resultado del análisis: {etiqueta}")
            datos = (etiqueta, nombreImagenBbox, nombreCamImagen)
            return etiqueta, nombreImagenBbox, nombreCamImagen
        except Exception as e:
            print(f"Error durante el procesamiento: {e}")
            return "Error de IA", "Error de IA", "Error de IA"

class PiCameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.picam2 = None

    def run(self):
        print("[Cámara] Iniciando Picamera2...")
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1280, 960)})
        self.picam2.configure(config)
        self.picam2.start()
        # Enfocar la cámara 
        self.picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 8.0})
        
        while not self.isInterruptionRequested():
            frame = self.picam2.capture_array()
            self.change_pixmap_signal.emit(frame)
    
    def stop(self):
        if self.picam2:
            print("[Cámara] Deteniendo Picamera2.")
            self.picam2.stop()
            self.picam2.close()

class GuidedCameraApp(QWidget):
    def __init__(self):
        super().__init__()
        
        self.sistema = SistemaDeHardwareRPi()
        
        BUCKET_URL = 'lunar-df556.firebasestorage.app'
        self.actualizador_firebase = None
        try:
            self.actualizador_firebase = ActualizadorFirebase(
                clave_servicio=clave_secreta_incrustada,
                bucket_url=BUCKET_URL
            )
        except Exception as e:
            print(f"‼️ ADVERTENCIA: No se pudo conectar a Firebase. La app funcionará en modo offline. Error: {e}")
        
        self.setWindowTitle("Asistente de Captura LUNAR (RPi)")
        self.setGeometry(100, 100, 1024, 768)
        self.setStyleSheet("background-color: #FFFFFF;")
        
        self.latest_cv_frame = None
        
        self.stack = QStackedWidget(self)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.stack)
        
        self.restart_timer = QTimer(self)
        self.restart_timer.setSingleShot(True)
        self.restart_timer.timeout.connect(self.restart_process)
        
        self.button_poll_timer = QTimer(self)
        self.button_poll_timer.timeout.connect(self.check_physical_button)
        self.button_poll_timer.start(100) # Revisa el estado del botón cada 100 ms

        self.state = ""
        self.go_to_state("START")

    def handle_button_press(self):
        """Función central para manejar la acción del botón (físico o Enter)"""
        if self.state == "START":
            self.go_to_state("WELCOME")
        elif self.state == "CAMERA_VIEW":
            self.capture_process_and_upload()
        elif self.state == "RESULTS":
            self.restart_process()

    def check_physical_button(self):
        """Revisa si el botón físico ha sido presionado."""
        if self.sistema.boton.is_pressed:
            time.sleep(0.2) # Pequeño anti-rebote
            print("[Botón] Botón físico presionado.")
            self.handle_button_press()

    def keyPressEvent(self, event):
        """Maneja la presión de la tecla Enter."""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            print("[Teclado] Tecla Enter presionada.")
            self.handle_button_press()

    def capture_process_and_upload(self):
        if self.state != "CAMERA_VIEW" or self.latest_cv_frame is None:
            return

        print("\n--- INICIANDO PROCESO DE CAPTURA Y ANÁLISIS ---")
        self.thread.requestInterruption() # Detiene el bucle de la cámara
        self.thread.stop() # Llama al método stop() para liberar la cámara
        self.thread.wait()
        
        self.sistema.desactivar_leds()
        archivo_path = self.sistema.guardar_foto(self.latest_cv_frame)
        resultado, bbox, cam = self.sistema.procesar_imagen(archivo_path)   #CAMPOS
        
        self.go_to_state("PROCESSING")
#############################################################
        if self.actualizador_firebase:
            print("\n--- INICIANDO SUBIDA A FIREBASE ---")
            nombre_en_storage = 'lunar_ejemplo.jpg'
            self.actualizador_firebase.actualizar_datos(
                coleccion='persona', documento='examen', 
                nombre_imagen_storage=nombre_en_storage,
                resultado_diagnostico=resultado,
                ruta_imagen_local=archivo_path,
                ruta_map = bbox,
                nombre_map = "image1.jpg",
                ruta_cam= cam,
                nombre_cam = "image2.jpg"
            )
        else:
            print("‼️ ADVERTENCIA: Omitiendo subida a Firebase.")

        QTimer.singleShot(500, lambda: self.go_to_state("RESULTS", analysis_result=resultado))

    def go_to_state(self, new_state, analysis_result=None):
        self.restart_timer.stop()
        self.state = new_state
        new_widget = self.create_state_widget(new_state, analysis_result)
        self.stack.addWidget(new_widget)
        self.animate_transition(self.stack.count() - 1)
        if self.stack.count() > 1:
            QTimer.singleShot(500, self.cleanup_old_widget)

    def create_state_widget(self, state, analysis_result=None):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        if state == "START":
            lunar_layout = QVBoxLayout()
            lunar_layout.addStretch(1)
            lunar_label = QLabel("LUNAR")
            lunar_label.setAlignment(Qt.AlignCenter)
            lunar_label.setFont(QFont('Segoe UI', 70, QFont.Bold))
            lunar_label.setStyleSheet("color: #007BFF; background-color: transparent;")
            lunar_layout.addWidget(lunar_label)
            lunar_layout.addStretch(1)
            layout.addLayout(lunar_layout)
            instruction_label = QLabel("Presione el botón para comenzar")
            instruction_label.setAlignment(Qt.AlignCenter)
            instruction_label.setFont(QFont('Segoe UI', 25, QFont.Bold))
            instruction_label.setStyleSheet("color: #343a40; background-color: transparent;")
            layout.addWidget(instruction_label)
        
        elif state == "WELCOME":
            self.add_message_to_layout(layout, "¡HOLA!", 50, "#007BFF")
            QTimer.singleShot(3000, lambda: self.go_to_state("PROMPT_ARM"))
        elif state == "PROMPT_ARM":
            self.add_message_to_layout(layout, "Ingresa tu brazo...", 40)
            QTimer.singleShot(3000, lambda: self.go_to_state("PROMPT_FOCUS"))
        elif state == "PROMPT_FOCUS":
            self.add_message_to_layout(layout, "Enfoca la zona a analizar...", 40)
            QTimer.singleShot(3000, lambda: self.go_to_state("CAMERA_VIEW"))
        elif state == "CAMERA_VIEW":
            self.setup_camera_view(layout)
        elif state == "PROCESSING":
            self.add_message_to_layout(layout, "Analizando y subiendo...", 40)
        elif state == "RESULTS":
            #
            # color_resultado = "#E74C3C" if analysis_result == "Cancer" else "#2ECC71"
            self.add_message_to_layout(layout, "Análisis Completado", 35, "#198754")
            #self.add_message_to_layout(layout, f"Resultado: {analysis_result}", 30, color_resultado) (No mostrar enn el raspberry)
            qr_pixmap = self.generate_qr_pixmap("http://lunar-df556.web.app", 250)
            if qr_pixmap:
                qr_label = QLabel(); qr_label.setPixmap(qr_pixmap); qr_label.setAlignment(Qt.AlignCenter); layout.addWidget(qr_label)
            self.add_message_to_layout(layout, "(Presione el botón para reiniciar)", 20, "#6c757d")
            self.restart_timer.start(30000)
        
        return widget

    def setup_camera_view(self, layout):
        self.sistema.activar_leds()
        layout.setContentsMargins(20, 20, 20, 20)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 8px;")
        layout.addWidget(self.video_label)
        
        info_label = QLabel("Presione el botón para tomar foto", self.video_label)
        info_label.setAlignment(Qt.AlignCenter); info_label.setFont(QFont('Segoe UI', 16, QFont.Bold))
        info_label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 0.5); padding: 10px; border-radius: 5px;")
        info_label.adjustSize()
        self.camera_instruction_label = info_label
        
        self.thread = PiCameraThread()
        self.thread.change_pixmap_signal.connect(self.update_frame)
        self.thread.start()
    
    def add_message_to_layout(self, layout, text, font_size, color="#343a40"):
        message_label = QLabel(text); message_label.setAlignment(Qt.AlignCenter)
        message_label.setFont(QFont('Segoe UI', font_size, QFont.Bold))
        message_label.setStyleSheet(f"color: {color}; background-color: transparent;")
        layout.addWidget(message_label)

    def restart_process(self):
        self.restart_timer.stop()
        self.sistema.desactivar_leds()
        self.go_to_state("START")

    def update_frame(self, cv_img):
        if not hasattr(self, 'video_label'): return
        self.latest_cv_frame = cv_img # Guardamos el frame RGB de Picamera2
        qt_img = self.convert_cv_qt(cv_img)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        if hasattr(self, 'camera_instruction_label'):
            self.camera_instruction_label.move(int((self.video_label.width() - self.camera_instruction_label.width()) / 2), int(self.video_label.height() - self.camera_instruction_label.height() - 20))

    def animate_transition(self, new_index):
        new_widget = self.stack.widget(new_index)
        new_widget.setGeometry(self.rect())
        start_pos = QPoint(0, self.height())
        new_widget.move(start_pos)
        self.pos_animation = QPropertyAnimation(new_widget, b"pos")
        self.pos_animation.setDuration(400)
        self.pos_animation.setStartValue(start_pos)
        self.pos_animation.setEndValue(self.rect().topLeft())
        self.pos_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.stack.setCurrentIndex(new_index)
        self.pos_animation.start()

    def cleanup_old_widget(self):
        if self.stack.count() > 1:
            old_widget = self.stack.widget(0)
            self.stack.removeWidget(old_widget)
            old_widget.deleteLater()

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

    def generate_qr_pixmap(self, data, size):
        try:
            qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
            qr.add_data(data)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white").convert('RGBA')
            qimage = QImage(img.tobytes("raw", "RGBA"), img.size[0], img.size[1], QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimage)
            return pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        except Exception as e:
            print(f"Error al generar QR: {e}")
            return None

    def closeEvent(self, event):
        print("Cerrando aplicación...")
        self.button_poll_timer.stop()
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.requestInterruption()
            self.thread.stop()
            self.thread.wait()
        self.sistema.desactivar_leds() # Apagar LEDs al cerrar
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GuidedCameraApp()
    window.showFullScreen()
    sys.exit(app.exec_())
