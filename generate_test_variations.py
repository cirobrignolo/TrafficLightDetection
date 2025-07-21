import cv2
import numpy as np
import os
import random
from pathlib import Path

def apply_sepia(image):
    """Aplica efecto sepia (luz cálida/atardecer)"""
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(image, kernel)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)

def apply_blue_tint(image, intensity=0.3):
    """Aplica tinte azul (simula noche/sombra)"""
    blue_tinted = image.copy().astype(np.float32)
    blue_tinted[:,:,0] = blue_tinted[:,:,0] * (1 + intensity)  # Aumentar canal azul
    blue_tinted[:,:,1] = blue_tinted[:,:,1] * (1 - intensity*0.3)  # Reducir verde
    blue_tinted[:,:,2] = blue_tinted[:,:,2] * (1 - intensity*0.5)  # Reducir rojo
    return np.clip(blue_tinted, 0, 255).astype(np.uint8)

def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """Ajusta brillo y contraste"""
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted

def add_fog_effect(image, intensity=0.4):
    """Simula efecto niebla"""
    # Crear máscara de niebla
    fog = np.ones_like(image, dtype=np.float32) * 255 * intensity
    # Aplicar blur para efecto difuso
    fog = cv2.GaussianBlur(fog, (15, 15), 0)
    # Mezclar con imagen original
    foggy = cv2.addWeighted(image.astype(np.float32), 1-intensity, fog, intensity, 0)
    return np.clip(foggy, 0, 255).astype(np.uint8)

def add_rain_effect(image, intensity=0.3):
    """Simula lluvia con líneas verticales"""
    rain_image = image.copy()
    height, width = image.shape[:2]
    
    # Número de gotas basado en intensidad
    num_drops = int(width * height * intensity / 1000)
    
    for _ in range(num_drops):
        x = random.randint(0, width-1)
        y1 = random.randint(0, height//2)
        y2 = y1 + random.randint(10, 30)
        
        # Línea semitransparente
        cv2.line(rain_image, (x, y1), (x, min(y2, height-1)), 
                (200, 200, 200), 1, cv2.LINE_AA)
    
    return rain_image

def add_snow_effect(image, intensity=0.2):
    """Simula nieve con puntos blancos aleatorios"""
    snow_image = image.copy()
    height, width = image.shape[:2]
    
    # Número de copos basado en intensidad
    num_flakes = int(width * height * intensity / 100)
    
    for _ in range(num_flakes):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        size = random.randint(1, 3)
        
        cv2.circle(snow_image, (x, y), size, (255, 255, 255), -1)
    
    return snow_image

def add_gaussian_noise(image, intensity=25):
    """Añade ruido gaussiano"""
    noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def compress_jpeg(image, quality=30):
    """Simula compresión JPEG agresiva"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    compressed = cv2.imdecode(encimg, 1)
    return compressed

def add_motion_blur(image, size=15):
    """Añade desenfoque de movimiento"""
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

# Diccionario de efectos
EFFECTS = {
    'sepia': apply_sepia,
    'blue_night': lambda img: apply_blue_tint(img, 0.4),
    'dark': lambda img: adjust_brightness_contrast(img, brightness=-50, contrast=0.7),
    'bright': lambda img: adjust_brightness_contrast(img, brightness=30, contrast=1.2),
    'low_contrast': lambda img: adjust_brightness_contrast(img, brightness=0, contrast=0.5),
    'high_contrast': lambda img: adjust_brightness_contrast(img, brightness=0, contrast=1.8),
    'fog_light': lambda img: add_fog_effect(img, 0.2),
    'fog_heavy': lambda img: add_fog_effect(img, 0.5),
    'rain_light': lambda img: add_rain_effect(img, 0.2),
    'rain_heavy': lambda img: add_rain_effect(img, 0.5),
    'snow_light': lambda img: add_snow_effect(img, 0.1),
    'snow_heavy': lambda img: add_snow_effect(img, 0.3),
    'noise_light': lambda img: add_gaussian_noise(img, 15),
    'noise_heavy': lambda img: add_gaussian_noise(img, 40),
    'jpeg_compression': lambda img: compress_jpeg(img, 20),
    'motion_blur': lambda img: add_motion_blur(img, 10)
}

def generate_variations(input_dir, output_base_dir, effects_to_apply=None):
    """
    Genera variaciones de todas las imágenes en input_dir
    
    Args:
        input_dir: Directorio con imágenes originales
        output_base_dir: Directorio base para guardar variaciones
        effects_to_apply: Lista de efectos a aplicar (None = todos)
    """
    
    if effects_to_apply is None:
        effects_to_apply = list(EFFECTS.keys())
    
    input_path = Path(input_dir)
    output_path = Path(output_base_dir)
    
    # Crear directorio de salida
    output_path.mkdir(exist_ok=True)
    
    # Procesar cada imagen
    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
    
    print(f"📸 Procesando {len(image_files)} imágenes...")
    print(f"🎨 Aplicando {len(effects_to_apply)} efectos: {', '.join(effects_to_apply)}")
    
    for img_file in image_files:
        print(f"\n🔄 Procesando: {img_file.name}")
        
        # Leer imagen original
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"❌ Error leyendo {img_file}")
            continue
        
        # Aplicar cada efecto
        for effect_name in effects_to_apply:
            if effect_name not in EFFECTS:
                print(f"⚠️  Efecto desconocido: {effect_name}")
                continue
            
            try:
                # Aplicar efecto
                modified_image = EFFECTS[effect_name](image)
                
                # Crear directorio para este efecto
                effect_dir = output_path / effect_name
                effect_dir.mkdir(exist_ok=True)
                
                # Guardar imagen modificada
                output_file = effect_dir / img_file.name
                cv2.imwrite(str(output_file), modified_image)
                
                print(f"  ✅ {effect_name} → {output_file}")
                
            except Exception as e:
                print(f"  ❌ Error aplicando {effect_name}: {e}")
    
    print(f"\n🎉 ¡Completado! Variaciones guardadas en: {output_path}")
    
    # Copiar projection_bboxes_master.txt a cada directorio
    bbox_file = input_path / 'projection_bboxes_master.txt'
    if bbox_file.exists():
        for effect_name in effects_to_apply:
            effect_dir = output_path / effect_name
            if effect_dir.exists():
                import shutil
                shutil.copy2(bbox_file, effect_dir / 'projection_bboxes_master.txt')
        print(f"📋 Archivo projection_bboxes_master.txt copiado a cada directorio")

if __name__ == "__main__":
    # Configuración
    INPUT_DIR = "frames_auto_labeled"
    OUTPUT_DIR = "frames_variations_test"
    
    # Efectos recomendados para pruebas de robustez
    RECOMMENDED_EFFECTS = [
        'sepia',           # Luz cálida/atardecer
        'blue_night',      # Condiciones nocturnas
        'dark',            # Baja iluminación
        'fog_light',       # Niebla ligera
        'rain_light',      # Lluvia ligera
        'noise_light',     # Ruido de cámara
        'jpeg_compression' # Compresión agresiva
    ]
    
    print("🚀 Generador de Variaciones para Pruebas de Robustez")
    print("=" * 50)
    
    # Generar variaciones
    generate_variations(INPUT_DIR, OUTPUT_DIR, RECOMMENDED_EFFECTS)
    
    print(f"\n📊 Para probar cada variación, ejecuta:")
    for effect in RECOMMENDED_EFFECTS:
        print(f"  python run_pipeline_debug_stages_fixed.py # Cambiar input_dir a '{OUTPUT_DIR}/{effect}'")