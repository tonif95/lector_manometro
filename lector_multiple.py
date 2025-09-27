import cv2
import numpy as np
import math
import argparse

# ==============================================================================
# 1. CONFIGURACIÓN CENTRALIZADA PARA TODOS LOS MANÓMETROS
# ==============================================================================
# Rellena aquí los valores específicos de cada uno de tus manómetros.
# Estos son los valores que tenías en tus 3 archivos separados.

CONFIGS = {
    'izquierda': {
        'MIN_VALUE': 0,
        'MAX_VALUE': 10,
        'START_ANGLE': 130,
        'END_ANGLE': 50,
        'INVERTIR_AGUJA': False 
    },
    'centro': {
        'MIN_VALUE': 0,
        'MAX_VALUE': 25, # Ejemplo
        'START_ANGLE': 135,  # Ejemplo
        'END_ANGLE': 50,    # Ejemplo
        'INVERTIR_AGUJA': False # Ejemplo
    },
    'derecha': {
        'MIN_VALUE': 0,
        'MAX_VALUE': 25, # Ejemplo
        'START_ANGLE': 130,   # Ejemplo
        'END_ANGLE': 50,     # Ejemplo
        'INVERTIR_AGUJA': False # Ejemplo
    }
}

# ==============================================================================
# 2. FUNCIÓN DE PROCESAMIENTO (LA LÓGICA QUE YA TENÍAMOS)
# ==============================================================================
# Esta función ahora recibe un trozo de imagen y su configuración específica.

def procesar_manometro(sub_imagen, config):
    """
    Analiza la imagen de un único manómetro y devuelve la imagen procesada y su valor.
    """
    img_display = sub_imagen.copy()
    gray = cv2.cvtColor(sub_imagen, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # --- Detección de Elipse ---
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_ellipse = None
    best_area = 0
    for cnt in contours:
        if len(cnt) < 20: continue
        area = cv2.contourArea(cnt)
        if area < 1000: continue
        try:
            ellipse = cv2.fitEllipse(cnt)
        except cv2.error:
            continue
            
        (center, axes, angle) = ellipse
        major_axis, minor_axis = max(axes), min(axes)
        if minor_axis == 0: continue
        aspect_ratio = major_axis / minor_axis
        if aspect_ratio > 1.8: continue
        if area > best_area:
            best_area = area
            best_ellipse = ellipse

    if best_ellipse is None:
        print("  -> No se pudo detectar elipse.")
        cv2.putText(img_display, "No detectado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img_display, None

    (center, axes, angle) = best_ellipse
    center_x, center_y = int(center[0]), int(center[1])
    radius = int((axes[0] + axes[1]) / 4)

    cv2.ellipse(img_display, best_ellipse, (0, 255, 0), 3)
    cv2.circle(img_display, (center_x, center_y), 5, (0, 0, 255), -1)

    # --- Líneas de Calibración ---
    start_angle_rad = math.radians(config['START_ANGLE'])
    end_angle_rad = math.radians(config['END_ANGLE'])
    line_radius = int((axes[0] + axes[1]) / 2.5)
    x_start = int(center_x + line_radius * math.cos(start_angle_rad))
    y_start = int(center_y + line_radius * math.sin(start_angle_rad))
    cv2.line(img_display, (center_x, center_y), (x_start, y_start), (255, 0, 255), 2)
    x_end = int(center_x + line_radius * math.cos(end_angle_rad))
    y_end = int(center_y + line_radius * math.sin(end_angle_rad))
    cv2.line(img_display, (center_x, center_y), (x_end, y_end), (0, 255, 255), 2)

    # --- Detección de Aguja ---
    mask = np.zeros_like(gray)
    cv2.ellipse(mask, best_ellipse, 255, -1)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=40, minLineLength=int(radius * 0.2), maxLineGap=10)

    if lines is None:
        print("  -> No se detectó aguja.")
        cv2.putText(img_display, "Aguja no detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return img_display, None

    best_line = None
    max_line_length = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        dist_to_center = np.sqrt((mid_x - center_x)**2 + (mid_y - center_y)**2)
        if dist_to_center > radius * 0.5: continue
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length > max_line_length:
            max_line_length = length
            best_line = (x1, y1, x2, y2)
    
    if best_line is None:
        print("  -> No se pudo determinar la línea de la aguja.")
        cv2.putText(img_display, "Aguja no determinada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return img_display, None

    x1, y1, x2, y2 = best_line
    cv2.line(img_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    dist1_from_center = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
    dist2_from_center = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)
    if dist1_from_center > dist2_from_center:
        tip_x_original, tip_y_original, base_x_original, base_y_original = x1, y1, x2, y2
    else:
        tip_x_original, tip_y_original, base_x_original, base_y_original = x2, y2, x1, y1
    
    if config['INVERTIR_AGUJA']:
        tip_x, tip_y = base_x_original, base_y_original
    else:
        tip_x, tip_y = tip_x_original, tip_y_original
    
    cv2.circle(img_display, (tip_x, tip_y), 5, (0, 165, 255), -1)

    # --- Mapeo de Ángulo a Presión ---
    angle_rad = math.atan2(tip_y - center_y, tip_x - center_x)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0: angle_deg += 360

    pressure_value = 0.0
    total_angle_sweep = (config['END_ANGLE'] - config['START_ANGLE'] + 360) % 360
    current_angle_relative = (angle_deg - config['START_ANGLE'] + 360) % 360
    if total_angle_sweep > 0 and current_angle_relative <= total_angle_sweep:
        pressure_percentage = current_angle_relative / total_angle_sweep
        pressure_value = config['MIN_VALUE'] + pressure_percentage * (config['MAX_VALUE'] - config['MIN_VALUE'])
    else:
        # Lógica simple para valores fuera de rango
        if current_angle_relative > 180: pressure_value = config['MIN_VALUE']
        else: pressure_value = config['MAX_VALUE']
    
    pressure_value = max(config['MIN_VALUE'], min(config['MAX_VALUE'], pressure_value))

    result_text = f"Presion: {pressure_value:.2f} bar" 
    cv2.putText(img_display, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return img_display, pressure_value

# ==============================================================================
# 3. SCRIPT PRINCIPAL
# ==============================================================================
# Este es el orquestador: corta la imagen y llama a la función de procesado.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lee 3 manómetros desde una única imagen.')
    parser.add_argument('-i', '--image', required=True, help='Ruta a la imagen ancha con los 3 manómetros.')
    parser.add_argument('-o', '--output', help='Ruta para guardar la imagen procesada con los 3 resultados.')
    args = parser.parse_args()

    # Cargar la imagen completa
    imagen_completa = cv2.imread(args.image)
    if imagen_completa is None:
        print(f"Error: no se pudo cargar la imagen desde {args.image}")
        exit()

    # Obtener dimensiones y cortar en 3
    alto, ancho, _ = imagen_completa.shape
    tercio_ancho = ancho // 3

    img_izquierda = imagen_completa[:, 0:tercio_ancho]
    img_centro = imagen_completa[:, tercio_ancho:2*tercio_ancho]
    img_derecha = imagen_completa[:, 2*tercio_ancho:ancho]

    # Lista para guardar las imágenes procesadas y los resultados
    imagenes_procesadas = []
    valores_presion = {}

    # Procesar cada manómetro con su configuración
    print("Procesando manómetro izquierdo...")
    img_izq_proc, val_izq = procesar_manometro(img_izquierda, CONFIGS['izquierda'])
    imagenes_procesadas.append(img_izq_proc)
    valores_presion['Izquierda'] = val_izq

    print("Procesando manómetro central...")
    img_cen_proc, val_cen = procesar_manometro(img_centro, CONFIGS['centro'])
    imagenes_procesadas.append(img_cen_proc)
    valores_presion['Centro'] = val_cen

    print("Procesando manómetro derecho...")
    img_der_proc, val_der = procesar_manometro(img_derecha, CONFIGS['derecha'])
    imagenes_procesadas.append(img_der_proc)
    valores_presion['Derecha'] = val_der

    # Unir las 3 imágenes procesadas en una sola
    imagen_final = np.hstack(imagenes_procesadas)

    # Mostrar resultados
    print("\n--- Resultados Finales ---")
    for nombre, valor in valores_presion.items():
        if valor is not None:
            print(f"  - Manómetro {nombre}: {valor:.2f} bar")
        else:
            print(f"  - Manómetro {nombre}: Lectura fallida")
    print("--------------------------\n")

    cv2.imshow('Resultados de los 3 Manometros', imagen_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if args.output:
        cv2.imwrite(args.output, imagen_final)
        print(f"Imagen de resultados guardada en: {args.output}")