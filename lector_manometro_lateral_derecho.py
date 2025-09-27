import cv2
import numpy as np
import math
import argparse

def get_pressure_reading(image_path, output_path=None):
    """
    Analiza la imagen de un manómetro (incluso si se ve como una elipse) 
    y devuelve el valor de la presión.
    """
    
    # --- 1. CONFIGURACIÓN Y CALIBRACIÓN DEL MANÓMETRO ---
    MIN_VALUE = 0
    MAX_VALUE = 25
    START_ANGLE = 130 # Ajustar con la línea magenta
    END_ANGLE = 50   # Ajustar con la línea amarilla
    
    # --- 2. CARGA Y PREPROCESAMIENTO ---
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Error: No se pudo cargar la imagen desde {image_path}")
        return None
        
    img_display = img_orig.copy()
    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # --- 3. DETECCIÓN DE LA ELIPSE Y DIBUJO DE GUÍAS ---
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_ellipse = None
    best_area = 0

    for cnt in contours:
        if len(cnt) < 20: # Mínimo 20 puntos para ajustar una elipse fiable
            continue

        area = cv2.contourArea(cnt)
        if area < 1000: # Filtra contornos muy pequeños
            continue

        ellipse = cv2.fitEllipse(cnt)
        
        (center, axes, angle) = ellipse
        major_axis, minor_axis = max(axes), min(axes)
        if major_axis == 0 or minor_axis == 0:
            continue
        
        aspect_ratio = major_axis / minor_axis
        if aspect_ratio > 1.5: # Filtra elipses muy alargadas
            continue

        if area > best_area:
            best_area = area
            best_ellipse = ellipse

    if best_ellipse is None:
        print("No se pudo detectar una elipse que parezca un manómetro.")
        cv2.imshow('Deteccion Fallida', img_orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

    (center, axes, angle) = best_ellipse
    center_x, center_y = int(center[0]), int(center[1])
    # Usamos el radio promedio de la elipse
    radius = int((axes[0] + axes[1]) / 4) 

    cv2.ellipse(img_display, best_ellipse, (0, 255, 0), 3)
    cv2.circle(img_display, (center_x, center_y), 5, (0, 0, 255), -1)

    # --- LÍNEAS DE CALIBRACIÓN ---
    start_angle_rad = math.radians(START_ANGLE)
    end_angle_rad = math.radians(END_ANGLE)
    
    line_radius = int((axes[0] + axes[1]) / 2.5) # Dibuja las líneas hasta un punto visible

    x_start = int(center_x + line_radius * math.cos(start_angle_rad))
    y_start = int(center_y + line_radius * math.sin(start_angle_rad))
    cv2.line(img_display, (center_x, center_y), (x_start, y_start), (255, 0, 255), 2) # Magenta

    x_end = int(center_x + line_radius * math.cos(end_angle_rad))
    y_end = int(center_y + line_radius * math.sin(end_angle_rad))
    cv2.line(img_display, (center_x, center_y), (x_end, y_end), (0, 255, 255), 2) # Amarilla

    # --- 4. DETECCIÓN DE LA AGUJA ---
    mask = np.zeros_like(gray)
    cv2.ellipse(mask, best_ellipse, 255, -1)
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 30, None,
                           int(radius * 0.2), int(radius * 1.5))

    if lines is None:
        print("No se detectó ninguna línea (aguja) dentro de la elipse.")
        cv2.putText(img_display, "Aguja no detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Manometro Detectado', img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

    best_line = None
    max_line_length = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        dist_to_center = np.sqrt((mid_x - center_x)**2 + (mid_y - center_y)**2)
        
        if dist_to_center > radius * 0.5: 
            continue

        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length > max_line_length:
            max_line_length = length
            best_line = (x1, y1, x2, y2)
    
    if best_line is None:
        print("No se pudo determinar la línea de la aguja.")
        return None

    x1, y1, x2, y2 = best_line
    cv2.line(img_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    dist1_from_center = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
    dist2_from_center = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)

    # Inicialmente, el punto más lejano es la punta y el más cercano es la base
    if dist1_from_center > dist2_from_center:
        tip_x_original, tip_y_original = x1, y1
        base_x_original, base_y_original = x2, y2
    else:
        tip_x_original, tip_y_original = x2, y2
        base_x_original, base_y_original = x1, y1
    
    # --- ¡CORRECCIÓN CLAVE AQUÍ! ---
    # Si la lógica anterior te da el contrapeso como punta (la parte trasera de la aguja),
    # entonces el punto real que indica la presión es el otro extremo (la base original).
    # Forzamos la inversión de la punta:
    tip_x, tip_y = base_x_original, base_y_original
    
    cv2.circle(img_display, (tip_x, tip_y), 5, (0, 165, 255), -1) # Punto naranja en la punta corregida

    # --- 5. MAPEAR EL ÁNGULO AL VALOR DE PRESIÓN ---
    angle_rad = math.atan2(tip_y - center_y, tip_x - center_x)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360

    pressure_value = 0.0
    total_angle_sweep = (END_ANGLE - START_ANGLE + 360) % 360
    current_angle_relative = (angle_deg - START_ANGLE + 360) % 360

    if total_angle_sweep > 0 and current_angle_relative <= total_angle_sweep:
        pressure_percentage = current_angle_relative / total_angle_sweep
        pressure_value = MIN_VALUE + pressure_percentage * (MAX_VALUE - MIN_VALUE)
    elif current_angle_relative > total_angle_sweep and total_angle_sweep != 0:
        if current_angle_relative > (total_angle_sweep / 2 + 180) % 360:
             pressure_value = MIN_VALUE
        else:
             pressure_value = MAX_VALUE
    
    pressure_value = max(MIN_VALUE, min(MAX_VALUE, pressure_value))

    # --- 6. MOSTRAR Y GUARDAR RESULTADOS ---
    result_text = f"Presion: {pressure_value:.2f} bar" 
    cv2.putText(img_display, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(result_text)

    cv2.imshow('Manometro Detectado', img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if output_path:
        cv2.imwrite(output_path, img_display)
        print(f"Imagen procesada guardada en: {output_path}")
    
    return pressure_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lee un manómetro analógico (circular o elíptico) desde una imagen.')
    parser.add_argument('-i', '--image', required=True, help='Ruta a la imagen de entrada.')
    parser.add_argument('-o', '--output', help='Ruta para guardar la imagen procesada.')
    args = parser.parse_args()
    get_pressure_reading(args.image, args.output)