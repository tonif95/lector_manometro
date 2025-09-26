import cv2
import numpy as np
import math
import argparse

def get_pressure_reading(image_path, output_path=None):
    """
    Analiza la imagen de un manómetro y devuelve el valor de la presión.
    También guarda una imagen con las detecciones visualizadas.
    """
    
    # --- 1. CONFIGURACIÓN Y CALIBRACIÓN DEL MANÓMETRO ---
    MIN_VALUE = 0
    MAX_VALUE = 25
    START_ANGLE = 210
    END_ANGLE = 330
    
    # --- 2. CARGA Y PREPROCESAMIENTO DE LA IMAGEN ---
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Error: No se pudo cargar la imagen desde {image_path}")
        return None
        
    img_display = img_orig.copy()
    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # --- 3. DETECCIÓN DEL CÍRCULO (CARA DEL MANÓMETRO) ---
    detected_circles = cv2.HoughCircles(blurred,
                                       cv2.HOUGH_GRADIENT,
                                       dp=1,
                                       minDist=50,
                                       param1=50,
                                       param2=60, 
                                       minRadius=50,
                                       maxRadius=300)

    if detected_circles is None:
        print("No se detectó ningún círculo.")
        return None

    circle = np.uint16(np.around(detected_circles))[0, 0]
    center_x, center_y, radius = circle[0], circle[1], circle[2]

    cv2.circle(img_display, (center_x, center_y), radius, (0, 255, 0), 3)
    cv2.circle(img_display, (center_x, center_y), 5, (0, 0, 255), -1)

    # --- 4. DETECCIÓN DE LA AGUJA (LÍNEA) ---
    mask = np.zeros_like(gray)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, None,
                           int(radius * 0.3), int(radius * 0.95)) 

    if lines is None:
        print("No se detectó ninguna línea (aguja) dentro del círculo.")
        return None

    # --- NUEVA LÓGICA MÁS ROBUSTA PARA ENCONTRAR LA AGUJA ---
    # Encontraremos la línea cuyo punto medio esté más cerca del centro del círculo.
    best_line = None
    min_dist_from_midpoint_to_center = float('inf')

    for line in lines:
        x1, y1, x2, y2 = line[0]
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        dist = np.sqrt((mid_x - center_x)**2 + (mid_y - center_y)**2)
        if dist < min_dist_from_midpoint_to_center:
            min_dist_from_midpoint_to_center = dist
            best_line = (x1, y1, x2, y2)
    
    if best_line is None:
        # Este error ahora es mucho menos probable que ocurra.
        print("No se pudo determinar la línea de la aguja con los criterios actuales.")
        return None

    x1, y1, x2, y2 = best_line
    cv2.line(img_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    dist1_from_center = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
    dist2_from_center = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)

    if dist1_from_center > dist2_from_center:
        tip_x, tip_y = x1, y1
    else:
        tip_x, tip_y = x2, y2
        
    cv2.circle(img_display, (tip_x, tip_y), 5, (0, 165, 255), -1)

    # --- 5. MAPEAR EL ÁNGULO AL VALOR DE PRESIÓN ---
    angle_rad = math.atan2(tip_y - center_y, tip_x - center_x)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360

    pressure_value = 0.0
    
    if END_ANGLE >= START_ANGLE:
        total_angle_sweep = END_ANGLE - START_ANGLE
        current_angle_relative = angle_deg - START_ANGLE
    else:
        total_angle_sweep = (360 - START_ANGLE) + END_ANGLE
        if angle_deg >= START_ANGLE:
            current_angle_relative = angle_deg - START_ANGLE
        else:
            current_angle_relative = (360 - START_ANGLE) + angle_deg

    if total_angle_sweep > 0:
        if current_angle_relative < 0:
            pressure_value = MIN_VALUE
        elif current_angle_relative > total_angle_sweep:
            pressure_value = MAX_VALUE
        else:
            pressure_percentage = current_angle_relative / total_angle_sweep
            pressure_value = MIN_VALUE + pressure_percentage * (MAX_VALUE - MIN_VALUE)

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
    parser = argparse.ArgumentParser(description='Lee un manómetro analógico desde una imagen.')
    parser.add_argument('-i', '--image', required=True, help='Ruta a la imagen de entrada.')
    parser.add_argument('-o', '--output', help='Ruta para guardar la imagen procesada.')
    args = parser.parse_args()
    get_pressure_reading(args.image, args.output)