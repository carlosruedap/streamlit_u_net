import numpy as np
import cv2


def adjust_mask(mask):

    if mask.dtype != np.uint8:
        mask = (mask * 255).astype("uint8")

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fix_mask = np.zeros_like(mask, dtype=np.uint8)

    for cont in contours:
        (x, y), radius = cv2.minEnclosingCircle(cont)
        center = (int(x), int(y))
        diameter = int(2 * radius)
        color = (255, 255, 255)
        cv2.circle(fix_mask, center, diameter // 2, color, -1)

    return fix_mask/255


def count_colonies(colonies_by_wells):
    colony_counts = []
    total_colonies = 0
    colonies_by_wells = np.array(colonies_by_wells)
    min_area = 20

    for well in colonies_by_wells:
        if len(well.shape) == 3 and well.shape[2] == 3:
            gray_well = cv2.cvtColor(well, cv2.COLOR_BGR2GRAY)
        else:
            gray_well = well

        # Aplicar un filtro Gaussiano para suavizar la imagen
        gauss = cv2.GaussianBlur(gray_well, (3, 3), 0)

        # Detectar los bordes Canny
        canny = cv2.Canny(gauss.astype('uint8'), 90, 120)

        # Buscar los contornos
        contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

        num_colonies = len(contours)
        # num_colonies = len(filtered_contours)

        colony_counts.append(num_colonies)
        total_colonies += num_colonies

    return colony_counts, total_colonies


def find_index_mask(mask_segment):

    mask_segment = adjust_mask(mask_segment)

    if len(mask_segment.shape) == 3 and mask_segment.shape[2] == 3:
        gray_mask = cv2.cvtColor(mask_segment, cv2.COLOR_BGR2GRAY)
    else:
        gray_mask = mask_segment

    if gray_mask.dtype != np.uint8:
        gray_mask = (gray_mask * 255).astype("uint8")

    # Aplicar un desenfoque Gaussiano para reducir el ruido
    # gray_mask = cv2.GaussianBlur(gray_mask, (9, 9), 0)

    gray_mask = cv2.blur(gray_mask, (9, 9))

    # Detectar círculos usando el método de Hough
    circle = cv2.HoughCircles(gray_mask, cv2.HOUGH_GRADIENT, 1, 100,
                              param1=250, param2=127, minRadius=0, maxRadius=0)

    if circle is not None:
        circle = np.round(circle[0, :]).astype("int")
        x_center, y_center, _ = circle[0]
        # print(f"Centro del círculo: ({x_center}, {y_center})")

        rows, columns = mask_segment.shape[:2]  # 100% rows ; 100% columns

        m_rows = np.round(rows / 2)  # 50% rows
        m_columns = np.round(columns / 2)  # 50% columns

        desviacion = abs(1 - (x_center / m_columns))

        if y_center < m_rows:  # Parte de arriba
            if desviacion > 0.2:
                if x_center < m_columns:
                    return 1  # Primer pozo
                else:
                    return 3  # Tercer pozo
            else:
                return 2  # Segundo pozo
        else:  # Parte de abajo
            if desviacion > 0.2:
                if x_center < m_columns:
                    return 4  # Cuarto pozo
                else:
                    return 6  # Sexto pozo
            else:
                return 5  # Quinto pozo
    print("No se detectaron circulos")
    return None









