import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
from src.utils import count_colonies, find_index_mask
from tensorflow.keras.models import load_model
import torch  # Asegúrate de importar PyTorch

# Forzar que los modelos usen la CPU
model1 = YOLO(r'src/model_weights/best.pt').to("cpu")  # Predicción sobre los pozos en las imágenes
model2 = YOLO(r'src/model_weights/best_colonies2.pt').to("cpu")  # Predicción sobre las colonias dentro de los pozos
model_path = 'model/unet_checkpoint_24.keras'
input_size = (224, 224)
model = load_model(model_path, compile=False)

def predict(image):
    # Predict with the model
    results = model1(image, save=True, conf=0.5, show_boxes=True, retina_masks=True)  # predict on an image
    individual_mask = []  # Se almacena cada máscara individual del pozo
    index_mask = []  # Se almacena el indice de cada pozo
    for i in range(results[0].masks.shape[0]):
        mask = results[0].masks[i].data[0].numpy()
        index = find_index_mask(mask)
        index_mask.append(index)
        individual_mask.append(mask)
        if i == 0:
            full_mask = mask
        else:
            full_mask = mask + full_mask
    for i in range(0, 6):
        print(f"{i+1}. {index_mask[i]}")

    return full_mask, individual_mask, index_mask


def predict_colonies(image, individual_mask):

    #----------------------------------------------------------------------------
    # Use the CNN model to predict on individual masks
    colonies_by_wells = []
    for i, mask in enumerate(individual_mask):
        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        
        # Resize the masked image to the input size of the CNN model
        resized_image = cv2.resize(masked_image, input_size)
        normalized_image = resized_image / 255.0
        input_image = np.expand_dims(normalized_image, axis=0)
        
        # Predict using the CNN model
        prediction = model.predict(input_image)[0, :, :, 0]
        prediction = (prediction > 0.5).astype(np.uint8) * 255
        
        # Resize the prediction back to the original mask size
        prediction_resized = cv2.resize(prediction, (mask.shape[1], mask.shape[0]))
        
        # Combine the prediction with the mask
        well = prediction_resized * mask
        well = cv2.normalize(well, None, 0, 255, cv2.NORM_MINMAX)
        colonies_by_wells.append(well)
        #----------------------------------------------------------------------------

    # Predict with the model
    results = model2(image, save=True, conf=0.3, show_boxes=True, retina_masks=True)  # predict on an image
    for i in range(results[0].masks.shape[0]):
        mask = results[0].masks[i].data[0].numpy()
        if i == 0:
            full_mask = mask
        else:
            full_mask = mask + full_mask  # Combinar todas las máscaras de las colonias

    colonies_by_wells = []
    for ind_mask in individual_mask:
        well = full_mask * ind_mask  # Aislar las colonias de cada pozo
        well = cv2.normalize(well, None, 0, 255, cv2.NORM_MINMAX)
        colonies_by_wells.append(well)
    
    '''print(f"Detallar {np.array(colonies_by_wells).shape}")
    print(f"Detallar {np.array(colonies_by_wells[0]).shape}")
    print(f"Min {np.min(colonies_by_wells[0])}, Max {np.max(colonies_by_wells[0])}")
    '''
    
    colony_counts, total_colonies = count_colonies(colonies_by_wells)

    return full_mask, colony_counts, total_colonies, colonies_by_wells


def get_result(mask, original_image):
    # Convertir imágenes de PIL a numpy arrays si es necesario
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)

    # Convertir la máscara a un array de 3 canales si es necesario
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Eliminar el canal alfa si está presente en la imagen original
    if original_image.shape[2] == 4:
        original_image = original_image[:, :, :3]

    # Verifica que ambas imágenes tengan el mismo tipo de datos
    if mask.dtype != original_image.dtype:
        mask = mask.astype(original_image.dtype)

    # Asegúrate de que ambas imágenes tengan el mismo tamaño
    if mask.shape[:2] != original_image.shape[:2]:
        raise ValueError("La máscara y la imagen original deben tener el mismo tamaño")

    print(mask.shape)
    print(original_image.shape)

    # Multiplicar
    result = cv2.multiply(mask, original_image)  # Mantener solo regiones segmentadas

    return result


def preprocessing(im, threshold1, threshold2):
    imagen_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # Definir los rangos de los colores a filtrar (conservar) en HSV
    rango_bajo = np.array([threshold1, 20, 20])
    rango_alto = np.array([threshold2, 255, 255])

    # Crear una máscara para los colores dentro del rango
    mascara = cv2.inRange(imagen_hsv, rango_bajo, rango_alto)

    # Aplicar la máscara a la imagen original
    color_filtrado = cv2.bitwise_and(im, im, mask=mascara)

    return color_filtrado
