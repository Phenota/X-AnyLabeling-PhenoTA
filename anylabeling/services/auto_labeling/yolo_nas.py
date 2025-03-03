import numbers
import os
from dataclasses import dataclass
from typing import Tuple, Optional, Union, Sized

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .engines.build_onnx_engine import OnnxBaseModel
from .model import Model
from .types import AutoLabelingResult

YOLO_NAS_DEFAULT_PROCESSING_STEPS = [
    {"DetLongMaxRescale": None},
    {"CenterPad": {"pad_value": 114}},
    # {"Standardize": {"max_value": 255.0}},
]

@dataclass
class PaddingCoordinates:
    top: int
    bottom: int
    left: int
    right: int


def _rescale_image(image: np.ndarray, target_shape: Tuple[int, int], interpolation_method: Optional = cv2.INTER_LINEAR) -> np.ndarray:
    """Rescale image to target_shape, without preserving aspect ratio.

    :param image:           Image to rescale. (H, W, C) or (H, W).
    :param target_shape:    Target shape to rescale to (H, W).
    :return:                Rescaled image.
    """
    height, width = target_shape[:2]
    return cv2.resize(image, dsize=(width, height), interpolation=interpolation_method)


def _get_bottom_right_padding_coordinates(input_shape: Tuple[int, int], output_shape: Tuple[int, int]) -> PaddingCoordinates:
    """Get parameters for padding an image to given output shape, in bottom right mode
    (i.e. image will be at top-left while bottom-right corner will be padded).

    :param input_shape:  Shape of the input image.
    :param output_shape: Shape to resize to.
    :return:             Padding parameters.
    """
    pad_height, pad_width = output_shape[0] - input_shape[0], output_shape[1] - input_shape[1]
    return PaddingCoordinates(top=0, bottom=pad_height, left=0, right=pad_width)


def _pad_image(image: np.ndarray, padding_coordinates: PaddingCoordinates, pad_value: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """Pad an image.

    :param image:       Image to shift. (H, W, C) or (H, W).
    :param pad_h:       Tuple of (padding_top, padding_bottom).
    :param pad_w:       Tuple of (padding_left, padding_right).
    :param pad_value:   Padding value. Can be a single scalar (Same value for all channels) or a tuple of values.
                        In the latter case, the tuple length must be equal to the number of channels.
    :return:            Image shifted according to padding coordinates.
    """
    pad_h = (padding_coordinates.top, padding_coordinates.bottom)
    pad_w = (padding_coordinates.left, padding_coordinates.right)

    if len(image.shape) == 3:
        _, _, num_channels = image.shape

        if isinstance(pad_value, numbers.Number):
            pad_value = tuple([pad_value] * num_channels)
        else:
            if isinstance(pad_value, Sized) and len(pad_value) != num_channels:
                raise ValueError(f"A pad_value tuple ({pad_value} length should be {num_channels} for an image with {num_channels} channels")

            pad_value = tuple(pad_value)

        constant_values = ((pad_value, pad_value), (pad_value, pad_value), (0, 0))
        # Fixes issue with numpy deprecation warning since constant_values is ragged array (Have to explicitly specify object dtype)
        constant_values = np.array(constant_values, dtype=np.object_)

        padding_values = (pad_h, pad_w, (0, 0))
    else:
        if isinstance(pad_value, numbers.Number):
            pass
        elif isinstance(pad_value, Sized):
            if len(pad_value) != 1:
                raise ValueError(f"A pad_value tuple ({pad_value} length should be 1 for a grayscale image")
            else:
                (pad_value,) = pad_value  # Unpack to a single scalar
        else:
            raise ValueError(f"Unsupported pad_value type {type(pad_value)}")

        constant_values = pad_value
        padding_values = (pad_h, pad_w)

    return np.pad(image, pad_width=padding_values, mode="constant", constant_values=constant_values)


def _rescale_and_pad_to_size(image: np.ndarray, output_shape: Tuple[int, int], swap: Tuple[int] = (2, 0, 1), pad_val: int = 114) -> Tuple[np.ndarray, float]:
    """
    Rescales image according to minimum ratio input height/width and output height/width rescaled_padded_image,
    pads the image to the target shape and finally swap axis.
    Note: Pads the image to corner, padding is not centered.

    :param image:           Image to be rescaled. (H, W, C) or (H, W).
    :param output_shape:    Target Shape.
    :param swap:            Axis's to be rearranged.
    :param pad_val:         Value to use for padding.
    :return:
        - Rescaled image while preserving aspect ratio, padded to fit output_shape and with axis swapped. By default, (C, H, W).
        - Minimum ratio between the input height/width and output height/width.
    """
    r = min(output_shape[0] / image.shape[0], output_shape[1] / image.shape[1])
    rescale_shape = (int(image.shape[0] * r), int(image.shape[1] * r))

    resized_image = _rescale_image(image=image, target_shape=rescale_shape)

    padding_coordinates = _get_bottom_right_padding_coordinates(input_shape=rescale_shape, output_shape=output_shape)
    padded_image = _pad_image(image=resized_image, padding_coordinates=padding_coordinates, pad_value=pad_val)

    padded_image = padded_image.transpose(swap)
    padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
    return padded_image, r


def scale_results_back(results,r):
    num_predictions, pred_boxes, pred_scores, pred_classes = results
    pred_boxes /= r
    return num_predictions, pred_boxes, pred_scores, pred_classes


class Preprocessing:
    """Preprocessing Handler

    Args:
        steps (List[Dict]): Preprocessing steps, list of dictionary contains name and args.
        out_shape (Tuple[int]): image out shapes [h, w].

    Examples:
        Simple preprocessing image

        >>> prep = Preprocessing([{"DetLongMaxRescale": None}])
        >>> prep(img)
    """

    def __init__(self, steps, out_shape):
        self.steps = steps
        self.out_shape = out_shape

    @staticmethod
    def __rescale_img(img, out_shape):
        """rescale func"""
        return cv2.resize(
            img, dsize=out_shape, interpolation=cv2.INTER_LINEAR
        ).astype(np.uint8)

    def _standarize(self, img, max_value):
        """standarize img based on max value"""
        # return (img / max_value).astype(np.float32), None
        return img, None

    def _det_rescale(self, img):
        """Rescale image to output based with scale factors"""
        scale_factor_h, scale_factor_w = (
            self.out_shape[0] / img.shape[0],
            self.out_shape[1] / img.shape[1],
        )
        return self.__rescale_img(img, self.out_shape), {
            "scale_factors": (scale_factor_w, scale_factor_h)
        }

    def _det_long_max_rescale(self, img):
        """Rescale image to output based on max size"""
        height, width = img.shape[:2]
        scale_factor = min(
            (self.out_shape[1] - 4) / height, (self.out_shape[0] - 4) / width
        )

        if scale_factor != 1.0:
            new_height, new_width = round(height * scale_factor), round(
                width * scale_factor
            )
            img = self.__rescale_img(img, (new_width, new_height))

        return img, {"scale_factors": (scale_factor, scale_factor)}

    def _bot_right_pad(self, img, pad_value):
        """Pad bottom and right only (palce image on top left)"""
        pad_height, pad_width = (
            self.out_shape[1] - img.shape[0],
            self.out_shape[0] - img.shape[1],
        )
        return cv2.copyMakeBorder(
            img,
            0,
            pad_height,
            0,
            pad_width,
            cv2.BORDER_CONSTANT,
            value=[pad_value] * img.shape[-1],
        ), {"padding": (0, pad_height, 0, pad_width)}

    def _center_pad(self, img, pad_value):
        """Pad center (palce image on center)"""
        pad_height, pad_width = (
            self.out_shape[1] - img.shape[0],
            self.out_shape[0] - img.shape[1],
        )
        pad_top, pad_left = pad_height // 2, pad_width // 2
        return cv2.copyMakeBorder(
            img,
            pad_top,
            pad_height - pad_top,
            pad_left,
            pad_width - pad_left,
            cv2.BORDER_CONSTANT,
            value=[pad_value] * img.shape[-1],
        ), {
            "padding": (
                pad_top,
                pad_height - pad_top,
                pad_left,
                pad_width - pad_left,
            )
        }

    def _normalize(self, img, mean, std):
        """Normalize image based on mean and stdev"""
        return (img - np.asarray(mean)) / np.asarray(std), None

    def _call_fn(self, name):
        """Call prep func from string name"""
        mapper = {
            "Standardize": self._standarize,
            "DetRescale": self._det_rescale,
            "DetLongMaxRescale": self._det_long_max_rescale,
            "BotRightPad": self._bot_right_pad,
            "CenterPad": self._center_pad,
            "Normalize": self._normalize,
        }
        return mapper[name]

    def __call__(self, img):
        """Do all preprocessing steps on single image"""
        img = img.copy()
        metadata = []
        for st in self.steps:
            if not st:  # if steps isn't None
                continue
            name, kwargs = list(st.items())[0]
            if not self._call_fn(name):
                continue
            img, meta = (
                self._call_fn(name)(img, **kwargs)
                if kwargs
                else self._call_fn(name)(img)
            )
            metadata.append(meta)

        img = cv2.dnn.blobFromImage(img, swapRB=True)
        img = img.astype(np.uint8)
        return img, metadata


class Postprocessing:
    """Postprocessing Handler

    Args:
        steps (List[Dict]): Preprocessing steps, list of dictionary contains name and args.
        iou_thres (float): Float representing NMS/IOU threshold.
        score_thres (float): image out shapes [h, w].

    Examples:
        Postprocessing outputs (boxes, scores)

        >>> postp = Postprocessing([{"DetLongMaxRescale": None}], o.45, 0.25)
        >>> prep(output, prep_metadata)
    """

    def __init__(self, steps, iou_thres, score_thres):
        self.steps = steps
        self.iou_thres = iou_thres
        self.score_thres = score_thres

    def _rescale_boxes(self, boxes, metadata):
        """Rescale boxes to original image size"""
        scale_factors_w, scale_factors_h = metadata["scale_factors"]
        boxes[:, [0, 2]] /= scale_factors_w
        boxes[:, [1, 3]] /= scale_factors_h
        return boxes

    def _shift_bboxes(self, boxes, metadata):
        """Shift boxes because of padding"""
        pad_top, _, pad_left, _ = metadata["padding"]
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top
        return boxes

    def _call_fn(self, name):
        """Call postp func from string name"""
        mapper = {
            "DetRescale": self._rescale_boxes,
            "DetLongMaxRescale": self._rescale_boxes,
            "BotRightPad": self._shift_bboxes,
            "CenterPad": self._shift_bboxes,
            "Standardize": None,
            "Normalize": None,
        }
        return mapper[name]

    def __call__(self, outputs, metadata):
        """Do all preprocessing steps on single output"""
        boxes, raw_scores = outputs
        boxes = np.squeeze(boxes, 0)

        metadata = metadata.copy()
        for st in reversed(self.steps):
            if not st:
                continue
            name, _ = list(st.items())[0]
            meta = metadata.pop()
            if not self._call_fn(name):
                continue
            boxes = self._call_fn(name)(boxes, meta)

        # change xyxy to xywh
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]

        scores = raw_scores.max(axis=2).flatten()
        classes = np.argmax(raw_scores, axis=2).flatten()
        return boxes, scores, classes


class YOLO_NAS(Model):
    """Object detection model using YOLO_NAS"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "confidence_threshold",
            "nms_threshold",
            "classes",
        ]
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "input_iou",
            "edit_iou",
            "toggle_preserve_existing_annotations",
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)
        model_name = self.config["type"]
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {model_name} model.",
                )
            )

        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        _, _, input_height, input_width = self.net.get_input_shape()
        self.preprocess = Preprocessing(
            YOLO_NAS_DEFAULT_PROCESSING_STEPS, (input_height, input_width)
        )
        self.postprocess = Postprocessing(
            YOLO_NAS_DEFAULT_PROCESSING_STEPS,
            self.config["nms_threshold"],
            self.config["confidence_threshold"],
        )
        self.filter_classes = self.config.get("filter_classes", [])
        self.nms_thres = self.config["nms_threshold"]
        self.conf_thres = self.config["confidence_threshold"]
        self.replace = True

    def set_auto_labeling_conf(self, value):
        """set auto labeling confidence threshold"""
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_iou(self, value):
        """set auto labeling iou threshold"""
        if value > 0:
            self.nms_thres = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        # blob, prep_meta = self.preprocess(image)
        # outputs = self.net.get_ort_inference(blob, extract=False)
        # boxes, scores, classes = self.postprocess(outputs, prep_meta)

        # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = image
        img, r = _rescale_and_pad_to_size(img, (640, 640))
        img = img.astype(np.uint8)
        image_bchw = np.expand_dims(img, 0)
        results = self.net.get_ort_inference(image_bchw, extract=False, squeeze=False)
        results = scale_results_back(results, r)
        num_predictions, pred_boxes, pred_scores, pred_classes = results
        num_predictions = int(num_predictions.item())
        boxes = pred_boxes[0, :num_predictions]
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]
        scores = pred_scores[0, :num_predictions]
        classes = pred_classes[0, :num_predictions]

        score_thres = self.conf_thres
        iou_thres = self.nms_thres
        selected = cv2.dnn.NMSBoxes(boxes, scores, score_thres, iou_thres)

        shapes = []
        for i in selected:
            score = float(scores[i])
            label = str(self.config["classes"][classes[i]])
            if self.filter_classes and label not in self.filter_classes:
                continue
            box = boxes[i, :].astype(np.int32).flatten()
            x, y, w, h = box[0], box[1], box[2], box[3]
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h
            shape = Shape(
                label=label, score=score, shape_type="rectangle", flags={}
            )
            shape.add_point(QtCore.QPointF(xmin, ymin))
            shape.add_point(QtCore.QPointF(xmax, ymin))
            shape.add_point(QtCore.QPointF(xmax, ymax))
            shape.add_point(QtCore.QPointF(xmin, ymax))
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=self.replace)

        return result

    def unload(self):
        del self.net
