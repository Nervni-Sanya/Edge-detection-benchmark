import numpy as np
import pytest
from vector_lighting import EdgeDetector, vector_lighting, __version__

@pytest.fixture
def sample_image():
    """Создаёт тестовое RGB изображение 64x64"""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48, 0] = 255  # Красный квадрат
    img[16:48, 16:48, 1] = 100  # Зелёный компонент
    return img

def test_version():
    assert __version__ == "1.0.0"

def test_sobel_output_shape(sample_image):
    result = EdgeDetector.sobel(sample_image)
    assert result.shape == (64, 64)
    assert result.dtype == np.uint8

def test_prewitt_output_shape(sample_image):
    result = EdgeDetector.prewitt(sample_image)
    assert result.shape == (64, 64)
    assert result.dtype == np.uint8

def test_canny_output_shape(sample_image):
    result = EdgeDetector.canny(sample_image)
    assert result.shape == (64, 64)
    assert result.dtype == np.uint8
    assert np.all(np.isin(result, [0, 255]))  # Бинарный результат

def test_vector_lighting_output_shape(sample_image):
    result = EdgeDetector.vector_lighting(sample_image)
    assert result.shape == (64, 64)
    assert result.dtype == np.uint8
    assert np.all(np.isin(result, [0, 255]))

def test_vector_lighting_debug_mode(sample_image):
    result = vector_lighting(sample_image, return_debug=True)
    assert isinstance(result, dict)
    assert 'result' in result
    assert 'merged' in result
    assert 'debug' in result
    assert result['result'].shape == (64, 64)

def test_vector_lighting_invalid_input():
    gray = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError):
        vector_lighting(gray)

def test_vector_lighting_parameters(sample_image):
    # Проверка разных режимов
    for mode in [0, 1, 2, 3]:
        result = vector_lighting(sample_image, mode=mode)
        assert result.shape == (64, 64)
    
    # Проверка отключения перестановок
    result = vector_lighting(sample_image, use_permutations=False)
    assert result.shape == (64, 64)