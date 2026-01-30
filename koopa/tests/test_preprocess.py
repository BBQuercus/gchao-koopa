import numpy as np
import pytest

from koopa import preprocess


@pytest.fixture
def image():
    image = np.zeros((20, 100, 100))
    image[9, 50:, 50:] = 1
    image[10] = np.random.random((100, 100))
    return image


def test_get_sharpest_slice(image):
    sharpest_slice = preprocess.get_sharpest_slice(image)
    assert (sharpest_slice == image[10]).all()


@pytest.mark.parametrize("method", ["maximum", "mean", "sharpest"])
def test_register_3d_image(image, method):
    output = preprocess.register_3d_image(
        np.expand_dims(image, axis=0), method
    ).squeeze()
    assert output.shape == (100, 100)


def test_crop_image(image):
    output = preprocess.crop_image(np.expand_dims(image, axis=0), 50, 100).squeeze()
    assert (output[9] == 1).all()


def test_bin_image(image):
    output = preprocess.bin_image(image, (1, 0.5, 0.5))
    assert output.shape == (20, 50, 50)

    # Add 1 padding for interpolation
    assert (output[9, 26:, 26:] == 1).all()


def test_trim_image(image):
    output = preprocess.trim_image(np.expand_dims(image, axis=0), 8, 16).squeeze()
    assert len(output) == 8
    assert (output[1] == image[9]).all()


class TestRegister3dImageZRange:
    """Tests for z_start/z_end parameters in register_3d_image."""

    @pytest.fixture
    def stack_4d(self):
        """Create a 4D image with distinct z-slices for testing.

        Shape: (2, 20, 50, 50) - 2 channels, 20 z-slices, 50x50 pixels.
        Each z-slice has a unique value equal to its index.
        """
        image = np.zeros((2, 20, 50, 50), dtype=np.float32)
        for z in range(20):
            image[:, z, :, :] = z
        return image

    def test_default_includes_all_slices(self, stack_4d):
        """With defaults (z_start=0, z_end=0), all z-slices are included."""
        result = preprocess.register_3d_image(stack_4d, "maximum")
        # Maximum across all z (0-19) should be 19
        assert result.shape == (2, 50, 50)
        assert np.allclose(result, 19)

    def test_z_start_skips_slices(self, stack_4d):
        """z_start skips the first N slices."""
        result = preprocess.register_3d_image(stack_4d, "maximum", z_start=10)
        # Maximum across z 10-19 should be 19
        assert result.shape == (2, 50, 50)
        assert np.allclose(result, 19)

    def test_z_end_limits_slices(self, stack_4d):
        """z_end limits to slices before z_end (exclusive)."""
        result = preprocess.register_3d_image(stack_4d, "maximum", z_end=10)
        # Maximum across z 0-9 should be 9
        assert result.shape == (2, 50, 50)
        assert np.allclose(result, 9)

    def test_z_start_and_z_end_selects_range(self, stack_4d):
        """Both z_start and z_end select a specific range."""
        result = preprocess.register_3d_image(stack_4d, "maximum", z_start=5, z_end=15)
        # Maximum across z 5-14 should be 14
        assert result.shape == (2, 50, 50)
        assert np.allclose(result, 14)

    def test_z_range_with_mean_method(self, stack_4d):
        """z_start/z_end work with mean projection method."""
        result = preprocess.register_3d_image(stack_4d, "mean", z_start=5, z_end=15)
        # Mean of z 5-14 = (5+6+7+8+9+10+11+12+13+14) / 10 = 9.5
        assert result.shape == (2, 50, 50)
        assert np.allclose(result, 9.5)

    def test_z_range_single_slice(self, stack_4d):
        """Selecting a single z-slice works."""
        result = preprocess.register_3d_image(stack_4d, "maximum", z_start=7, z_end=8)
        # Only z=7 is included
        assert result.shape == (2, 50, 50)
        assert np.allclose(result, 7)

    def test_z_start_zero_is_same_as_default(self, stack_4d):
        """z_start=0 should be equivalent to not specifying it."""
        result_default = preprocess.register_3d_image(stack_4d, "maximum")
        result_explicit = preprocess.register_3d_image(stack_4d, "maximum", z_start=0)
        assert np.allclose(result_default, result_explicit)

    def test_z_end_zero_is_same_as_default(self, stack_4d):
        """z_end=0 should be equivalent to not specifying it."""
        result_default = preprocess.register_3d_image(stack_4d, "maximum")
        result_explicit = preprocess.register_3d_image(stack_4d, "maximum", z_end=0)
        assert np.allclose(result_default, result_explicit)
