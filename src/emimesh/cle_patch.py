from pyclesperanto_prototype._tier0 import Image, plugin_function, create_labels_like, create_like
from pyclesperanto_prototype._tier1 import detect_label_edges, binary_not, mask, minimum_sphere, minimum_box, not_equal_constant
from pyclesperanto_prototype._tier3 import relabel_sequential
from pyclesperanto_prototype._tier5 import connected_components_labeling_diamond
from pyclesperanto_prototype._tier1 import copy, erode_sphere, erode_box, multiply_images
from pyclesperanto_prototype._tier4 import dilate_labels


@plugin_function(categories=['label processing', 'in assistant'], output_creator=create_labels_like)
def erode_labels(labels_input : Image, labels_destination : Image = None, radius: int = 1, relabel_islands : bool = False) -> Image:
    """Erodes labels to a smaller size. Note: Depending on the label image and the radius,
    labels may disappear and labels may split into multiple islands. Thus, overlapping labels of input and output may
    not have the same identifier.

    Notes
    -----
    * This operation assumes input images are isotropic.

    Parameters
    ----------
    labels_input : Image
        label image to erode
    labels_destination : Image, optional
        result
    radius : int, optional
    relabel_islands : Boolean, optional
        True: Make sure that the resulting label image has connected components labeled individually
        and all label indices exist.

    Returns
    -------
    labels_destination

    See Also
    --------
    ..[1] https://clij.github.io/clij2-docs/reference_erodeLabels
    """
    if radius <= 0:
        copy(labels_input, labels_destination)
        return labels_destination

    # make a gap between labels == erosion by one pixel
    temp = create_labels_like(labels_input)
    detect_label_edges(labels_input, temp)
    temp1 = binary_not(temp)
    mask(labels_input, temp1, temp)
    del temp1

    if radius == 1:
        copy(temp, labels_destination)
    else:
        for i in range(0, int(radius) - 1):
            if i % 2 == 0:
                minimum_sphere(temp, labels_destination, 1, 1, 1)
            else:
                minimum_box(labels_destination, temp, 1, 1, 1)
    if relabel_islands:
        if radius % 2 != 0:
            copy(temp, labels_destination)
        not_equal_constant(labels_destination, temp)
        connected_components_labeling_diamond(temp, labels_destination)
    else:
        if radius % 2 == 0:
            copy(labels_destination, temp)

    return labels_destination


def opening_labels(labels_input: Image, labels_destination: Image = None, radius: int = 0) -> Image:
    """Apply a morphological opening operation to a label image.

    The operation consists of iterative erosion and dilation of the labels.
    With every iteration, box and diamond/sphere structuring elements are used
    and thus, the operation has an octagon as structuring element.

    Notes
    -----
    * This operation assumes input images are isotropic.

    Parameters
    ----------
    labels_input: Image
    labels_destination: Image, optional
    radius: int, optional

    Returns
    -------
    labels_destination: Image
    """
    if radius == 0:
        return copy(labels_input, labels_destination)

    temp = erode_labels(labels_input, radius=radius)
    return dilate_labels(temp, labels_destination, radius=radius)

def closing_labels(labels_input: Image, labels_destination: Image = None, radius: int = 0) -> Image:
    """Apply a morphological closing operation to a label image.

    The operation consists of iterative dilation and erosion of the labels.
    With every iteration, box and diamond/sphere structuring elements are used
    and thus, the operation has an octagon as structuring element.

    Notes
    -----
    * This operation assumes input images are isotropic.

    Parameters
    ----------
    labels_input: Image
    labels_destination: Image, optional
    radius: int, optional

    Returns
    -------
    labels_destination: Image
    """
    if radius == 0:
        return copy(labels_input, labels_destination)

    temp = dilate_labels(labels_input, radius=radius)

    flip = temp > 0
    flop = create_like(temp)
    for i in range(int(radius)):
        if i % 2 == 0:
            erode_sphere(flip, flop)
        else:
            erode_box(flop, flip)
    if radius % 2 == 0:
        return multiply_images(flip, temp, labels_destination)
    else:
        return multiply_images(flop, temp, labels_destination)