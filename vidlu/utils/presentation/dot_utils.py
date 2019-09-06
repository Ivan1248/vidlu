from io import BytesIO

import matplotlib.image as mpimg

import pydot


def to_numpy_image(graph: pydot.Dot, *args, **kwargs):
    png_str = graph.create_png(*args, **kwargs, prog='dot')
    sio = BytesIO()
    sio.write(png_str)
    sio.seek(0)
    return mpimg.imread(sio)

def show(graph: pydot.Dot, *args, **kwargs):
    import matplotlib.pyplot as plt
    img = to_numpy_image(graph, *args, **kwargs)
    plt.imshow(img, interpolation="bilinear")
    plt.tight_layout(0)
    plt.axis('off')
    plt.show()

