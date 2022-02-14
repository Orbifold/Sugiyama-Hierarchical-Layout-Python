class VertexViewer(object):
    """
    The VertexViewer class is used as the default provider of
    Vertex dimensions (w,h) and position (xy).
    In most cases it should be replaced by *view* instances associated
    with a ui widgets library, allowing to get dimensions and
    set position directly on the widget.
    """

    def __init__(self, w=2, h=2, data=None):
        self.w = w
        self.h = h
        self.data = data
        self.xy = None

    def __str__(self, *args, **kwargs):
        return "VertexViewer (xy: %s) w: %s h: %s" % (self.xy, self.w, self.h)