# -*- coding: utf-8 -*-


class BoundingBox(object):
    def __init__(self, *, x1: float, x2: float, y1: float, y2: float):
        """
        A bounding box is a face defined by 4 edges on a 2D plane. It must have
        a non-zero width and height.
        :param x1: Left edge of the bounding box
        :param x2: Right edge of the bounding box
        :param y1: Bottom edge of the bounding box
        :param y2: Top edge of the bounding box
        """

        # Force the provided values to the correct edges
        self.x_left = float(min(x1, x2))
        self.x_right = float(max(x1, x2))
        self.y_bottom = float(min(y1, y2))
        self.y_top = float(max(y1, y2))

        if self.x_right <= self.x_left:
            raise ValueError('Bounding box has zero width')
        elif self.y_top <= self.y_bottom:
            raise ValueError('Bounding box has zero height')

    @property
    def area(self) -> float:
        """ Return the area of the bounding box in natural units """
        return (self.x_right - self.x_left) * (self.y_top - self.y_bottom)

    def intersection(self, *, bb2: 'BoundingBox') -> float:
        """
        Calculates the intersection area between this bounding box and a
        second, axis aligned, bounding box. Returns 0 if the two bounding boxes
        do not intersect.
        :param bb2: The second bounding box
        :return: The intersection area in natural units
        """

        # Get the face that is the intersection between the 8 edges
        x_left = max(self.x_left, bb2.x_left)
        x_right = min(self.x_right, bb2.x_right)

        y_bottom = max(self.y_bottom, bb2.y_bottom)
        y_top = min(self.y_top, bb2.y_top)

        if x_right <= x_left or y_top <= y_bottom:
            # The two bounding boxes do not intersect
            return 0.0
        else:
            return BoundingBox(
                x1=x_left, x2=x_right, y1=y_bottom, y2=y_top
            ).area

    def union(self, *, bb2: 'BoundingBox') -> float:
        """
        Calculates the union between this bounding box and a second,
        axis aligned, bounding box.
        :param bb2: The second bounding box
        :return: The union area in natural units
        """
        return self.area + bb2.area - self.intersection(bb2=bb2)

    def intersection_over_union(self, *, bb2: 'BoundingBox') -> float:
        """
        Calculates the intersection over union between this bounding box and a
        second, axis aligned, bounding box.
        :param bb2: The second bounding box
        :return: The intersection over union in natural units
        """
        return self.intersection(bb2=bb2) / self.union(bb2=bb2)
