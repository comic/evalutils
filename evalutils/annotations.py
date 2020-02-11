import logging

logger = logging.getLogger(__name__)


class BoundingBox:
    def __init__(self, *, x1: float, x2: float, y1: float, y2: float):
        """
        A bounding box is a face defined by 4 edges on a 2D plane. It must have
        a non-zero width and height.

        Parameters
        ----------
        x1
            Left edge of the bounding box
        x2
            Right edge of the bounding box
        y1
            Bottom edge of the bounding box
        y2
            Top edge of the bounding box

        Raises
        ------
        ValueError
            If the bounding box has zero width or height

        """

        # Force the provided values to the correct edges
        self.x_left = float(min(x1, x2))
        self.x_right = float(max(x1, x2))
        self.y_bottom = float(min(y1, y2))
        self.y_top = float(max(y1, y2))

        if self.x_right <= self.x_left:
            raise ValueError("Bounding box has zero width")
        elif self.y_top <= self.y_bottom:
            raise ValueError("Bounding box has zero height")

    def __repr__(self):
        return (
            f"{self.__class__.__qualname__}"
            f"("
            f"x1={self.x_left}, "
            f"x2={self.x_right}, "
            f"y1={self.y_bottom}, "
            f"y2={self.y_top}"
            f")"
        )

    def __hash__(self):
        return hash((self.x_left, self.x_right, self.y_bottom, self.y_top))

    def __eq__(self, other: "BoundingBox"):
        if self.__class__ is other.__class__:
            return (
                self.x_left == other.x_left
                and self.x_right == other.x_right
                and self.y_bottom == other.y_bottom
                and self.y_top == other.y_top
            )
        return NotImplemented

    @property
    def area(self) -> float:
        """ Return the area of the bounding box in natural units """
        return (self.x_right - self.x_left) * (self.y_top - self.y_bottom)

    def intersection(self, *, other: "BoundingBox") -> float:
        """
        Calculates the intersection area between this bounding box and
        another, axis aligned, bounding box.

        Parameters
        ----------
        other
            The other bounding box

        Returns
        -------
        float
            The intersection area in natural units if the two bounding boxes
            overlap, zero otherwise.

        """

        # Get the face that is the intersection between the 8 edges
        x_left = max(self.x_left, other.x_left)
        x_right = min(self.x_right, other.x_right)

        y_bottom = max(self.y_bottom, other.y_bottom)
        y_top = min(self.y_top, other.y_top)

        if x_right <= x_left or y_top <= y_bottom:
            logger.warning(
                f"The bounding boxes do not intersect. bb1={self}, "
                f"other={other}."
            )
            return 0.0
        else:
            return BoundingBox(
                x1=x_left, x2=x_right, y1=y_bottom, y2=y_top
            ).area

    def union(self, *, other: "BoundingBox") -> float:
        """
        Calculates the union between this bounding box and another,
        axis aligned, bounding box.

        Parameters
        ----------
        other
            The other bounding box

        Returns
        -------
        float
            The union area in natural units

        """
        return self.area + other.area - self.intersection(other=other)

    def jaccard_index(self, *, other: "BoundingBox") -> float:
        """
        Calculates the intersection over union between this bounding box and a
        second, axis aligned, bounding box.

        Parameters
        ----------
        other
            The other bounding box

        Returns
        -------
        float
            The intersection over union in natural units

        """
        return self.intersection(other=other) / self.union(other=other)
