class Court:
    def __init__(self):
        self.length = 23.77  # Full length of the court in meters
        self.width = 8.23  # Full width of the court in meters (singles court)
        self.net_height_posts = 1.07  # Height of the net in meters at the posts
        self.net_height_center = 0.914  # Height of the net in meters at the center
        self.service_line_distance = 6.40  # Distance from the net to the service line in meters

    @property
    def net_y(self) -> float:
        """y-coordinate of the net measured from the server baseline."""
        return self.length / 2.0
        