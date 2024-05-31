class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = None
        self.labels = None
        self.continent_names = []

    def update_data(self, data):
        self.data = data

    def update_labels(self, labels):
        self.labels = labels

    def update_continent_names(self, names):
        self.continent_names = names
