# tests/test_map_generator.py
import unittest
from src.procedural.map_generator import MapGenerator

class TestMapGeneration(unittest.TestCase):
    def test_map_generation(self):
        width, height = 100, 100
        map_gen = MapGenerator(width, height)
        world = map_gen.generate()

        self.assertEqual(world.shape, (width, height))
        self.assertTrue((world >= -1).all() and (world <= 1).all(), "Map values should be between -1 and 1")

if __name__ == '__main__':
    unittest.main()
