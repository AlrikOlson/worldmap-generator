import pygame
import threading
from src.ui.button import Button
from src.core.world import World
from src.core.world_generator import WorldGenerator
from src.core.world_renderer import WorldRenderer
from src.core.continent_labeler import ContinentLabeler

# Constants
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 50
BUTTON_MARGIN = 10
QUIT_TEXT_SIZE = 36
CONTINENT_TEXT_SIZE = 24
TEXT_X_OFFSET = 200
START_Y_OFFSET = 20
CONTINENT_TEXT_Y_PADDING = 30
PROGRESS_TEXT_X = 10
PROGRESS_TEXT_Y = 10
BACKGROUND_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
QUIT_BUTTON_COLOR = (150, 0, 0)
QUIT_BUTTON_HOVER_COLOR = (200, 0, 0)
FPS = 60


class Game:
    def __init__(self, screen, map_width, map_height):
        self.screen = screen
        self.map_width = map_width
        self.map_height = map_height
        self.clock = pygame.time.Clock()
        self.running = True
        self.world = World(map_width, map_height)
        self.world_generator = WorldGenerator(map_width, map_height, scale=400.0, octaves=48)
        self.continent_labeler = ContinentLabeler(self.world)
        self.world_renderer = WorldRenderer(self.world, self.screen)
        self.progress = 0
        self.progress_text = ""
        self.generate_world()

        self.quit_button = Button(
            rect=(map_width - BUTTON_WIDTH - BUTTON_MARGIN, map_height - BUTTON_HEIGHT - BUTTON_MARGIN, BUTTON_WIDTH,
                  BUTTON_HEIGHT),
            color=QUIT_BUTTON_COLOR,
            hover_color=QUIT_BUTTON_HOVER_COLOR,
            text='Quit',
            font=pygame.font.Font(None, QUIT_TEXT_SIZE),
            text_color=TEXT_COLOR,
            callback=self.quit_game
        )

    def generate_world(self):
        self.progress = 0
        self.progress_text = ""
        threading.Thread(target=self.run_map_generation).start()

    def run_map_generation(self):
        world_data = self.world_generator.generate(self.update_progress)
        self.world.update_data(world_data)
        self.continent_labeler.label_continents()
        self.world_renderer.update_world_surface()

    def update_progress(self, progress, text=""):
        self.progress = progress
        self.progress_text = text

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.generate_world()
            self.quit_button.handle_event(event)

    def update(self):
        pass

    def render(self):
        self.world_renderer.render()
        self.render_progress()
        self.quit_button.draw(self.screen)
        pygame.display.flip()

    def render_progress(self):
        if self.progress < 100:
            font = pygame.font.Font(None, QUIT_TEXT_SIZE)
            text = font.render(f'{self.progress_text}: {self.progress:.1f}%', True, TEXT_COLOR)
            self.screen.blit(text, (PROGRESS_TEXT_X, PROGRESS_TEXT_Y))

    def quit_game(self):
        self.running = False
