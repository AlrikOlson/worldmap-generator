# src/main.py
import pygame
from core.game import Game  # Relative import

def main():
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))

    game = Game(screen)
    game.run()

    pygame.quit()

if __name__ == "__main__":
    main()
