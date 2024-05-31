import pygame
from core.game import Game


def main():
    pygame.init()

    # Get the display's current resolution
    display_info = pygame.display.Info()
    screen_width, screen_height = display_info.current_w, display_info.current_h

    # Set the display mode to fullscreen and borderless
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN | pygame.NOFRAME)

    game = Game(screen, screen_width, screen_height)
    game.run()

    pygame.quit()


if __name__ == "__main__":
    main()
