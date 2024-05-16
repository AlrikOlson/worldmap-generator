import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import os

# Initialize Pygame and OpenGL
pygame.init()
screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
pygame.display.set_caption('Spherical Map Projection')

def load_texture(image_path):
    image = pygame.image.load(image_path)
    texture_data = pygame.image.tostring(image, "RGB", True)
    width, height = image.get_size()
    
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    
    glGenerateMipmap(GL_TEXTURE_2D)
    
    return texture

def generate_sphere(radius, lats, longs):
    vertices = []
    tex_coords = []
    for i in range(lats + 1):
        lat0 = np.pi * (-0.5 + float(i - 1) / lats)
        z0 = np.sin(lat0)
        zr0 = np.cos(lat0)
        
        lat1 = np.pi * (-0.5 + float(i) / lats)
        z1 = np.sin(lat1)
        zr1 = np.cos(lat1)
        
        for j in range(longs + 1):
            lng = 2 * np.pi * float(j - 1) / longs
            x = np.cos(lng)
            y = np.sin(lng)
            
            vertices.append((x * zr0 * radius, y * zr0 * radius, z0 * radius))
            tex_coords.append((1 - float(j) / longs, 1 - float(i) / lats))
            
            vertices.append((x * zr1 * radius, y * zr1 * radius, z1 * radius))
            tex_coords.append((1 - float(j) / longs, 1 - float(i + 1) / lats))
            
    return vertices, tex_coords

def draw_sphere(vertices, tex_coords):
    glEnable(GL_TEXTURE_2D)
    glBegin(GL_QUAD_STRIP)
    for i, (vertex, tex_coord) in enumerate(zip(vertices, tex_coords)):
        glTexCoord2fv(tex_coord)
        glVertex3fv(vertex)
    glEnd()
    glDisable(GL_TEXTURE_2D)

# Load the image texture
image_path = os.path.join("assets", "Marshy_04-512x512.png")
texture = load_texture(image_path)

# Generate sphere vertices and texture coordinates
vertices, tex_coords = generate_sphere(2.0, 60, 60)

# Main loop
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_PROJECTION)
gluPerspective(45, (800 / 600), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)
glTranslatef(0.0, 0.0, -5)

running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glRotatef(1, 3, 1, 1)
    
    glBindTexture(GL_TEXTURE_2D, texture)
    draw_sphere(vertices, tex_coords)
    
    pygame.display.flip()
    pygame.time.wait(10)

pygame.quit()
