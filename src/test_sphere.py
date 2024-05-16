import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import os
import ctypes

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
            tex_coords.append((float(j) / longs, float(i) / lats))
            
            vertices.append((x * zr1 * radius, y * zr1 * radius, z1 * radius))
            tex_coords.append((float(j) / longs, float(i + 1) / lats))
            
    return vertices, tex_coords

def create_vbo(vertices, tex_coords):
    vertices = np.array(vertices, dtype=np.float32)
    tex_coords = np.array(tex_coords, dtype=np.float32)

    vbo_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes + tex_coords.nbytes, None, GL_STATIC_DRAW)
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
    glBufferSubData(GL_ARRAY_BUFFER, vertices.nbytes, tex_coords.nbytes, tex_coords)

    return vbo_id

def draw_sphere(vbo_id, num_vertices):
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_TEXTURE_COORD_ARRAY)

    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
    glVertexPointer(3, GL_FLOAT, 0, None)
    glTexCoordPointer(2, GL_FLOAT, 0, ctypes.c_void_p(num_vertices * 12))

    glDrawArrays(GL_TRIANGLE_STRIP, 0, num_vertices)

    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_TEXTURE_COORD_ARRAY)

# Load the image texture
image_path = os.path.join("assets", "Marshy_04-512x512.png")
texture = load_texture(image_path)

# Generate sphere vertices and texture coordinates
vertices, tex_coords = generate_sphere(2.0, 30, 30)
vbo_id = create_vbo(vertices, tex_coords)
num_vertices = len(vertices)

# Main loop
glEnable(GL_DEPTH_TEST)
glEnable(GL_TEXTURE_2D)  # Ensure texturing is enabled
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
    draw_sphere(vbo_id, num_vertices)
    
    pygame.display.flip()
    pygame.time.wait(10)

pygame.quit()
