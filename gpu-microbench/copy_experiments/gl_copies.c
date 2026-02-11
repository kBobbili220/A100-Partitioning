// Built off the Triangle and Checkered Triangle OpenGL examples by Ray Toal
// (https://cs.lmu.edu/~ray/notes/openglexamples/)
#include <stdio.h>

#include <GL/glut.h>

#define red {0xff, 0x00, 0x00}
#define yellow {0xff, 0xff, 0x00}
#define magenta {0xff, 0, 0xff}

GLubyte* texture;

const unsigned side = 1024*10; // 30 MiB

// Clears the current window and draws a triangle.
void display() {

  // Set every pixel in the frame buffer to the current clear color.
  glClear(GL_COLOR_BUFFER_BIT);

  // Re-upload the texture on every frame
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, side, side, GL_RGB, GL_UNSIGNED_BYTE, texture);

  // Drawing is done by specifying a sequence of vertices.  The way these
  // vertices are connected (or not connected) depends on the argument to
  // glBegin.  GL_POLYGON constructs a filled polygon.
  glBegin(GL_POLYGON);
  glTexCoord2f(0.5, side); glVertex3f(-0.6, -0.75, 0.5);
  glTexCoord2f(0.0, 0.0); glVertex3f(0.6, -0.75, 0);
  glTexCoord2f(side, 0.0); glVertex3f(0, 0.75, 0);
  glEnd();

  // Flush drawing command buffer to make drawing happen as soon as possible.
  glFlush();

  // Request a redraw on the next loop
  glutPostRedisplay();
}

// Initializes GLUT, the display mode, and main window; registers callbacks;
// enters the main event loop.
int main(int argc, char** argv) {
  // No arguments are accepted
  if (argc != 1) {
    fprintf(stderr, "Usage: %s\n", argv[0]);
    return 1;
  }

  // 30 MiB image
  texture = malloc(side*side*3u);

  // Add a checkered pattern to the texture
  for (unsigned i = 0; i < side*side; i++) {
    if (i % 2 == 0) { // Yellow
      texture[i*3] = 0xff;
      texture[i*3+1] = 0xff;
    } else // Red
      texture[i*3] = 0xff;
  }

  // Use a single buffered window in RGB mode (as opposed to a double-buffered
  // window or color-index mode).
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);

  // Position window at (80,80)-(480,380) and give it a title.
  glutInitWindowPosition(80, 80);
  glutInitWindowSize(400, 300);
  glutCreateWindow("A Simple Triangle");

  // Tell GLUT that whenever the main window needs to be repainted that it
  // should call the function display().
  glutDisplayFunc(display);

  glEnable(GL_TEXTURE_2D);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D,
               0,                    // level 0
               3,                    // use only R, G, and B components
               side, side,           // texture has 2x2 texels
               0,                    // no border
               GL_RGB,               // texels are in RGB format
               GL_UNSIGNED_BYTE,     // color components are unsigned bytes
               texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // Tell GLUT to start reading and processing events.  This function
  // never returns; the program only exits when the user closes the main
  // window or kills the process.
  glutMainLoop();
}
