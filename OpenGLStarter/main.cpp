#ifdef _WIN32
#include <stdlib.h> // MUST BE BEFORE GLUT ON WINDOWS
#include <gl/glut.h>
#else
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#endif
#include "perlin.h"
#include "GLUtil.h"
#include "StdWilUtil.h"
#include "Vectorf.h"
#include "Geometry.h"
#include "Timer.h"

#include <vector>
using namespace std;

Timer t ;
int w=768, h=768 ;
#include "Message.h"

static float mx, my, sbd=10.f,ptSize=1.f, lineWidth=1.f ;
vector<VertexPC> verts ;
Vector4f lightPos0, lightPos1, lightPos2, lightPos3 ;


void init() // Called before main loop to set up the program
{
  glClearColor( 0.1, 0.1, 0.1, 0.1 ) ;
  for( int i = 0 ; i < 3*1000 ; i++ )
    verts.push_back( VertexPC( Vector3f::random(0,1), Vector4f::random() ) ) ;
  
  msg( "OpenGL program", Vector2f( 20, h-20 ), White, 1.f ) ;  
}


void draw()
{
  if( !verts.size() )
  {
    error( "NOT READY" ) ; 
    return ;
  }
  t.reset() ;
  
  glEnable( GL_DEPTH_TEST ) ;
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ) ;
  
  glEnable( GL_BLEND ) ;
  glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ) ;
  
  glEnable( GL_COLOR_MATERIAL ) ;

  glViewport( 0, 0, w, h ) ;
  glMatrixMode( GL_PROJECTION ) ;
  glLoadIdentity();
  //glOrtho( -5, 5, -5, 5, 5, -5 ) ;
  gluPerspective( 45.0, 1.0, 0.5, 1000.0 ) ;
  
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt( 0, 0, sbd,   0, 0, 0,   0, 1, 0 ) ;
  
  glRotatef( my, 1, 0, 0 ) ;
  glRotatef( mx, 0, 1, 0 ) ;
  
  drawAxisLines() ;
  drawDebug() ;
  
  glEnable( GL_LIGHTING ) ;
  glEnable( GL_LIGHT0 ) ;
  glEnable( GL_LIGHT1 ) ;
  glEnable( GL_LIGHT2 ) ;
  glEnable( GL_LIGHT3 ) ;

  float ld = 50.f ;
  lightPos0 = Vector4f(  ld,  ld,  ld, 1 ) ;
  lightPos1 = Vector4f( -ld, -ld, -ld, 1 ) ;
  lightPos2 = Vector4f(   0,  ld,   0, 1 ) ;
  lightPos3 = Vector4f( -ld,   0,   0, 1 ) ;
  glLightfv( GL_LIGHT0, GL_POSITION, &lightPos0.x ) ;
  glLightfv( GL_LIGHT1, GL_POSITION, &lightPos1.x ) ;
  glLightfv( GL_LIGHT2, GL_POSITION, &lightPos2.x ) ;
  glLightfv( GL_LIGHT3, GL_POSITION, &lightPos3.x ) ;
  
  float white[4] = {1,1,1,1};
  glLightfv( GL_LIGHT1, GL_DIFFUSE, white ) ;
  glLightfv( GL_LIGHT2, GL_DIFFUSE, white ) ;
  glLightfv( GL_LIGHT3, GL_DIFFUSE, white ) ;

  Vector4f spec(1,1,1,75) ;
  //glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, &spec.x ) ;
  //glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, spec.w ) ;
  
  drawPC( verts, GL_POINTS ) ;
  
  msgDraw() ;
  
  glutSwapBuffers();
}

// Called every time a window is resized to resize the projection matrix
void resize( int newWidth, int newHeight )
{
  w = newWidth ;
  h = newHeight ;
}

static int lastX=0, lastY=0 ;
static int mmode=0;
void mouseMotion( int x, int y )
{
  int dx = x-lastX, dy=y-lastY ;
  
  // LMB
  if( mmode == GLUT_LEFT_BUTTON )
  {
    mx += dx, my += dy ;
  }
  else if( mmode == GLUT_RIGHT_BUTTON )
  {
    // dolly
    sbd +=0.01*(-dx+dy) ;
    clamp( sbd, 1, 100 ) ;
  }
  
  lastX=x,lastY=y;
}

void mouse( int button, int state, int x, int y )
{
  lastX = x ;
  lastY = y ;
  
  mmode=button; // 0 for LMB, 2 for RMB
  //printf( "%d %d %d %d\n", button,state,x,y ) ;
  //msg( makeString( "%s click @(%d,%d)", (state?"Up":"Down"),x,y ) ) ;
}

void keyboard( unsigned char key, int x, int y )
{
  
  switch( key )
  {
  case '2':
    {
    int pMode[2];
    glGetIntegerv( GL_POLYGON_MODE, pMode ) ;
    if( pMode[0] == GL_FILL )  glPolygonMode( GL_FRONT_AND_BACK, GL_LINE ) ;
    else  glPolygonMode( GL_FRONT_AND_BACK, GL_FILL ) ;
    }
    break ;
    
  case 'c':
  CLEAR:
    debugPointsPerm.clear() ;
    debugLinesPerm.clear() ;
    debugTrisPerm.clear();
    break ;
  
  case 'l':
    lineWidth++;
    glLineWidth( lineWidth ) ;
    break ;
  case 'L':
    lineWidth--;
    if( lineWidth < 1 ) lineWidth = 1;
    glLineWidth( lineWidth ) ;
    break ;
  
  case 'p':
    ptSize++;
    glPointSize( ptSize ) ;
    break ;
  case 'P':
    ptSize--;
    if( ptSize < 1 )  ptSize=1.f;
    glPointSize( ptSize ) ;
    break ;


  case 27:
    exit(0);
    break;
    
  default:
    break;
  }
}

int main( int argc, char **argv )
{
  glutInit( &argc, argv ) ; // Initializes glut

  glutInitDisplayMode( GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA ) ;
  glutInitWindowSize( w, h ) ;
  glutInitWindowPosition( 0, 0 ) ;
  glutCreateWindow( "OpenGL" ) ;
  glutReshapeFunc( resize ) ;
  glutDisplayFunc( draw ) ;
  glutIdleFunc( draw ) ;
  
  glutMotionFunc( mouseMotion ) ;
  glutMouseFunc( mouse ) ;
  
  glutKeyboardFunc( keyboard ) ;

  init();

  glutMainLoop();
  return 0;
}












