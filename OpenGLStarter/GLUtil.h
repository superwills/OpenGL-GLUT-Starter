#ifndef GLUTIL_H
#define GLUTIL_H

#ifdef _WIN32
#include <stdlib.h> // MUST BE BEFORE GLUT ON WINDOWS
#include <gl/glut.h>
#else
#include <OpenGL/gl.h>
#endif
#include "StdWilUtil.h"
#include "Vectorf.h"

struct Ray ;
extern map<int,const char*> glErrName ;

inline bool GL_OK()
{
  GLenum err = glGetError() ;
  if( err != GL_NO_ERROR )
    error( "GLERROR %d %s", err, glErrName[ err ] ) ;
  return err == GL_NO_ERROR ;
}

inline bool GL_OK( int line, const char* file )
{
  GLenum err = glGetError() ;
  if( err != GL_NO_ERROR )
    error( "GLERROR %d %s, line=%d of file=%s", err, glErrName[ err ], line, file ) ;
  return err == GL_NO_ERROR ;
}

inline bool CHECK( bool cond, const char* errMsg )
{
  if( !cond )  error( errMsg ) ;
  return cond ;
}

// WITH LINE NUMBERS
#define CHECK_GL GL_OK( __LINE__, __FILE__ ) 

void drawDebug() ;
void drawAxisLines() ;

void drawPC( const vector<VertexPC>& verts, GLenum drawMode ) ;
void drawPNC( const vector<VertexPNC>& verts, GLenum drawMode ) ;

// helpers for annoying tasks
extern void addDebugPoint( const Vector3f& a, const Vector4f& color ) ;
extern void addPermDebugPoint( const Vector3f& a, const Vector4f& color ) ;

extern void addDebugLine( const Vector3f& a, const Vector3f& b, const Vector4f& color ) ;
extern void addDebugLine( const Vector3f& a, const Vector4f& cA, const Vector3f& b, const Vector4f& cB ) ;
extern void addPermDebugLine( const Vector3f& a, const Vector3f& b, const Vector4f& color ) ;
extern void addPermDebugLine( const Vector3f& a, const Vector4f& cA, const Vector3f& b, const Vector4f& cB ) ;

extern void addDebugRay( const Ray& ray, const Vector4f& colorStart, const Vector4f& colorEnd ) ;
extern void addPermDebugRay( const Ray& ray, const Vector4f& colorStart, const Vector4f& colorEnd ) ;

extern void addDebugTriLine( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector4f& color ) ;
extern void addPermDebugTriLine( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector4f& color ) ;

extern void addDebugTriSolid( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector4f& color ) ;
extern void addPermDebugTriSolid( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector4f& color ) ;

extern void addDebugQuadSolid( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d, const Vector4f& color ) ;
extern void addPermDebugQuadSolid( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d, const Vector4f& color ) ;

extern void addDebugSphere( const Vector3f& center, float r, const Vector4f& color ) ;
extern void addPermDebugSphere( const Vector3f& center, float r, const Vector4f& color ) ;

extern vector<VertexPC> debugPoints, debugPointsPerm, debugLines, debugLinesPerm ;
extern vector<VertexPNC> debugTris, debugTrisPerm ;

#endif