#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "GLUtil.h"

struct Geometry
{
  // makes a spherical mesh from an icosahedron.
  static void makeSphere( vector<Vector3f>& verts, float r )
  {
    //http://en.wikipedia.org/wiki/Icosahedron
    //(0, ±1, ±φ)
    //(±1, ±φ, 0)
    //(±φ, 0, ±1)
    //where φ = (1 + √5) / 2 

    const float t = ( 1 + sqrt( 5.0 ) ) / 2.0 ;
    Vector3f v[12]; // 12 base verts
    
    for( int i = 0 ; i < 4; i++ )
      //v[ i ] = Vector( 0, -(i&2), -(i&1)*t ) ; 
      v[ i ] = Vector3f( 0, i&2?-1:1, i&1?-t:t ) * r ;

    for( int i = 4 ; i < 8; i++ )
      //v[ i ] = Vector( -(i&2), -(i&1)*t, 0 ) ; 
      v[ i ] = Vector3f( i&2?-1:1, i&1?-t:t, 0 ) * r ;

    for( int i = 8 ; i < 12; i++ )
      //v[ i ] = Vector( -(i&1)*t, 0, -(i&2) ) ; 
      v[ i ] = Vector3f( i&1?-t:t, 0, i&2?-1:1 ) * r ;
      
    // these are the faces.
    addTri( verts, v[0], v[2], v[8] ) ;
    addTri( verts, v[0], v[8], v[4] ) ;
    addTri( verts, v[0], v[4], v[6] ) ;
    addTri( verts, v[0], v[6], v[9] ) ;
    addTri( verts, v[0], v[9], v[2] ) ;

    addTri( verts, v[2], v[7], v[5] ) ;
    addTri( verts, v[2], v[5], v[8] ) ;
    addTri( verts, v[2], v[9], v[7] ) ;
      
    addTri( verts, v[8], v[5], v[10] ) ;
    addTri( verts, v[8], v[10], v[4] ) ;
    
    addTri( verts, v[10], v[5], v[3] ) ;
    addTri( verts, v[10], v[3], v[1] ) ;
    addTri( verts, v[10], v[1], v[4] ) ;
    
    addTri( verts, v[1], v[6], v[4] ) ;
    addTri( verts, v[1], v[3], v[11] ) ;
    addTri( verts, v[1], v[11], v[6] ) ;

    addTri( verts, v[6], v[11], v[9] ) ;

    addTri( verts, v[11], v[3], v[7] ) ;
    addTri( verts, v[11], v[7], v[9] ) ;

    addTri( verts, v[3], v[5], v[7] ) ;
    
    
  }
  
  template <typename T> static void makeSphere( vector<T>& verts, const Vector3f& center, float r, const Vector4f& color )
  {
    const float t = ( 1 + sqrt( 5.0 ) ) / 2.0 ;
    T v[12]; // 12 base verts
    
    for( int i = 0 ; i < 4; i++ )
      v[ i ].normal = v[ i ].pos = Vector3f( 0, i&2?-1:1, i&1?-t:t ) ;

    for( int i = 4 ; i < 8; i++ )
      v[ i ].normal = v[ i ].pos = Vector3f( i&2?-1:1, i&1?-t:t, 0 ) ;

    for( int i = 8 ; i < 12; i++ )
      v[ i ].normal = v[ i ].pos = Vector3f( i&1?-t:t, 0, i&2?-1:1 ) ;
      
    for( int i = 0 ; i < 12 ; i++ )
    {
      v[ i ].color = color ;
      v[ i ].pos *= r ;
      v[ i ].pos += center ; // offset after scaling
    }
    
    // these are the faces.
    addTri( verts, v[0], v[2], v[8] ) ;
    addTri( verts, v[0], v[8], v[4] ) ;
    addTri( verts, v[0], v[4], v[6] ) ;
    addTri( verts, v[0], v[6], v[9] ) ;
    addTri( verts, v[0], v[9], v[2] ) ;

    addTri( verts, v[2], v[7], v[5] ) ;
    addTri( verts, v[2], v[5], v[8] ) ;
    addTri( verts, v[2], v[9], v[7] ) ;
      
    addTri( verts, v[8], v[5], v[10] ) ;
    addTri( verts, v[8], v[10], v[4] ) ;
    
    addTri( verts, v[10], v[5], v[3] ) ;
    addTri( verts, v[10], v[3], v[1] ) ;
    addTri( verts, v[10], v[1], v[4] ) ;
    
    addTri( verts, v[1], v[6], v[4] ) ;
    addTri( verts, v[1], v[3], v[11] ) ;
    addTri( verts, v[1], v[11], v[6] ) ;

    addTri( verts, v[6], v[11], v[9] ) ;

    addTri( verts, v[11], v[3], v[7] ) ;
    addTri( verts, v[11], v[7], v[9] ) ;

    addTri( verts, v[3], v[5], v[7] ) ;
  }

  template <typename T>
  static void makeIcosahedronVerts( T* v, float r )
  {
    //http://en.wikipedia.org/wiki/Icosahedron
    //(0, ±1, ±φ)
    //(±1, ±φ, 0)
    //(±φ, 0, ±1)
    //where φ = (1 + √5) / 2
    const static float t = ( 1 + sqrtf( 5.0f ) ) / 2.0 ;
    const static float L = sqrtf( 2.f / (5.f+sqrtf(5.f)) ) ;
    const static float S = t*L;
    
    for( int i = 0 ; i < 4; i++ )
      v[ i ].pos = Vector3f( 0, i&2?-L:L, i&1?-S:S ) * r ;

    for( int i = 4 ; i < 8; i++ )
      v[ i ].pos = Vector3f( i&2?-L:L, i&1?-S:S, 0 ) * r ;

    for( int i = 8 ; i < 12; i++ )
      v[ i ].pos = Vector3f( i&1?-S:S, 0, i&2?-L:L ) * r ;
  }
  
  static void makeIcosahedronVerts( Vector3f* v, float r )
  {
    //http://en.wikipedia.org/wiki/Icosahedron
    //(0, ±1, ±φ)
    //(±1, ±φ, 0)
    //(±φ, 0, ±1)
    //where φ = (1 + √5) / 2
    const static float t = ( 1 + sqrtf( 5.0f ) ) / 2.0 ;
    const static float L = sqrtf( 2.f / (5.f+sqrtf(5.f)) ) ;
    const static float S = t*L;
    
    for( int i = 0 ; i < 4; i++ )
      v[ i ] = Vector3f( 0, i&2?-L:L, i&1?-S:S ) * r ;

    for( int i = 4 ; i < 8; i++ )
      v[ i ] = Vector3f( i&2?-L:L, i&1?-S:S, 0 ) * r ;

    for( int i = 8 ; i < 12; i++ )
      v[ i ] = Vector3f( i&1?-S:S, 0, i&2?-L:L ) * r ;
  }

  static void makeWireframeSphere( vector<Vector3f>& verts, const Vector3f& center, float r )
  {
    Vector3f v[12]; // 12 base verts
    
    makeIcosahedronVerts( v, r ) ;
    
    for( int i = 0 ; i < 12 ; i++ )
      v[ i ] += center ; // offset after scaling
      
    // draw 30 lines!
    addEdge( verts, v[0], v[2] ) ;  addEdge( verts, v[2], v[8] ) ;
    addEdge( verts, v[0], v[8] ) ;  addEdge( verts, v[4], v[8] ) ;
    addEdge( verts, v[0], v[4] ) ;  addEdge( verts, v[4], v[6] ) ;
    addEdge( verts, v[0], v[6] ) ;  addEdge( verts, v[6], v[9] ) ;
    addEdge( verts, v[0], v[9] ) ;  addEdge( verts, v[2], v[9] ) ;
    
    addEdge( verts, v[2], v[7] ) ;  addEdge( verts, v[7], v[9] ) ;
    addEdge( verts, v[2], v[5] ) ;  addEdge( verts, v[5], v[7] ) ;
    addEdge( verts, v[5], v[8] ) ;  addEdge( verts, v[3], v[7] ) ;
    addEdge( verts, v[3], v[5] ) ;  addEdge( verts, v[3], v[10] ) ;
    addEdge( verts, v[5], v[10] ) ; addEdge( verts, v[8], v[10] ) ;
    
    addEdge( verts, v[4], v[10] ) ; addEdge( verts, v[1], v[10] ) ;
    addEdge( verts, v[1], v[3] ) ;  addEdge( verts, v[1], v[11] ) ;
    addEdge( verts, v[3], v[11] ) ; addEdge( verts, v[7], v[11] ) ;
    addEdge( verts, v[1], v[4] ) ;  addEdge( verts, v[1], v[6] ) ;
    addEdge( verts, v[6], v[11] ) ; addEdge( verts, v[9], v[11] ) ;
  }
  
  template <typename T>
  static void makeWireframeSphere( vector<T>& verts, const Vector3f& center, float r, const Vector4f& color )
  {
    T v[12]; // 12 base verts
    makeIcosahedronVerts( &v[0], r ) ;
    for( int i = 0 ; i < 12 ; i++ )
    {
      v[ i ].color = color ;
      v[ i ].pos += center ;
    }
    
    // draw 30 lines!
    addEdge( verts, v[0], v[2] ) ;  addEdge( verts, v[2], v[8] ) ;
    addEdge( verts, v[0], v[8] ) ;  addEdge( verts, v[4], v[8] ) ;
    addEdge( verts, v[0], v[4] ) ;  addEdge( verts, v[4], v[6] ) ;
    addEdge( verts, v[0], v[6] ) ;  addEdge( verts, v[6], v[9] ) ;
    addEdge( verts, v[0], v[9] ) ;  addEdge( verts, v[2], v[9] ) ;
    
    addEdge( verts, v[2], v[7] ) ;  addEdge( verts, v[7], v[9] ) ;
    addEdge( verts, v[2], v[5] ) ;  addEdge( verts, v[5], v[7] ) ;
    addEdge( verts, v[5], v[8] ) ;  addEdge( verts, v[3], v[7] ) ;
    addEdge( verts, v[3], v[5] ) ;  addEdge( verts, v[3], v[10] ) ;
    addEdge( verts, v[5], v[10] ) ;  addEdge( verts, v[8], v[10] ) ;
    
    addEdge( verts, v[4], v[10] ) ;  addEdge( verts, v[1], v[10] ) ;
    addEdge( verts, v[1], v[3] ) ;  addEdge( verts, v[1], v[11] ) ;
    addEdge( verts, v[3], v[11] ) ;  addEdge( verts, v[7], v[11] ) ;
    addEdge( verts, v[1], v[4] ) ;  addEdge( verts, v[1], v[6] ) ;
    addEdge( verts, v[6], v[11] ) ;  addEdge( verts, v[9], v[11] ) ;
    
    // Cost is 60 VERTS = 60*(3 float pos + 4 float color) = 420 floats
    
    // If you use index buffers, then cost is only
    // 12 verts = 84 floats, + 60 indices (shorts)= eq 84 + 30 float size = 114.
    

  }


  // For the vertex types that do not support normals
  template <typename T>
  static void makeSphereNoNormal( vector<T>& verts, float r, const Vector4f& color )
  {
    T v[12];
    makeIcosahedronVerts( v, r ) ;
      
    for( int i = 0 ; i < 12 ; i++ )
    {
      v[ i ].color = color ;
      v[ i ].pos *= r ;
    }
      
    // these are the faces.
    addTri( verts, v[0], v[2], v[8] ) ;
    addTri( verts, v[0], v[8], v[4] ) ;
    addTri( verts, v[0], v[4], v[6] ) ;
    addTri( verts, v[0], v[6], v[9] ) ;
    addTri( verts, v[0], v[9], v[2] ) ;

    addTri( verts, v[2], v[7], v[5] ) ;
    addTri( verts, v[2], v[5], v[8] ) ;
    addTri( verts, v[2], v[9], v[7] ) ;
      
    addTri( verts, v[8], v[5], v[10] ) ;
    addTri( verts, v[8], v[10], v[4] ) ;
    
    addTri( verts, v[10], v[5], v[3] ) ;
    addTri( verts, v[10], v[3], v[1] ) ;
    addTri( verts, v[10], v[1], v[4] ) ;
    
    addTri( verts, v[1], v[6], v[4] ) ;
    addTri( verts, v[1], v[3], v[11] ) ;
    addTri( verts, v[1], v[11], v[6] ) ;

    addTri( verts, v[6], v[11], v[9] ) ;

    addTri( verts, v[11], v[3], v[7] ) ;
    addTri( verts, v[11], v[7], v[9] ) ;

    addTri( verts, v[3], v[5], v[7] ) ;
    
  }

  template <typename T>
  static void makeOctahedron( vector<T>& verts, const T& baseVertex, const Matrix4f& mat, const Vector3f& offset )
  {

    float bwx=0.055, bwy=0.055, heado=0, tailo=8;
    float midp = 2.0*tailo/3.0 ;
    T PX,NX,PY,NY,HEAD,TAIL;
    //set up the base properties
    PX=NX=PY=NY=HEAD=TAIL=baseVertex ;
    
    PX.pos = Vector3f( bwx, 0, midp ), NX.pos = Vector3f( -bwx, 0, midp ), 
    PY.pos = Vector3f( 0, bwy, midp ), NY.pos = Vector3f( 0, -bwy, midp ),
    HEAD.pos = Vector3f(0,0,-heado), TAIL.pos = Vector3f( 0,0,tailo ) ;
    
    // The offset from base center, like "shoot from right cannon"
    // means you must offset the octahedron +x a bit, and -y a bit first.
    PX.pos += offset, PY.pos += offset, NX.pos += offset, NY.pos += offset,
    HEAD.pos += offset, TAIL.pos += offset ;
    
    // Orienting the octahedron in space
    PX.pos = mat*PX.pos ;
    PY.pos = mat*PY.pos ;
    NX.pos = mat*NX.pos ;
    NY.pos = mat*NY.pos ;
    HEAD.pos = mat*HEAD.pos ;
    TAIL.pos = mat*TAIL.pos ;

    addTri( verts, TAIL, PX, PY ) ;
    addTri( verts, TAIL, PY, NX ) ;
    addTri( verts, TAIL, NX, NY ) ;
    addTri( verts, TAIL, NY, PX ) ;

    addTri( verts, HEAD, NX, PY ) ;
    addTri( verts, HEAD, PY, PX ) ;
    addTri( verts, HEAD, PX, NY ) ;
    addTri( verts, HEAD, NY, NX ) ;

  }
  
  template <typename T> static void addEdge( vector<T>& verts, const T& a, const T& b ) {
    verts.push_back( a ) ;  verts.push_back( b ) ;
  }
  
  template <typename T> static void addTri( vector<T>& verts, const T& A, const T& B, const T& C ) {
    verts.push_back( A ) ;  verts.push_back( B ) ;  verts.push_back( C ) ;
  }
  
  // (0,1)
  // D----C (1,1)
  // | __/|
  // |/   |
  // A----B (1,0)
  // (0,0)
  template <typename T> static void spinQuad( vector<T>& verts, const T& A, const T& B, const T& C, const T& D )
  {
    addTri( verts, A, B, C ) ;
    addTri( verts, A, C, D ) ;
  }
  
  template <typename T> static void spinQuadGenUVNormal( vector<T>& verts,
    T A, T B, T C, T D,
    const Vector2f& minTex, const Vector2f& maxTex )
  {
    // (0,1)
    // D----C (1,1)
    // | __/|
    // |/   |
    // A----B (1,0)
    // (0,0)
    
    A.tex = minTex ;
    B.tex = Vector2f( maxTex.x, minTex.y );
    C.tex = maxTex ;
    D.tex = Vector2f( minTex.x, maxTex.y );
    
    // Sets the normal 
    setFaceNormalQuad( A, B, C, D ) ;
    
    spinQuad( verts, A, B, C, D ) ;
  }
  
  
  template <typename T> static void makeSquare( vector<T>& verts, const Vector3f& min, const Vector3f& max, const Vector4f& color,
    const Vector2f& minTex, const Vector2f& maxTex )
  {
    T A,B,C,D ;

    // D----C
    // | __/|
    // |/   |
    // A----B 

    A.pos = min ;
    B.pos = Vector3f( max.x, min.y, max.z ) ; // use max's z all the time
    C.pos = max ;
    D.pos = Vector3f( min.x, max.y, max.z ) ;

    A.color=B.color=C.color=D.color= color ;
    
    spinQuadGenUVNormal( verts, A, B, C, D, minTex, maxTex ) ;
  }
  
  template <typename T> static void makeCube( vector<T>& verts, const Vector3f& min, const Vector3f& max, const Vector4f& color,
    const Vector2f& minTex, const Vector2f& maxTex,
    bool facingOut )
  {
    T A,B,C,D,E,F,G,H;
    
    //Vector3f A( min ),B( min.x, min.y, max.z ),C( min.x, max.y, min.z ),D( min.x, max.y, max.z ),
    //  E( max.x, min.y, min.z ),F( max.x, min.y, max.z ),G( max.x, max.y, min.z ),H( max );
    A.pos = min ;
    B.pos = Vector3f( min.x, min.y, max.z ) ;
    C.pos = Vector3f( min.x, max.y, min.z ) ;
    D.pos = Vector3f( min.x, max.y, max.z ) ;
    
    E.pos = Vector3f( max.x, min.y, min.z ) ;
    F.pos = Vector3f( max.x, min.y, max.z ) ;
    G.pos = Vector3f( max.x, max.y, min.z ) ;
    H.pos = max ;
    
    A.color=B.color=C.color=D.color=E.color=F.color=G.color=H.color= color ;
    
    //       y
    //     ^
    //     |
    //    C----G
    //   /|   /|
    //  D-A--H E  -> x
    //  |/   |/ 
    //  B----F  
    //  /
    // z
    
    // 6 faces
    if( facingOut )
    {
      // CCW out
      spinQuadGenUVNormal( verts, F, E, G, H, minTex, maxTex ) ; //PX
      spinQuadGenUVNormal( verts, A, B, D, C, minTex, maxTex ) ; //NX
      spinQuadGenUVNormal( verts, G, C, D, H, minTex, maxTex ) ; //PY
      spinQuadGenUVNormal( verts, F, B, A, E, minTex, maxTex ) ; //NY
      spinQuadGenUVNormal( verts, B, F, H, D, minTex, maxTex ) ; //PZ
      spinQuadGenUVNormal( verts, E, A, C, G, minTex, maxTex ) ; //NZ
    }
    else
    {
      // normals face IN, CCW in.
      spinQuadGenUVNormal( verts, E, F, H, G, minTex, maxTex ) ; //PX
      spinQuadGenUVNormal( verts, B, A, C, D, minTex, maxTex ) ; //NX
      spinQuadGenUVNormal( verts, C, G, H, D, minTex, maxTex ) ; //PY
      spinQuadGenUVNormal( verts, B, F, E, A, minTex, maxTex ) ; //NY
      spinQuadGenUVNormal( verts, F, B, D, H, minTex, maxTex ) ; //PZ
      spinQuadGenUVNormal( verts, A, E, G, C, minTex, maxTex ) ; //NZ
    }
  }
  
  
  // related to makeCube, this function CHANGES the texcoords in verts
  // to being minTex/maxTex as specified.
  // This is needed because as you choose an itembox to create,
  // the skin has to be selected from the main texture
  template <typename T> static void cubeChangeTexcoords( vector<T>& verts,
    const Vector2f& minTex, const Vector2f& maxTex )
  {
    // now use the same visitation order (ABC, ACD)
    for( int i = 0 ; i < verts.size() ; i+=6 )
    {
      // (0,1)
      // D----C (1,1)
      // | __/|
      // |/   |
      // A----B (1,0)
      // (0,0)
      verts[i].tex = minTex ; //A
      verts[i+1].tex = Vector2f(maxTex.x,minTex.y) ;
      verts[i+2].tex = maxTex ;
      
      verts[i+3].tex = minTex ;
      verts[i+4].tex = maxTex ;
      verts[i+5].tex = Vector2f(minTex.x,maxTex.y) ;
    }
  }
  
  static void addCube( vector<VertexPC>& cmVerts, const Vector3f& center, float s, const Vector4f& color )
  {
    s /= 2.f;
    /*

      C----G
     /|   /|
    D-A--H E
    |/   |/
    B----F

       D--H
       |  |
    D--C--G--H--D
    |  |  |  |  |
    B--A--E--F--B
       |  |
       B--F

    */
    Vector3f A( -s, -s, -s ),  B( -s, -s,  s ),  C( -s,  s, -s ),  D( -s,  s,  s ),
      E(  s, -s, -s ),  F(  s, -s,  s ),  G(  s,  s, -s ),  H(  s,  s,  s ) ;
    A+=center ;  B+=center ;  C+=center ;  D+=center ;
    E+=center ;  F+=center ;  G+=center ;  H+=center ;
    // right face PX
    Geometry::spinQuad( cmVerts, VertexPC( E,color ), VertexPC( F,color ), VertexPC( H,color ), VertexPC( G,color ) ) ; // IN
    
    // left NX
    Geometry::spinQuad( cmVerts, VertexPC( B,color ), VertexPC( A,color ), VertexPC( C,color ), VertexPC( D,color ) ) ; // IN

    // top face PY
    Geometry::spinQuad( cmVerts, VertexPC( C,color ), VertexPC( G,color ), VertexPC( H,color ), VertexPC( D,color ) ) ; // IN
    
    // bottom NY
    Geometry::spinQuad( cmVerts, VertexPC( B,color ), VertexPC( F,color ), VertexPC( E,color ), VertexPC( A,color ) ) ; // IN
    
    // back face PZ
    Geometry::spinQuad( cmVerts, VertexPC( A,color ), VertexPC( E,color ), VertexPC( G,color ), VertexPC( C,color ) ) ; // IN
    
    // front face NZ
    Geometry::spinQuad( cmVerts, VertexPC( F,color ), VertexPC( B,color ), VertexPC( D,color ), VertexPC( H,color ) ) ;
  }
  
  static void addCube( vector<VertexPNC>& cmVerts, const Vector3f& center, float s, const Vector3f& norm, const Vector4f& color )
  {
    s /= 2.f;
    Vector3f A( -s, -s, -s ),  B( -s, -s,  s ),  C( -s,  s, -s ),  D( -s,  s,  s ),
      E(  s, -s, -s ),  F(  s, -s,  s ),  G(  s,  s, -s ),  H(  s,  s,  s ) ;
    A+=center ;  B+=center ;  C+=center ;  D+=center ;
    E+=center ;  F+=center ;  G+=center ;  H+=center ;
    Geometry::spinQuad( cmVerts, VertexPNC( E,norm,color ), VertexPNC( F,norm,color ), VertexPNC( H,norm,color ), VertexPNC( G,norm,color ) ) ;
    Geometry::spinQuad( cmVerts, VertexPNC( B,norm,color ), VertexPNC( A,norm,color ), VertexPNC( C,norm,color ), VertexPNC( D,norm,color ) ) ;
    Geometry::spinQuad( cmVerts, VertexPNC( C,norm,color ), VertexPNC( G,norm,color ), VertexPNC( H,norm,color ), VertexPNC( D,norm,color ) ) ;
    Geometry::spinQuad( cmVerts, VertexPNC( B,norm,color ), VertexPNC( F,norm,color ), VertexPNC( E,norm,color ), VertexPNC( A,norm,color ) ) ;
    Geometry::spinQuad( cmVerts, VertexPNC( A,norm,color ), VertexPNC( E,norm,color ), VertexPNC( G,norm,color ), VertexPNC( C,norm,color ) ) ;
    Geometry::spinQuad( cmVerts, VertexPNC( F,norm,color ), VertexPNC( B,norm,color ), VertexPNC( D,norm,color ), VertexPNC( H,norm,color ) ) ;
  }

  static void addCube( vector<VertexPNC>& cmVerts, const Vector3f& center, float s, bool softNormals, const Vector4f& color )
  {
    s /= 2.f;
    float n = 1.0f/sqrtf( 3.0f ) ;
    VertexPNC A( Vector3f(-s, -s, -s ), Vector3f( -n,-n,-n ), color ),
              B( Vector3f(-s, -s,  s ), Vector3f( -n,-n, n ), color ),
              C( Vector3f(-s,  s, -s ), Vector3f( -n, n,-n ), color ),
              D( Vector3f(-s,  s,  s ), Vector3f( -n, n, n ), color ),
              E( Vector3f( s, -s, -s ), Vector3f(  n,-n,-n ), color ),
              F( Vector3f( s, -s,  s ), Vector3f(  n,-n, n ), color ),
              G( Vector3f( s,  s, -s ), Vector3f(  n, n,-n ), color ),
              H( Vector3f( s,  s,  s ), Vector3f(  n, n, n ), color )
    ;
    A.pos+=center ;  B.pos+=center ;  C.pos+=center ;  D.pos+=center ;
    E.pos+=center ;  F.pos+=center ;  G.pos+=center ;  H.pos+=center ;
    
    Geometry::spinQuad( cmVerts, E, F, H, G ) ;
    Geometry::spinQuad( cmVerts, B, A, C, D ) ;
    Geometry::spinQuad( cmVerts, C, G, H, D ) ;
    Geometry::spinQuad( cmVerts, B, F, E, A ) ; // IN
    Geometry::spinQuad( cmVerts, A, E, G, C ) ; // IN
    Geometry::spinQuad( cmVerts, F, F, D, H ) ;
  }

  static void addCube( vector<VertexPNC>& cmVerts, const Vector3f& center, float s, const Vector4f& color )
  {
    s /= 2.f;
    /*

      C----G
     /|   /|
    D-A--H E
    |/   |/
    B----F

       D--H
       |  |
    D--C--G--H--D
    |  |  |  |  |
    B--A--E--F--B
       |  |
       B--F

    */
    Vector3f A( -s, -s, -s ),  B( -s, -s,  s ),  C( -s,  s, -s ),  D( -s,  s,  s ),
      E(  s, -s, -s ),  F(  s, -s,  s ),  G(  s,  s, -s ),  H(  s,  s,  s ) ;
    A+=center ;  B+=center ;  C+=center ;  D+=center ;
    E+=center ;  F+=center ;  G+=center ;  H+=center ;
    // right face PX
    Vector3f norm( 1,0,0 ) ;
    Geometry::spinQuad( cmVerts, VertexPNC( E,norm,color ), VertexPNC( F,norm,color ), VertexPNC( H,norm,color ), VertexPNC( G,norm,color ) ) ; // IN
    
    // left NX
    norm.x=-1;
    Geometry::spinQuad( cmVerts, VertexPNC( B,norm,color ), VertexPNC( A,norm,color ), VertexPNC( C,norm,color ), VertexPNC( D,norm,color ) ) ; // IN

    // top face PY
    norm.x=0,norm.y=1;
    Geometry::spinQuad( cmVerts, VertexPNC( C,norm,color ), VertexPNC( G,norm,color ), VertexPNC( H,norm,color ), VertexPNC( D,norm,color ) ) ; // IN
    
    // bottom NY
    norm.y=-1;
    Geometry::spinQuad( cmVerts, VertexPNC( B,norm,color ), VertexPNC( F,norm,color ), VertexPNC( E,norm,color ), VertexPNC( A,norm,color ) ) ; // IN
    
    // back face PZ
    norm.y=0,norm.z=1;
    Geometry::spinQuad( cmVerts, VertexPNC( A,norm,color ), VertexPNC( E,norm,color ), VertexPNC( G,norm,color ), VertexPNC( C,norm,color ) ) ; // IN
    
    // front face NZ
    norm.z=-1;
    Geometry::spinQuad( cmVerts, VertexPNC( F,norm,color ), VertexPNC( B,norm,color ), VertexPNC( D,norm,color ), VertexPNC( H,norm,color ) ) ;
  }
} ;


#endif