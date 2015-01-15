#include "AABB.h"
#include "Intersectable.h"

vector<Vector3f> AABB::SATAxes = AABB::initSATAxes() ;

AABB::AABB( const Sphere& sphere ) {
  min = sphere.c - sphere.r ;
  max = sphere.c + sphere.r ;
}

template <typename T> AABB::AABB( const vector<T>& verts ) {
  bound( verts ) ; 
}





// Non const.
// Bounding
template <typename T> void AABB::bound( const vector<T>& verts ) {
  for( int i = 0 ; i < verts.size() ; i++ )
    bound( verts[i].pos ) ;
}
void AABB::bound( const Vector3f& vertex ) {
  //min.clamp( min, vertex ) ;
  //max.clamp( vertex, max ) ;

  // let the min bound be pushed out
  // by the vertices of the model
  if( min.x > vertex.x ) min.x = vertex.x ; // pushed out to reach vertex.x - ie min.x clamped BELOW by vertex.x
  if( min.y > vertex.y ) min.y = vertex.y ;
  if( min.z > vertex.z ) min.z = vertex.z ;

  if( max.x < vertex.x ) max.x = vertex.x ; // max.x pushed out to include vertex.x - ie max.x clamped ABOVE by vertex.x
  if( max.y < vertex.y ) max.y = vertex.y ;
  if( max.z < vertex.z ) max.z = vertex.z ;
}


// Here we return true IFF the tri's 3 pts
// are completely contained within the AABB
inline bool AABB::containsTri( const Triangle& tri ) const {
  return tri.a.isBetween( min, max ) &&
         tri.b.isBetween( min, max ) &&
         tri.c.isBetween( min, max ) ;
}

// Intersection methods:
bool AABB::intersectsAABB( const AABB& o ) const 
{
  // intersection requires overlap in all 3 axes
  if( max.x < o.min.x || // my max is left of his min, miss x overlap OR
      o.max.x < min.x )  // his max is left of my min
    return false ;
  if( max.y < o.min.y ||
      o.max.y < min.y )
    return false ;
  if( max.z < o.min.z ||
      o.max.z < min.z )
    return false ;

  // if nothing missed it's a hit
  return true;
}

bool AABB::intersectsSphere( const Sphere& s ) const
{
  // get the closest pt on the box to the sphere.
  Vector3f boxPt = s.c ;
  boxPt.clampComponent( min, max ) ;
  
  // see if the pt on the box is inside the sphere
  return s.contains( boxPt ) ;
}

// no one needs teh intersection point for the aabb,
// it's just a boolean check
//bool intersects( const Ray& ray, Intersection* intn ) ;
bool AABB::intersectsRay( const Ray& ray, Vector3f& pt ) const
{
  bool rayStartInside = containsPoint( ray.start ) ;

  // Rays starting and ending in an octree node must be considered
  // to hit the node.
  // This is the behavior you USUALLY want.
  if( rayStartInside && containsPoint( ray.end ) )
  {
    pt=ray.start;
    return true ; 
  }
  
  // the algorithm says, find 3 t's,
  Vector3f t ;

  if( rayStartInside )
  {
    for( int i = 0 ; i < 3 ; i++ )
    {
      if( ray.dir.elts[i] > 0 ) // RAY GOING + and we are inside the box.. CULL BACK FACE (mins)
      {
        t.elts[i] = ( max.elts[i] - ray.start.elts[i] ) / ray.dir.elts[i] ;
      }
      else // RAY GOING - and we are inside the box.. so cull the MAXES
      {
        t.elts[i] = ( min.elts[i] - ray.start.elts[i] ) / ray.dir.elts[i] ;
      }
    }
  }
  else
  {
    // LARGEST t is the only only we need to test if it's on the face.
    for( int i = 0 ; i < 3 ; i++ )
    {
      if( ray.dir.elts[i] > 0 ) // RAY GOING +.. CULL BACK FACE (maxes)
      {
        //    ^ |
        //   /  |
        //  *   |
        // -----+-------
        //      |
        // A ray with ray.dir.x>0 means IT WOULD HIT MINFACES FIRST.
        // __BUT__, if the ray STARTS AFTER THE MINFACE (ray.start.x > min.x)
        // THEN THE RAY STARTED INSIDE THE BOX.  to still detect the intersection
        // you would use the MAX FACE in that case
        t.elts[i] = ( min.elts[i] - ray.start.elts[i] ) / ray.dir.elts[i] ;
      }
      else // RAY GOING -.. so cull the MINS
      {
        // IF RAY LEFT OF MAXFACE, USE MINFACE
        t.elts[i] = ( max.elts[i] - ray.start.elts[i] ) / ray.dir.elts[i] ;
      }
      
      // The above code works fine for when you are between min/max in 2 dimensions or less,
      // because the detected hits for those dimensions will be BEHIND THE RAY START
      // so it's kind of like "using a loophole".. if you are between max/min in 2 dimensions,
      // THEN YOU CAN ONLY HIT THE BOX IN THE DIMENSION YOU ARE __NOT__ between max/min in,
      // unless you're already inside the box, which is hte cases handled above.
    }
  }
  
  // The right answer will be 
  int tIndex ;
  if( rayStartInside )  tIndex = t.minIndex() ;
  else  tIndex = t.maxIndex() ;
  
  if( isBetweenOrdered( t.elts[tIndex], 0, ray.len ) )
  {
    pt = ray.at( t.elts[tIndex] ) ;

    // check it's in the box in other 2 dimensions
    int o1 = OTHERAXIS1(tIndex) ; // i=0: o1=1, o2=2, i=1: o1=2,o2=0 etc.
    int o2 = OTHERAXIS2(tIndex) ;

    return isBetweenOrdered( pt.elts[o1], min.elts[o1], max.elts[o1] ) &&
           isBetweenOrdered( pt.elts[o2], min.elts[o2], max.elts[o2] ) ;
  }
  
  return false ;
}

bool AABB::intersectsRay( const Ray& ray ) const
{
  // if the start point OR the end point is in the AABB, then it's a hit.
  // For the BOOLEAN check where you don't want the hit point, we use an OR (vs an AND)
  bool rayStartInside = containsPoint( ray.start ) ;
  if( rayStartInside || containsPoint( ray.end ) )
    return true ; // I don't have to get you the exact point.
  
  Vector3f t ;

  if( rayStartInside )
    for( int i = 0 ; i < 3 ; i++ )
      if( ray.dir.elts[i] > 0 )
        t.elts[i] = ( max.elts[i] - ray.start.elts[i] ) / ray.dir.elts[i] ;
      else
        t.elts[i] = ( min.elts[i] - ray.start.elts[i] ) / ray.dir.elts[i] ;
  else
    for( int i = 0 ; i < 3 ; i++ )
      if( ray.dir.elts[i] > 0 )
        t.elts[i] = ( min.elts[i] - ray.start.elts[i] ) / ray.dir.elts[i] ;
      else
        t.elts[i] = ( max.elts[i] - ray.start.elts[i] ) / ray.dir.elts[i] ;
  
  int tIndex ;
  if( rayStartInside )  tIndex = t.minIndex() ;
  else  tIndex = t.maxIndex() ;
  if( isBetweenOrdered( t.elts[tIndex], 0, ray.len ) )
  {
    Vector3f pt = ray.at( t.elts[tIndex] ) ;
    int o1 = OTHERAXIS1(tIndex) ;
    int o2 = OTHERAXIS2(tIndex) ;
    return isBetweenOrdered( pt.elts[o1], min.elts[o1], max.elts[o1] ) &&
           isBetweenOrdered( pt.elts[o2], min.elts[o2], max.elts[o2] ) ;
  }
  
  return false ;
}

// Gives you a new AABB that describes where
// one AABB intersects another.  You get
// an EMPTY (point) AABB if they don't intersect
// at all (max=min=0,0,0)
AABB AABB::getIntersectionVolume( const AABB& o ) const {
  AABB res ;
  for( int c = 0 ; c < 3 ; c++ )
  {
    // get the overlap in this component.

    // Total miss cases
    // ==tttttt======oooooo===========>
    // ==oooooo======tttttt===========>
    if( max.elts[c] < o.min.elts[c] ||
        min.elts[c] > o.max.elts[c] )
      return AABB(0,0) ;//MISS

    // ===============================>
    //         ttttttttttttttt
    //             ooooooo
    else if( isBetweenOrdered( o.min.elts[c], min.elts[c], max.elts[c] ) && 
        isBetweenOrdered( o.max.elts[c], min.elts[c], max.elts[c] ) )
    {
      res.min.elts[c] = o.min.elts[c] ;
      res.max.elts[c] = o.max.elts[c] ;
    }

    // ===============================>
    //         ooooooooooooooo
    //             ttttttt
    else if( isBetweenOrdered( min.elts[c], o.min.elts[c], o.max.elts[c] ) && 
        isBetweenOrdered( max.elts[c], o.min.elts[c], o.max.elts[c] ) )
    {
      res.min.elts[c] = min.elts[c] ;
      res.max.elts[c] = max.elts[c] ;
    }

    // ===============================>
    //  tttttttttttt
    //          oooooooo
    else if( isBetweenOrdered( max.elts[c], o.min.elts[c], o.max.elts[c] ) )
    {
      //o.min->t.max
      res.min.elts[c] = o.min.elts[c] ;
      res.max.elts[c] = max.elts[c] ;
    }

    // ===============================>
    //          tttttttt
    //    oooooooooo
    else 
    {
      //t.min->o.max
      res.min.elts[c] = min.elts[c] ;
      res.max.elts[c] = o.max.elts[c] ;
    }
  }

  return res ;
}

// splits on a given axis into 2 aabb's,
// used in kd-tree construction
// Push 0 for x, 1 for y, 2 for z.
vector<AABB> AABB::split2( int axisIndex, float val ) const
{
  vector<AABB> subAABB ;
  if( !isBetweenOrdered(val, min.elts[axisIndex], max.elts[axisIndex] ) )
  {
    error( "split plane OOB AABB" ) ;
    subAABB.push_back( *this ) ;
    return subAABB ; // ASUS RMA (you get the same one back)
  }

  Vector3f newmin = min ;
  newmin.elts[ axisIndex ] = val ;
  Vector3f newmax = max ;
  newmax.elts[ axisIndex ] = val ;

  subAABB.push_back( AABB( min, newmax ) ) ;
  subAABB.push_back( AABB( newmin, max ) ) ;

  return subAABB ;
}

// splits into 8 AABBs,
// used mainly for octree construction
vector<AABB> AABB::split8() const
{
  Vector3f c = (min + max) / 2 ;

  vector<AABB> subAABB ;

  //     ________
  //    /   /   /|
  //   /___/___/ |
  //  /   /   /| |
  // /___/___/ |/|
  // |   |   | | |
  // |___|___|/|/
  // |   |   | |
  // |___|___|/


  //   _______
  //  / 5  8 /
  // | 6  7   
  // |  1  4  
  // | 2  3  /

  subAABB.push_back( AABB( min, c ) ) ;
  subAABB.push_back( AABB( Vector3f(min.x,min.y,c.z), Vector3f(c.x,c.y,max.z) ) ) ;
  subAABB.push_back( AABB( Vector3f(c.x,min.y,c.z), Vector3f(max.x,c.y,max.z) ) ) ;
  subAABB.push_back( AABB( Vector3f(c.x,min.y,min.z), Vector3f(max.x,c.y,max.z) ) ) ;

  subAABB.push_back( AABB( Vector3f(min.x,c.y,min.z), Vector3f(c.x,max.y,c.z) ) ) ;
  subAABB.push_back( AABB( Vector3f(min.x,c.y,c.z), Vector3f(c.x,max.y,max.z) ) ) ;
  subAABB.push_back( AABB( c, max ) ) ;
  subAABB.push_back( AABB( Vector3f(c.x,c.y,min.z), Vector3f(max.x,max.y,c.z) ) ) ;

  return subAABB ;
}


// don't use an offset
void AABB::drawDebug( const Vector3f& color ) const
{
  
  
  /*
  Geometry::makeLineCube( 
  addPermDebugLine( a, Vector4f(1,0,0,1), b, Vector4f(1,1,0,1) );
  addPermDebugLine( b, Vector4f(1,1,0,1), c, Vector4f(1,0,1,1) );
  addPermDebugLine( c, Vector4f(1,0,1,1), d, Vector4f(1,1,1,1) );
  addPermDebugLine( d, Vector4f(1,1,1,1), a, Vector4f(1,0,0,1) );
  
  // draws the far plane.
  addPermDebugLine( farA, Vector4f(1,0,0,1), farB, Vector4f(1,1,0,1) );
  addPermDebugLine( farB, Vector4f(1,1,0,1), farC, Vector4f(1,0,1,1) );
  addPermDebugLine( farC, Vector4f(1,0,1,1), farD, Vector4f(1,1,1,1) );
  addPermDebugLine( farD, Vector4f(1,1,1,1), farA, Vector4f(1,0,0,1) );
  
  addPermDebugLine( a, Vector4f(1,0,0,1), farA, Vector4f(1,0,0,1) );
  addPermDebugLine( b, Vector4f(1,1,0,1), farB, Vector4f(1,1,0,1) );
  addPermDebugLine( c, Vector4f(1,0,1,1), farC, Vector4f(1,0,1,1) );
  addPermDebugLine( d, Vector4f(1,1,1,1), farD, Vector4f(1,1,1,1) );
  */
}

// A convenience method to get the lines
//void AABB::generateDebugLines( const Vector3f& offset, const Vector3f& color ) const ;
