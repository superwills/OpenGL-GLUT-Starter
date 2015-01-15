#ifndef AABB_H
#define AABB_H

#include "Vectorf.h"

struct Triangle ;
struct Ray ;
struct Sphere ;

//#define HUGE 1e30f // actually defined as MAXFLOAT
//can use FLT_MAX instead
extern void addPermDebugLine( const Vector3f& a, const Vector4f& cA, const Vector3f& b, const Vector4f& cB ) ;

struct AABB
{

  Vector3f min, max ;
  // NOT caching 'extents' because it's not used a lot
  // and it's like more memory for no reason.
  
  // These are used for SAT testing REPEATEDLY
  vector<Vector3f> corners ;

  AABB() {
    resetInsideOut() ;
  }
  AABB( const Vector3f& iMin, const Vector3f& iMax ) {
    min=(iMin), max=(iMax);
    
    if( max.x < min.x )  swap( max.x, min.x ) ;
    if( max.y < min.y )  swap( max.y, min.y ) ;
    if( max.z < min.z )  swap( max.z, min.z ) ;
    
    recomputeCorners() ;
  }
  // boxes a sphere
  AABB( const Sphere& sphere ) ;
  static AABB FromCenterAndExtents( const Vector3f& center, Vector3f extents )
  {
    Vector3f halfExts = extents.fabs()/2.f ; // you're not allowed negative extents.
    return AABB( center - halfExts, center + halfExts ) ;
  }

  template <typename T> AABB( const vector<T>& verts ) ;
  inline void resetInsideOut() {
    min.x=min.y=min.z= HUGE ;
    max.x=max.y=max.z=-HUGE ;
  }
  void recomputeCorners()
  {
    corners.clear() ;

    corners.push_back( Vector3f( min.x,min.y,min.z ) ) ; // A
    corners.push_back( Vector3f( min.x,min.y,max.z ) ) ; // B, 1 nxnypz
    corners.push_back( Vector3f( min.x,max.y,min.z ) ) ; // C, 2 nxpynz
    corners.push_back( Vector3f( min.x,max.y,max.z ) ) ; // D
    
    corners.push_back( Vector3f( max.x,min.y,min.z ) ) ; // E, 5 pxnynz
    corners.push_back( Vector3f( max.x,min.y,max.z ) ) ; // F
    corners.push_back( Vector3f( max.x,max.y,min.z ) ) ; // G, 
    corners.push_back( Vector3f( max.x,max.y,max.z ) ) ; // H, 7 pxpypz
  }
  
  static vector<Vector3f> initSATAxes()
  {
    vector<Vector3f> axes ;
    axes.push_back( Vector3f(1,0,0) ) ; 
    axes.push_back( Vector3f(0,1,0) ) ; 
    axes.push_back( Vector3f(0,0,1) ) ; 
    return axes ;
  }
  static vector<Vector3f> SATAxes ;
  
  
  // Non const.
  // Bounding
  template <typename T> void bound( const vector<T>& verts ) ;
  void bound( const Vector3f& vertex ) ;
  
  inline AABB& operator*=( const Vector3f& scale ) {
    min*=scale ;  max*=scale ;
    return *this ;
  }
  inline AABB& operator/=( const Vector3f& scale ) {
    min/=scale ;  max/=scale ;
    return *this ;
  }
  
  //////////////// CONST FUNCTIONS
  // Extents and volume measure
  inline float xExtents() const { return max.x - min.x ; }
  inline float yExtents() const { return max.y - min.y ; }
  inline float zExtents() const { return max.z - min.z ; }
  inline float extents( int axisIndex ) const {
    return max.elts[axisIndex] - min.elts[axisIndex] ;
  }
  inline Vector3f extents() const {
    return max - min ;
  }
  // Gives you the volume of an AABB.  This is 0
  // if the AABB is a point, or if the aabb is
  // infinitely thin in any direction
  inline float volume() const {
    return xExtents()*yExtents()*zExtents();
  }

  // ie the aabb is a point, not a box anymore.
  inline bool isZeroVolume() const { return max==min ; }
  
  inline Vector3f random() const {
    return Vector3f::random( min, max ) ;
  }

  // point in the middle of box
  inline Vector3f mid() const {
    return ( min + max ) / 2 ;
  }
  
  // named contains because that's what it really is.
  // you can "intersect" a point, but its better to say
  // the AABB CONTAINS it.
  inline bool containsPoint( const Vector3f& point ) const {
    // the point is in the box of space between min and max
    return point.isBetween( min, max ) ;
  }

  // Here we return true IFF the tri's 3 pts
  // are completely contained within the AABB
  inline bool containsTri( const Triangle& tri ) const ;

  inline bool containsAABB( const AABB& oaabb ) const {
    return oaabb.min.isBetween( min, max ) && 
           oaabb.max.isBetween( min, max ) ;
  }
  
  
  
  // Intersection methods:
  bool intersectsAABB( const AABB& o ) const ;
  
  bool intersectsSphere( const Sphere& s ) const ;
  
  // no one needs teh intersection point for the aabb,
  // it's just a boolean check
  //bool intersects( const Ray& ray, Intersection* intn ) ;
  bool intersectsRay( const Ray& ray ) const ;
  
  bool intersectsRay( const Ray& ray, Vector3f& pt ) const ;

  // Gives you a new AABB that describes where
  // one AABB intersects another.  You get
  // an EMPTY (point) AABB if they don't intersect
  // at all (max=min=0,0,0)
  AABB getIntersectionVolume( const AABB& o ) const ;
  
  // splits on a given axis into 2 aabb's,
  // used in kd-tree construction
  // Push 0 for x, 1 for y, 2 for z.
  vector<AABB> split2( int axisIndex, float val ) const ;
  
  // splits into 8 AABBs,
  // used mainly for octree construction
  vector<AABB> split8() const ;
  
  AABB operator+( const Vector3f & translation ) const {
    return AABB( min+translation, max+translation ) ;
  }
  AABB operator-( const Vector3f & s ) const{
    return AABB( min-s, max-s ) ;
  }
  AABB operator*( const Vector3f & s ) const {
    return AABB( min*s, max*s ) ;
  }
  AABB operator/( const Vector3f & s ) const {
    return AABB( min/s, max/s ) ;
  }
  
  AABB scale( float scale ) const{
    // Keep it CENTERED
    Vector3f center = mid() ;
    Vector3f range = extents() ;
    range *= scale ; // change the RANGE scale.
    range/=2; // push center only by half range to get max/min
    return AABB( center-range, center+range ) ;
  }
  
  // don't use an offset
  void drawDebug( const Vector3f& color ) const ;

  // A convenience method to get the lines
  //void drawDebug( const Vector3f& offset, const Vector3f& color ) const ;

} ;




#endif