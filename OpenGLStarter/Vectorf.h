#ifndef VECTORF_H
#define VECTORF_H

#include <stdio.h>

#define _USE_MATH_DEFINES
#include <math.h>
#import "StdWilUtil.h"

// See https://gist.github.com/superwills/6159033
// for the matrix characters

struct Vector2f
{
  float x,y;
  Vector2f():x(0.f),y(0.f){}
  Vector2f( float ix, float iy ):x(ix),y(iy){}
  //Vector2f( const CGPoint& o ):x( o. {  }
  Vector2f( float iv ):x(iv),y(iv){}
  
  //CONST
  inline void print() const {
    printf( "(%.2f %.2f) ",x,y ) ;
  }
  inline void print( const char* msg ) const {
    printf( "%s (%.2f %.2f)",msg,x,y ) ;
  }
  inline void println() const {
    printf( "(%.2f %.2f)\n",x,y ) ;
  }
  inline void println( const char* msg ) const {
    printf( "%s (%.2f %.2f)\n",msg,x,y ) ;
  }
  
  inline float max() const { return ::max( x,y ) ; }
  inline float min() const { return ::min( x,y ) ; }
  inline float getAvg() const { return (x+y)/2.f ; }
 
  inline Vector2f operator+( const Vector2f& o ) const {
    return Vector2f(x+o.x,y+o.y);
  }
  inline Vector2f operator-() const {
    return Vector2f(-x,-y);
  }
  inline Vector2f operator-( const Vector2f& o ) const {
    return Vector2f(x-o.x,y-o.y);
  }
  
  inline Vector2f operator*( const Vector2f& o ) const {
    return Vector2f(x*o.x,y*o.y);
  }
  inline Vector2f operator*( float s ) const {
    return Vector2f(x*s,y*s);
  }
  inline Vector2f operator/( const Vector2f& o ) const {
    return Vector2f(x/o.x,y/o.y);
  }
  inline Vector2f operator/( float s ) const {
    return Vector2f(x/s,y/s);
  }
  inline float cross( const Vector2f& o ) const {
    return x*o.y-y*o.x ;
  }
  inline float dot( const Vector2f& o ) const {
    return x*o.x+y*o.y ;
  }
  inline float len() const {
    return sqrtf( x*x+y*y ) ;
  }
  inline float len2() const {
    return x*x+y*y ;
  }
  inline Vector2f normalizedCopy() const {
    return Vector2f( *this ).normalize() ;
  }
  inline float angleWith( const Vector2f& o ) const {
    return acosf( this->normalizedCopy().dot(o.normalizedCopy()) ) ;
  }
  // Returns + if this leads o.
  // more expensive than unsigned angle,
  // (2x acos)
  inline float signedAngleWith( const Vector2f& o ) const {
    float aThis = atan2f( y, x );
    float aO = atan2f( o.y, o.x ) ;
    
    //info( "lead=%.2f lag=%.2f, ", aThis, aO ) ;
    return aThis - aO ;
    // When 
  }
  // proximity
  inline bool isNear( const Vector2f& o ) const{
    return fabsf(x-o.x)<EPS_MIN && fabsf(y-o.y)<EPS_MIN ;
  }
  inline bool isNear( const Vector2f& o, const Vector2f& maxDiff ) const{
    return fabsf(x-o.x)<maxDiff.x && fabsf(y-o.y)<maxDiff.y ;
  }
  inline bool isCanonical() const {
    return -1.f <= x && x <= 1.f && 
           -1.f <= y && y <= 1.f ;
  }
  inline bool isNaN() const {
    return isnan(x) || isnan(y) ;
  }
  // Exact equality
  inline bool operator==( const Vector2f& o ) const{
    return x==o.x && y==o.y ;
  }
  inline bool operator!=( const Vector2f& o ) const{
    return x!=o.x || y!=o.y ;
  }
  
  // Returns TRUE if "this" is closer to A
  // than B
  inline bool isCloserTo( const Vector2f& a, const Vector2f& thanB ){
    return ( *this-a ).len2() < ( *this-thanB ).len2() ;
  }
  
  // gets the tangential and normal components of THIS along fN
  // f MUST BE NORMALIZED.
  // It gives you the component of this along fN.
  // If that's negative, THEN THIS IS MORE THAN 90 DEG FROM FN.
  inline float parallelPerp( const Vector2f& fN, Vector2f &vParallel, Vector2f &vPerp ) const
  {
    // assuming fN is normalized
    float compParallel = fN.dot( *this ) ;
    vParallel = fN * compParallel ;
    vPerp = (*this) - vParallel ;
    return compParallel ;
  }
  
  // The perpendicular vector is 
  inline void parallelPerpComponents( const Vector2f& fN, float &compParallel, float &compPerp ) const
  {
    // assuming fN is normalized
    compParallel = fN.dot( *this ) ;
    compPerp = fN.cross( *this ) ;
  }
  
  // This is the CCW 90 deg rotated perp. ( y,-x ) is CW rotated.
  inline Vector2f getPerpendicular() const {
    return Vector2f( -y,x ) ;
  }
  
  
  
  
  
  //NON-CONST
  inline Vector2f& normalize(){
    float length = len() ;
    
    // I added this debug check man, never take it out.
    if( !length ) {
      error( "Vector2f::normalize() attempt to divide by 0" ) ;
      return *this ;
    }
    
    return (*this)/=length ;
  }
  inline float safeNormalize(){
    float length = len() ;
    if( length )  (*this)/=length ;
    return length ;
  }
  inline Vector2f& operator+=( const Vector2f& o ){
    x+=o.x,y+=o.y;
    return *this ;
  }
  inline Vector2f& operator-=( const Vector2f& o ){
    x-=o.x,y-=o.y;
    return *this ;
  }
  inline Vector2f& operator*=( const Vector2f& o ){
    x*=o.x,y*=o.y;
    return *this ;
  }
  inline Vector2f& operator*=( float s ){
    x*=s,y*=s;
    return *this ;
  }
  inline Vector2f& operator/=( const Vector2f& o ){
    x/=o.x,y/=o.y;
    return *this ;
  }
  inline Vector2f& operator/=( float s ){
    x/=s,y/=s;
    return *this ;
  }
  inline Vector2f& clampLen( float maxLen ){
    float length = len() ;
    if( length > maxLen ) // also means length > 0, hopefully
      return normalize()*=maxLen ;
    
    return *this ;
  }
  inline Vector2f& clampComponent( float minVal, float maxVal )
  {
    ::clamp( x,minVal,maxVal ) ;
    ::clamp( y,minVal,maxVal ) ;
    return *this ;
  }
  inline Vector2f& clampComponentBelow( float below )
  {
    if( x < below ) x=below ;
    if( y < below ) y=below ;
    return *this ;
  }
  inline Vector2f& clampComponentAbove( float above )
  {
    if( x > above ) x=above ;
    if( y > above ) y=above ;
    return *this ;
  }
  inline Vector2f& clampBelow( const Vector2f& below )
  {
    if( x < below.x ) x=below.x ;
    if( y < below.y ) y=below.y ;
    return *this ;
  }
  inline Vector2f& clampAbove( const Vector2f& above )
  {
    if( x > above.x ) x=above.x ;
    if( y > above.y ) y=above.y ;
    return *this ;
  }
  
} ;


inline Vector2f operator-( const Vector2f& v, float s )
{
  return Vector2f(v.x-s,v.y-s);
}

inline Vector2f operator-( float s, const Vector2f& v )
{
  return Vector2f(s-v.x,s-v.y);
}







union Vector3f ;
struct Vector3i
{
  int x,y,z ;
  
  Vector3i():x(0),y(0),z(0){}
  Vector3i( int ix, int iy, int iz ):x(ix),y(iy),z(iz){}
  Vector3i( int iv ):x(iv),y(iv),z(iv){}
  Vector3i( const Vector3f& v ) ;
  static Vector3i UnitVector[3] ;
  
  inline void print() const {
    printf( "(%d %d %d)",x,y,z ) ;
  }
  inline void print( const char* msg ) const {
    printf( "%s (%d %d %d)",msg,x,y,z ) ;
  }
  inline void println() const {
    printf( "(%d %d %d)\n",x,y,z ) ;
  }
  inline void println(const char* msg) const {
    printf( "%s (%d %d %d)\n",msg,x,y,z ) ;
  }

  inline bool operator==( const Vector3i& o ) const {
    return x==o.x && y==o.y && z==o.z ;
  }
  
  // same face at least, IF NOT same edge.
  inline bool atLeastOneEqual( const Vector3i& o ) const {
    return x==o.x || y==o.y || z==o.z ;
  }
  
  // Same face
  inline bool oneEqual( const Vector3i& o ) const {
    return (x==o.x && y!=o.y && z!=o.z) || 
           (x!=o.x && y==o.y && z!=o.z) || 
           (x!=o.x && y!=o.y && z==o.z)
    ;
  }
  
  // same edge
  inline bool twoEqual( const Vector3i& o ) const {
    return (x==o.x && y==o.y && z!=o.z) ||
           (x==o.x && y!=o.y && z==o.z) ||
           (x!=o.x && y==o.y && z==o.z)
    ;
  }
  
  // Operators are provided for Vector3i ONLY here.
  // Vector3i/Vector3f cross operations are provided for
  // OUTSIDE of both classes.
  inline Vector3i operator+( const Vector3i& o ) const {
    return Vector3i(x+o.x,y+o.y,z+o.z);
  }
  inline Vector3i operator+( int i ) const {
    return Vector3i(x+i,y+i,z+i);
  }
  inline Vector3i operator-() const{
    return Vector3i(-x,-y,-z);
  }
  inline Vector3i operator-( const Vector3i& o ) const {
    return Vector3i(x-o.x,y-o.y,z-o.z);
  }
  inline Vector3i operator-( int i ) const {
    return Vector3i(x-i,y-i,z-i);
  }
  inline Vector3i operator*( const Vector3i& o ) const {
    return Vector3i(x*o.x,y*o.y,z*o.z);
  }
  inline Vector3i operator*( int s ) const {
    return Vector3i(x*s,y*s,z*s);
  }
  inline Vector3i operator/( const Vector3i& o ) const {
    return Vector3i(x/o.x,y/o.y,z/o.z);
  }
  inline Vector3i operator/( int s ) const {
    return Vector3i(x/s,y/s,z/s);
  }
  inline Vector3i operator%( const Vector3i& o ) const {
    return Vector3i(x%o.x,y%o.y,z%o.z);
  }
  inline Vector3i operator%( int s ) const {
    return Vector3i(x%s,y%s,z%s);
  }
  inline Vector3i cross( const Vector3i& o ) const {
    return Vector3i( y*o.z-o.y*z, z*o.x-x*o.z, x*o.y-o.x*y ) ;
  }
  inline int dot( const Vector3i& o ) const {
    return x*o.x+y*o.y+z*o.z ;
  }
  

  // NON CONST
  inline Vector3i& operator+=( const Vector3i& o ){
    x+=o.x,y+=o.y,z+=o.z;
    return *this ;
  }
  inline Vector3i& operator-=( const Vector3i& o ){
    x-=o.x,y-=o.y,z-=o.z;
    return *this ;
  }
  inline Vector3i& operator*=( const Vector3i& o ){
    x*=o.x,y*=o.y,z*=o.z;
    return *this ;
  }
  inline Vector3i& operator*=( int s ){
    x*=s,y*=s,z*=s;
    return *this ;
  }
  inline Vector3i& operator/=( const Vector3i& o ){
    x/=o.x,y/=o.y,z/=o.z;
    return *this ;
  }
  inline Vector3i& operator/=( int s ){
    x/=s,y/=s,z/=s;
    return *this ;
  }
  inline Vector3i& operator%=( const Vector3i& o ){
    x%=o.x,y%=o.y,z%=o.z;
    return *this ;
  }
  inline Vector3i& operator%=( int s ){
    x%=s,y%=s,z%=s;
    return *this ;
  }
} ;

struct Vector4i
{
  int x,y,z,w ;
} ;

// >0 (+ve) means on + side of plane
// =0 means in plane
// <0 (-ve) means on - side of plane (side opposite normal)
inline int planeSide( Vector3i* pts, int a, int b, int c, const Vector3i& testPoint )
{
  Vector3i AB = pts[b] - pts[a] ;
  Vector3i AC = pts[c] - pts[a] ;
  Vector3i n = AB.cross( AC ) ;
  int dPlane = -n.dot( pts[a] ) ;
  return n.dot( testPoint ) + dPlane ;
}

union Vector3f
{
  struct{ float x,y,z ; } ;
  float elts[3];

  Vector3f():x(0.f),y(0.f),z(0.f){}
  Vector3f( const Vector3i& v ):x(v.x),y(v.y),z(v.z){}
  Vector3f( const Vector2f& v ):x(v.x),y(v.y),z(0.f){}
  Vector3f( float ix, float iy, float iz ):x(ix),y(iy),z(iz){}
  Vector3f( float iv ):x(iv),y(iv),z(iv){}
  
  static Vector3f UnitVectors[ 3 ] ;
  
  // macho linker errors
  static inline Vector3f random() { return Vector3f( randFloat(), randFloat(), randFloat() ) ;  }
  
  static inline Vector3f random(float min, float max) {
    return Vector3f( randFloat(min,max), randFloat(min,max), randFloat(min,max) ) ;
  }
  static inline Vector3f random(const Vector3f& min, const Vector3f& max) {
    return Vector3f( randFloat(min.x,max.x), randFloat(min.y,max.y), randFloat(min.z,max.z) ) ;
  }
  
  // graphics gems page 24
  static inline Vector3f randBary() {
    float s=sqrtf( randFloat() ), t=randFloat() ;
    Vector3f bary ;
    bary.x = 1.f - t ;
    bary.y = ( 1.f - s ) * t ;
    bary.z = s*t;
    return bary ;
  }
  
  static Vector3f randomSpherical( float r ){
    float tEl = acosf( 1.f - 2.f*randFloat() ) ;
    float pAz = randFloat( 0, 2.f*M_PI ) ;
    
    return Vector3f(
      r*sin(tEl)* //proj xz plane
        cos(pAz),
      r*cos(tEl), // independent of pAzimuth, pAzimuth goes around the y-axis
      r*sin(tEl)*sin(pAz)
    ) ;
  }
  static Vector3f lerp( const Vector3f& t, const Vector3f& A, const Vector3f& B ){
    return A + (B-A)*t ; // == A*(1.f-t) + B*t ;
  }
  static Vector3f lerp( float t, const Vector3f& A, const Vector3f& B ){
    return A + (B-A)*t ; // == A*(1.f-t) + B*t ;
  }
  static Vector3f unlerp( const Vector3f& v, const Vector3f& A, const Vector3f& B ){
    return (v-A)/(B-A) ;  
  }
  
  static Vector3f linearSpline( float t, const Vector3f& c1, const Vector3f& c2 )
  {
    return (c1*t + c2*(1.f-t)) ;
  }

  static Vector3f quadraticSpline( float t, const Vector3f& c1, const Vector3f& c2, const Vector3f& c3 )
  {
    float _t = 1.f-t ;
    return c1*_t*_t + c2*2.f*_t*t + c3*t*t ;
  }

  static Vector3f cubicSpline( float t, const Vector3f& c1, const Vector3f& c2, const Vector3f& c3, const Vector3f& c4 )
  {
    float _t = 1.f-t ;
    return c1 *_t*_t*_t + 
      c2 *3.f*_t*_t*t + 
      c3 *3.f*_t*t*t +
      c4 *t*t*t ;
  }

  static Vector3f quarticSpline( float t, const Vector3f& c1, const Vector3f& c2, const Vector3f& c3, const Vector3f& c4, const Vector3f& c5 )
  {
    float _t = 1.f-t;
    return c1 *_t*_t*_t*_t + 
      c2 *4.f*_t*_t*_t*t + 
      c3 *6.f*_t*_t*t*t +
      c4 *4.f*_t*t*t*t + 
      c5 *t*t*t*t ;
  }

  static Vector3f quinticSpline( float t, const Vector3f& c1, const Vector3f& c2, const Vector3f& c3,
    const Vector3f& c4, const Vector3f& c5, const Vector3f& c6 )
  {
    float _t = 1.f-t;
    Vector3f color = c1 * _t*_t*_t*_t*_t + 
      c2 * 5.f *_t*_t*_t*_t*t + 
      c3 * 10.f*_t*_t*_t*t*t +
      c4 * 10.f*_t*_t*t*t*t + 
      c5 * 5.f *_t*t*t*t*t +
      c6 * t*t*t*t*t ;

    return color ;
  }
  
  
  ///////////////////////////////
  //           CONST           //
  inline void print() const {
    printf( "(%.2f %.2f %.2f)",x,y,z ) ;
  }
  inline void print( const char* msg ) const {
    printf( "%s (%.2f %.2f %.2f)",msg,x,y,z ) ;
  }
  inline void println() const {
    printf( "(%.2f %.2f %.2f)\n",x,y,z ) ;
  }
  inline void println(const char* msg) const {
    printf( "%s (%.2f %.2f %.2f)\n",msg,x,y,z ) ;
  }
  inline bool all( float val ) const { return x==val && y==val && z==val ; }
  inline bool allEqual() const { return x==y && x==z; }
  inline bool nonzero() const { return x||y||z ; }
  // the SQUARE must be nonzero. because for REALLY small x,
  // x*x can be 0.
  inline bool nonzero2() const { return x*x||y*y||z*z ; }
  // Note: negative (-0.0) works EXACTLY like +0.0
  inline bool allzero() const { return !x&&!y&&!z ; }
  inline int minIndex() const { 
    if( x <= y && x <= z )  return 0 ;
    else if( y <= x && y <= z ) return 1 ;
    else return 2 ;
  }
  inline int maxIndex() const { 
    if( x >= y && x >= z )  return 0 ;
    else if( y >= x && y >= z ) return 1 ;
    else return 2 ;
  }
  inline int middleIndex() const { 
    // y <= x <= z  OR  z <= x <= y
    if( ::isBetween( x, y, z ) )  return 0 ;
    else if( ::isBetween( y, x, z ) )  return 1 ;
    else return 2 ;
  }
  inline int fabsMinIndex() const { 
    Vector3f c = fabsCopy() ;
    if( c.x <= c.y && c.x <= c.z )  return 0 ;
    else if( c.y <= c.x && c.y <= c.z ) return 1 ;
    else return 2 ;
  }
  inline int fabsMaxIndex() const { 
    Vector3f c = fabsCopy() ;
    if( c.x >= c.y && c.x >= c.z )  return 0 ;
    else if( c.y >= c.x && c.y >= c.z ) return 1 ;
    else return 2 ;
  }
  inline int fabsMiddleIndex() const { 
    // y <= x <= z  OR  z <= x <= y
    Vector3f c = fabsCopy() ;
    if( ::isBetween( c.x, c.y, c.z ) )  return 0 ;
    else if( ::isBetween( c.y, c.x, c.z ) )  return 1 ;
    else return 2 ;
  }
  inline float max() const { return max3( x,y,z ) ; }
  inline float min() const { return min3( x,y,z ) ; }
  inline float getAvg() const { return (x+y+z)/3.f ; }
 
  
  // proximity
  inline bool isNear( const Vector3f& o ) const {
    return fabsf(x-o.x)<EPS_MIN && fabsf(y-o.y)<EPS_MIN && fabsf(z-o.z)<EPS_MIN ;
  }

  // you pick the epsilon
  inline bool isNear( const Vector3f& o, float eps ) const {
    return fabsf(x-o.x)<eps && fabsf(y-o.y)<eps && fabsf(z-o.z)<eps ;
  }
  
  inline bool isBetween( const Vector3f& min, const Vector3f& max ) const {
    return ::isBetweenOrdered( x, min.x, max.x ) &&
           ::isBetweenOrdered( y, min.y, max.y ) &&
           ::isBetweenOrdered( z, min.z, max.z ) ;
  }

  // any are near?
  inline bool anyNear( const Vector3f& o, float eps ) const {
    return ::isNear( x, o.x, eps ) || ::isNear( y, o.y, eps ) || ::isNear( z, o.z, eps ) ;
  }
  
  
  inline bool isCanonical() const {
    return -1.f <= x && x <= 1.f && 
           -1.f <= y && y <= 1.f && 
           -1.f <= z && z <= 1.f ;
  }
  inline bool isNaN() const {
    return isnan(x) || isnan(y) || isnan(z) ;
  }
  // for set ordering
  inline bool operator<( const Vector3f& o ) const {
    // uses lexical ordering.
    if( z < o.z ) return 1 ;
    else if( o.z < z ) return 0 ;
    // else z's are equal
    
    if( y < o.y ) return 1 ;
    else if( o.y < y ) return 0 ;
    // else y's are equal
    
    if( x < o.x ) return 1 ;
    else //if( o.x < x ) return 0 ; // no need to check, we know either equal
      return 0 ; 
    // x's are equal
  }
  // Exact equality
  inline bool operator==( const Vector3f& o ) const{
    return x==o.x && y==o.y && z==o.z ;
  }
  // Exact inequality
  inline bool operator!=( const Vector3f& o ) const{
    return x!=o.x || y!=o.y || z!=o.z ;
  }
  
  inline Vector3f operator+( const Vector3f& o ) const {
    return Vector3f(x+o.x,y+o.y,z+o.z);
  }
  inline Vector3f operator+( float o ) const {
    return Vector3f(x+o,y+o,z+o);
  }
  
  inline Vector3f operator-() const{
    return Vector3f(-x,-y,-z);
  }
  inline Vector3f operator-( const Vector3f& o ) const {
    return Vector3f(x-o.x,y-o.y,z-o.z);
  }
  inline Vector3f operator-( const Vector3i& o ) const {
    return Vector3f(x-o.x,y-o.y,z-o.z);
  }
  inline Vector3f operator-( float o ) const {
    return Vector3f(x-o,y-o,z-o);
  }
  
  inline Vector3f operator*( const Vector3f& o ) const {
    return Vector3f(x*o.x,y*o.y,z*o.z);
  }
  inline Vector3f operator*( const Vector3i& o ) const {
    return Vector3f(x*o.x,y*o.y,z*o.z);
  }
  inline Vector3f operator*( float s ) const {
    return Vector3f(x*s,y*s,z*s);
  }
  
  inline Vector3f operator/( const Vector3f& o ) const {
    return Vector3f(x/o.x,y/o.y,z/o.z);
  }
  inline Vector3f operator/( const Vector3i& o ) const {
    return Vector3f(x/o.x,y/o.y,z/o.z);
  }
  inline Vector3f operator/( float s ) const {
    return Vector3f(x/s,y/s,z/s);
  }
  // 9 op
  inline Vector3f cross( const Vector3f& o ) const {
    return Vector3f( y*o.z-o.y*z, z*o.x-x*o.z, x*o.y-o.x*y ) ;
  }
  // 5 op
  inline float dot( const Vector3f& o ) const{
    return x*o.x+y*o.y+z*o.z ;
  }
  //inline Vector3f proj( const Vector3f& u ) const {
    // projv u = |u| cos( v.angleWith(u) ) 
    //         = |u| ( u.dot(v) / |u||v| )
    //         = u.dot(v) / |v|
    //         = u.dot(v) IF v is normalized.
  //}
  inline float len() const {
    return sqrtf( x*x+y*y+z*z ) ;
  }
  inline float len2() const {
    return x*x+y*y+z*z ;
  }
  
  inline Vector2f getXY() const { return Vector2f(x,y); }
  inline Vector2f getXZ() const { return Vector2f(x,z); }
  
  inline Vector3f& withLen( float newLen ) const {
    return normalizedCopy()*=newLen ; // or return *this*=newLen/len();
  }
  
  inline Vector3f flippedCopy( const Vector3f& N ) const {
    // really negative of reflect, but with how it works easier to type code
    return N * 2.f*( this->dot( N ) ) - *this ;  // opposite to reflection
  }

  inline Vector3f reflectedCopy( const Vector3f& N ) const {
    return *this - N * 2.f*( this->dot( N ) ) ;
  }
  inline Vector3f reflectedDecayedCopy( const Vector3f& N, float scale ) const {
    return (*this - N * 2.f*( this->dot( N ) )) * scale ;
  }

  inline Vector3f normalizedCopy() const {
    return Vector3f( *this ).normalize() ;
  }
  inline Vector3f floorCopy() const {
    return Vector3f( floorf(x), floorf(y), floorf(z) ) ;
  }
  inline Vector3f ceilCopy() const {
    return Vector3f( ceilf(x), ceilf(y), ceilf(z) ) ;
  }
  inline Vector3i signum() const {
    return Vector3i( ::signum(x), ::signum(y), ::signum(z) ) ;
  }
  inline float angleWith( const Vector3f& o ) const {
    float v = this->normalizedCopy().dot(o.normalizedCopy()) ;
    ::clamp(v, -1.f, 1.f) ; // IF v is > 1, acosf errs out with NaN. Isn't that stupid?
    return acosf( v ) ;
  }
  
  // rotate THIS about AXIS, ANGLE degrees, to get that.
  inline void angleAndAxisToGet( const Vector3f& that, float&angle, Vector3f&axis ) const {
    // the axis to rotate about is cross
    axis = this->cross( that ) ;
    angle = angleWith( that ) ;
  }
  // wraps to world size, (-s,s) in all coordinates
  /*
  inline Vector3f wrappedCopy( float mins, float maxs ) const {
    Vector3f c = *this;
    c.wrap( mins, maxs ) ;
    return c ;
  }
  */
  inline Vector3f wrappedCopy( const Vector3f& worldSize ) const {
    Vector3f c = *this;
    return c.wrap( worldSize ) ;
  }
  
  inline bool needsWrap( float mins, float maxs ) const {
    return x < mins || y < mins || z < mins || x > maxs || y > maxs || z > maxs ;
  }
  
  // gets the tangential and normal components of THIS along fN
  // f MUST BE NORMALIZED.
  inline void parallelPerp( const Vector3f& fN, Vector3f &vParallel, Vector3f &vPerp ) const
  {
    // assuming fN is normalized
    vParallel = fN * this->dot( fN ) ;
    vPerp = (*this) - vParallel ;
  }
  
  inline int getDominantAxis() const {
    if( fabsf(x) > fabsf(y) && fabsf(x) > fabsf(z) )  return 0 ;
    else if( fabsf(y) > fabsf(x) && fabsf(y) > fabsf(z) )  return 1 ;
    else if( fabsf(z) > fabsf(x) && fabsf(z) > fabsf(y) )  return 2 ;
    else return -1 ; // (none -- at least two axes equal.
    // this happens for (0,0,0), (1,1,0) etc.
  }

  // Returns a dominant axis even when two of the axes
  // are equal. The "lowest" axis is returned
  // (e.g. (1,1,0) has x as the dominant axis)
  inline int getDominantAxisEq() const {
    if( all(0) ) return -1 ; // first check that they're all not 0
    else if( fabsf(x) >= fabsf(y) && fabsf(x) >= fabsf(z) )  return 0 ; // now this bins (1,1,0) with just axis 0
    else if( fabsf(y) >= fabsf(x) && fabsf(y) >= fabsf(z) )  return 1 ;
    else if( fabsf(z) >= fabsf(x) && fabsf(z) >= fabsf(y) )  return 2 ;
    else return -1 ; // (none -- at least two axes equal.
    // this happens for (0,0,0), (1,1,0) etc.
  }

  /// Gets you a perpendicular vector.  Tries to
  /// use the dominant axis so ( 0, 0, 1 ) will
  /// return ( -1, 0, 0 ) (a correct perpendicular)
  /// not ( -0, 0, 1 ) (which is useless, same vector)
  inline Vector3f getPerpendicular() const {
    // in the below switch, we
    // detect the dominant axis, and make sure
    // to use that component in the swap.
    // also we AVOID using a zero component for
    // the swap:  (-0.7, 0, 0.6) => (-0, -0.7, 0.6) is NOT a perpendicular
    // (but it would have been if the y component was non-zero)

    // (0,0,0) returns a vector (0, -0, 0)

    // unintuitively, we must zero a component
    // to get the perpendicular.
    switch( getDominantAxis() )
    {
    case 0: // x should be involved.  (1, 0, 0) --> ( -0, 1, 0 )
      if( y == 0.0f ) // avoid y
        return Vector3f( z, 0, -x ) ; // swap x, z. apply negation to x
      else
        return Vector3f( y, -x, 0 ) ; // swap x, y
      break ;
    case 1: // y should be involved:  (-1, 1, -1) --> (-1,-1, -1)
      // avoid swapping a zero component
      if( x == 0.0f ) // avoid x
        return Vector3f( 0, z, -y ) ; // y,z
      else
        return Vector3f( -y, x, 0 ) ; // x,y
    case 2: // z should be involved
    case -1: // No matter which we switch.
    default:
      if( x == 0.0f ) // avoid x
        return Vector3f( 0, -z, y ) ; // y,z [(0,0,0) case]
      else
        return Vector3f( -z, 0, x ) ; // swap -z,x
    }
  }
  inline Vector3f fabsCopy() const {
    return Vector3f( fabsf(x), fabsf(y), fabsf(z) ) ;
  }
  
  //NON-CONST
  // XX YOU ARE RESPONSIBLE FOR NOT SENDING A ZERO LENGTH VECTOR
  // XX (if you're not sure the vector will have a length, check it
  // XX first yourself.)
  inline Vector3f& normalize(){
    float length = len() ;
    
    // I added this debug check man, never take it out.
    if( !length ) {
      error( "Vector3f::normalize() attempt to divide by 0" ) ;
      return *this ;
    }
    
    return (*this)/=length ;
  }
  // reflects THIS about normal N.
  inline Vector3f& reflect( const Vector3f& N ) {
    return *this = *this - N * 2.f*( this->dot( N ) ) ;
  }
  // Reflects but also decays the vector
  inline Vector3f& reflectDecay( const Vector3f& N, float scale ) {
    return *this = (*this - N * 2.f*( this->dot( N ) ))*scale ;
  }
  
  inline Vector3f& floor() {
    x=floorf(x), y=floorf(y), z=floorf(z) ;
    return *this ;
  }
  inline Vector3f& ceil() {
    x=ceilf(x), y=ceilf(y), z=ceilf(z) ;
    return *this ;
  }
  // I'm aware len might be 0, and that's ok, it means I havea  0 vector (which I'm fie with)
  // other method is for when vector SHOULD NEVER BE 0
  // ret val 0 means wasn't normalized.. it's a 0 vector
  inline float safeNormalize(){
    float length = len() ;
    if( length )  (*this)/=length ;
    return length ;
  }
  inline Vector3f& operator+=( const Vector3f& o ){
    x+=o.x,y+=o.y,z+=o.z;
    return *this ;
  }
  inline Vector3f& operator+=( float s ){
    x+=s,y+=s,z+=s;
    return *this ;
  }
  inline Vector3f& operator-=( const Vector3f& o ){
    x-=o.x,y-=o.y,z-=o.z;
    return *this ;
  }
  inline Vector3f& operator-=( float s ){
    x-=s,y-=s,z-=s;
    return *this ;
  }
  inline Vector3f& operator*=( const Vector3f& o ){
    x*=o.x,y*=o.y,z*=o.z;
    return *this ;
  }
  inline Vector3f& operator*=( float s ){
    x*=s,y*=s,z*=s;
    return *this ;
  }
  inline Vector3f& operator/=( const Vector3f& o ){
    x/=o.x,y/=o.y,z/=o.z;
    return *this ;
  }
  inline Vector3f& operator/=( float s ){
    x/=s,y/=s,z/=s;
    return *this ;
  }
  inline Vector3f& operator%=( float s ){
    x=fmodf( x,s );  y=fmodf( y,s );  z=fmodf( z,s );
    return *this ;
  }
  inline Vector3f& operator%=( const Vector3f &o ){
    x=fmodf( x,o.x );  y=fmodf( y,o.y );  z=fmodf( z,o.z );
    return *this ;
  }
  // I didn't provide .transform functions because a copy must
  // be made anyway.  May as well use vec = mat*vec format.
  ///inline Vector3f& transformBy( const Matrix4f& mat ){}
  // A bit misleading, as it does not clampLen like the other func does
  inline Vector3f& clampComponent( float minVal, float maxVal ) {
    ::clamp( x,minVal,maxVal ) ;
    ::clamp( y,minVal,maxVal ) ;
    ::clamp( z,minVal,maxVal ) ;
    return *this ;
  }
  inline Vector3f& clampComponent( const Vector3f& mins, const Vector3f& maxes ) {
    ::clamp( x,mins.x,maxes.x ) ;
    ::clamp( y,mins.y,maxes.y ) ;
    ::clamp( z,mins.z,maxes.z ) ;
    return *this ;
  }
  
  
  inline Vector3f& clampLen( float minLen, float maxLen ){
    float length = len() ;
    if( length < minLen )
      return normalize()*=minLen ;
    if( length > maxLen ) // also means length > 0, hopefully
      return normalize()*=maxLen ;
    
    return *this ;
  }
  
  inline Vector3f& clampLen( float maxLen ){
    float length = len() ;
    if( length > maxLen ) // also means length > 0, hopefully
      return normalize()*=maxLen ;
    
    return *this ;
  }
  
  inline Vector3f& setLen( float newLen ){
    return normalize()*=newLen ; // or return *this*=newLen/len();
  }
  
  inline Vector3f& reduceLen( float by ){
    float length = len() ; // 5
    
    // I added this debug check man, never take it out.
    if( !length ) {
      error( "Vector3f::reduceLen() attempt to divide by 0" ) ;
      return *this ;
    }
    //setLen( length - by ) ; // this is inefficient, because it finds len() again to normalize.
    float scale = (length - by) / length ; // scale = (5 - 3) / 5 = 0.4, and 3 is 60% of 5, so
    // reduce len by 3 is reduction by 60% which is eq to *.4
    // if it's negative, it turns the vector around.  That's the behavior we want.
    *this *= scale ; // 5*.4 = 2
    return *this ;
    
    // if you are really close to your target (1)
    // and you reduce len by 5, then you will TURN AROUND
    // because the scale will be
    // (1-5)/1 = -4
  }
  
  
  // wraps to world size, in all coordinates
  /*
  inline Vector3f& wrap( float mins, float maxs ){
    float ws = maxs-mins;
    if( x > maxs ) x -= ws ;
    else if( x < mins ) x += ws ;
    if( y > maxs ) y -= ws ;
    else if( y < mins ) y += ws ;
    if( z > maxs ) z -= ws ;
    else if( z < mins ) z += ws ;
    return *this ;
  }
  */
  // This may be faster, but it requires you use the +octant
  // (0,0,0)->(worldSize,worldSize,worldSize)
  inline Vector3f& wrap( const Vector3f& worldSize ){
    *this += worldSize ;
    return *this %= worldSize ;
  }
  
  inline Vector2f& xy() {
    return (Vector2f&)x ;
  }
  // assignment from Vector2
  inline Vector3f& xy( const Vector2f& o ){
    x=o.x,y=o.y; return *this ;
  }
  inline Vector3f& yx( const Vector2f& o ){
    y=o.x,x=o.y; return *this ;
  }
  inline Vector3f& xy( float ix, float iy ){
    x=ix,y=iy; return *this ;
  }
  inline Vector3f& xz( const Vector2f& o ){
    x=o.x,z=o.y; return *this ;
  }
  inline Vector3f& zx( const Vector2f& o ){
    z=o.x,x=o.y; return *this ;
  }
  inline Vector3f& xz( float ix, float iz ){
    x=ix,z=iz; return *this ;
  }
  inline Vector3f& yz( const Vector2f& o ){
    y=o.x,z=o.y; return *this ;
  }
  inline Vector3f& zy( const Vector2f& o ){
    z=o.x,y=o.y; return *this ;
  }
  inline Vector3f& yz( float iy, float iz ){
    y=iy,z=iz; return *this ;
  }
  
  
  
  
  
  
  // r,g,b values are from 0 to 1
  // h = [0,360], s = [0,1], v = [0,1]
  //		if s == 0, then h = -1 (undefined)
  static Vector3f RGBtoHSV( float r, float g, float b )
  {
    Vector3f hsv ;
	  
    float min, max, delta;

    // if you have a problem here because you're using Gdiplus,
    // just remove std:: from in front and let the compiler use
    // the min, max fcns defined in Windef.h.
	  min = std::min( std::min( r, g ), b ) ; // MIN( r, g, b );
	  max = std::max( std::max( r, g ), b ) ; // MAX( r, g, b );
	  hsv.z = max;				// v
	  
    delta = max - min;
	  if( max != 0 )
		  hsv.y = delta / max;		// s
	  else {
		  // r = g = b = 0		// s = 0, v is undefined
		  hsv.y = 0;  //s
		  hsv.x = -1; //h
		  return hsv ;
	  }
	  if( r == max )
		  hsv.x = ( g - b ) / delta;		// h: between yellow & magenta
	  else if( g == max )
		  hsv.x = 2 + ( b - r ) / delta;	// h: between cyan & yellow
	  else
		  hsv.x = 4 + ( r - g ) / delta;	// h: between magenta & cyan

    hsv.x *= 60;				// h: degrees (it would have been 0 to 6).
	  if( hsv.x < 0 )
		  hsv.x += 360; //h:

      return hsv;
  }
  
  Vector3f toHSV() const {
	  return RGBtoHSV( x, y, z ) ;
  }
  Vector3f toRGB() const {
	  return HSVtoRGB( x, y, z ) ;
  }
  static Vector3f HSVtoRGB( float h, float s, float v )
  {
    Vector3f rgb ;

	  int i;
	  float f, p, q, t;
	  if( s == 0 ) {
		  // achromatic (grey)
		  rgb.x = rgb.y = rgb.z = v;
		  return rgb ;
	  }

    h /= 60;			// sector 0 to 5
	  i = ::floorf( h );
	  f = h - i;			// factorial part of h
	  p = v * ( 1 - s );
	  q = v * ( 1 - s * f );
	  t = v * ( 1 - s * ( 1 - f ) );
	  switch( i ) {
		  case 0:
			  rgb.x = v;
			  rgb.y = t;
			  rgb.z = p;
			  break;
		  case 1:
			  rgb.x = q;
			  rgb.y = v;
			  rgb.z = p;
			  break;
		  case 2:
			  rgb.x = p;
			  rgb.y = v;
			  rgb.z = t;
			  break;
		  case 3:
			  rgb.x = p;
			  rgb.y = q;
			  rgb.z = v;
			  break;
		  case 4:
			  rgb.x = t;
			  rgb.y = p;
			  rgb.z = v;
			  break;
		  default:		// case 5:
			  rgb.x = v;
			  rgb.y = p;
			  rgb.z = q;
			  break;
	  }

    return rgb ;
  }

  inline Vector3f& fabs() {
    x=fabsf(x) ;  y=fabsf(y); z=fabsf(z) ;
    return *this ;
  }

} ;

struct SVector
{
  float r,
    tElevation, // angle from +y
    pAzimuth ;  // angle from +x axis
  
  // Radius 1 default
  SVector() : r(1.0f),pAzimuth(0.0f),tElevation(0.0f) {}
  SVector( float ir, float iTElevation, float iPAzimuth ):
    r(ir),tElevation(iTElevation),pAzimuth(iPAzimuth) { }
  Vector3f toCartesian() const
  {
    return Vector3f(
      r*sinf(tElevation)* //proj xz plane
        cosf(pAzimuth),
      r*cosf(tElevation), // independent of pAzimuth, pAzimuth goes around the y-axis
      r*sinf(tElevation)*sinf(pAzimuth)
    ) ;
  }

  // Gives a random point on the sphere
  static SVector random( float radius )
  {
    return SVector( radius,
      acosf( 1 - 2*randFloat() ),
      randFloat( 0, 2*M_PI )
    ) ;
  }

  // Gets you a random vector on a hemisphere about
  // the normal.
  static Vector3f randomHemi( const Vector3f& normal )
  {
    if( normal.len2() == 0 ) {
      error( "Bad normal" ) ;
      return Vector3f() ;
    }
    Vector3f cart = SVector::random( 1 ).toCartesian() ;
    
    do cart = SVector::random( 1 ).toCartesian() ;
    while( cart.dot( normal ) < 0 ) ;

    return cart ;
  }

  // This gives you a random distribution
  // actually in a lobe about the normal.
  static SVector randomHemiCosine( float radius )
  {
    return SVector( radius,
      acos( sqrt( randFloat() ) ),
      2*M_PI*randFloat()
    ) ;
  }

//  static Vector3f randomHemiCosine( const Vector3f& normal )
//  {
//    SVector sv = randomHemiCosine( 1.0 ) ;
//    Vector3f y(0,1,0);
//    Vector3f axis = y.cross( normal ) ;
//    float rads = y.angleWith( normal ) ;
//    Matrix3f mat = Matrix3f::rotation( axis, rads ) ;
//    return mat * sv.toCartesian() ;
//  }
} ;


// COULD provide cross class operators here, so both ways are supported.
// But I think there's an advantage to only allowing them to happen ONE WAY
// (which is 3f*3i right now, NOT 3i*3f, that's illegal)
//Vector3f operator+( const Vector3f& v3f, const Vector3i& v3i ){ }
//Vector3f operator+( const Vector3i& v3i, const Vector3f& v3f ){ }

// name collide with std::distance, annoying.
inline float distance1( const Vector3f& a, const Vector3f& b ) {
  return ( a - b ).len() ;
}

// squared distance
inline float distance2( const Vector3f& a, const Vector3f& b ) {
  return ( a - b ).len2() ;
}


union Vector4f
{
  struct{ float x,y,z,w ; } ;
  struct{ float r,g,b,a ; } ;
  float elts[4];
  
  Vector4f():x(0.f),y(0.f),z(0.f),w(1.f){}
  Vector4f( float ix, float iy, float iz ):x(ix),y(iy),z(iz),w(1.f){}
  Vector4f( float ix, float iy, float iz, float iw ):x(ix),y(iy),z(iz),w(iw){}
  Vector4f( const Vector3f& o, float iw ):x(o.x),y(o.y),z(o.z),w(iw){}
  Vector4f( const Vector3f& v3f ):x(v3f.x),y(v3f.y),z(v3f.z),w(1.0f){}
  Vector4f( const Vector2f& v2f ):x(v2f.x),y(v2f.y),z(0.0f),w(1.0f){}
  Vector4f( float iv ):x(iv),y(iv),z(iv),w(iv){}
  
  static inline Vector4f random() { return Vector4f( randFloat(), randFloat(), randFloat(), 1.f ) ;  }
  
  static inline Vector4f random(float min, float max) {
    return Vector4f( randFloat(min,max), randFloat(min,max), randFloat(min,max), 1.f ) ;
  }
  static Vector4f lerp( float t, const Vector4f& A, const Vector4f& B ){
    return A*(1.f-t) + B*t ;
  }
  //CONST
  inline void print() const {
    printf( "(%.2f %.2f %.2f %.2f) ",x,y,z,w ) ;
  }
  inline void println( const char* msg ) const {
    printf( "%s (%.2f %.2f %.2f %.2f)\n",msg,x,y,z,w ) ;
  }
  inline void println() const {
    printf( "(%.2f %.2f %.2f %.2f)\n",x,y,z,w ) ;
  }
  // Good for checking if all 0
  inline bool all( float val ){
    return x==val && y==val && z==val && w==val ;
  }
  inline Vector4f operator+( const Vector4f& o ) const {
    return Vector4f(x+o.x,y+o.y,z+o.z,w+o.w);
  }
  inline Vector4f operator-() const{
    return Vector4f(-x,-y,-z,-w);
  }
  inline Vector4f operator-( const Vector4f& o ) const {
    return Vector4f(x-o.x,y-o.y,z-o.z,w-o.w);
  }
  inline Vector4f operator*( const Vector4f& o ) const {
    return Vector4f(x*o.x,y*o.y,z*o.z,w*o.w);
  }
  inline Vector4f operator*( float s ) const {
    return Vector4f(x*s,y*s,z*s,w*s);
  }
  inline Vector4f operator/( const Vector4f& o ) const {
    return Vector4f(x/o.x,y/o.y,z/o.z,w/o.w);
  }
  inline Vector4f operator/( float s ) const {
    return Vector4f(x/s,y/s,z/s,w/s);
  }
  // 5 op
  inline float dot( const Vector4f& o ) const{
    return x*o.x+y*o.y+z*o.z+w*o.w ;
  }
  
  // proximity
  inline bool isNear( const Vector4f& o ) const{
    return fabsf(x-o.x)<EPS_MIN && fabsf(y-o.y)<EPS_MIN && fabsf(z-o.z)<EPS_MIN && fabsf(w-o.w)<EPS_MIN ;
  }
  inline bool isCanonical() const {
    return -1.f <= x && x <= 1.f && 
           -1.f <= y && y <= 1.f && 
           -1.f <= z && z <= 1.f && 
           -1.f <= w && w <= 1.f ;
  }
  // Exact equality
  inline bool operator==( const Vector4f& o ) const{
    return x==o.x && y==o.y && z==o.z && w==o.w;
  }
  
  Vector4f toHSV() const {
	  return Vector4f( Vector3f::RGBtoHSV( x, y, z ), w ) ;
  }
  Vector4f toRGB() const {
	  return Vector4f( Vector3f::HSVtoRGB( x, y, z ), w ) ;
  }
  
  inline Vector3f& xyz(){
    return (Vector3f&)x ;
  }
  
  inline Vector4f& operator+=( const Vector4f& o ){
    x+=o.x,y+=o.y,z+=o.z,w+=o.w;
    return *this ;
  }
  inline Vector4f& operator-=( const Vector4f& o ){
    x-=o.x,y-=o.y,z-=o.z,w-=o.w;
    return *this ;
  }
  inline Vector4f& operator*=( const Vector4f& o ){
    x*=o.x,y*=o.y,z*=o.z,w*=o.w;
    return *this ;
  }
  inline Vector4f& operator*=( float s ){
    x*=s,y*=s,z*=s,w*=s;
    return *this ;
  }
  inline Vector4f& operator/=( const Vector4f& o ){
    x/=o.x,y/=o.y,z/=o.z,w/=o.w;
    return *this ;
  }
  inline Vector4f& operator/=( float s ){
    x/=s,y/=s,z/=s,w/=s;
    return *this ;
  }
  inline Vector4f& clampComponent( float minVal, float maxVal )
  {
    ::clamp( x,minVal,maxVal ) ;
    ::clamp( y,minVal,maxVal ) ;
    ::clamp( z,minVal,maxVal ) ;
    ::clamp( w,minVal,maxVal ) ;
    return *this ;
  }
  
  // You only need __4__ components to do a regular perspective projection, not 16
  static Vector4f persp(float fovyRadians, float aspect, float nearZ, float farZ)
  {
    float yScale = 1.0 / tanf( fovyRadians / 2.0 ) ;
    float xScale = yScale / aspect ;
    
    // This is D3DXMatrixPerspectiveFovRH: http://msdn.microsoft.com/en-us/library/bb205351(VS.85).aspx
    //((TRANSPOSED)):  THE FOLLOWING MATRIX IS COLUMN MAJOR:
    //xScale     0          0              0
    //0        yScale       0              0
    //0        0        zf/(zn-zf)        zn*zf/(zn-zf)
    //0        0           -1              0
    //where:
    //yScale = cot(fovY/2)
    //xScale = yScale / aspect ratio

    // You don't actually need a whole matrix to do
    // perspective projection.  You only need a vec4.
    return Vector4f( xScale,
                     yScale,
                     (farZ) / (nearZ - farZ),
                     (farZ * nearZ) / (nearZ - farZ)
                   ) ;
                   
    // So:
    // xT = v.x * x ;
    // yT = v.y * y ;
    // zT = v.z * z + v.w ;
    // wT = -z ;
    //gl_Position.x = eyeSpace.x * perspProj.x ;
    //gl_Position.y = eyeSpace.y * perspProj.y ;
    //gl_Position.z = eyeSpace.z * perspProj.z + perspProj.w ;
    //gl_Position.w = -eyeSpace.z ;
  }
} ;


extern Vector4f Red, DarkRed, 
  Green, DarkGreen,
  Blue, DarkBlue,
  Purple, Orange,
  White, Gray, DarkGray, Black, 
  Magenta, Cyan, Yellow,
  TWhite, TBlack
;


union Matrix3f
{
  struct { float m00,m01,m02,
                 m10,m11,m12,
                 m20,m21,m22 ; } ;
  float elts[9] ;
  
  Matrix3f():
    // IDENTITY
    m00(1),m01(0),m02(0),
    m10(0),m11(1),m12(0),
    m20(0),m21(0),m22(1)
  { }

  Matrix3f( float im00, float im01, float im02,
            float im10, float im11, float im12,
            float im20, float im21, float im22 ) :
    m00(im00), m01(im01), m02(im02),
    m10(im10), m11(im11), m12(im12),
    m20(im20), m21(im21), m22(im22)
  { }
  
  Matrix3f( const Vector3f& right, const Vector3f& up, const Vector3f& forward ) :
    m00(right.x), m01(up.x), m02(-forward.x),
    m10(right.y), m11(up.y), m12(-forward.y),
    m20(right.z), m21(up.z), m22(-forward.z)
  { }
  
  inline int getIndex( int iCol, int iRow ) const {
    return iCol*3 + iRow ;
  }
  inline Vector3f row( int iRow ) const {
    return Vector3f( elts[getIndex(0,iRow)], elts[getIndex(1,iRow)], elts[getIndex(2,iRow)] ) ;
  }
  inline Vector3f col( int iCol ) const {
    return Vector3f( elts[getIndex(iCol,0)], elts[getIndex(iCol,1)], elts[getIndex(iCol,2)] ) ;
  }
  inline Matrix3f& transpose() {
    swap( m10, m01 ) ;
    swap( m20, m02 ) ;
    swap( m21, m12 ) ;
    return *this ;
  }

  inline static Matrix3f rotation( const Vector3f& u, float radians )
  {
    float c = cosf( radians ) ;
    float l_c = 1 - c ;
    float s = sinf( radians ) ;
    
    // COLUMN MAJOR
    return Matrix3f(
      u.x*u.x + (1.f - u.x*u.x)*c,   u.x*u.y*l_c - u.z*s,   u.x*u.z*l_c + u.y*s,
      u.x*u.y*l_c + u.z*s,   u.y*u.y+(1.f - u.y*u.y)*c,   u.y*u.z*l_c - u.x*s,
      u.x*u.z*l_c - u.y*s,   u.y*u.z*l_c + u.x*s,   u.z*u.z + (1.f - u.z*u.z)*c
    )   ;
  }
  
  inline static Matrix3f rotationX( float radians )
  {
    float c = cosf( radians ) ;
    float s = sinf( radians ) ;
    
    //   ^ y
    //   |
    // <-o
    // z  x
    
    // COLUMN MAJOR, RH
    return Matrix3f(
      1, 0, 0, // COLUMN 1
      0, c, s,
      0,-s, c
    ) ;
  }
  
  inline static Matrix3f rotationY( float radians )
  {
    float c = cosf( radians ) ;
    float s = sinf( radians ) ;
    
    //z
    // <-oy
    //   |
    //   vx
    //
    
    // COLUMN MAJOR, [ 1 0 0 ] ->90y-> [ 0 0 -1 ]
    //   so z value has -sin, ie 3rd row
    return Matrix3f(
      c, 0,-s, // COLUMN 1
      0, 1, 0,
      s, 0, c
    ) ;
  }
  
  inline static Matrix3f rotationZ( float radians )
  {
    float c = cosf( radians ) ;
    float s = sinf( radians ) ;
    
    // ^y
    // |
    // o->x
    //z
    
    // COLUMN MAJOR
    return Matrix3f(
      c, s, 0, // COLUMN 1
     -s, c, 0,
      0, 0, 1
    ) ;
  }
  
  // Y,X,Z
  inline static Matrix3f rotationYawPitchRoll( float radiansYaw, float radiansPitch, float radiansRoll )
  {
    // COLUMN MAJOR so yaw first, yaw at right
    return rotationZ( radiansRoll ) * rotationX( radiansPitch ) * rotationY( radiansYaw ) ;
  }
  
  // COLUMN MAJOR:
  // FIRST INDEX=COLUMN, SECOND INDEX=ROW
  // m00  m10  m20  [o.x]
  // m01  m11  m12  [o.y]
  // m02  m12  m22  [o.z]
  // post multiply only
  inline Vector3f operator*( const Vector3f& o ) const
  {
    // ┌             ┐   ┌   ┐   ┌                       ┐
    // │ m00 m10 m20 │   │ x │   │ m00*x + m10*y + m20*z │
    // │ m01 m11 m21 │ * │ y │ = │ m01*x + m11*y + m21*z │
    // │ m02 m12 m22 │   │ z │   │ m02*x + m12*y + m22*z │
    // └             ┘   └   ┘   └                       ┘
    return Vector3f(
      o.x*m00 + o.y*m10 + o.z*m20,  //o.dot( *(Vector3f*)(&m00) ), // this won't inline
      o.x*m01 + o.y*m11 + o.z*m21,
      o.x*m02 + o.y*m12 + o.z*m22
    ) ;
  }
  
  Matrix3f operator*( const Matrix3f& o ) const
  {
    Matrix3f m ;
    
    // 0 4  8 12
    // 1 5  9 13
    // 2 6 10 14
    // 3 7 11 15
    
    // 0 3 6   0 3 6
    // 1 4 7   1 4 7
    // 2 5 8   2 5 8
    
    m.elts[0]  = elts[0] * o.elts[0]  + elts[3] * o.elts[1]  + elts[6] * o.elts[2] ;
    m.elts[1]  = elts[1] * o.elts[0]  + elts[4] * o.elts[1]  + elts[7] * o.elts[2] ;
    m.elts[2]  = elts[2] * o.elts[0]  + elts[5] * o.elts[1]  + elts[8] * o.elts[2] ;
    
    m.elts[3]  = elts[0] * o.elts[3]  + elts[3] * o.elts[4]  + elts[6] * o.elts[5] ;
    m.elts[4]  = elts[1] * o.elts[3]  + elts[4] * o.elts[4]  + elts[7] * o.elts[5] ;
    m.elts[5]  = elts[2] * o.elts[3]  + elts[5] * o.elts[4]  + elts[8] * o.elts[5] ;
    
    m.elts[6]  = elts[0] * o.elts[6]  + elts[3] * o.elts[7]  + elts[6] * o.elts[8] ;
    m.elts[7]  = elts[1] * o.elts[6]  + elts[4] * o.elts[7]  + elts[7] * o.elts[8] ;
    m.elts[8]  = elts[2] * o.elts[6]  + elts[5] * o.elts[7]  + elts[8] * o.elts[8] ;
    
    return m;
  }
  
  void println() const {
    for( int i = 0 ; i < 3 ; i++ )
    {
      // Because printf works horizontally, we have to print out each ROW.
      for( int j = 0 ; j < 3 ; j++ )
        printf( "%8.3f ", elts[ getIndex( j, i ) ] ) ;
      puts("");
    }
    puts("");
  }
} ;

// COLUMN MAJOR TO FIT with OpenGL.
union Matrix4f
{
  // ACTUAL LAYOUT ON PAPER (since column major):
  //
  // upper left 3x3 ROTATION.
  // [ m00  m10  m20 | m30 ]}
  // [ m01  m11  m21 | m31 ]} m30,m31,m32 TRANSLATION.
  // [ m02  m12  m22 | m32 ]}
  //   -------------   --- 
  // [ m03  m13  m23 | m33 ]
  
  struct { float m00,m01,m02,m03,
                 m10,m11,m12,m13,
                 m20,m21,m22,m23,
                 m30,m31,m32,m33 ; } ;
  float elts[16] ; 
  
  Matrix4f():
    m00(1), m01(0), m02(0), m03(0),
    m10(0), m11(1), m12(0), m13(0),
    m20(0), m21(0), m22(1), m23(0),
    m30(0), m31(0), m32(0), m33(1)
  {}
    
  Matrix4f( float im00, float im01, float im02, float im03,
            float im10, float im11, float im12, float im13,
            float im20, float im21, float im22, float im23,
            float im30, float im31, float im32, float im33 ) :
    m00(im00), m01(im01), m02(im02), m03(im03),
    m10(im10), m11(im11), m12(im12), m13(im13),
    m20(im20), m21(im21), m22(im22), m23(im23),
    m30(im30), m31(im31), m32(im32), m33(im33)
  {}
  
  Matrix4f( const Matrix3f& o ) :
    m00( o.m00 ), m01( o.m01 ), m02( o.m02 ), m03( 0 ),
    m10( o.m10 ), m11( o.m11 ), m12( o.m12 ), m13( 0 ),
    m20( o.m20 ), m21( o.m21 ), m22( o.m22 ), m23( 0 ),
    m30( 0 ),     m31( 0 ),     m32( 0 ),     m33( 1 )
  {}
  Matrix4f( const Matrix3f& o, const Vector3f& t ) :
    m00( o.m00 ), m01( o.m01 ), m02( o.m02 ), m03( 0 ),
    m10( o.m10 ), m11( o.m11 ), m12( o.m12 ), m13( 0 ),
    m20( o.m20 ), m21( o.m21 ), m22( o.m22 ), m23( 0 ),
    m30( t.x ),   m31( t.y ),   m32( t.z ),   m33( 1 )
  {}
  
  Matrix4f( const Matrix4f& o ) :
    m00( o.m00 ), m01( o.m01 ), m02( o.m02 ), m03( o.m03 ),
    m10( o.m10 ), m11( o.m11 ), m12( o.m12 ), m13( o.m13 ),
    m20( o.m20 ), m21( o.m21 ), m22( o.m22 ), m23( o.m23 ),
    m30( o.m30 ), m31( o.m31 ), m32( o.m32 ), m33( o.m33 )
  {}
  inline int getIndex( int iCol, int iRow ) const {
    return iCol*4 + iRow ;
  }
  inline Vector3f row( int iRow ) const {
    return Vector3f( elts[getIndex(0,iRow)], elts[getIndex(1,iRow)], elts[getIndex(2,iRow)] ) ;
  }
  inline Vector3f col( int iCol ) const {
    return Vector3f( elts[getIndex(iCol,0)], elts[getIndex(iCol,1)], elts[getIndex(iCol,2)] ) ;
  }
  // This is the translation component of the matrix
  inline Vector3f getTranslation() const {
    return Vector3f( m30, m31, m32 ) ;
  }
  /*
  Vector3f right() {
    return *this ;
  }
  Vector3f up() {
    return *this ;
  }
  Vector3f forward() {
    return *this ;
  }
  */
  // rotation component only as a 3x3
  inline Matrix3f getRotation() const {
    return Matrix3f(
      m00,m01,m02,
      m10,m11,m12,
      m20,m21,m22
    );
  }
  // Transposes the rotation component as it passes it back to you
  inline Matrix3f getInvertedRotation() const {
    return Matrix3f(
      m00,m10,m20,
      m01,m11,m21,
      m02,m12,m22
    );
  }
  
  static Matrix4f LookAt( const Vector3f& eye, const Vector3f& look, const Vector3f& up )
  {
    // The reason for all this crossing is to get 3 mutually perpendicular vectors,
    // with the stipulation that f (the forward vector) is along from eye to look.
    // (actually it is backwards here, because the lookat matrix changes handedness for you)
    Vector3f f = (eye - look).normalize();  // negated forward
    
    // if forward were NOT backwards, then these would be written properly for a right-handed system
    // as right = forward × up
    //    up = right × forward
    // But since f = -f, think of these as LEFT HANDED CROSS PRODUCTS.
    Vector3f r = up.cross( f ) ;
    Vector3f u = f.cross( r ) ;
    
    // A LOOKAT matrix is a TRANSLATION, TO PUT THE EYE AT THE ORIGIN,
    // followed by a ROTATION to swing the world around so
    // the eye points down -z.
    //
    // Oh you like matrix math?
    // MODELVIEW = ROT^T * TRANS  (trans then (inverse rot) in column major)
    // So, ROT*TRANS here is the transpose of the forward rotation matrix, times the rotation that goes -eye (-t here)
    // ┌               ┐   ┌            ┐   ┌                  ┐
    // │ r.x r.y r.z 0 │   │ 1 0 0 -t.x │   │ r.x r.y r.z -t•r │
    // │ u.x u.y u.z 0 │ * │ 0 1 0 -t.y │ = │ u.x u.y u.z -t•u │
    // │ f.x f.y f.z 0 │   │ 0 0 1 -t.z │   │ f.x f.y f.z -t•f │
    // │  0   0   0  1 │   │ 0 0 0   1  │   │  0   0   0    1  │
    // └               ┘   └            ┘   └                  ┘
    // This may seem INSANE, but the LookAt matrix doesn't actually move..
    // the eye (or camera).  Instead, it moves EVERY VERTEX IN THE WORLD
    // __BACKWARDS__ -eye units, THEN "UNROTATES" THEM.
    // If you had a camera and a bowl of fruit, and the camera was laying face up on a table,
    // you could either pick up the camera, move and rotate it to point at the subject.
    // Or, if you were stupid and could move your house but the camera was cosmically nailed to
    // the point in space it started in, you could just move your entire house (and everything in it)
    // BY THE OPPOSITE TRANSFORMATION you wanted to apply to the camera.  That would give you the same picture.
    // 
    // Expensive way:
    // return TransformToFace( u,v,n ).transpose() * Translation( -eye ) ;
    // Because we can just 
    
    // TRANS THEN ROT.. IN FOR A FOT
    //   - trans by -eye
    //   - unrotate by lookat matrix
    // ON PAPER as a column major matrix:
    // ┌                             ┐
    // │ r.x  r.y  r.z │ -r.dot(eye) │
    // │ u.x  u.y  u.z │ -u.dot(eye) │
    // │ f.x  f.y  f.z │ -f.dot(eye) │
    // │ ──────────────┼──────────── │
    // │  0    0    0  │      1      │
    // └                             ┘
    Matrix4f m( r.x, u.x, f.x, 0,
                r.y, u.y, f.y, 0,
                r.z, u.z, f.z, 0,
                 -(r.dot( eye )),
                 -(u.dot( eye )),
                 -(f.dot( eye )),
                 1 );
    
    return m;
  }
  
  // The lookat matrix is actually inverted.
  // this one does a forward transformation given some eye, look, up vectors
  static Matrix4f LookAtFORWARD( const Vector3f& eye, const Vector3f& look, const Vector3f& up )
  {
    // LookAtFORWARD is the transformation to apply ROTATION FOLLOWED BY TRANSLATION
    // (TRANS*ROT for column major)
    // that would be used on an object to move it where the camera IS.?
    
    // These are right and they reverse handedness.
    Vector3f f = (eye - look).normalize(); // LOOKATPT->EYE .. BACKWARDS on purpose
    Vector3f r = up.cross( f ) ;
    Vector3f u = f.cross( r ) ;
    
    // rot then trans (rot on right since column major) is simple appand
    // ┌           ┐   ┌                ┐   ┌                 ┐
    // │ 1 0 0 t.x │   │ r.x u.x f.x  0 │   │ r.x u.x f.x t.x │
    // │ 0 1 0 t.y │ * │ r.y u.y f.y  0 │ = │ r.y u.y f.y t.y │
    // │ 0 0 1 t.z │   │ r.z u.z f.z  0 │   │ r.z u.z f.z t.z │
    // │ 0 0 0  1  │   │  0   0   0   1 │   │  0   0   0   1  │
    // └           ┘   └                ┘   └                 ┘
    // backwardsded. this is WRONG so don't use it.
    //Vector3f n = (look - eye).normalize();
    //Vector3f u = n.cross( up ) ;
    //Vector3f v = u.cross( n );
    Matrix4f m( r.x, r.y, r.z, 0,
                u.x, u.y, u.z, 0,
                f.x, f.y, f.z, 0,
                eye.x, eye.y, eye.z, 1 );
    
    return m;
  }
  
  static Matrix4f Translation( const Vector3f& t )
  {
    // ┌           ┐
    // │ 1 0 0 t.x │
    // │ 0 1 0 t.y │
    // │ 0 0 1 t.z │
    // │ 0 0 0  1  │
    // └           ┘
    return Matrix4f(
        1,   0,   0, 0,
        0,   1,   0, 0,
        0,   0,   1, 0,
      t.x, t.y, t.z, 1
    ) ;
  }
  
  // Thsi doesn't work as expected
  //Matrix4f invertedLookAt() const {
  //  return Matrix4f( getInvertedRotation(), -getTranslation() ) ;
  //}
  
  // Keeps handedness
  static Matrix4f TransformToFace( const Vector3f& right, const Vector3f& up, const Vector3f& forward )
  {
    // COLUMN MAJOR: The new basis vectors in terms
    // of the original basis are the COLUMNS of M.
    // ON PAPER:
    // ┌                              ┐
    // │ right.x  up.x  forward.x │ 0 │
    // │ right.y  up.y  forward.y │ 0 │
    // │ right.z  up.z  forward.z │ 0 │
    // │ ─────────────────────────┼── │
    // │    0       0          0  │ 1 │
    // └                              ┘
    return Matrix4f(
         right.x,    right.y,    right.z, 0,
            up.x,       up.y,       up.z, 0,
       forward.x,  forward.y,  forward.z, 0,
               0,          0,          0, 1
    ) ;
  }

  // Like OpenGL, this embeds a change of handedness
  static Matrix4f TransformToFaceChangeHandedness( const Vector3f& right, const Vector3f& up, const Vector3f& forward )
  {
    // COLUMN MAJOR: The new basis vectors in terms
    // of the original basis are the COLUMNS of M, but I quietly backwardsize fwd to change handedness.
    return Matrix4f(
         right.x,    right.y,    right.z, 0,
            up.x,       up.y,       up.z, 0,
      -forward.x, -forward.y, -forward.z, 0,
               0,          0,          0, 1
    ) ;
  }
  
  // If you pass nearZ=1, farZ=-1, you get NO CHANGE to your z (fsn=-2,fan=0).
  static Matrix4f Ortho(float left, float right,
                        float bottom, float top,
                        float nearZ, float farZ)
  {
    float ral = right + left;
    float rsl = right - left;
    float tab = top + bottom;
    float tsb = top - bottom;
    float fan = farZ + nearZ;
    float fsn = farZ - nearZ; //FARSUBNEAR
    
    Matrix4f m(  2.f / rsl,        0.f,         0.f, 0.f,
                       0.f,  2.f / tsb,         0.f, 0.f,
                       0.f,        0.f,  -2.f / fsn, 0.f,
                -ral / rsl, -tab / tsb,  -fan / fsn, 1.f );
                     
    return m;
  }
  
  // Just so you don't have to remember NEAR is +1, FAR is -1
  // "looking down -ve z.."
  static Matrix4f Ortho(float left, float right,
                        float bottom, float top )
  {
    // this actually says the NEAR PLANE is 1 unit in front of me (at z=-1)
    // and the FAR PLANE is 1 unit BEHIND me (at z=+1) WHICH MEANS z=-1 is
    // THE CLOSEST THE EYE POSSIBLE and Z=+1 is the FURTHEST from the eye possible.
    //This screw up is there b/c the GPU is LH in canonical space.
    return Ortho( left, right, bottom, top, 1.f, -1.f ) ;
  }
  
  static Matrix4f persp(float fovyRadians, float aspect, float nearZ, float farZ)
  {
    float yScale = 1.0 / tanf( fovyRadians / 2.0 ) ;
    float xScale = yScale / aspect ;
    
    // Shitt
    /*
    return Matrix4f( xScale,      0,                                      0,  0,
                          0, yScale,                                      0,  0,
                          0,      0,        (farZ + nearZ) / (nearZ - farZ), -1,
                          0,      0, (2.0f * farZ * nearZ) / (nearZ - farZ),  0 );
    */
    
    // This is D3DXMatrixPerspectiveFovRH
    // You don't actually need a whole matrix to do
    // perspective projection.  You only need a vec4.
    // http://msdn.microsoft.com/en-us/library/bb205351(VS.85).aspx
    return Matrix4f( xScale,      0,                               0,  0,
                          0, yScale,                               0,  0,
                          0,      0,         (farZ) / (nearZ - farZ), -1,
                          0,      0, (farZ * nearZ) / (nearZ - farZ),  0 );
  }
  
  // MATRIX MODIFIER FUNCTIONS
  Matrix4f& translate( const Vector3f& t )
  {
    m30 += t.x ;
    m31 += t.y ;
    m32 += t.z ;
    return *this ;
  }
  
  Matrix4f& translate( float x, float y, float z )
  {
    m30 += x ;
    m31 += y ;
    m32 += z ;
    return *this ;
  }  

  Matrix4f& scale( const Vector3f& t )
  {
    m00 *= t.x ;
    m11 *= t.y ;
    m22 *= t.z ;
    return *this ;
  }  

  Matrix4f& scale( float x, float y, float z )
  {
    m00 *= x ;
    m11 *= y ;
    m22 *= z ;
    return *this ;
  }  

  // m00  m10  m20  m30 
  // m01  m11  m21  m31
  // m02  m21  m22  m32
  // m03  m31  m32  m33
  
  // CONST
  // Because this is ambiguous, you COULD consider
  // renaming this to rotateAndTranslate, and
  // the other to rotateNoTranslate
  Vector3f operator*( const Vector3f& o ) const
  {
    // the 'w' component in o is implicitly 1,
    // you get a Vector3f back.
    return Vector3f(
      m00*o.x + m10*o.y + m20*o.z + m30,
      m01*o.x + m11*o.y + m21*o.z + m31,
      m02*o.x + m12*o.y + m22*o.z + m32
    ) ;
  }
  
  // For when you want to access only the rotation
  // components of this matrix and not translate.
  Vector3f rotNoTrans( const Vector3f& o ) const
  {
    return Vector3f(
      // NO TRANSLATION.
      m00*o.x + m10*o.y + m20*o.z,
      m01*o.x + m11*o.y + m21*o.z,
      m02*o.x + m12*o.y + m22*o.z
    ) ;
  }
  
  Vector4f operator*( const Vector4f& o ) const
  {
    return Vector4f(
      m00*o.x + m10*o.y + m20*o.z + m30*o.w,
      m01*o.x + m11*o.y + m21*o.z + m31*o.w,
      m02*o.x + m12*o.y + m22*o.z + m32*o.w,
      m03*o.x + m13*o.y + m23*o.z + m33*o.w
    ) ;
  }
  
  Matrix4f operator*( const Matrix4f& o ) const
  {
    Matrix4f m ;
    
    m.elts[0]  = elts[0] * o.elts[0]  + elts[4] * o.elts[1]  + elts[8] * o.elts[2]   + elts[12] * o.elts[3];
    m.elts[4]  = elts[0] * o.elts[4]  + elts[4] * o.elts[5]  + elts[8] * o.elts[6]   + elts[12] * o.elts[7];
    m.elts[8]  = elts[0] * o.elts[8]  + elts[4] * o.elts[9]  + elts[8] * o.elts[10]  + elts[12] * o.elts[11];
    m.elts[12] = elts[0] * o.elts[12] + elts[4] * o.elts[13] + elts[8] * o.elts[14]  + elts[12] * o.elts[15];
      
    m.elts[1]  = elts[1] * o.elts[0]  + elts[5] * o.elts[1]  + elts[9] * o.elts[2]   + elts[13] * o.elts[3];
    m.elts[5]  = elts[1] * o.elts[4]  + elts[5] * o.elts[5]  + elts[9] * o.elts[6]   + elts[13] * o.elts[7];
    m.elts[9]  = elts[1] * o.elts[8]  + elts[5] * o.elts[9]  + elts[9] * o.elts[10]  + elts[13] * o.elts[11];
    m.elts[13] = elts[1] * o.elts[12] + elts[5] * o.elts[13] + elts[9] * o.elts[14]  + elts[13] * o.elts[15];
      
    m.elts[2]  = elts[2] * o.elts[0]  + elts[6] * o.elts[1]  + elts[10] * o.elts[2]  + elts[14] * o.elts[3];
    m.elts[6]  = elts[2] * o.elts[4]  + elts[6] * o.elts[5]  + elts[10] * o.elts[6]  + elts[14] * o.elts[7];
    m.elts[10] = elts[2] * o.elts[8]  + elts[6] * o.elts[9]  + elts[10] * o.elts[10] + elts[14] * o.elts[11];
    m.elts[14] = elts[2] * o.elts[12] + elts[6] * o.elts[13] + elts[10] * o.elts[14] + elts[14] * o.elts[15];
      
    m.elts[3]  = elts[3] * o.elts[0]  + elts[7] * o.elts[1]  + elts[11] * o.elts[2]  + elts[15] * o.elts[3];
    m.elts[7]  = elts[3] * o.elts[4]  + elts[7] * o.elts[5]  + elts[11] * o.elts[6]  + elts[15] * o.elts[7];
    m.elts[11] = elts[3] * o.elts[8]  + elts[7] * o.elts[9]  + elts[11] * o.elts[10] + elts[15] * o.elts[11];
    m.elts[15] = elts[3] * o.elts[12] + elts[7] * o.elts[13] + elts[11] * o.elts[14] + elts[15] * o.elts[15];
    
    return m;
  }
  
  void println(const char* msg) const {
    puts( msg ) ;
    println() ;
  }
  
  void println() const {
    for( int i = 0 ; i < 4 ; i++ )
    {
      // Because printf works horizontally, we have to print out each ROW.
      for( int j = 0 ; j < 4 ; j++ )
        printf( "%8.3f ", elts[ getIndex( j, i ) ] ) ;
      puts("");
    }
    puts("");
  }
  
 
} ;






// A local axis
struct Axis
{
  Vector3f pos, forward, right, up ;
  
  Axis():pos(0,0,0),forward(0,0,-1),right(1,0,0),up(0,1,0)
  {
    
  }
  
  // This applies an offset to the position
  // BEFORE rotation.  You use it in a situation
  // where you want the matrices you ask for getTransformation*
  // to ALREADY HAVE BEEN OFFSET from the origin BEFORE
  // rotation (ie like a turret bay is offset from
  // the plane centroid by a certain vector)
  void offsetFromModelSpace( const Vector3f& offset )
  {
    pos += transBeforeRot( offset ) ;
  }

  void roll( float radians )
  {
    // rotates about FORWARD
    // Because of the turn-around in projection,
    // we have to rotate about -forward.
    // If we don't do this, then NDC will be
    // backwards.
    Matrix3f m = Matrix3f::rotation( forward, -radians ) ;
    
    right = m*right ;
    up = m*up ;
    renormalize() ;
  }

  void yaw( float radians )
  {
    // rotates about UP
    Matrix3f m = Matrix3f::rotation( up, radians ) ;
    
    right = m*right ;
    forward = m*forward ;
    renormalize() ;
  }

  void pitch( float radians )
  {
    // rotates about RIGHT.
    Matrix3f m = Matrix3f::rotation( right, radians ) ;
    
    up = m * up ; // note how up*m is undefined (we want it that way -- column major matrices, vectors are column vectors)
    forward = m * forward ;
    renormalize() ;
  }
  
  // Sets the new forward vector
  void setForward( const Vector3f& newFwd )
  {
    // angle to rotate
    float angle = forward.angleWith( newFwd ) ;
    
    // The angle throttle actually makes the missile WAY
    // more realistic (because it can't instantly turn around),
    // you don't have to program ANYTHING, (initially I thought
    // I'd have to program an interpolation using quaternions.
    // Turns out all you have to do is CLAMP the angle and
    // recompute the rotation matrix each frame, and that will
    // effectively "interpolate" for you.)
    
    // Too big to jump in one crossing
    if( angle > .25 ) angle = .25 ;
    
    // axis to rotate about
    Vector3f axis = newFwd.cross( forward ) ; // I had this backwards but this way seems to work.
    // otherwise the missile faced backwards.
    
    Matrix3f rot = Matrix3f::rotation(axis, angle) ;
    
    right = rot * right ;
    up = rot* up ;
    forward = rot *forward ;
    
    
    // the renormalization seems to "dampen" steering effects
    renormalize();
  }

  // abruptly sets the new forward vector
  void setForwardSHOCK( const Vector3f& newFwd )
  {
    // angle to rotate
    float angle = forward.angleWith( newFwd ) ;
    
    // The angle throttle actually makes the missile WAY
    // more realistic (because it can't instantly turn around),
    // you don't have to program ANYTHING, (initially I thought
    // I'd have to program an interpolation using quaternions.
    // Turns out all you have to do is CLAMP the angle and
    // recompute the rotation matrix each frame, and that will
    // effectively "interpolate" for you.)
    //if( angle > 1e-1 ) angle = 1e-1 ;  // Too big to jump in one crossing
    
    // axis to rotate about
    Vector3f axis = newFwd.cross( forward ) ; // I had this backwards but this way seems to work.
    // otherwise the missile faced backwards.
    
    Matrix3f rot = Matrix3f::rotation(axis, angle) ;
    
    right = rot * right ;
    up = rot* up ;
    forward = rot *forward ;
    
    // the renormalization seems to "dampen" steering effects
    renormalize();
  }

  void renormalize()
  {
    // leave forward the same direction, only be sure it's normalized
    forward.normalize() ;
    
    // recompute right first,
    // based on forward and up,
    right = forward.cross( up ).normalize() ;
    
    // RIGHT is now guaranteed orthogonal to 
    // both FORWARD and UP 
    // recompute up, now, as being normal to forward and right
    up = right.cross(forward) ;
    // we won't re-normalize up, because 
  }

  // world transform
  // change of basis:  newRight, newUp, newFwd,
  // transpose because the viewing matrix is inverse transform.
  Matrix3f getViewingMatrix3f() const
  {
    return Matrix3f(
      right.x,  up.x, -forward.x, // column 1
      right.y,  up.y, -forward.y, // column 2
      right.z,  up.z, -forward.z  // column 3
    ) ;
  }

  // This is how you VIEW from this Axis' point of view.
  // It is the COMPLETE INVERSE of the transformation matrix
  // the object uses, Logical INVERSE TRANSLATE (the entire world)
  // THEN INVERSE ROTATE (the entire world)
  Matrix4f getViewingMatrix4f() const
  {
    // COLUMN MAJOR: The new basis vectors in terms
    // of the original basis are the columns of M.
    
    // Finding t here is premultiplying
    Vector3f t = transBeforeRot( -pos ) ;
    
    return Matrix4f(
      right.x,  up.x, -forward.x, 0, // column 1
      right.y,  up.y, -forward.y, 0, // column 2
      right.z,  up.z, -forward.z, 0, // column 3
          t.x,   t.y,        t.z, 1 ) ;  // column 4
    
    // negate the eye, because what's actually happening is we don't
    // move the eye, we move the world instead.
    // To move the eye to (0,0,5), we
    // move the world by (0,0,-5).
  }
  
  // Logical INV TRANS THEN INV ROT
  // which is INVROT*INVTRANS for column major.
  // Allows you to work in an additional offset, for turret FROM
  // ship, or pilot eye position FROM ship center
  Matrix4f getViewingMatrix4f( const Vector3f& additionalOffset ) const
  {
    // COLUMN MAJOR: The new basis vectors in terms
    // of the original basis are the columns of M.
    
    // This represents a translate in modelspace BEFORE the rotate.
    Vector3f t = transBeforeRot( -pos ) ;
    
    t -= additionalOffset ;
    
    return Matrix4f(
      right.x,  up.x, -forward.x, 0, // column 1
      right.y,  up.y, -forward.y, 0, // column 2
      right.z,  up.z, -forward.z, 0, // column 3
          t.x,   t.y,        t.z, 1 ) ;  // column 4
  }
  
  
  // As a rule these ALWAYS should be normalized,
  // this will NEVER scale, so you can safely use it
  // for normal transformation as well.
  Matrix3f getTransformationMatrix3f() const
  {
    return Matrix3f(
         right.x,     right.y,    right.z, // column 1
            up.x,        up.y,       up.z, // column 2
      -forward.x,  -forward.y, -forward.z  // column 3
    ) ;
  }

  // This ORIENTS a point to line up and face
  // This places the object with this Axis in the world
  // Logical ROTATE model around origin, THEN TRANSLATE OUT to pos in world
  // column major TRANS*ROT is SIMPLE APPEND
  Matrix4f getTransformationMatrix4f() const
  {
    return Matrix4f(
         right.x,     right.y,    right.z, 0, // column 1
            up.x,        up.y,       up.z, 0, // column 2
      -forward.x,  -forward.y, -forward.z, 0, // column 3
           pos.x,       pos.y,      pos.z, 1 ) ;  // column 4
  }
  
  // logical
  // (trans additionalOffset) (rotate) (trans pos into world)
  // because column major,
  // (trans pos into world)*(rotate)*(trans additionalOffset)
  // The rot*trans requires the messy dot producting.
  Matrix4f getTransformationMatrix4f( const Vector3f& additionalOffset ) const
  {
    // (rotate)*(trans additional offset)
    Vector3f t = transBeforeRot( additionalOffset ) ;
    
    // (trans pos into world)*(rotate) (ROT THEN TRANS is easy APPAND) 
    t += pos ;
    return Matrix4f(
         right.x,     right.y,    right.z, 0, // column 1
            up.x,        up.y,       up.z, 0, // column 2
      -forward.x,  -forward.y, -forward.z, 0, // column 3
             t.x,         t.y,        t.z, 1 ) ;  // column 4
  }

  // ROT THEN TRANS IS SIMPLE APPAND
  // BUT TRANS THAN ROT AND YOU'RE IN FOR A FOT
  // You use this function when you need to translate
  // BEFORE rotation by the rotation matrix described by this axis.
  // This is also just VIEWINGMATRIX*v
  Vector3f transBeforeRot( const Vector3f& v ) const
  {
    return Vector3f(
      v.dot( right ),
      v.dot( up ),
      v.dot( -forward )
    ) ;
  }
  
  // Instead of doing getTransformationMatrix3f() * v,
  // you call this method.
  // FORWARDMATRIX * v
  Vector3f forwardTransform( const Vector3f& v ) const 
  {
    return Vector3f(
      right.x*v.x + up.x*v.y - forward.x*v.z,
      right.y*v.x + up.y*v.y - forward.y*v.z,
      right.z*v.x + up.z*v.y - forward.z*v.z
    ) ;
  }
  
  //Axis operator+( const Axis& o ) const
  //{
  //  allows you to concatenate the transformations performed by two axes..
  //  actually don't need to do this
  //}
} ;




// The vertex types are:
// VertexPcNCT, VertexPcCT, VertexPCT, VertexPCTT
struct VertexPNC
{
  Vector3f pos ;
  Vector3f normal ;
  Vector4f color ;
  
  VertexPNC(){}
    
  VertexPNC( const Vector3f& iPos, const Vector3f& iNormal, const Vector4f& iColor ) :
    pos( iPos ), normal( iNormal ), color( iColor )
  {
    
  }
} ;

// Actually PcNCT (position, centroid, normal, color, texture),
// but 'c' is implied as part of "position"
struct VertexPcNC
{
  Vector3f pos ;
  Vector4f centroid ;
  Vector3f normal ;
  Vector4f color ;
    
  VertexPcNC(){}
  
  VertexPcNC( const Vector3f& iPos, const Vector4f& iCentroid, const Vector3f& iNormal, const Vector4f& iColor ) :
              pos(iPos),centroid(iCentroid),normal(iNormal),color(iColor)
  {
  }
              
} ;
// Actually PcNCT (position, centroid, normal, color, texture),
// but 'c' is implied as part of "position"
struct VertexPcNCT
{
  Vector3f pos ;
  Vector4f centroid ;//the centroid of the owner.
  // although redundant throughout a geometry,
  // it's what makes wrapping clean on the GPU (the
  // centroid is wrapped.)
  // Without this member, the vertex has no sense
  // of the body it's attached to.
  Vector3f normal ;
  Vector4f color ;
  Vector4f tex ;  // the last 2 texcoords are used for z="percent damage (0 to 1)" and w="NOTHING"
    
  VertexPcNCT(){}
  
  VertexPcNCT( const Vector3f& iPos, const Vector3f& iNormal,
               const Vector4f& iColor, const Vector2f& iTex ) :
               pos(iPos),normal(iNormal),tex(iTex),color(iColor)
  {
  }
              
} ;

struct VertexPcCT
{
  Vector3f pos ;
  Vector4f centroid ; //required for wrapping
  Vector4f color ;
  Vector4f tex ;  // the last 2 texcoords are used for z="percent damage (0 to 1)" and w="NOTHING"
    
  VertexPcCT(){}
  
  VertexPcCT( const Vector3f& iPos, const Vector4f& iColor, const Vector4f& iTex ) :
             pos(iPos),color(iColor),tex(iTex)
  {
  }
  
  // Assumes white
  VertexPcCT( const Vector3f& iPos, const Vector4f& iTex ) :
             pos(iPos),color(Vector4f(1.0f,1.0f,1.0f,1.0f)),tex(iTex)
  {
  }
  
  VertexPcCT( const VertexPcNCT& o ) :
    pos(o.pos),tex(o.tex.x,o.tex.y,0,0),color(o.color)
  { }
              
} ;


// FOR TEXT and stuff that doesn't use the centroid wrap transformation
struct VertexPCT // TEX2f,
{
  Vector3f pos ;
  Vector4f color ;
  Vector2f tex ;
  
  VertexPCT(){}
  
  VertexPCT( const Vector3f& iPos, const Vector4f& iColor, const Vector2f& iTex ) :
             pos(iPos),color(iColor),tex(iTex)
  {
  }
  
  // Assumes white
  VertexPCT( const Vector3f& iPos, const Vector2f& iTex ) :
             pos(iPos),color(Vector4f(1.0f,1.0f,1.0f,1.0f)),tex(iTex)
  {
  }
  
  VertexPCT( const VertexPcNCT& o ) :
    pos(o.pos),tex(o.tex.x,o.tex.y),color(o.color)
  { }
              
} ;

// For rendering quads with BOTH text from a texture atlas AND
// some other texture.
struct VertexPCTT
{
  Vector3f pos ;
  Vector4f color ;
  Vector2f tex0, tex1 ;
  
  VertexPCTT(){}
  
  VertexPCTT( const Vector3f& iPos, const Vector4f& iColor, const Vector2f& iTex0, const Vector2f& iTex1 ) :
             pos(iPos),color(iColor),tex0(iTex0),tex1(iTex1)
  {
  }
} ;


struct VertexPCCT
{
  Vector3f pos ;
  Vector4f color ;
  Vector4f colorAdd ; // each vertex has a color To ADD to its "real" base color.
  // because of s/w instancing, this ISN'T easy to do as a UNIFORM parameter (want to draw 600 x 6 element arrays
  // and update a uniform variable in between each draw??)
  // Instead we draw the 3600 elements in rapid succession, colorAdd is 0 if there is no highlight.
  Vector2f tex ;
} ;

struct VertexPC
{
  Vector3f pos ;
  Vector4f color ;
  
  VertexPC(){}
    
  VertexPC( const Vector3f& iPos, const Vector4f& iColor ) :
    pos( iPos ), color( iColor )
  {
    
  }
} ;


struct VertexPcC
{
  Vector3f pos ;
  Vector4f centroid ; //required for wrapping
  Vector4f color ;
  
  VertexPcC(){}
    
  VertexPcC( const Vector3f& iPos, const Vector4f& iColor ) :
    pos( iPos ), centroid(pos), color( iColor )
  {
    
  }
  
  VertexPcC( const VertexPcNCT& o ) :
    pos(o.pos),centroid(o.centroid),color(o.color)
  { }
} ;


/*
struct VertexPSiC
{
  Vector3f pos ;
  float pointSize ;
  Vector4f color ;
  
  VertexPSiC(){}
  VertexPSiC( const Vector3f& iPos, float iPointSize, const Vector4f& iColor ) :
    pos( iPos ), pointSize( iPointSize ), color( iColor )
  {
    
  }

  VertexPSiC( const VertexPcNCT& o ) :
    pos(o.pos),color(o.color)
  { }  
} ;

struct VertexPNC
{
  Vector3f pos ;
  Vector4f centroid ; //required for wrapping
  Vector3f normal ;
  Vector4f color ;
  
  VertexPNC(){}
  VertexPNC( const Vector3f& iPos, const Vector3f& iNormal, const Vector4f& iColor ) :
    pos( iPos ), normal( iNormal ), color( iColor )
  {
    
  }
  
  VertexPNC( const VertexPcNCT& o ) :
    pos(o.pos),normal(o.normal),color(o.color)
  { }
} ;
*/



template <typename> struct hasNormalTrait : std::false_type { };
template <> struct hasNormalTrait<VertexPcNCT> : std::true_type { };

template <typename> struct hasCentroidTrait : std::false_type { };
template <> struct hasCentroidTrait<VertexPcNCT> : std::true_type { };
template <> struct hasCentroidTrait<VertexPcCT> : std::true_type { };
template <> struct hasCentroidTrait<VertexPcC> : std::true_type { };



// A RECTF is supposed to be MEASURED IN POINTS.
// You can convert a PTS x,y,w,h value to
// CANONICAL COORDS based on the SIZE OF THE VIEWPORT
// YOU INTEND TO DRAW INTO.  Canonical coords are converted
// by OpenGL into the PX space.
// PTS => CANONICAL COORDS => PX
//
// Touch events are in PTS,
struct RectF
{
  // x,y is BOTTOM LEFT corner
  float x,y,w,h ;
  static float DW, DH ; 
  static RectF window ;
  ///static const RectF& Window() { return window ; } //protects Window from modification
  RectF():x(0),y(0),w(0),h(0){}
  RectF( float ixPts, float iyPts, float iwPts, float ihPts ):x(ixPts),y(iyPts),w(iwPts),h(ihPts){}
  RectF( const Vector2f& pos, const Vector2f& size ):
    x(pos.x),y(pos.y),w(size.x),h(size.y){}
  
  Vector2f randomPoint() const {
    return Vector2f( randFloat( x, x+w ), randFloat( y, y+w ) ) ;
  }
  RectF randomRect( float width, float height ) const {
    return RectF( randFloat( x, x+w-width ), randFloat( y, y+h-height ), width, height ) ;
  }
  
  // Allows you to position your rectangle from the TL corner
  // instead of usual BL
  static RectF FromTopLeft( float ixPts, float iyPts, float iwPts, float ihPts ){
    return RectF( ixPts, iyPts-ihPts, iwPts, ihPts ) ;
  }
  
  inline bool hit( const Vector2f& v ) const {
    return ( v.x >= left() && v.x <= right() && v.y >= bottom() && v.y <= top() ) ;
  }
  inline bool hitX( float ix ) const {
    return ( ix >= left() && ix <= right() ) ;
  }
  inline bool hitY( float iy ) const {
    return ( iy >= bottom() && iy <= top() ) ;
  }
  
  inline bool hit( const RectF& o ) const {
    // These are the miss situations
    bool hitted = !( right() < o.left()   || // definite miss
              left() > o.right()   ||
              bottom() > o.top()   ||
              top() < o.bottom() ) ;
              
    //if( hitted )      puts( "HIT" ) ;
    //else       puts( "MISS" ) ;
    return hitted ;
  }
  
  // HOW MUCH does THIS escape O by?
  inline Vector2f overflows( const RectF& container ) const {
    // overflows R, T, L, B ( x axis, y axis, -x, -y )
    Vector4f disp;
    // ______________
    // |     |______|
    // |________|
    //
    disp.x = container.right() - right() ; // NEED MOVE -ve x
    disp.y = container.top() - top() ;
    disp.z = container.left() - left() ;
    disp.w = container.bottom() - bottom() ;
    
    //disp.println( "DISP" ) ;
    // Clamp off displacements that are not needed (bounded)
    if( disp.x > 0 ) disp.x=0;    
    if( disp.y > 0 ) disp.y=0;
    if( disp.z < 0 ) disp.z=0;
    if( disp.w < 0 ) disp.w=0;
    //disp.println( "DISP c" ) ;
        
    // Addup the required x displacements and y displacements
    // to get the total suggested displacement ot keep THIS within bounds of O.
    // if THIS is bigger than O the method should fail or its behavior is "undefined"
    return Vector2f( disp.x + disp.z, disp.y + disp.w ) ;
  }
  
  // WHAT IS THE OUTSIDE WALLS DISTANCE BETWEEN THE 2?
  inline Vector2f outsideWallDistance( const RectF& o ) const {
    // ________   ________
    // | this |   |   o  |
    // |______|   |      |
    //            |______|
    // 
    
    // If the walls OVERLAP, the distance is 0.
    Vector2f dist ;
    if( isLeftOf(o) )
      dist.x = o.left() - right() ; // +++ - I have to go +x to reach o
    else if( isRightOf(o) )
      dist.x = o.right() - left() ; // --- I have to go -x to reach o
      
    if( isBelowOf( o ) )
      dist.y = o.bottom() - top() ; // +++ I have to go +y to reach o
    else if( isOnTopOf( o ) )
      dist.y = o.top() - bottom() ; // --- I have to go -y to reach o
      
    return dist ;
  }
  
  inline bool isLeftOf( const RectF& o ) const {
    // ________   ________
    // | this |   |   o  |
    // |______|   |      |
    //            |______|
    // 
    return right() < o.left() ;
  }
  
  inline bool isRightOf( const RectF& o ) const {
    // my left wall is bigger than your right wall means i'm right of you
    // ________   ________
    // | o    |   | this |
    // |______|   |      |
    //            |______|
    // 
    return left() > o.right() ;
  }
  
  inline bool isBelowOf( const RectF& o ) const {
    // my top is less than your bottom
    return top() < o.bottom() ;
  }
  
  inline bool isOnTopOf( const RectF& o ) const {
    // my bottom wall is bigger than your top wall
    return bottom() > o.top() ;
  }
  
  // Default considers origin BOTTOM LEFT.
  inline float left() const { return x ; }
  inline float right() const { return x+w ; }
  
  //!! The GL coordinate system is ALWAYS origin in lowerleft.
  // so if you want to work in upside down coordinates for textboxes,
  // you still have to invert it back at EOD
  inline float bottom() const { return y ; }
  inline float top() const { return y + h ; } 
  
  inline float midX() const { return x + w/2 ; }
  inline float midY() const { return y + h/2 ; }
  
  inline bool isCanonical() const { 
    return -1.f <= x && x <= 1.f && 
           -1.f <= y && y <= 1.f && 
              0 <= w && w <= 2.f && 
              0 <= h && h <= 2.f ;
  }
  
  // Consider this as a vec4
  inline const Vector4f* vec4c() const {
    return (const Vector4f*)this;
  }
  inline Vector4f* vec4() const {
    return (Vector4f*)this;
  }
  // Considers origin @ bottom left.
  inline Vector2f bottomLeft() const {
    return Vector2f( left(), bottom() ) ;
  }
  inline Vector2f bottomRight() const {
    return Vector2f( right(), bottom() ) ;
  }
  inline Vector2f topLeft() const {
    return Vector2f( left(), top() ) ;
  }
  inline Vector2f topRight() const {
    return Vector2f( right(), top() ) ;
  }
  
  // BL corner
  inline Vector2f pos() const {
    return Vector2f( x,y ) ;
  }
  inline Vector2f& xy() {
    return (Vector2f&)x ;
  }
  inline Vector2f size() const {
    return Vector2f( w,h ) ;
  }
  inline Vector2f& wh() {
    return (Vector2f&)w ;
  }
  inline Vector2f centroid() const {
    return Vector2f( midX(), midY() ) ;
  }
  inline Vector2f offsetToCenter( const RectF& inner ) const {
    return centroid() - inner.centroid() ;
  }
  
  // Centers SOME OTHER RECT inside THIS
  inline RectF center( const RectF& inner ) const {
    Vector2f diff = offsetToCenter( inner ) ;
    return RectF( inner.x+diff.x, inner.y+diff.y, inner.w, inner.h ) ;
  }
  inline RectF center( const RectF& inner, const Vector2f& offsets ) const {
    Vector2f diff = offsetToCenter( inner ) ;
    return RectF( inner.x+diff.x+offsets.x, inner.y+diff.y+offsets.y, inner.w, inner.h ) ;
  }
  inline RectF centered( const Vector2f& sizes ) const
  {
    return RectF( x+w/2 - sizes.x/2, y+h/2 - sizes.y/2, sizes.x, sizes.y ) ;
  }
  
  // Sets the rect sent in the bottom left of (this) the parent rect
  inline RectF setInBottomLeft( const RectF& inner, const Vector2f& margin ) const {
    return RectF( left() + margin.x, bottom() + margin.y, inner.w, inner.h ) ;
  }
  inline RectF setInBottomRight( const RectF& inner, const Vector2f& margin ) const {
    return RectF( right() - inner.w - margin.x, bottom() + margin.y, inner.w, inner.h ) ;
  }
  inline RectF setInCenterRight( const RectF& inner, float marginx ) const {
    return RectF( right() - inner.w - marginx, (bottom() + top() - inner.h)/2, inner.w, inner.h ) ;
  }
  inline RectF setInTopLeft( const RectF& inner, const Vector2f& margin ) const {
    return RectF( left() + margin.x, top() - inner.h - margin.y, inner.w, inner.h ) ;
  }
  inline RectF setInTopCenter( const RectF& inner, float margintop ) const {
    return RectF( (left()+right())/2 - inner.w/2, top() - inner.h - margintop, inner.w, inner.h ) ;
  }
  inline RectF setInTopRight( const RectF& inner, const Vector2f& margin ) const {
    return RectF( right() - inner.w - margin.x, top() - inner.h - margin.y, inner.w, inner.h ) ;
  }
  inline RectF setInCenterLeft( const RectF& inner, float marginx ) const {
    return RectF( left() + marginx, (bottom() + top() - inner.h)/2, inner.w, inner.h ) ;
  }
    
  inline RectF operator+( const Vector2f& offset ) {
    return RectF( x+offset.x, y+offset.y, w,h ) ;
  }
  
  inline RectF& setTop( float t ) {
    //if( upsideDown )  y = t ;  else
    h = t-y;
    return *this ;
  }
  inline RectF& setBottom( float b ) {
    //if( upsideDown )  h = y-b ; else
    y=b;
    return *this ;
  }
  inline RectF& setLeft( float l ) {
    x=l;
    return *this ;
  }
  inline RectF& setRight( float r ) {
    w = r-x;
    return *this ;
  }
  inline RectF& setCenter( const Vector2f& v ) {
    Vector2f halfSize = size()/2;
    // when you set the position of the CENTER,
    setLeft( v.x - halfSize.x ) ;
    setRight( v.x + halfSize.x ) ;
    setBottom( v.y - halfSize.y ) ;
    setTop( v.y + halfSize.y ) ;
    return *this ;
  }
  
  inline RectF& expandRight( float r ) {
    // take width and expand it so x+w is AT LEAST r.
    w = max( w, r-x ) ;
    return *this ;
  }
  inline RectF& expandLeft( float l ) {
    x = min( x, l ) ;
    return *this ;
  }
  inline RectF& expandTop( float t ) {
    //if( upsideDown )  y = min( y, t ) ;  else
    h = max( h, t-y ) ;
    return *this ;
  }
  inline RectF& expandBottom( float b ) {
    //if( upsideDown )  h = max( b-y, h ); // moves the HEIGHT, NOT the origin
    //else
    y = min( y, b ) ;
    return *this ;
  }
  
  inline void setMinMax( const Vector2f& min, const Vector2f& max )
  {
    x=min.x;  y=min.y;
    w=max.x-min.x;
    h=max.y-min.y;
  }
  
  // Expands THIS to contain O completely.
  inline RectF& expand( const RectF& o ) {
    ///*
    // THis is more complicated than just doing min/max
    // because max depends on min (because of width parameter).
    // You have to find the maximal span 
    float maxRight = max( right(), o.right() ) ;
    float minLeft = min( x, o.x ) ;
    w = maxRight - minLeft ;
    x = min( x, o.x ) ;
    
    // what if maxtop is MY top but min is YOUR BOT.
    // We have to x-check this.
    float maxTop = max( top(), o.top() ) ;
    float minBot = min( y, o.y ) ;
    h = maxTop - minBot ;
    y = min( y, o.y ) ;
    //*/
    
    /*
    // another way to do it:
    // PUsh min by min of other box.
    // Min may be no larger than bottomLeft
    Vector2f min = pos() ;
    min.clampAbove( o.bottomLeft() ) ;
    
    Vector2f max = pos() + size() ;
    // Max may be no smaller than topRight
    max.clampBelow( o.topRight() ) ;
    setMinMax( min,max ) ;
    */
    return *this ;
  }
  
  inline RectF& applyDisplacement( const Vector2f& v ) {
    return (*this += v) ;
  }
  
  inline RectF& applyDisplacement( float ix, float iy ) {
    x+=ix ;   y+=iy ;  return *this ;
  }
  
  // For offsetting a rectangle by a vector.
  inline RectF& operator+=( const Vector2f& v ) {
    x+=v.x ;  y+=v.y ; return *this ;
  }
  
  inline RectF& pad( float amount ) {
    float half=amount/2.f ;
    x -= half ;  w += amount ;
    y -= half ;  h += amount ;
    return *this ;
  }
  inline RectF& pad( float amountX, float amountY ) {
    x -= amountX/2.f ;  w += amountX ;
    y -= amountY/2.f ;  h += amountY ;
    return *this ;
  }
  inline RectF& pad( Vector2f amount ) {
    x -= amount.x / 2.f ;  w += amount.x ;
    y -= amount.y / 2.f ;  h += amount.y ;
    return *this ;
  }
  
  // bounds %= 2 makes bounds 2% of itself
  inline RectF& operator%=( const Vector2f& percentage ) {
    Vector2f amt = size()*(1.f-percentage/100.f) ;
    return pad( -amt ) ;
  }
  
  inline RectF paddedCopy( float amount ) const {
    RectF padded = *this;
    return padded.pad( amount ) ;
  }
  inline RectF paddedCopy( float amountX, float amountY ) const {
    RectF padded = *this;
    return padded.pad( amountX, amountY ) ;
  }
  inline RectF paddedCopy( Vector2f amount ) const {
    RectF padded = *this;
    return padded.pad( amount ) ;
  }  
  
  // You get a new rectangle adjusted as you specify
  inline RectF copy( float dx, float dy, float dw, float dh ) const {
    return RectF( x+dx, y+dy, w+dw, h+dh ) ;
  }
  
  inline RectF sub( float dx, float dy, float newWidth, float newHeight ) const {
    return RectF( x+dx, y+dy, newWidth, newHeight ) ;
  }
  
  // mx is margin x
  inline RectF subTopRight( float mx, float my, float newWidth, float newHeight ) const {
    return RectF( right()-(mx+newWidth), top()-(my+newHeight), newWidth, newHeight ) ;
  }
  // mx is margin x
  inline RectF subTopLeft( float mx, float my, float newWidth, float newHeight ) const {
    return RectF( left()+mx, top()-(my+newHeight), newWidth, newHeight ) ;
  }
  // Quad split, 
  void subQuad( RectF subrectsTlTrBrBl[4] ) const
  {
    float halfW=w/2.f, halfH=h/2.f;
    subrectsTlTrBrBl[0] = sub(     0, halfH, halfW, halfH ) ; //tl
    subrectsTlTrBrBl[1] = sub( halfW, halfH, halfW, halfH ) ; //tr
    subrectsTlTrBrBl[2] = sub( halfW,     0, halfW, halfH ) ; //br
    subrectsTlTrBrBl[3] = sub(     0,     0, halfW, halfH ) ; //bl
  }
  void println( const char* msg ) const {
    //printf( "%s: x=%.2f y=%.2f w=%.2f h=%.2f\n", msg,x,y,w,h ) ;
    printf( "%s: l=%.2f r=%.2f b=%.2f t=%.2f (w=%.2f, h=%.2f)\n", msg,left(),right(),bottom(),top(),w,h ) ;
  }
  void println() const {
    this->println( "" ) ;
  }
} ;


// bounds % 20 or 20%bounds takes 20% of bounds
inline RectF operator%( RectF rect, const Vector2f& percentage ) {
  Vector2f amt = rect.size()*(1.f-percentage/100.f) ;
  return rect.pad( -amt ) ;
}

inline RectF operator%( const Vector2f& percentage, RectF rect ) {
  Vector2f amt = rect.size()*(1.f-percentage/100.f) ;
  return rect.pad( -amt ) ;
}






enum AnimationCycles
{
  NoCycles,
  YesCycles
} ;
enum AnimationRocks
{
  NoRocks,
  YesRocks
} ;

//1D bezier spline.
struct Bezier
{
  vector<double> points ;
  double startT, responseLength ;
  bool fwd ; // currenlty in fwd or backward transition state
  bool cycles ; // cycle when finished or CLAMP.
  bool rocks ;  // when cycling, ROCKS back and forth, OR just cycles fwd,fwd,fwd
  
  void defaults(){
    startT=(0.0),responseLength=(1.0);
    fwd=1;
    // BY DEFAULT it will cycle and rock.
    cycles=1;
    rocks=1;
  }
  
  // Can't construct a bezier without these 4 pts man.
  Bezier( double A, double B, double C, double D )
  {
    defaults();
    points.push_back( A ) ;
    points.push_back( B ) ;
    points.push_back( C ) ;
    points.push_back( D ) ;
  }

  // For constructing just a bezier curve, but you cannot use it until some
  // interaction takes place (so you don't know the start time and end time yet).
  Bezier( const Vector4f& coeff ) : Bezier( coeff.x,coeff.y,coeff.z,coeff.w ) { }
  
  Bezier( double A, double B, double C, double D, double iStartT, double iResponseLen ) :
    Bezier( A,B,C,D ) {
    startT = iStartT ;
    responseLength = iResponseLen ;
  }
  
  Bezier( double A, double B, double C, double D, double iStartT, double iResponseLen, AnimationCycles iCycles, AnimationRocks iRocks ) :
    Bezier( A,B,C,D ) {
    startT = iStartT ;
    responseLength = iResponseLen ;
    cycles = iCycles ;
    rocks  = iRocks ;
  }
  
  inline bool isEnded( double t ) {
    return t - startT >= responseLength ;
  }
private:
  // straight eval the bez spline for a t.
  inline float eval( double t )
  {
    double r = 0 ;
    int n = (int)points.size()-1;
    for( int i = 0 ; i < points.size() ; i++ )
      r += points[i] * binomial( n, i )   *   pow( t, i )   *   pow( 1-t, n-i ) ;
    return r ;
  }

public:
  // t will be converted to a value between 0 and 1.
  // you will get a result between 0 and 1.
  inline float operator()( double t )
  {
    // change t so that startT is accounted for as t=0
    t-=startT;
    
    if( t > responseLength )
    {
      // we have finished a cycle.
      if( ! cycles )  return points[ points.size()-1 ]; // clamp at last pt value.
      
      // Now we can either modify startT
      startT += responseLength ;
      // and take t back into range (just this time)
      t -= responseLength ;
      // and flip fwd, if it rocks between fwd/backwd
      if( rocks )
        fwd = !fwd ;
      
      // OR we can fmodf t so that it is between [0, responseLength)
      //t = fmodf( t, responseLength ) ;
      // and increment some kind of period cycle counter so that we don't flip FWD every time
      // OR use even/odd results of division,   t/responseLength is even on FWD cycles, t/responseLength is ODD on backcycles.
      //fwd = ((int)(t/responseLength))%2 ;
    }
    
    // Now make t periodic in the response length
    
    t/=responseLength; // normalizes t so that t is between 0 and 1.
    
    //return eval( t ) ;
    return fwd?eval( t ):eval( 1.-t ) ;
  }
  
  // [] walks backwards
  //inline float operator[]( double t ) {
  //  t-=startT;
  //  t/=responseLength; // t is now between 0 and 1
  //  return eval( 1.-t ) ; // t now between 1 and 0.
  //}

} ;



// A Transition is actually just a type of Bezier wrapper.
// Provides
template <typename T>
struct BezierTransition : public Bezier
{
  // Could also accept a generalized function
  //function< T (double t) > transitionFunc ;
  
  //Bezier bez ; // can use this instead of generalized func
  T startVal, endVal, dir, lastVal ;
  
  BezierTransition( double bezA, double bezB, double bezC, double bezD ) :
    Bezier( bezA, bezB, bezC, bezD ) { }
  
  BezierTransition( const Vector4f& coeff ) :
    Bezier( coeff.x,coeff.y,coeff.z,coeff.w ) { }
  
  BezierTransition( const Vector4f& coeff,
                    double iStartT, double iResponseLength, const T& iStartVal, const T& iEndVal,
                    AnimationCycles iCycles, AnimationRocks iRocks ) :
    Bezier( coeff.x,coeff.y,coeff.z,coeff.w, iStartT, iResponseLength, iCycles, iRocks )
  {
    reset( startT, iResponseLength, iStartVal, iEndVal ) ;
  }
  
  void reset( double newStartTime, double newResponseLen, const T& iStartValue, const T& iEndValue ){
    startT = newStartTime ;
    responseLength = newResponseLen ;
    
    startVal = iStartValue ;
    endVal = iEndValue ;
    
    dir = endVal - startVal ;
    lastVal = (*this)(startT); // INITIALIZE THIS. Otherwise lastVal has a random value!
  }
  
  // I give you a NET DISPLACEMENT to apply, relative to some starting pos.
  // Since these are used in the GPU, 
  // DOES NOT update lastVal, b/c/ well, I need lastVal to be intact in d.
  T operator()( double t ) {
    return startVal + dir*(Bezier::operator()( t )) ;
  }
  
  // Gets you the DISPLACEMENT since the last value.
  T d( double t ) {
    T newVal = (*this)( t ) ;
    T diff = newVal - lastVal ;
    lastVal = newVal ;
    return diff ;
  }
} ;

// This is a superglobal variable for passing a single Vector3f
// between 2 files that both include Vectorf.h, but do not have any other
// common includes.  Used for drawing debug lines andwhatnot
extern Vector3f debugPASS1 ;

#endif

