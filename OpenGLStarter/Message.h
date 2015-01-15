#ifndef MESSAGE_H
#define MESSAGE_H

static float defLifeTime=2.f;
static Vector4f defColor(1,1,1,1) ;
static float defMarginX=20.f;

struct Message
{
  Vector2f pos;
  string msg;
  Vector4f color;
  float lifeTime ;
  bool manPos ;
  
  Message( const string& imsg, const Vector2f& iPos, const Vector4f& icolor, float ilifeTime )
  {
    msg=imsg;
    pos=iPos;
    color=icolor;
    lifeTime=ilifeTime;
    manPos=0;
  }
  void debugprint() const
  {
    printf( "Msg `%s` pos=(%f %f), color=(%f %f %f %f), lifeTime=%f\n", msg.c_str(), pos.x,pos.y,
      color.x,color.y,color.z,color.w, lifeTime ) ;
  }
  float decay( float dt )
  {
    lifeTime -= dt ;
    color.w = lifeTime ; // fade out
    return lifeTime ;
  }
  void print() const
  {
    glColor4fv( &color.x ) ; // must be before the other one
    glRasterPos2fv( &pos.x ) ;

    const char * p = msg.c_str() ;
    do glutBitmapCharacter( GLUT_BITMAP_HELVETICA_18, *p ); while( *(++p) ) ;
  }
  
} ;
map<string,Message*> msgs;

void msg( const string& imsg, const Vector2f& iPos, const Vector4f& icolor, float ilifeTime ){
  map<string,Message*>::iterator iter = msgs.find( imsg ) ;
  if( iter == msgs.end() ) // not found
    msgs[ imsg ] = new Message( imsg, iPos, icolor, ilifeTime ) ;
  else{
    iter->second->msg = imsg ;
    iter->second->color=icolor;
    iter->second->lifeTime=ilifeTime;
  }
}


float getLowestMsg(){
  float ly=h; //margin for height of 18
  for( const pair<string,Message*> &p : msgs )
    if( !p.second->manPos && p.second->pos.y < ly )
      ly=p.second->pos.y;
  
  return ly - 18.f ;
}
void msg( const string& imsg, const Vector4f& icolor, float ilifeTime ) {
  msg( imsg, Vector2f( defMarginX, getLowestMsg() ), icolor, ilifeTime ) ;
}
void msg( const string& imsg, float ilifeTime ) {
  msg( imsg, Vector2f( defMarginX, getLowestMsg() ), defColor, ilifeTime ) ;
}
void msg( const string& imsg, const Vector4f& icolor ) {
  msg( imsg, Vector2f( defMarginX, getLowestMsg() ), icolor, defLifeTime ) ;
}
void msg( const string& imsg ) {
  msg( imsg, Vector2f( defMarginX, getLowestMsg() ), defColor, defLifeTime ) ;
}
void msg( const string& mapId, const string& imsg )
{
  map<string,Message*>::iterator iter = msgs.find( mapId ) ;
  if( iter == msgs.end() ) // not found
    msgs[ mapId ] = new Message( imsg, Vector2f( defMarginX, getLowestMsg() ), defColor, defLifeTime ) ;
  else
  {
    iter->second->msg = imsg ;
    iter->second->lifeTime = defLifeTime ; //renew
  }
}
void msg( const string& mapId, const string& imsg, float x, float y, float ilifetime )
{
  map<string,Message*>::iterator iter = msgs.find( mapId ) ;
  if( iter == msgs.end() ) // not found
    msgs[ mapId ] = new Message( imsg, Vector2f( x, y ), defColor, ilifetime ) ;
  else
  {
    iter->second->msg = imsg ;
    iter->second->lifeTime = ilifetime ;
    iter->second->manPos=1; //MANUALLY POSITIONED, does not count for getting "lowest" msg
  }
}

void msgDraw()
{
  glDisable( GL_LIGHTING ) ;  
  
  //TEXT
  glMatrixMode( GL_PROJECTION ) ;
  glLoadIdentity();
  glOrtho( 0, w, 0, h, 10, -10 ) ;
  glMatrixMode( GL_MODELVIEW ) ;
  glLoadIdentity() ;
  
  glDisable( GL_DEPTH_TEST ) ;
  
  
  for( map<string,Message*>::iterator iter=msgs.begin(); iter!=msgs.end(); )
  {
    iter->second->print() ;
    //iter->second->debugprint() ;
    if( iter->second->decay( t.getTime() ) <= 0 )
    {
      delete iter->second ;
      iter = msgs.erase( iter ) ;
    }
    else ++iter ;
  }
}

#endif